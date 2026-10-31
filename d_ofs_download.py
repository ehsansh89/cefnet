"""Bulk download NOAA OFS stations.forecast archive and build per-station files.

For each matched (NOAA gauge, OFS model) pair in data/ofs/crosswalk.csv, pull
every stations.forecast.nc cycle in the window, extract the single save-point
slice for that gauge, and append to a per-station NetCDF at
  data/ofs/per_station/{station_id}__{model}.nc
Files are written monthly (one intermediate per month in tmp/) and then
concatenated per station at the end of each model. Monthly files that already
exist are skipped on restart, so the run is resume-safe at the month level.

Two archive sources (--source):
  ncei : NCEI autoindex (canonical, deep archive; reads data/ofs/inventory_ncei.csv).
  s3   : NODD public bucket noaa-nos-ofs-pds (not rate-limited like NCEI; cycles
         discovered live by listing). Required for DBOFS, whose stations.forecast
         product starts 2024-06-22 and is not served by NCEI for that period.

Schema per output file (dims: cycle_time, lead_step):
  vars:     zeta, uwind, vwind   (float32)
  coords:   cycle_time (datetime64[ns]), lead_minutes (int16, 0..N*6)
  attrs:    station_id, ofs_model, ofs_lat, ofs_lon, station_lat, station_lon,
            dist_km, horizon_h, cadence_min, source='NCEI'

Native 6-min cadence is kept. Window per model = max(2021-01-01, archive_start)
→ archive_end. Downloads are parallel (4 concurrent) per model; only one model
runs at a time to keep RAM bounded.
"""

import argparse
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

NCEI_BASE = ("https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-"
             "hydrodynamic-model-systems-co-ops/access/")

# NODD public S3 mirror. Used as an alternative source to NCEI: it is not
# rate-limited the way NCEI is, and for some models (e.g. DBOFS) it carries the
# stations.forecast product that NCEI no longer serves for the relevant period.
S3_BUCKET = "https://noaa-nos-ofs-pds.s3.amazonaws.com/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
# S3 stations.forecast product first appears at these dates (per-model floor).
# NYOFS is intentionally absent: its S3 'nyofs_fg' product carries a different
# 20-station set (not our gauges) and its zeta is all-fill from 2025 on
# (legacy NYOFS decommissioning), so it is not a usable S3 source here.
S3_PRODUCT_FLOOR = {"dbofs": "2024-06-22", "sfbofs": "2024-06-21",
                    "gomofs": "2024-09-08", "wcofs": "2024-07-02"}

MODEL_DIRS = {
    "cbofs":  "chesapeake-bay-operational-forecast-system-cbofs",
    "dbofs":  "delaware-bay-operational-forecast-system-dbofs",
    "gomofs": "gulf-of-maine-operational-forecast-system-gomofs",
    "ngofs2": "northern-gulf-of-america-2-operational-forecast-system-ngofs2",
    "nyofs":  "port-of-new-york-and-new-jersey-operational-forecast-system-nyofs",
    "sfbofs": "san-francisco-bay-operational-forecast-system-sfbofs",
    "tbofs":  "tampa-bay-operational-forecast-system-tbofs",
    "wcofs":  "west-coast-operational-forecast-system-wcofs",
}

# Variable-name normalization: ROMS / FVCOM / POM -> (zeta, uwind, vwind)
VAR_ALIASES = {
    "zeta":  ["zeta", "el"],
    "uwind": ["Uwind", "uwind_speed", "air_u"],
    "vwind": ["Vwind", "vwind_speed", "air_v"],
}

WINDOW_FLOOR = "2021-01-01"
TMP_ROOT = "data/ofs/per_station/_tmp"
OUT_ROOT = "data/ofs/per_station"
DOWNLOAD_WORKERS = 4


def find_var(ds, canonical):
    for cand in VAR_ALIASES[canonical]:
        if cand in ds.variables:
            return cand
    return None


def time_axis(ds):
    for t in ("ocean_time", "time", "Times"):
        if t in ds.variables:
            return t
    raise KeyError(f"no recognized time axis in {list(ds.variables)[:15]}")


def ncei_url_candidates(model, date_str, cycle_str):
    """Return both filename conventions NCEI uses (they coexist in the same folder).

    Modern: nos.{model}.stations.forecast.{date}.{cycle}.nc
    Legacy: {model}.{cycle}.{date}.stations.forecast.nc
    """
    yyyy, mm = date_str[:4], date_str[4:6]
    folder = f"{NCEI_BASE}{MODEL_DIRS[model]}/{yyyy}/{mm}/"
    return [
        f"{folder}nos.{model}.stations.forecast.{date_str}.{cycle_str}.nc",
        f"{folder}{model}.{cycle_str}.{date_str}.stations.forecast.nc",
    ]


def s3_list(prefix, delimiter=None, token=None, retries=3, timeout=60):
    """One ListObjectsV2 page from the NODD bucket. Returns (keys, common_prefixes,
    next_token)."""
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    if delimiter:
        params["delimiter"] = delimiter
    if token:
        params["continuation-token"] = token
    url = S3_BUCKET + "?" + urlencode(params)
    last = None
    for _ in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "conus_ai_ofs_download/1.0"})
            with urlopen(req, timeout=timeout) as r:
                xml = r.read()
            root = ET.fromstring(xml)
            keys = [e.text for e in root.findall(f"{S3_NS}Contents/{S3_NS}Key")]
            pfx = [e.text for e in root.findall(f"{S3_NS}CommonPrefixes/{S3_NS}Prefix")]
            nt = root.find(f"{S3_NS}NextContinuationToken")
            return keys, pfx, (nt.text if nt is not None else None)
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"s3_list failed: {last} url={url}")


def parse_sf_name(name):
    """Pull (date 'YYYYMMDD', cycle int) from a stations.forecast filename, in
    either the legacy ({model}.t{cc}z.{date}.stations.forecast.nc) or modern
    (nos.{model}.stations.forecast.{date}.t{cc}z.nc) convention. None if no match."""
    m = re.search(r"\.t(\d{2})z\.(\d{8})\.stations\.forecast\.nc$", name)
    if m:
        return m.group(2), int(m.group(1))
    m = re.search(r"\.stations\.forecast\.(\d{8})\.t(\d{2})z\.nc$", name)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _collect_sf_under(prefix, found):
    """List stations.forecast keys directly under `prefix` and one CommonPrefixes
    level below it (handles flat YYYYMM/ folders and nested YYYY/MM/DD/ days).
    Mutates `found`: {(date,cycle): key}."""
    contents, subpfx, token = s3_list(prefix, delimiter="/")
    while True:
        for k in contents:
            if k.endswith(".stations.forecast.nc"):
                p = parse_sf_name(k.rsplit("/", 1)[-1])
                if p:
                    found[p] = k
        if not token:
            break
        contents, _, token = s3_list(prefix, delimiter="/", token=token)
    for sp in subpfx:  # one level down (e.g. day folders)
        tok = None
        while True:
            ks, _, tok = s3_list(sp, token=tok)
            for k in ks:
                if k.endswith(".stations.forecast.nc"):
                    p = parse_sf_name(k.rsplit("/", 1)[-1])
                    if p:
                        found[p] = k
            if not tok:
                break


def s3_month_cycles(model, year, month):
    """Discover stations.forecast cycles for (model, year, month) on the NODD S3
    mirror. Handles both the flat `netcdf/YYYYMM/` and nested `netcdf/YYYY/MM/DD/`
    layouts that coexist in the bucket. Returns rows with date/cycle/key."""
    found = {}
    _collect_sf_under(f"{model}/netcdf/{year:04d}{month:02d}/", found)   # flat
    _collect_sf_under(f"{model}/netcdf/{year:04d}/{month:02d}/", found)  # nested
    rows = []
    for (date, cyc), key in found.items():
        rows.append({"date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                     "cycle": f"t{cyc:02d}z", "key": key})
    rows.sort(key=lambda r: (r["date"], r["cycle"]))
    return rows


def download_one(urls, out_path, retries=3, timeout=180):
    """Try each candidate URL with retries; return True on first success."""
    if isinstance(urls, str):
        urls = [urls]
    last = None
    for url in urls:
        for _ in range(retries):
            try:
                req = Request(url, headers={"User-Agent": "conus_ai_ofs_download/1.0"})
                with urlopen(req, timeout=timeout) as r:
                    data = r.read()
                with open(out_path, "wb") as f:
                    f.write(data)
                return True
            except Exception as e:
                last = e
                if "404" in str(e):
                    break  # don't retry a 404, try next candidate
                time.sleep(2.0)
    print(f"    DOWNLOAD-FAIL {urls[0]}: {last}", flush=True)
    return False


# Coordinate-variable precedence MUST match x_ofs_crosswalk.extract_save_points so
# the crosswalk's ofs_idx (an index into the *valid* save-points) lines up with the
# station axis sliced here. GOMOFS masks ~16 of its 126 save-points (fill ~9.97e36);
# without this filter, ofs_idx would point at the wrong physical station.
SAVE_POINT_COORDS = [("lon_rho", "lat_rho"), ("lon", "lat"),
                     ("longitude", "latitude"), ("x", "y")]


def valid_station_mask(ds):
    """Boolean mask over the raw station axis selecting the save-points the crosswalk
    kept (drops fill/invalid lon-lat). None if no recognized coords (then no filtering).
    For models with no masked points this is all-True, i.e. a no-op."""
    for lo, la in SAVE_POINT_COORDS:
        if lo in ds.variables and la in ds.variables:
            lon = np.asarray(ds[lo].values, dtype=float).ravel()
            lat = np.asarray(ds[la].values, dtype=float).ravel()
            return (np.isfinite(lon) & np.isfinite(lat)
                    & (np.abs(lat) <= 90) & (np.abs(lon) <= 360))
    return None


def extract_cycle(nc_path, station_indices):
    """Open cycle file; return zeta, uwind, vwind arrays shaped (lead_step, n_stations)
    and the init time (numpy datetime64), plus horizon_h and cadence_min.
    """
    ds = xr.open_dataset(nc_path, decode_times=False, mask_and_scale=True)
    try:
        tname = time_axis(ds)
        t = np.asarray(ds[tname].values, dtype=float).ravel()
        tu = ds[tname].attrs.get("units", "")
        # Parse the reference date from units. Tolerate the variants OFS models use:
        #   ROMS  "seconds since 2024-07-01 00:00:00"
        #   NYOFS "days since 2008-01-01  0:00:00 00:00"  (1-digit hour, double space, tz)
        m = re.search(r"since\s+(\d{4}-\d{2}-\d{2})(?:[ T]+(\d{1,2}):(\d{2}):(\d{2}))?", tu)
        if not m:
            raise ValueError(f"bad time units '{tu}' in {nc_path}")
        if m.group(2) is None:
            ref = np.datetime64(m.group(1))
        else:
            ref = np.datetime64(f"{m.group(1)}T{int(m.group(2)):02d}:{m.group(3)}:{m.group(4)}")
        if "second" in tu:
            ts = ref + (t * 1e9).astype("timedelta64[ns]")
        elif "day" in tu:
            ts = ref + (t * 86400 * 1e9).astype("timedelta64[ns]")
        else:
            raise ValueError(f"unknown time units '{tu}'")
        cycle_time = ts[0]
        lead_minutes = ((ts - cycle_time).astype("timedelta64[s]").astype(np.int64) // 60).astype(np.int32)

        smask = valid_station_mask(ds)  # align raw station axis with crosswalk ofs_idx
        out = {}
        for canonical in ("zeta", "uwind", "vwind"):
            v = find_var(ds, canonical)
            if v is None:
                out[canonical] = np.full((len(t), len(station_indices)), np.nan, dtype=np.float32)
                continue
            arr = ds[v].values
            # shape is typically (time, station). If station is first, transpose.
            if arr.ndim == 2 and arr.shape[0] != len(t) and arr.shape[1] == len(t):
                arr = arr.T
            if arr.ndim != 2:
                raise ValueError(f"{v} has unexpected ndim={arr.ndim} shape={arr.shape}")
            if smask is not None and arr.shape[1] == smask.size:
                arr = arr[:, smask]  # drop masked save-points before applying ofs_idx
            arr = arr[:, station_indices].astype(np.float32)
            out[canonical] = arr
        return cycle_time, lead_minutes, out
    finally:
        ds.close()


def build_monthly(model, matches_df, year, month, outdir, workers,
                  source="ncei", start=None, end=None):
    """Download all cycles in (year, month), write one xarray Dataset per station to outdir.
    Returns number of cycles successfully processed.

    `source` selects the archive: "ncei" reads the precomputed inventory CSV and
    builds NCEI URLs; "s3" discovers cycles live from the NODD bucket. `start`/`end`
    (pd.Timestamp or None) clip to exact days for partial boundary months."""
    if source == "s3":
        rows = s3_month_cycles(model, year, month)
    else:
        inv = pd.read_csv("data/ofs/inventory_ncei.csv")
        inv = inv[(inv["model"] == model)]
        inv["dt"] = pd.to_datetime(inv["date"])
        inv_m = inv[(inv["dt"].dt.year == year) & (inv["dt"].dt.month == month)].copy()
        rows = inv_m.to_dict("records")
    if start is not None or end is not None:
        lo = pd.Timestamp(start) if start is not None else None
        hi = pd.Timestamp(end) if end is not None else None
        rows = [r for r in rows
                if (lo is None or pd.Timestamp(r["date"]) >= lo)
                and (hi is None or pd.Timestamp(r["date"]) <= hi)]
    if not rows:
        return 0

    station_ids = matches_df["station_id"].astype(str).tolist()
    station_indices = matches_df["ofs_idx"].astype(int).to_numpy()

    # All stations for one (model, month) live in memory until flush
    per_stn = {sid: {"cycle_time": [], "zeta": [], "uwind": [], "vwind": [], "lead_minutes": None}
               for sid in station_ids}

    dl_dir = os.path.join(TMP_ROOT, model, f"{year:04d}{month:02d}_dl")
    os.makedirs(dl_dir, exist_ok=True)

    def download_job(row):
        date_s = row["date"].replace("-", "")
        cycle_s = row["cycle"]
        if source == "s3":
            urls = [S3_BUCKET + row["key"]]
        else:
            urls = ncei_url_candidates(model, date_s, cycle_s)
        fn = os.path.join(dl_dir, f"{date_s}_{cycle_s}.nc")
        if os.path.exists(fn) and os.path.getsize(fn) > 1024:
            return fn
        if download_one(urls, fn):
            return fn
        return None

    ok = 0
    # Download in parallel (IO-bound); extract sequentially (HDF5 not thread-safe).
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download_job, r) for r in rows]
        for fut in as_completed(futures):
            fn = fut.result()
            if fn is None:
                continue
            try:
                cycle_time, lead_minutes, vdict = extract_cycle(fn, station_indices)
            except Exception as e:
                print(f"    EXTRACT-FAIL {fn}: {e}", flush=True)
                try: os.remove(fn)
                except OSError: pass
                continue
            for i, sid in enumerate(station_ids):
                ps = per_stn[sid]
                if ps["lead_minutes"] is None:
                    ps["lead_minutes"] = lead_minutes
                elif len(lead_minutes) != len(ps["lead_minutes"]):
                    # horizon changed; pad / truncate to the longer one
                    n = max(len(lead_minutes), len(ps["lead_minutes"]))
                    # keep the longer baseline; pad new with NaN if shorter
                    if len(lead_minutes) < n:
                        pad = n - len(lead_minutes)
                        for k in ("zeta", "uwind", "vwind"):
                            vdict[k] = np.concatenate([vdict[k], np.full((pad, len(station_indices)), np.nan, dtype=np.float32)], axis=0)
                ps["cycle_time"].append(cycle_time)
                ps["zeta"].append(vdict["zeta"][:, i])
                ps["uwind"].append(vdict["uwind"][:, i])
                ps["vwind"].append(vdict["vwind"][:, i])
            os.remove(fn)
            ok += 1

    # Flush one file per station
    for sid in station_ids:
        ps = per_stn[sid]
        if not ps["cycle_time"]:
            continue
        order = np.argsort(np.asarray(ps["cycle_time"], dtype="datetime64[ns]"))
        ct = np.asarray(ps["cycle_time"], dtype="datetime64[ns]")[order]
        # normalize lead_step length across cycles (pad shorter with NaN)
        max_len = max(arr.shape[0] for arr in ps["zeta"])
        def stack(name):
            out = np.full((len(ct), max_len), np.nan, dtype=np.float32)
            for i_row, idx in enumerate(order):
                arr = ps[name][idx]
                out[i_row, :arr.shape[0]] = arr
            return out
        z, u, v = stack("zeta"), stack("uwind"), stack("vwind")
        lead_minutes = np.arange(max_len, dtype=np.int32) * 6
        meta = matches_df[matches_df["station_id"].astype(str) == sid].iloc[0]
        ds = xr.Dataset(
            data_vars={
                "zeta":  (("cycle_time", "lead_step"), z),
                "uwind": (("cycle_time", "lead_step"), u),
                "vwind": (("cycle_time", "lead_step"), v),
            },
            coords={
                "cycle_time":   ct,
                "lead_minutes": ("lead_step", lead_minutes),
            },
            attrs={
                "station_id":   str(sid),
                "ofs_model":    model,
                "ofs_lat":      float(meta["ofs_lat"]),
                "ofs_lon":      float(meta["ofs_lon"]),
                "station_lat":  float(meta["station_lat"]),
                "station_lon":  float(meta["station_lon"]),
                "dist_km":      float(meta["distance_km"]),
                "cadence_min":  6,
                "horizon_h":    float((max_len - 1) * 6 / 60.0),
                "source":       ("NODD-S3 stations.forecast" if source == "s3"
                                 else "NCEI stations.forecast"),
            },
        )
        out_path = os.path.join(outdir, f"{sid}__{model}__{year:04d}{month:02d}.nc")
        enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ("zeta", "uwind", "vwind")}
        ds.to_netcdf(out_path, encoding=enc)
        ds.close()

    shutil.rmtree(dl_dir, ignore_errors=True)
    return ok


def concat_station(station_id, model, monthly_dir, out_path):
    files = sorted(f for f in os.listdir(monthly_dir)
                   if f.startswith(f"{station_id}__{model}__") and f.endswith(".nc"))
    if not files:
        return False
    dsets = [xr.open_dataset(os.path.join(monthly_dir, f)) for f in files]
    # Align lead_step to the max across months
    max_len = max(ds.sizes["lead_step"] for ds in dsets)
    padded = []
    for ds in dsets:
        if ds.sizes["lead_step"] < max_len:
            ds = ds.reindex(lead_step=np.arange(max_len))
        padded.append(ds)
    merged = xr.concat(padded, dim="cycle_time")
    merged = merged.sortby("cycle_time")
    merged.attrs = dsets[0].attrs  # keep first file's attrs (station meta)
    merged.attrs["horizon_h"] = float((max_len - 1) * 6 / 60.0)
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
           for v in ("zeta", "uwind", "vwind")}
    merged.to_netcdf(out_path, encoding=enc)
    for ds in dsets:
        ds.close()
    merged.close()
    return True


def window_for_model(model, floor=None, ceil=None):
    inv = pd.read_csv("data/ofs/inventory_ncei_summary.csv")
    row = inv[inv["model"] == model].iloc[0]
    start = max(pd.Timestamp(floor or WINDOW_FLOOR), pd.Timestamp(row["archive_start"]))
    end = pd.Timestamp(row["archive_end"])
    if ceil is not None:
        end = min(end, pd.Timestamp(ceil))
    return start, end


def run_model(model, source="ncei", months_limit=None, workers=DOWNLOAD_WORKERS,
              start=None, end=None, exclude_covered_by=None):
    crosswalk = pd.read_csv("data/ofs/crosswalk.csv")
    matches = crosswalk[(crosswalk["ofs_model"] == model) & (crosswalk["matched"])].copy()
    if matches.empty:
        print(f"[{model}] no matched stations, skipping.", flush=True)
        return

    # Drop gauges already covered by a higher-priority (regional) model so a basin-
    # scale model like WCOFS does not store stations SFBOFS already handles. One OFS
    # baseline per gauge: prefer the regional model.
    if exclude_covered_by:
        covered = set(crosswalk[(crosswalk["ofs_model"].isin(exclude_covered_by))
                                & (crosswalk["matched"])]["station_id"].astype(str))
        n_before = len(matches)
        matches = matches[~matches["station_id"].astype(str).isin(covered)].copy()
        print(f"[{model}] excluded {n_before - len(matches)} gauge(s) already covered by "
              f"{','.join(exclude_covered_by)}", flush=True)

    if source == "s3":
        # No reliable summary CSV for S3; bound by the per-model product floor and
        # the requested start/end (default end = today).
        floor = pd.Timestamp(S3_PRODUCT_FLOOR.get(model, "2000-01-01"))
        w_start = max(pd.Timestamp(start), floor) if start else floor
        w_end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    else:
        w_start, w_end = window_for_model(model, floor=start, ceil=end)
    months = pd.date_range(w_start.replace(day=1), w_end.replace(day=1), freq="MS")
    if months_limit:
        months = months[:months_limit]
    print(f"[{model}] src={source}, {len(matches)} matched stations, "
          f"{len(months)} months {w_start.date()} -> {w_end.date()}", flush=True)

    monthly_dir = os.path.join(TMP_ROOT, model, "monthly")
    os.makedirs(monthly_dir, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    t_model = time.time()
    for mdt in months:
        y, m = mdt.year, mdt.month
        # skip months where all station files for this (model, month) already exist
        expected = [f"{sid}__{model}__{y:04d}{m:02d}.nc" for sid in matches["station_id"].astype(str)]
        if all(os.path.exists(os.path.join(monthly_dir, fn)) for fn in expected):
            continue
        t0 = time.time()
        ok = build_monthly(model, matches, y, m, monthly_dir, workers,
                           source=source, start=w_start, end=w_end)
        print(f"  [{model}] {y}-{m:02d}: {ok} cycles, {time.time()-t0:.1f}s", flush=True)

    # Consolidate per-station across all months
    for sid in matches["station_id"].astype(str):
        final = os.path.join(OUT_ROOT, f"{sid}__{model}.nc")
        if os.path.exists(final):
            os.remove(final)
        concat_station(sid, model, monthly_dir, final)
    print(f"[{model}] DONE in {time.time()-t_model:.1f}s", flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_DIRS.keys()) + ["all"], default="all")
    ap.add_argument("--source", choices=["ncei", "s3"], default="ncei",
                    help="Archive source: NCEI (canonical) or the NODD S3 mirror.")
    ap.add_argument("--months", type=int, default=None,
                    help="Limit to first N months (pilot).")
    ap.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD inclusive floor")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD inclusive ceil")
    ap.add_argument("--exclude-covered-by", type=str, default=None,
                    help="Comma-separated OFS models; skip gauges already matched to them "
                         "(e.g. wcofs --exclude-covered-by sfbofs).")
    args = ap.parse_args()

    exclude = ([s.strip() for s in args.exclude_covered_by.split(",")]
               if args.exclude_covered_by else None)
    todo = list(MODEL_DIRS.keys()) if args.model == "all" else [args.model]
    t0 = time.time()
    for m in todo:
        run_model(m, source=args.source, months_limit=args.months, workers=args.workers,
                  start=args.start, end=args.end, exclude_covered_by=exclude)
    print(f"\nAll done in {time.time()-t0:.1f}s", flush=True)
