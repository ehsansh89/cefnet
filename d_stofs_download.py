"""Bulk download STOFS-2D-Global points archive and build per-station NetCDFs.

Reads `data/stofs/inventory.csv` and `data/stofs/crosswalk.csv`, downloads the
points.* products per cycle from `noaa-gestofs-pds`, slices each cycle to the
matched STOFS save-point indices, and accumulates per-station time-series.

Output schema (one file per matched gauge):
  data/stofs/per_station/{station_id}__stofs2d.nc

  dims:     cycle_time (N), lead_step (=1860, native 6-min cadence)
  coords:   cycle_time (datetime64[ns]),
            lead_minutes (int32, -354..+10800)   # ~ -5.9 h nowcast to +180 h forecast
  vars:     cwl            (float32, combined water level, m)
            swl            (float32, sub-tidal/surge residual, m, optional)
            cwl_noanomaly  (float32, raw CWL prior to v2.1.7 bias correction,
                            only populated for cycles on/after 2024-05-14 12Z)
  attrs:    station_id, stofs_idx, stofs_station_name, stofs_lat, stofs_lon,
            station_lat, station_lon, distance_km, match_type,
            cadence_min=6, horizon_h_forecast=180, nowcast_h=6,
            vertical_reference='sea_surface_height_above_geoid',
            source='NODD noaa-gestofs-pds'

Important: STOFS's first time step is at cycle - 354 min (not -360). Downstream
pair-builders MUST select by `lead_minutes` value, never by `lead_step`
positional index, because the 6-min grid is not perfectly aligned to the hour.

Products downloaded per cycle (controlled by CONFIG['products']):
  - 'cwl'           -> stofs_2d_glo.tCCz.points.cwl.nc           (always exists)
  - 'swl'           -> stofs_2d_glo.tCCz.points.swl.nc           (always exists)
  - 'cwl_noanomaly' -> stofs_2d_glo.tCCz.points.cwl.noanomaly.nc (only on/after 2024-05-14 12Z)

Resume-safety is at the month level. For each calendar month inside the run
window, all cycles are processed in memory, then a per-station monthly NC is
written under `data/stofs/per_station/_tmp/monthly/`. A month is skipped on
restart if all expected monthly NCs already exist. After all months finish,
each station's monthly files are concatenated into the final per-station NC.
Memory peak is bounded to one month (~120 cycles x 156 stations x 1860 leads
x 3 vars x float32 ~ 420 MB), regardless of run length.
"""

import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

BUCKET_HOST = "noaa-gestofs-pds.s3.amazonaws.com"
# SCN 23-124 v2.1.7 effective with the 2024-05-14 12Z cycle. The 00Z and 06Z
# cycles on that day are still pre-v2.1.7 (no noanomaly product).
V217_BOUNDARY = pd.Timestamp("2024-05-14T12:00")
OUT_ROOT = "data/stofs/per_station"
TMP_ROOT = "data/stofs/per_station/_tmp"
DL_ROOT = os.path.join(TMP_ROOT, "dl")
MONTHLY_ROOT = os.path.join(TMP_ROOT, "monthly")

CONFIG = {
    # Full pull: 2024-01-01 -> 2025-11-30, matching the AI v1 test split.
    # Pre-v2.1.7 cycles (Jan 1 -> May 14 06Z 2024) have no cwl_noanomaly product;
    # those rows in the final NCs will have cwl_noanomaly = NaN by design.
    "max_cycles":      None,
    # Inclusive [start, end] in YYYY-MM-DD. None = use full inventory.
    "start_date":      "2024-01-01",
    "end_date":        "2025-11-30",
    # Which point products to fetch.
    "products":        ("cwl", "swl"),
    # If True, also fetch cwl_noanomaly when the cycle is post-v2.1.7 12Z.
    "fetch_noanomaly": True,
    # Parallel cycle downloads. HDF5 is not thread-safe, so extraction runs
    # sequentially regardless.
    "download_workers": 4,
    # Keep per-cycle download files in _tmp/dl/ after extraction. Default False
    # (delete) so the full pull does not accumulate ~140-200 GB. Set True
    # locally if you want the network cache for repeated smoke-test reruns.
    "keep_tmp":        False,
}

PRODUCT_FILENAME = {
    "cwl":           "points.cwl.nc",
    "swl":           "points.swl.nc",
    "cwl_noanomaly": "points.cwl.noanomaly.nc",
}


def cycle_url(date_str, cycle, product_filename):
    return (f"https://{BUCKET_HOST}/stofs_2d_glo.{date_str}/"
            f"stofs_2d_glo.{cycle}.{product_filename}")


def http_get(url, out_path, retries=3, timeout=300):
    last = None
    for _ in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "conus_ai_stofs_download/1.0"})
            with urlopen(req, timeout=timeout) as r:
                data = r.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            last = e
            if "404" in str(e):
                return False  # don't retry on 404
            time.sleep(2.0)
    print(f"    DOWNLOAD-FAIL {url}: {last}", flush=True)
    return False


def parse_time_axis(ds):
    """STOFS points file: `time` axis is days/seconds since a reference. Return
    (numpy datetime64[ns] array)."""
    tname = "time" if "time" in ds.variables else ("ocean_time" if "ocean_time" in ds.variables else None)
    if tname is None:
        raise KeyError(f"no recognized time axis in {list(ds.variables)[:15]}")
    t = np.asarray(ds[tname].values, dtype=float).ravel()
    tu = ds[tname].attrs.get("units", "")
    m = re.search(r"since\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)", tu)
    if not m:
        raise ValueError(f"bad time units '{tu}'")
    ref_str = m.group(1).replace(" ", "T")
    if len(ref_str) == 16:  # 'YYYY-MM-DDTHH:MM'
        ref_str += ":00"
    ref = np.datetime64(ref_str)
    if "second" in tu:
        ts = ref + (t * 1e9).astype("timedelta64[ns]")
    elif "minute" in tu:
        ts = ref + (t * 60 * 1e9).astype("timedelta64[ns]")
    elif "hour" in tu:
        ts = ref + (t * 3600 * 1e9).astype("timedelta64[ns]")
    elif "day" in tu:
        ts = ref + (t * 86400 * 1e9).astype("timedelta64[ns]")
    else:
        raise ValueError(f"unknown time units '{tu}'")
    return ts


def extract_zeta(nc_path, station_indices):
    """Return (valid_times[ns], zeta[lead_step, n_stations]).

    STOFS source files declare `_FillValue=-99999.0`, but in some cycles a
    packing artifact lets the sentinel through as ~-99998.9 after mask_and_scale.
    Apply a defensive ±1000 m sentinel mask — no real surge/tide value can
    exceed that, so anything outside the band is provably a fill that survived.
    """
    ds = xr.open_dataset(nc_path, decode_times=False, mask_and_scale=True)
    try:
        ts = parse_time_axis(ds)
        if "zeta" not in ds.variables:
            raise KeyError(f"no 'zeta' var; vars={list(ds.variables)[:15]}")
        arr = ds["zeta"].values.astype(np.float32, copy=True)
        if arr.ndim != 2:
            raise ValueError(f"zeta has unexpected ndim={arr.ndim} shape={arr.shape}")
        if arr.shape[0] != len(ts) and arr.shape[1] == len(ts):
            arr = arr.T
        if arr.shape[0] != len(ts):
            raise ValueError(f"zeta shape {arr.shape} does not match time axis {len(ts)}")
        # Sentinel mask — catches the -99998.9 leak past mask_and_scale.
        arr[np.abs(arr) > 1000.0] = np.nan
        return ts, arr[:, station_indices]
    finally:
        ds.close()


def cycle_init_dt(date_iso, cycle):
    """Cycle init datetime from inventory ('YYYY-MM-DD', 'tCCz') -> np.datetime64[ns]."""
    hh = int(cycle[1:3])
    return np.datetime64(f"{date_iso}T{hh:02d}:00:00")


def is_post_v217(date_iso, cycle):
    hh = int(cycle[1:3])
    return pd.Timestamp(f"{date_iso}T{hh:02d}:00") >= V217_BOUNDARY


def select_cycles(inv, cfg):
    """Apply CONFIG date filters and max_cycles; return ordered cycle rows."""
    inv = inv.copy()
    inv["dt"] = pd.to_datetime(inv["date"])
    if cfg.get("start_date"):
        inv = inv[inv["dt"] >= pd.Timestamp(cfg["start_date"])]
    if cfg.get("end_date"):
        inv = inv[inv["dt"] <= pd.Timestamp(cfg["end_date"])]
    inv = inv.sort_values(["dt", "cycle"]).reset_index(drop=True)
    if cfg.get("max_cycles"):
        inv = inv.head(int(cfg["max_cycles"]))
    return inv


def fetch_cycle_products(date_iso, cycle, products, fetch_noanomaly, workers, dl_dir):
    """Download requested products for one cycle in parallel. Return {prod: local_path}."""
    date_str = date_iso.replace("-", "")
    want = list(products)
    if fetch_noanomaly and "cwl_noanomaly" not in want and is_post_v217(date_iso, cycle):
        want = want + ["cwl_noanomaly"]
    elif "cwl_noanomaly" in want and not is_post_v217(date_iso, cycle):
        want = [p for p in want if p != "cwl_noanomaly"]

    jobs, cached, downloaded = [], {}, {}
    for prod in want:
        fn = os.path.join(dl_dir, f"{date_str}_{cycle}__{prod}.nc")
        if os.path.exists(fn) and os.path.getsize(fn) > 1_000_000:
            cached[prod] = fn
            continue
        jobs.append((prod, fn, cycle_url(date_str, cycle, PRODUCT_FILENAME[prod])))

    downloaded.update(cached)
    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(http_get, url, fn): (prod, fn)
                    for (prod, fn, url) in jobs}
            for fut in as_completed(futs):
                prod, fn = futs[fut]
                if fut.result():
                    downloaded[prod] = fn
                else:
                    print(f"    {date_str} {cycle} {prod}: missing", flush=True)
    return downloaded


def expected_monthly_files(year, month, station_ids):
    return [f"{sid}__stofs2d__{year:04d}{month:02d}.nc" for sid in station_ids]


def build_monthly(year, month, cycles_m, matches, cfg, monthly_dir, dl_dir):
    """Download every cycle in (year, month), extract per-station slices, and
    write one per-station monthly NC under monthly_dir. Returns count of cycles
    successfully processed. Memory peak is bounded to one month.
    """
    station_ids = matches["station_id"].astype(str).tolist()
    station_idx = matches["stofs_idx"].astype(int).to_numpy()

    per = {sid: {"cycle_time": [], "lead_minutes_ref": None,
                 "cwl": [], "swl": [], "cwl_noanomaly": []}
           for sid in station_ids}

    n_ok = 0
    for _, row in cycles_m.iterrows():
        date_iso = row["date"]
        cycle = row["cycle"]
        downloaded = fetch_cycle_products(
            date_iso, cycle, cfg["products"], cfg["fetch_noanomaly"],
            cfg["download_workers"], dl_dir)

        if "cwl" not in downloaded:
            continue

        try:
            ts, cwl_arr = extract_zeta(downloaded["cwl"], station_idx)
        except Exception as e:
            print(f"    EXTRACT-FAIL cwl {date_iso} {cycle}: {e}", flush=True)
            continue
        swl_arr = None
        nan_arr = None
        if "swl" in downloaded:
            try:
                _, swl_arr = extract_zeta(downloaded["swl"], station_idx)
            except Exception as e:
                print(f"    EXTRACT-FAIL swl {date_iso} {cycle}: {e}", flush=True)
        if "cwl_noanomaly" in downloaded:
            try:
                _, nan_arr = extract_zeta(downloaded["cwl_noanomaly"], station_idx)
            except Exception as e:
                print(f"    EXTRACT-FAIL cwl_noanomaly {date_iso} {cycle}: {e}", flush=True)

        cycle_t = cycle_init_dt(date_iso, cycle)
        lead_min = ((ts - cycle_t).astype("timedelta64[s]").astype(np.int64) // 60).astype(np.int32)

        for i, sid in enumerate(station_ids):
            ps = per[sid]
            if ps["lead_minutes_ref"] is None:
                ps["lead_minutes_ref"] = lead_min
            elif len(lead_min) != len(ps["lead_minutes_ref"]):
                n_ref = len(ps["lead_minutes_ref"])
                if len(lead_min) < n_ref:
                    pad = n_ref - len(lead_min)
                    cwl_arr = np.concatenate([cwl_arr, np.full((pad, len(station_idx)), np.nan, dtype=np.float32)], axis=0)
                    if swl_arr is not None:
                        swl_arr = np.concatenate([swl_arr, np.full((pad, len(station_idx)), np.nan, dtype=np.float32)], axis=0)
                    if nan_arr is not None:
                        nan_arr = np.concatenate([nan_arr, np.full((pad, len(station_idx)), np.nan, dtype=np.float32)], axis=0)
                else:
                    cwl_arr = cwl_arr[:n_ref]
                    if swl_arr is not None:
                        swl_arr = swl_arr[:n_ref]
                    if nan_arr is not None:
                        nan_arr = nan_arr[:n_ref]
            ps["cycle_time"].append(cycle_t)
            ps["cwl"].append(cwl_arr[:, i])
            ps["swl"].append(swl_arr[:, i] if swl_arr is not None
                             else np.full(len(ps["lead_minutes_ref"]), np.nan, dtype=np.float32))
            ps["cwl_noanomaly"].append(nan_arr[:, i] if nan_arr is not None
                                       else np.full(len(ps["lead_minutes_ref"]), np.nan, dtype=np.float32))

        if not cfg.get("keep_tmp", False):
            for prod, fn in downloaded.items():
                try: os.remove(fn)
                except OSError: pass

        n_ok += 1
        print(f"  ok  {date_iso.replace('-','')} {cycle}  "
              f"(products: {','.join(downloaded.keys())})", flush=True)

    # Flush one monthly NC per station.
    written = 0
    for _, meta in matches.iterrows():
        sid = str(meta["station_id"])
        ps = per[sid]
        if not ps["cycle_time"]:
            continue
        order = np.argsort(np.asarray(ps["cycle_time"], dtype="datetime64[ns]"))
        ct = np.asarray(ps["cycle_time"], dtype="datetime64[ns]")[order]
        cwl = np.stack([ps["cwl"][i] for i in order]).astype(np.float32)
        swl = np.stack([ps["swl"][i] for i in order]).astype(np.float32)
        nan = np.stack([ps["cwl_noanomaly"][i] for i in order]).astype(np.float32)
        lead_min_ref = ps["lead_minutes_ref"].astype(np.int32)

        ds = xr.Dataset(
            data_vars={
                "cwl":           (("cycle_time", "lead_step"), cwl),
                "swl":           (("cycle_time", "lead_step"), swl),
                "cwl_noanomaly": (("cycle_time", "lead_step"), nan),
            },
            coords={
                "cycle_time":   ct,
                "lead_minutes": ("lead_step", lead_min_ref),
            },
            attrs=_station_attrs(meta),
        )
        out_path = os.path.join(monthly_dir, f"{sid}__stofs2d__{year:04d}{month:02d}.nc")
        enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
               for v in ("cwl", "swl", "cwl_noanomaly")}
        ds.to_netcdf(out_path, encoding=enc)
        ds.close()
        written += 1
    return n_ok, written


def _station_attrs(meta):
    return {
        "station_id":         str(meta["station_id"]),
        "stofs_idx":          int(meta["stofs_idx"]),
        "stofs_station_name": str(meta["stofs_station_name"]),
        "stofs_lat":          float(meta["stofs_lat"]),
        "stofs_lon":          float(meta["stofs_lon"]),
        "station_lat":        float(meta["station_lat"]),
        "station_lon":        float(meta["station_lon"]),
        "distance_km":        float(meta["distance_km"]),
        "match_type":         str(meta["match_type"]),
        "cadence_min":        6,
        "nowcast_h":          6,
        "horizon_h_forecast": 180,
        "vertical_reference": "sea_surface_height_above_geoid",
        "source":             "NODD noaa-gestofs-pds STOFS-2D-Global points.cwl/.swl/.cwl.noanomaly",
    }


def concat_station(sid, monthly_dir, out_path):
    """Concatenate all per-station monthly NCs for `sid` into one final NC."""
    files = sorted(f for f in os.listdir(monthly_dir)
                   if f.startswith(f"{sid}__stofs2d__") and f.endswith(".nc"))
    if not files:
        return False
    dsets = [xr.open_dataset(os.path.join(monthly_dir, f)) for f in files]
    # All months share STOFS's stable 6-min grid (1860 leads). Defensive
    # reindex if any month has a shorter axis (e.g. a partial last cycle).
    max_len = max(ds.sizes["lead_step"] for ds in dsets)
    padded = []
    for ds in dsets:
        if ds.sizes["lead_step"] < max_len:
            ds = ds.reindex(lead_step=np.arange(max_len))
        padded.append(ds)
    merged = xr.concat(padded, dim="cycle_time")
    merged = merged.sortby("cycle_time")
    merged.attrs = dsets[0].attrs
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
           for v in ("cwl", "swl", "cwl_noanomaly")}
    merged.to_netcdf(out_path, encoding=enc)
    for ds in dsets:
        ds.close()
    merged.close()
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data/stofs", exist_ok=True)
    os.makedirs(DL_ROOT, exist_ok=True)
    os.makedirs(MONTHLY_ROOT, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    t0 = time.time()
    inv = pd.read_csv("data/stofs/inventory.csv")
    xw = pd.read_csv("data/stofs/crosswalk.csv")
    matches = xw[xw["matched"]].copy()
    station_ids = matches["station_id"].astype(str).tolist()
    print(f"[{datetime.now():%H:%M:%S}] inventory rows={len(inv)}, "
          f"matched stations={len(matches)}", flush=True)

    cycles = select_cycles(inv, CONFIG)
    if cycles.empty:
        raise SystemExit("No cycles selected by CONFIG; nothing to do.")
    print(f"Cycles selected: {len(cycles)}  "
          f"({cycles['date'].min()} -> {cycles['date'].max()})", flush=True)

    # Process month-by-month with resume-safe skip.
    cycles["dt"] = pd.to_datetime(cycles["date"])
    months = sorted(set((d.year, d.month) for d in cycles["dt"]))
    total_ok = 0
    for (y, m) in months:
        cm = cycles[(cycles["dt"].dt.year == y) & (cycles["dt"].dt.month == m)]
        expected = expected_monthly_files(y, m, station_ids)
        if all(os.path.exists(os.path.join(MONTHLY_ROOT, fn)) for fn in expected):
            print(f"[{y}-{m:02d}] already complete, skipping ({len(cm)} cycles)", flush=True)
            continue
        t_m = time.time()
        print(f"[{y}-{m:02d}] processing {len(cm)} cycles ...", flush=True)
        n_ok, n_written = build_monthly(y, m, cm, matches, CONFIG, MONTHLY_ROOT, DL_ROOT)
        total_ok += n_ok
        print(f"[{y}-{m:02d}] cycles={n_ok}/{len(cm)}, monthly files written={n_written}, "
              f"{time.time()-t_m:.1f}s", flush=True)

    # Final concat per station.
    print(f"\nConcatenating monthly files into {OUT_ROOT}/ ...", flush=True)
    n_final = 0
    for sid in station_ids:
        final = os.path.join(OUT_ROOT, f"{sid}__stofs2d.nc")
        if os.path.exists(final):
            os.remove(final)
        if concat_station(sid, MONTHLY_ROOT, final):
            n_final += 1

    # Clean up the per-cycle download cache unless explicitly kept.
    if not CONFIG.get("keep_tmp", False) and os.path.isdir(DL_ROOT):
        shutil.rmtree(DL_ROOT, ignore_errors=True)
        os.makedirs(DL_ROOT, exist_ok=True)

    print(f"\nWrote {n_final} final per-station files under {OUT_ROOT}", flush=True)
    print(f"Total walltime: {time.time()-t0:.1f}s", flush=True)
    print(f"Total cycles processed (this run): {total_ok}", flush=True)
