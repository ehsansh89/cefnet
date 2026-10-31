"""NOAA OFS save-point to NOAA CO-OPS gauge crosswalk.

For each priority OFS model, download one stations.forecast.nc sample from NCEI,
extract the (lon, lat) save-point coordinates, and nearest-neighbor match each
NOAA gauge to the closest OFS save-point. Output: per-gauge crosswalk and
per-model coverage summary. Gates the whole OFS-comparison track.

NGOFS2 stations.forecast confirmed on NCEI from 2021-03-22 onward (was missing
on the S3 NODD bucket). All 8 priority models are now inventoried uniformly.
"""

import os
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

NCEI_BASE = ("https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-"
             "hydrodynamic-model-systems-co-ops/access/")

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

DISTANCE_THRESHOLD_KM = 5.0
EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def pick_ncei_sample_url(model, inventory_csv="data/ofs/inventory_ncei.csv"):
    """Pick the first (date, cycle) for `model` from the NCEI inventory CSV,
    return the full NCEI URL to that stations.forecast.nc file."""
    inv = pd.read_csv(inventory_csv)
    inv = inv[inv["model"] == model]
    if inv.empty:
        raise RuntimeError(f"no inventory rows for {model}")
    row = inv.iloc[0]
    date = str(row["date"]).replace("-", "")
    cycle = str(row["cycle"])
    yyyy, mm = date[:4], date[4:6]
    long_name = MODEL_DIRS[model]
    return (f"{NCEI_BASE}{long_name}/{yyyy}/{mm}/"
            f"nos.{model}.stations.forecast.{date}.{cycle}.nc")


def download_sample(url, out_path):
    req = Request(url, headers={"User-Agent": "conus_ai_ofs_crosswalk/1.0"})
    with urlopen(req, timeout=120) as r:
        data = r.read()
    with open(out_path, "wb") as f:
        f.write(data)


def fetch_sample(model, cache_dir="data/ofs/samples"):
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"{model}_sample.nc")
    if os.path.exists(fn) and os.path.getsize(fn) > 1024:
        return fn
    url = pick_ncei_sample_url(model)
    print(f"  downloading sample for {model}: {url}", flush=True)
    download_sample(url, fn)
    return fn


def extract_save_points(nc_file):
    """Return (lon, lat) arrays for save-points.

    Normalizes longitude to [-180, 180] (SFBOFS / WCOFS store 0-360 on the ROMS
    rho grid). Drops fill-valued points (GOMOFS has masked save-points with
    values ~9.97e36). Both are required for the gauge bbox filter to work.
    """
    ds = xr.open_dataset(nc_file, decode_times=False, mask_and_scale=True)
    try:
        for lo_name, la_name in [
            ("lon_rho", "lat_rho"),      # ROMS (CBOFS, SFBOFS, GOMOFS, WCOFS, DBOFS, TBOFS)
            ("lon", "lat"),              # FVCOM / generic
            ("longitude", "latitude"),   # POM / NYOFS style
            ("x", "y"),
        ]:
            if lo_name in ds.variables and la_name in ds.variables:
                lon = np.asarray(ds[lo_name].values, dtype=float).ravel()
                lat = np.asarray(ds[la_name].values, dtype=float).ravel()
                valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
                lon, lat = lon[valid], lat[valid]
                lon = np.where(lon > 180, lon - 360, lon)
                if lon.size == 0:
                    raise ValueError(f"all save-points masked/invalid in {nc_file}")
                return lon, lat
        raise KeyError(f"no recognized lon/lat vars in {nc_file}; vars={list(ds.variables)[:20]}")
    finally:
        ds.close()


def match_gauges_to_points(gauges, lon, lat, threshold_km):
    """For each gauge row, compute nearest save-point and match flag."""
    rows = []
    for _, g in gauges.iterrows():
        d = haversine_km(g["lat"], g["lon"], lat, lon)
        idx = int(np.argmin(d))
        dist = float(d[idx])
        rows.append({
            "station_id": g["station_id"],
            "station_name": g.get("name", ""),
            "station_lat": float(g["lat"]),
            "station_lon": float(g["lon"]),
            "region": int(g["region"]) if pd.notna(g.get("region")) else -1,
            "ofs_idx": idx,
            "ofs_lat": float(lat[idx]),
            "ofs_lon": float(lon[idx]),
            "distance_km": round(dist, 3),
            "matched": bool(dist <= threshold_km),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    gauges = pd.read_csv("data/stations_processed/stations_processed.csv")
    print(f"Loaded {len(gauges)} NOAA gauges from stations_processed.csv", flush=True)

    os.makedirs("data/ofs", exist_ok=True)
    results = []
    summary = []

    for model in MODEL_DIRS.keys():
        print(f"\n=== {model} ===", flush=True)
        t0 = time.time()
        try:
            fn = fetch_sample(model)
            lon, lat = extract_save_points(fn)
            size_mb = os.path.getsize(fn) / 1048576
            print(f"  sample: {os.path.basename(fn)} ({size_mb:.1f} MB); save-points: {len(lon)}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            summary.append({
                "ofs_model": model, "sample_size_mb": None, "save_points": 0,
                "gauges_in_bbox": 0, "gauges_matched_5km": 0, "match_rate_pct": 0.0,
                "mean_dist_matched_km": None, "error": str(e)[:200],
            })
            continue

        lon_min, lon_max = float(lon.min()) - 0.5, float(lon.max()) + 0.5
        lat_min, lat_max = float(lat.min()) - 0.5, float(lat.max()) + 0.5
        in_box = gauges[
            gauges["lat"].between(lat_min, lat_max)
            & gauges["lon"].between(lon_min, lon_max)
        ].copy()
        print(f"  bbox lat=[{lat_min:.2f},{lat_max:.2f}] lon=[{lon_min:.2f},{lon_max:.2f}]; "
              f"gauges in bbox: {len(in_box)}", flush=True)

        if len(in_box) == 0:
            summary.append({
                "ofs_model": model, "sample_size_mb": round(size_mb, 2),
                "save_points": len(lon), "gauges_in_bbox": 0,
                "gauges_matched_5km": 0, "match_rate_pct": 0.0,
                "mean_dist_matched_km": None, "error": "",
            })
            continue

        matches = match_gauges_to_points(in_box, lon, lat, DISTANCE_THRESHOLD_KM)
        matches.insert(0, "ofs_model", model)
        results.append(matches)

        n_match = int(matches["matched"].sum())
        rate = round(100.0 * n_match / max(1, len(in_box)), 1)
        mean_d = round(float(matches.loc[matches["matched"], "distance_km"].mean()), 2) if n_match else None
        print(f"  matched (<= {DISTANCE_THRESHOLD_KM} km): {n_match}/{len(in_box)} ({rate}%)"
              f"{f'; mean dist {mean_d} km' if mean_d is not None else ''}", flush=True)
        summary.append({
            "ofs_model": model, "sample_size_mb": round(size_mb, 2),
            "save_points": len(lon), "gauges_in_bbox": int(len(in_box)),
            "gauges_matched_5km": n_match, "match_rate_pct": rate,
            "mean_dist_matched_km": mean_d, "error": "",
        })
        print(f"  elapsed {time.time() - t0:.1f}s", flush=True)

    if results:
        all_matches = pd.concat(results, ignore_index=True)
        all_matches.to_csv("data/ofs/crosswalk.csv", index=False)
    else:
        all_matches = pd.DataFrame()
    pd.DataFrame(summary).to_csv("data/ofs/crosswalk_summary.csv", index=False)

    print("\n=== Crosswalk summary ===")
    print(pd.DataFrame(summary).to_string(index=False))

    if not all_matches.empty:
        print("\n=== Region x model matched-gauge count (<= 5 km) ===")
        piv = (all_matches[all_matches["matched"]]
               .pivot_table(index="region", columns="ofs_model",
                            values="station_id", aggfunc="count", fill_value=0))
        print(piv.to_string())
