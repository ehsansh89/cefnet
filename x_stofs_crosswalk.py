"""NOAA CO-OPS gauge to STOFS-2D-Global save-point crosswalk.

STOFS-2D-Global's `points.cwl.nc` file holds 1,688 monitoring stations, of which
~495 are NOAA CO-OPS gauges with their 7-digit station IDs embedded in the
`station_name` string. We can therefore match the project's gauges to STOFS
save-points by direct ID parse, with a nearest-neighbor fallback for any project
gauge that does not appear by ID.

Outputs:
  data/stofs/samples/stofs2d_sample.nc  (the inspected cycle, cached for reuse)
  data/stofs/crosswalk.csv              (one row per project gauge)
  data/stofs/crosswalk_summary.csv      (1-row summary; coverage stats)

The crosswalk schema mirrors `data/ofs/crosswalk.csv`:
  station_id, station_name, station_lat, station_lon, region,
  stofs_idx, stofs_station_name, stofs_lat, stofs_lon,
  distance_km, match_type ('direct_id' | 'nearest' | 'none'), matched (bool)
"""

import os
import re
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

BUCKET_HOST = "noaa-gestofs-pds.s3.amazonaws.com"
SAMPLE_DATE = "20260511"          # any modern cycle works; this one is verified
SAMPLE_CYCLE = "t00z"
SAMPLE_PRODUCT = "points.cwl.nc"
SAMPLE_URL = (f"https://{BUCKET_HOST}/stofs_2d_glo.{SAMPLE_DATE}/"
              f"stofs_2d_glo.{SAMPLE_CYCLE}.{SAMPLE_PRODUCT}")
SAMPLE_CACHE = "data/stofs/samples/stofs2d_sample.nc"

DISTANCE_THRESHOLD_KM = 5.0      # accept nearest-neighbor up to this far
EARTH_R_KM = 6371.0088
COOPS_ID_RE = re.compile(r"\b(\d{7})\b")    # 7-digit NOAA CO-OPS gauge IDs


def haversine_km(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def download_sample(url, out_path, timeout=300):
    req = Request(url, headers={"User-Agent": "conus_ai_stofs_crosswalk/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    with open(out_path, "wb") as f:
        f.write(data)


def fetch_sample():
    os.makedirs(os.path.dirname(SAMPLE_CACHE), exist_ok=True)
    if os.path.exists(SAMPLE_CACHE) and os.path.getsize(SAMPLE_CACHE) > 1_000_000:
        return SAMPLE_CACHE
    print(f"  downloading sample: {SAMPLE_URL}", flush=True)
    download_sample(SAMPLE_URL, SAMPLE_CACHE)
    return SAMPLE_CACHE


def decode_station_names(arr):
    """`station_name` may be bytes, fixed-width S-dtype, or object array. Return list[str]."""
    out = []
    for s in np.asarray(arr).ravel():
        if isinstance(s, bytes):
            s = s.decode("ascii", errors="replace")
        out.append(str(s).strip())
    return out


def parse_coops_ids(names):
    """For each station_name string, return the first embedded 7-digit ID or None."""
    ids = []
    for n in names:
        m = COOPS_ID_RE.search(n)
        ids.append(m.group(1) if m else None)
    return ids


def load_stofs_points(nc_path):
    ds = xr.open_dataset(nc_path, decode_times=False, mask_and_scale=True)
    try:
        names = decode_station_names(ds["station_name"].values)
        lon = np.asarray(ds["x"].values, dtype=float).ravel()
        lat = np.asarray(ds["y"].values, dtype=float).ravel()
    finally:
        ds.close()
    # Normalize longitude to [-180, 180] in case any points are stored 0-360.
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return names, lon, lat


def build_crosswalk(gauges, names, lon, lat, threshold_km):
    coops_ids = parse_coops_ids(names)
    id_to_idx = {}
    for i, sid in enumerate(coops_ids):
        if sid is None:
            continue
        # If a CO-OPS ID appears at multiple STOFS save-points (rare), keep the first.
        id_to_idx.setdefault(sid, i)

    rows = []
    for _, g in gauges.iterrows():
        sid = str(g["station_id"])
        if sid in id_to_idx:
            idx = id_to_idx[sid]
            d = float(haversine_km(g["lat"], g["lon"], lat[idx], lon[idx]))
            rows.append({
                "station_id":         sid,
                "station_name":       g.get("name", ""),
                "station_lat":        float(g["lat"]),
                "station_lon":        float(g["lon"]),
                "region":             int(g["region"]) if pd.notna(g.get("region")) else -1,
                "stofs_idx":          int(idx),
                "stofs_station_name": names[idx],
                "stofs_lat":          float(lat[idx]),
                "stofs_lon":          float(lon[idx]),
                "distance_km":        round(d, 3),
                "match_type":         "direct_id",
                "matched":            True,
            })
        else:
            d = haversine_km(g["lat"], g["lon"], lat, lon)
            j = int(np.argmin(d))
            dist = float(d[j])
            rows.append({
                "station_id":         sid,
                "station_name":       g.get("name", ""),
                "station_lat":        float(g["lat"]),
                "station_lon":        float(g["lon"]),
                "region":             int(g["region"]) if pd.notna(g.get("region")) else -1,
                "stofs_idx":          j,
                "stofs_station_name": names[j],
                "stofs_lat":          float(lat[j]),
                "stofs_lon":          float(lon[j]),
                "distance_km":        round(dist, 3),
                "match_type":         "nearest" if dist <= threshold_km else "none",
                "matched":            bool(dist <= threshold_km),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data/stofs", exist_ok=True)

    t0 = time.time()
    gauges = pd.read_csv("data/stations_processed/stations_processed.csv")
    print(f"Loaded {len(gauges)} NOAA gauges from stations_processed.csv", flush=True)

    fn = fetch_sample()
    size_mb = os.path.getsize(fn) / 1048576
    print(f"Sample file: {fn}  ({size_mb:.1f} MB)", flush=True)

    names, lon, lat = load_stofs_points(fn)
    coops_ids = parse_coops_ids(names)
    n_unique_coops = len(set(i for i in coops_ids if i is not None))
    print(f"STOFS save-points: {len(names)} total, "
          f"{sum(1 for i in coops_ids if i)} with embedded CO-OPS IDs "
          f"({n_unique_coops} unique)", flush=True)

    xw = build_crosswalk(gauges, names, lon, lat, DISTANCE_THRESHOLD_KM)
    xw.to_csv("data/stofs/crosswalk.csv", index=False)

    n_direct  = int((xw["match_type"] == "direct_id").sum())
    n_nearest = int((xw["match_type"] == "nearest").sum())
    n_none    = int((xw["match_type"] == "none").sum())
    n_matched = int(xw["matched"].sum())
    mean_d_direct  = xw.loc[xw["match_type"] == "direct_id", "distance_km"].mean()
    mean_d_nearest = xw.loc[xw["match_type"] == "nearest",   "distance_km"].mean()

    summary = pd.DataFrame([{
        "model":                "stofs2d_global",
        "sample_url":           SAMPLE_URL,
        "sample_size_mb":       round(size_mb, 2),
        "stofs_total_points":   len(names),
        "stofs_coops_points":   n_unique_coops,
        "project_gauges":       len(gauges),
        "matched_direct_id":    n_direct,
        "matched_nearest_5km":  n_nearest,
        "unmatched":            n_none,
        "match_rate_pct":       round(100.0 * n_matched / max(1, len(gauges)), 1),
        "mean_dist_direct_km":  round(float(mean_d_direct), 3) if n_direct else None,
        "mean_dist_nearest_km": round(float(mean_d_nearest), 3) if n_nearest else None,
        "distance_threshold_km": DISTANCE_THRESHOLD_KM,
    }])
    summary.to_csv("data/stofs/crosswalk_summary.csv", index=False)

    print(f"\nMatched: direct_id={n_direct}, nearest<=5km={n_nearest}, unmatched={n_none}")
    print(f"Overall match rate: {summary.iloc[0]['match_rate_pct']}%")
    print(f"\n=== Crosswalk summary ===")
    print(summary.to_string(index=False))

    if n_none > 0:
        print(f"\n=== Unmatched gauges ({n_none}) and nearest STOFS points ===")
        print(xw[xw["match_type"] == "none"]
              [["station_id", "region", "station_name", "stofs_station_name", "distance_km"]]
              .to_string(index=False))

    print(f"\nElapsed {time.time()-t0:.1f}s")
