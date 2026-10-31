"""
d_atmospheric_forecast_download.py — Download archived GFS forecast atmosphere
to per-region monthly zarr.

For each (init_time, lead_hour) in the inventory manifest, HTTP-range-fetch
the 3 GRIB messages (U10, V10, PRES:surface) from the public NOAA GFS S3
mirror, decode them with cfgrib **once per cycle**, then fan out: crop +
bilinear-interpolate to each pending region's 64×64 LR grid, convert pressure
to hPa, cast to float16, and write per-region monthly zarrs at:

    {OUT_BASE_DIR}/{region_name}/YYYYMM.zarr

Indexed by (init_time, lead_hour, y, x). See `docs/note_gfs_download_plan.md`
§3.2/§4 for the schema and §6.2 for the rationale behind the multi-region
fan-out (each global GRIB message is ~1 MiB; pulling per-region duplicated
the same byte ranges N times).

Resumable: zarrs that already exist on disk for a given (region, month) are
skipped when `OVERWRITE=false`. The manifest's `status` column flips to
'downloaded' after each month's zarrs land.

INIT_TIME_FILTER lets you run a single-cycle Gate-1 sanity test against the
same code path as the bulk pull.
"""

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

# --- Windows DLL bootstrap for eccodes ---------------------------------------
# When this script is launched outside an activated conda env (e.g. directly
# via `python.exe`), the env's `Library\bin` is not on PATH and eccodes can't
# find its DLLs. Fix that before any cfgrib/eccodes import.
if os.name == 'nt':
    _libbin = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin')
    if os.path.isdir(_libbin):
        try:
            os.add_dll_directory(_libbin)
        except (AttributeError, OSError):
            pass
        os.environ['PATH'] = _libbin + os.pathsep + os.environ.get('PATH', '')

import numpy as np
import pandas as pd
import requests
import xarray as xr
import zarr
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


CONFIG_PATH = 'd_atmospheric_forecast_download_config.json'

# cfgrib variable shortNames in GFS .pgrb2.0p25 messages.
GFS_U10_NAME = 'u10'
GFS_V10_NAME = 'v10'
GFS_PRES_NAME = 'sp'   # PRES:surface


def make_session(pool_size, retries):
    """HTTP session with keep-alive + retry on 5xx / 429."""
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=0.4,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=('GET', 'HEAD'))
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size,
                          max_retries=retry)
    s.mount('https://', adapter); s.mount('http://', adapter)
    return s


def fetch_lead_bytes(session, grib_url, ranges, timeout,
                     max_attempts=10, max_backoff=60.0):
    """Fetch the configured byte ranges from one GFS GRIB file and concatenate.

    `ranges` is a list of (byte_start, byte_end) inclusive. byte_end == -1
    means open-ended ("to end of file"). Returns a single bytes blob whose
    chunks are concatenated in the order given.

    Retries on transient network errors — covers mid-stream read timeouts
    (urllib3 Retry adapter doesn't recover from chunked-body interruptions),
    dropped connections, and brief DNS outages on the local side. Default
    schedule of 10 attempts with backoff capped at 60 s gives ~5 min of
    total tolerance per byte range.
    """
    parts = []
    for bs, be in ranges:
        hdr = f"bytes={bs}-" if be < 0 else f"bytes={bs}-{be}"
        last_err = None
        for attempt in range(max_attempts):
            try:
                r = session.get(grib_url, headers={'Range': hdr},
                                timeout=timeout)
                if r.status_code not in (200, 206):
                    raise RuntimeError(
                        f"HTTP {r.status_code} for {grib_url} {hdr}")
                blob = r.content   # may raise mid-stream
                parts.append(blob)
                break
            except (requests.exceptions.RequestException,
                    RuntimeError) as e:
                last_err = e
                if attempt + 1 < max_attempts:
                    time.sleep(min(2 ** attempt, max_backoff))
        else:
            raise RuntimeError(
                f"Failed after {max_attempts} attempts for {grib_url} "
                f"{hdr}: {last_err}")
    return b''.join(parts)


def fetch_cycle_bytes(session, cycle_rows, max_workers, timeout):
    """Parallel-fetch all (lead × var) byte chunks for one cycle.

    Returns a dict `{lead_hour: bytes_blob}` where each blob holds the cycle's
    GRIB messages for that lead in the order (PRES, UGRD, VGRD) — but order
    doesn't matter to cfgrib, only that each blob is a valid sequence of
    self-contained GRIB2 messages.
    """
    by_lead = {}
    for lead, sub in cycle_rows.groupby('lead_hour'):
        sub = sub.sort_values('variable')   # alphabetical: PRES, UGRD, VGRD
        ranges = list(zip(sub['byte_start'].astype(int).tolist(),
                          sub['byte_end'].astype(int).tolist()))
        url = sub['grib_url'].iloc[0]
        by_lead[int(lead)] = (url, ranges)

    out = {}
    def _worker(lead):
        url, ranges = by_lead[lead]
        return lead, fetch_lead_bytes(session, url, ranges, timeout)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker, l) for l in by_lead]
        for fut in as_completed(futs):
            lead, blob = fut.result()
            out[lead] = blob
    return out


def decode_cycle_to_global(blob_by_lead):
    """Decode a cycle's concatenated GRIB messages into global arrays.

    Returns:
        leads     : list[int]   sorted ascending
        u, v      : np.ndarray  (n_lead, ny_global, nx_global) float32
        p         : np.ndarray  (n_lead, ny_global, nx_global) float32, Pa
        src_lat   : np.ndarray  (ny_global,) float32, decreasing N→S
        src_lon   : np.ndarray  (nx_global,) float32, in 0..360

    The byte payload represents the full global GFS field for each lead;
    cropping to a region bbox is a separate cheap step
    (`crop_global_to_region`) so a single decode can fan out to many
    regions.
    """
    import cfgrib

    leads = sorted(blob_by_lead.keys())
    big = b''.join(blob_by_lead[l] for l in leads)

    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tf:
        tf.write(big); tf.flush()
        tmp_path = tf.name
    try:
        # backend_kwargs avoids polluting tempdir with .idx sidecars.
        ds_list = cfgrib.open_datasets(
            tmp_path, backend_kwargs={'indexpath': ''})
        # We expect 2 datasets: typeOfLevel='heightAboveGround' (u10, v10) and
        # 'surface' (sp). Walk both and harvest by shortName.
        u_full = v_full = p_full = None
        src_lat = src_lon = None
        for ds in ds_list:
            if 'latitude' in ds.coords and src_lat is None:
                src_lat = ds['latitude'].values
            if 'longitude' in ds.coords and src_lon is None:
                src_lon = ds['longitude'].values
            if GFS_U10_NAME in ds: u_full = ds[GFS_U10_NAME].values
            if GFS_V10_NAME in ds: v_full = ds[GFS_V10_NAME].values
            if GFS_PRES_NAME in ds: p_full = ds[GFS_PRES_NAME].values
            ds.close()
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    if u_full is None or v_full is None or p_full is None:
        raise RuntimeError("Decoded GRIB missing one of U10 / V10 / PRES:surface")

    src_lat = np.asarray(src_lat, dtype=np.float32)
    src_lon = np.asarray(src_lon, dtype=np.float32)
    return leads, u_full, v_full, p_full, src_lat, src_lon


def crop_global_to_region(u_full, v_full, p_full, src_lat, src_lon, bbox):
    """Crop global decoded arrays to one region's bbox.

    `bbox = (lat_min, lat_max, lon_min, lon_max)` in -180..180. Returns
    regional subsets plus lat/lon coords in the region grid convention
    (lat decreasing N→S, lon in -180..180, monotonic ascending).

    Lon mask is built in src's native 0..360 frame so we never materialize
    a reordered global array. Inputs are NOT mutated — same global decode
    can be cropped to many regions.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_mask = (src_lat >= lat_min) & (src_lat <= lat_max)
    lo_360 = lon_min % 360.0
    hi_360 = lon_max % 360.0
    if lo_360 <= hi_360:
        lon_mask = (src_lon >= lo_360) & (src_lon <= hi_360)
        wraps = False
    else:
        lon_mask = (src_lon >= lo_360) | (src_lon <= hi_360)
        wraps = True

    if not lat_mask.any() or not lon_mask.any():
        raise RuntimeError(f"Region bbox {bbox} has no overlap with GFS grid")

    sub_lat = src_lat[lat_mask]
    sub_lon = src_lon[lon_mask]
    u_sub = u_full[:, lat_mask][:, :, lon_mask]
    v_sub = v_full[:, lat_mask][:, :, lon_mask]
    p_sub = p_full[:, lat_mask][:, :, lon_mask]

    sub_lon = np.where(sub_lon > 180.0,
                       sub_lon - 360.0, sub_lon).astype(np.float32)
    if wraps:
        order = np.argsort(sub_lon)
        sub_lon = sub_lon[order]
        u_sub = u_sub[..., order]
        v_sub = v_sub[..., order]
        p_sub = p_sub[..., order]

    return (u_sub.astype(np.float32), v_sub.astype(np.float32),
            p_sub.astype(np.float32), sub_lat, sub_lon)


def interp_to_region(arr_3d, src_lats, src_lons, dst_lats, dst_lons):
    """Bilinear interp from rectilinear (lat, lon) source to curvilinear
    (y, x) target. `src_lats` decreasing; xr.interp handles that.
    Returns (n_lead, ny_dst, nx_dst) float32.
    """
    da = xr.DataArray(
        arr_3d, dims=('lead', 'lat', 'lon'),
        coords={'lat': src_lats, 'lon': src_lons},
    )
    out = da.interp(
        lat=xr.DataArray(dst_lats, dims=('y', 'x')),
        lon=xr.DataArray(dst_lons, dims=('y', 'x')),
        method='linear',
    )
    return out.values.astype(np.float32)


def write_monthly_zarr(out_path, init_times, leads, dst_lats, dst_lons,
                        u, v, p, region_name):
    """Persist (n_init, n_lead, y, x) tensors as a monthly zarr.

    Schema follows docs/note_gfs_download_plan.md §4. Pressure is in hPa,
    fields are float16, valid_time = init_time + lead_hour.
    """
    init_arr = np.asarray(init_times, dtype='datetime64[ns]')
    lead_arr = np.asarray(leads, dtype=np.int8)
    valid_time = (
        init_arr.astype('datetime64[h]')[:, None]
        + lead_arr.astype('timedelta64[h]')[None, :]
    ).astype('datetime64[ns]')

    ds = xr.Dataset(
        data_vars=dict(
            U10=(('init_time', 'lead_hour', 'y', 'x'), u.astype(np.float16)),
            V10=(('init_time', 'lead_hour', 'y', 'x'), v.astype(np.float16)),
            PSFC=(('init_time', 'lead_hour', 'y', 'x'), p.astype(np.float16)),
        ),
        coords=dict(
            init_time=init_arr,
            lead_hour=lead_arr,
            valid_time=(('init_time', 'lead_hour'), valid_time),
            lat=(('y', 'x'), dst_lats.astype(np.float32)),
            lon=(('y', 'x'), dst_lons.astype(np.float32)),
        ),
        attrs=dict(
            source='NOAA GFS',
            source_bucket='s3://noaa-gfs-bdp-pds',
            grid=f'{region_name}_lr_64x64',
            grid_resolution_km=16,
            lead_hour_max=int(lead_arr.max()),
            pressure_variable='PRES:surface',
            pressure_units='hPa',
            wind_units='m s-1',
            interpolation='bilinear to region lr grid',
            created_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        ),
    )
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = {v: {'compressor': compressor} for v in ('U10', 'V10', 'PSFC')}
    if os.path.exists(out_path):
        import shutil; shutil.rmtree(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_zarr(out_path, mode='w', encoding=encoding, consolidated=True)


def load_month_manifest(inventory_dir, month_str):
    path = os.path.join(inventory_dir, f"{month_str}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Inventory partition not found: {path}")
    return path, pd.read_parquet(path)


def update_manifest_status(parquet_path, df_full, processed_keys):
    """Mark processed (init_time, lead_hour, variable) rows as 'downloaded'."""
    if not processed_keys:
        return
    keys_set = set(processed_keys)
    mask = df_full.apply(
        lambda r: (r['init_time'], int(r['lead_hour']), r['variable']) in keys_set,
        axis=1,
    )
    df_full.loc[mask, 'status'] = 'downloaded'
    df_full.to_parquet(parquet_path, index=False)


def _load_region_specs(cfg):
    """Resolve `cfg['REGIONS']` to a list of dicts with grid + bbox info."""
    buf = float(cfg.get('BBOX_BUFFER_DEG', 1.0))
    grid_dir = cfg.get('REGION_GRID_DIR', 'GIS/regions')
    out = []
    for name in cfg['REGIONS']:
        npz = os.path.join(grid_dir, name, f'{name}_lr_latlon.npz')
        grid = np.load(npz)
        lats, lons = grid['lats'], grid['lons']
        bbox = (float(lats.min()) - buf, float(lats.max()) + buf,
                float(lons.min()) - buf, float(lons.max()) + buf)
        out.append({'name': name, 'lats': lats, 'lons': lons, 'bbox': bbox})
    return out


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)

    region_specs = _load_region_specs(cfg)
    print(f"[gfs-download] regions={[r['name'] for r in region_specs]}")
    for r in region_specs:
        print(f"  {r['name']}  grid={r['lats'].shape}  bbox={r['bbox']}")

    init_filter = cfg.get('INIT_TIME_FILTER')
    init_filter = ([pd.Timestamp(t) for t in init_filter]
                   if init_filter else None)

    lead_min = int(cfg['LEAD_MIN'])
    lead_max = int(cfg['LEAD_MAX'])
    leads_all = list(range(lead_min, lead_max + 1))
    n_lead = len(leads_all)

    session = make_session(
        pool_size=int(cfg['DOWNLOAD_WORKERS']),
        retries=int(cfg.get('HTTP_RETRIES', 3)),
    )
    timeout = int(cfg.get('REQUEST_TIMEOUT', 60))
    workers = int(cfg['DOWNLOAD_WORKERS'])
    overwrite = bool(cfg.get('OVERWRITE', False))

    for r in region_specs:
        os.makedirs(os.path.join(cfg['OUT_BASE_DIR'], r['name']), exist_ok=True)

    for month_str in cfg['MONTHS']:
        # Skip months where every pending region's zarr already exists.
        pending = [r for r in region_specs
                   if overwrite or not os.path.exists(
                       os.path.join(cfg['OUT_BASE_DIR'], r['name'],
                                    f"{month_str}.zarr"))]
        if not pending:
            print(f"[skip] {month_str}: all region zarrs exist")
            continue

        parquet_path, df_full = load_month_manifest(
            cfg['INVENTORY_DIR'], month_str)
        df = df_full[(df_full['lead_hour'] >= lead_min)
                      & (df_full['lead_hour'] <= lead_max)
                      & (df_full['status'].isin(('planned', 'downloaded')))]
        if init_filter is not None:
            df = df[df['init_time'].isin(init_filter)]

        cycles = sorted(df['init_time'].unique())
        if not cycles:
            print(f"[skip] {month_str}: no rows match the filter")
            continue
        n_init = len(cycles)

        # One accumulator per pending region.
        accums = {
            r['name']: {
                'U': np.empty((n_init, n_lead, *r['lats'].shape), dtype=np.float32),
                'V': np.empty((n_init, n_lead, *r['lats'].shape), dtype=np.float32),
                'P': np.empty((n_init, n_lead, *r['lats'].shape), dtype=np.float32),
            } for r in pending
        }

        processed_keys = []
        print(f"[gfs-download] {month_str}: {n_init} cycles x {n_lead} leads "
              f"x 3 vars = {n_init * n_lead * 3:,} GRIB messages  "
              f"-> {len(pending)} region(s): "
              f"{', '.join(r['name'] for r in pending)}")

        for i, ts in enumerate(tqdm(cycles, desc=month_str)):
            cyc_rows = df[df['init_time'] == ts]

            t0 = time.time()
            blob_by_lead = fetch_cycle_bytes(session, cyc_rows, workers, timeout)
            t_fetch = time.time() - t0

            t0 = time.time()
            leads_got, u_g, v_g, p_g, src_lat, src_lon = \
                decode_cycle_to_global(blob_by_lead)
            t_decode = time.time() - t0
            lead_idx = {l: k for k, l in enumerate(leads_got)}

            # Fan out: one decode → many regional crops + interpolations.
            t0 = time.time()
            for r in pending:
                u_sub, v_sub, p_sub, slat, slon = crop_global_to_region(
                    u_g, v_g, p_g, src_lat, src_lon, r['bbox'])
                u_r = interp_to_region(u_sub, slat, slon, r['lats'], r['lons'])
                v_r = interp_to_region(v_sub, slat, slon, r['lats'], r['lons'])
                p_r = interp_to_region(p_sub, slat, slon, r['lats'], r['lons']) / 100.0
                acc = accums[r['name']]
                for k, l in enumerate(leads_all):
                    if l not in lead_idx:
                        acc['U'][i, k] = np.nan
                        acc['V'][i, k] = np.nan
                        acc['P'][i, k] = np.nan
                        continue
                    src_k = lead_idx[l]
                    acc['U'][i, k] = u_r[src_k]
                    acc['V'][i, k] = v_r[src_k]
                    acc['P'][i, k] = p_r[src_k]
            t_interp = time.time() - t0

            del u_g, v_g, p_g

            for _, row in cyc_rows.iterrows():
                processed_keys.append(
                    (row['init_time'], int(row['lead_hour']), row['variable']))

            sample = pending[0]['name']
            sample_acc = accums[sample]
            tqdm.write(
                f"  {ts}  fetch={t_fetch:.1f}s  decode={t_decode:.1f}s  "
                f"interp×{len(pending)}={t_interp:.1f}s  "
                f"[{sample}] |U|max="
                f"{np.nanmax(np.abs(sample_acc['U'][i])):.1f}  "
                f"PSFC[{np.nanmin(sample_acc['P'][i]):.1f},"
                f"{np.nanmax(sample_acc['P'][i]):.1f}] hPa"
            )

        # Write per-region zarrs at end of month.
        for r in pending:
            acc = accums[r['name']]
            zp = os.path.join(cfg['OUT_BASE_DIR'], r['name'],
                              f"{month_str}.zarr")
            write_monthly_zarr(zp, cycles, leads_all, r['lats'], r['lons'],
                               acc['U'], acc['V'], acc['P'], r['name'])
            print(f"[done] {month_str}  {r['name']}  zarr -> {zp}")

        update_manifest_status(parquet_path, df_full, processed_keys)
        print(f"[done] {month_str}  manifest updated -> {parquet_path}")

    print("[gfs-download] complete.")
