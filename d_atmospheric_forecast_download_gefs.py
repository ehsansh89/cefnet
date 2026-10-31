"""
d_atmospheric_forecast_download_gefs.py — Download archived GEFS forecast
atmosphere (31 members) to per-region monthly zarr.

For each (init_time, member, lead_hour) in the inventory manifest,
HTTP-range-fetch the 4 GRIB messages (U10, V10, PRES:surface,
PRMSL:mean sea level) from the public NOAA GEFS S3 mirror, decode them
with cfgrib **once per cycle-member**, then fan out: crop +
bilinear-interpolate to each pending region's 64×64 LR grid, convert
pressure to hPa, cast to float16, and assign into per-region monthly
zarrs at:

    {OUT_BASE_DIR}/{region_name}/YYYYMM.zarr

CRITICAL: the GRIB byte payload is fetched ONCE per (cycle, member, lead)
and fanned out to all configured regions in memory. Do NOT loop over
regions on the outer level — that was the bug that wasted 3 TiB of GFS
bandwidth before we fixed it.

Zarr schema (per region per month):

    dimensions: init_time, member, lead_hour, y, x
    coordinates:
        init_time   (n_init,)            datetime64[ns]
        member      (n_member,)          str         e.g. 'gec00', 'gep01'..
        member_idx  (n_member,)          int8        0 control, 1..30 perturbed
        lead_hour   (n_lead,)            int8        0, 3, 6, ..., 84
        valid_time  (n_init, n_lead)     datetime64[ns]
        lat, lon    (y, x)               float32
    variables: U10, V10, PSFC, MSLP  (n_init, n_member, n_lead, y, x) float16
    chunks:    (1, n_member, n_lead, y, x)   one chunk per init_time

Memory: a cycle is processed as a slab of shape (1, n_member, n_lead, y, x)
per variable per region. Peak memory per cycle ≈ 3.5 GB (one member's
decoded global float32 array + all blobs of the cycle + slab buffers).
Members within a cycle are processed sequentially after a single parallel
byte-range fetch.

Resume: a `_DOWNLOAD_COMPLETE` sentinel file inside each region's
monthly zarr marks successful completion. When `OVERWRITE=false`,
months whose sentinel exists are skipped. Zarrs that exist without the
sentinel are inspected per-cycle: if their init_time / member /
lead_hour coords match the current config, cycle slabs already written
(detected by chunk-file presence across all 4 vars) are preserved and
the run resumes from the first missing cycle. If any coord disagrees
(e.g. the inventory has changed), the zarr is wiped and recreated to
avoid mixing inconsistent data.

Per-cycle error isolation: each member is fetched, decoded, and
written sequentially. If a member's decode fails the slab stays NaN
on disk AND that member's manifest rows are NOT marked as downloaded,
so a future rerun can attempt them again. Within each member we still
parallel-fetch its 29 leads across `DOWNLOAD_WORKERS` workers.

INIT_TIME_FILTER lets you run a single-cycle Gate-1 sanity test.
"""

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# Default config path matches the VSCode interactive workflow. To run
# multiple instances in parallel (one per month subset), set the
# `GEFS_DOWNLOAD_CONFIG_PATH` environment variable before invoking python.
CONFIG_PATH = os.environ.get(
    'GEFS_DOWNLOAD_CONFIG_PATH',
    'd_atmospheric_forecast_download_gefs_config.json',
)

# cfgrib variable shortNames in GEFS pgrb2sp25 messages.
GEFS_U10_NAME = 'u10'
GEFS_V10_NAME = 'v10'
GEFS_PSFC_NAME = 'sp'      # PRES:surface
GEFS_MSLP_NAME = 'prmsl'   # PRMSL:mean sea level


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
    """Fetch the configured byte ranges from one GEFS GRIB file and concatenate.

    `ranges` is a list of (byte_start, byte_end) inclusive. `byte_end == -1`
    means open-ended ("to end of file") — common for PRMSL which is the
    last record in pgrb2sp25. Returns a single bytes blob whose chunks are
    concatenated in the order given.

    Retries on transient network errors. Default 10 attempts with backoff
    capped at 60 s gives ~5 min of total tolerance per byte range.
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
                blob = r.content
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


def fetch_member_bytes(session, member_rows, max_workers, timeout):
    """Parallel-fetch all (lead × var) byte chunks for one (cycle, member).

    Returns dict `{lead_hour: bytes_blob}`. Streams one member at a time —
    peak in-memory blob size is the member's 29 leads × 4 vars × ~0.7 MiB
    ≈ 80 MiB, vs ~2.5 GB if we fetched the whole cycle (31 members) at once.
    """
    by_lead = {}
    for lead, sub in member_rows.groupby('lead_hour'):
        sub = sub.sort_values('variable')
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


def decode_member_to_global(blob_by_lead, leads_want):
    """Decode one member's concatenated GRIB messages into global arrays.

    `blob_by_lead` is `{lead: blob}` for a single (cycle, member). Returns:
        leads_got : list[int]   sorted ascending, leads actually present
        u, v      : np.ndarray  (n_lead, ny_global, nx_global) float32
        p         : np.ndarray  (n_lead, ny_global, nx_global) float32, Pa
        m         : np.ndarray  (n_lead, ny_global, nx_global) float32, Pa
        src_lat   : np.ndarray  (ny_global,) float32, decreasing N→S
        src_lon   : np.ndarray  (nx_global,) float32, in 0..360
    """
    import cfgrib

    leads_got = sorted(blob_by_lead.keys())
    big = b''.join(blob_by_lead[l] for l in leads_got)

    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tf:
        tf.write(big); tf.flush()
        tmp_path = tf.name
    try:
        ds_list = cfgrib.open_datasets(
            tmp_path, backend_kwargs={'indexpath': ''})
        # pgrb2sp25 spans three typeOfLevel groups:
        #   heightAboveGround (u10, v10), surface (sp), meanSea (prmsl).
        u_full = v_full = p_full = m_full = None
        src_lat = src_lon = None
        for ds in ds_list:
            if 'latitude' in ds.coords and src_lat is None:
                src_lat = ds['latitude'].values
            if 'longitude' in ds.coords and src_lon is None:
                src_lon = ds['longitude'].values
            if GEFS_U10_NAME in ds: u_full = ds[GEFS_U10_NAME].values
            if GEFS_V10_NAME in ds: v_full = ds[GEFS_V10_NAME].values
            if GEFS_PSFC_NAME in ds: p_full = ds[GEFS_PSFC_NAME].values
            if GEFS_MSLP_NAME in ds: m_full = ds[GEFS_MSLP_NAME].values
            ds.close()
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    if any(x is None for x in (u_full, v_full, p_full, m_full)):
        missing = [n for n, x in zip(
            ('u10', 'v10', 'sp', 'prmsl'),
            (u_full, v_full, p_full, m_full)) if x is None]
        raise RuntimeError(f"Decoded GRIB missing variables: {missing}")

    src_lat = np.asarray(src_lat, dtype=np.float32)
    src_lon = np.asarray(src_lon, dtype=np.float32)
    return leads_got, u_full, v_full, p_full, m_full, src_lat, src_lon


def crop_global_to_region(u_full, v_full, p_full, m_full,
                          src_lat, src_lon, bbox):
    """Crop global decoded arrays to one region's bbox.

    `bbox = (lat_min, lat_max, lon_min, lon_max)` in -180..180. Returns
    regional subsets plus lat/lon coords in the region grid convention
    (lat decreasing N→S, lon in -180..180, monotonic ascending).

    Lon mask is built in src's native 0..360 frame so we never materialize
    a reordered global array. Inputs are NOT mutated — the same global
    decode can be cropped to many regions.
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
        raise RuntimeError(f"Region bbox {bbox} has no overlap with GEFS grid")

    sub_lat = src_lat[lat_mask]
    sub_lon = src_lon[lon_mask]
    u_sub = u_full[:, lat_mask][:, :, lon_mask]
    v_sub = v_full[:, lat_mask][:, :, lon_mask]
    p_sub = p_full[:, lat_mask][:, :, lon_mask]
    m_sub = m_full[:, lat_mask][:, :, lon_mask]

    sub_lon = np.where(sub_lon > 180.0,
                       sub_lon - 360.0, sub_lon).astype(np.float32)
    if wraps:
        order = np.argsort(sub_lon)
        sub_lon = sub_lon[order]
        u_sub = u_sub[..., order]
        v_sub = v_sub[..., order]
        p_sub = p_sub[..., order]
        m_sub = m_sub[..., order]

    return (u_sub.astype(np.float32), v_sub.astype(np.float32),
            p_sub.astype(np.float32), m_sub.astype(np.float32),
            sub_lat, sub_lon)


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


def create_empty_monthly_zarr(zarr_path, cycles, members, leads_all,
                               dst_lats, dst_lons, region_name):
    """Create a NaN-filled monthly zarr with chunks=(1, n_member, n_lead, y, x).

    We pre-create the full month's structure once at the start so that
    each cycle write is a clean `to_zarr(region=...)` slab update. The
    fill_value is NaN, so any cycle that fails to download stays NaN
    on disk (sampler-side scaling already replaces NaN with 0).
    """
    n_init = len(cycles)
    n_member = len(members)
    n_lead = len(leads_all)
    ny, nx = dst_lats.shape

    init_arr = np.asarray(cycles, dtype='datetime64[ns]')
    lead_arr = np.asarray(leads_all, dtype=np.int8)
    member_arr = np.asarray(members, dtype=object)
    member_idx_arr = np.asarray(
        [0 if m == 'gec00' else int(m[3:]) for m in members], dtype=np.int8)
    valid_time = (
        init_arr.astype('datetime64[h]')[:, None]
        + lead_arr.astype('timedelta64[h]')[None, :]
    ).astype('datetime64[ns]')

    if os.path.exists(zarr_path):
        import shutil; shutil.rmtree(zarr_path)
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # Build the dataset with empty (NaN, fill_value-backed) data arrays via
    # dask so to_zarr writes only metadata; chunk writes happen later.
    import dask.array as da
    chunks_data = (1, n_member, n_lead, ny, nx)
    nan_da = da.full((n_init, n_member, n_lead, ny, nx),
                     np.nan, dtype='float16', chunks=chunks_data)

    ds = xr.Dataset(
        data_vars=dict(
            U10=(('init_time', 'member', 'lead_hour', 'y', 'x'), nan_da),
            V10=(('init_time', 'member', 'lead_hour', 'y', 'x'), nan_da),
            PSFC=(('init_time', 'member', 'lead_hour', 'y', 'x'), nan_da),
            MSLP=(('init_time', 'member', 'lead_hour', 'y', 'x'), nan_da),
        ),
        coords=dict(
            init_time=init_arr,
            member=member_arr,
            member_idx=('member', member_idx_arr),
            lead_hour=lead_arr,
            valid_time=(('init_time', 'lead_hour'), valid_time),
            lat=(('y', 'x'), dst_lats.astype(np.float32)),
            lon=(('y', 'x'), dst_lons.astype(np.float32)),
        ),
        attrs=dict(
            source='NOAA GEFS',
            source_bucket='s3://noaa-gefs-pds',
            product='atmos/pgrb2sp25 (0.25 deg select)',
            grid=f'{region_name}_lr_64x64',
            grid_resolution_km=16,
            n_members=int(n_member),
            lead_hour_min=int(lead_arr.min()),
            lead_hour_max=int(lead_arr.max()),
            lead_hour_step=(int(lead_arr[1] - lead_arr[0]) if len(lead_arr) > 1 else 0),
            pressure_units='hPa',
            wind_units='m s-1',
            cadence='native 3-hourly (NOT interpolated to hourly)',
            interpolation='bilinear to region lr grid',
            created_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        ),
    )
    encoding = {v: {'compressor': compressor, 'chunks': chunks_data,
                    'dtype': 'float16'}
                for v in ('U10', 'V10', 'PSFC', 'MSLP')}
    ds.to_zarr(zarr_path, mode='w', encoding=encoding,
               consolidated=True, compute=False)


def write_cycle_slab(zarr_path, i_cycle, cycle_ts,
                     u_slab, v_slab, p_slab, m_slab):
    """Write one cycle's (n_member, n_lead, y, x) slabs to the monthly zarr.

    Uses `to_zarr(region=...)` to update only the i_cycle-th init_time
    slice. The slabs must already be float16 with NaN for missing entries.
    """
    cycle_arr = np.asarray([cycle_ts], dtype='datetime64[ns]')
    n_member, n_lead, ny, nx = u_slab.shape
    slab = xr.Dataset(
        data_vars=dict(
            U10=(('init_time', 'member', 'lead_hour', 'y', 'x'),
                  u_slab[None].astype(np.float16)),
            V10=(('init_time', 'member', 'lead_hour', 'y', 'x'),
                  v_slab[None].astype(np.float16)),
            PSFC=(('init_time', 'member', 'lead_hour', 'y', 'x'),
                   p_slab[None].astype(np.float16)),
            MSLP=(('init_time', 'member', 'lead_hour', 'y', 'x'),
                   m_slab[None].astype(np.float16)),
        ),
        coords=dict(init_time=cycle_arr),
    )
    slab.to_zarr(zarr_path, region={'init_time': slice(i_cycle, i_cycle + 1)})


def _scan_done_cycles_via_chunks(zarr_path,
                                  var_names=('U10', 'V10', 'PSFC', 'MSLP')):
    """Return the set of cycle indices whose chunk file is on disk for
    every data variable. Filesystem-level scan — no decompression.

    Each variable is chunked by `(1, n_member, n_lead, y, x)`, so init_time
    `i` corresponds to chunk file `<var>/i.0.0.0.0`. A cycle counts as
    "done" only if all `var_names` chunks exist for that init_time, which
    matches the atomicity of `write_cycle_slab` (all 4 vars written in
    one xarray.to_zarr region call). A partial mid-write that left fewer
    than 4 chunks for a cycle correctly resolves to "not done".
    """
    chunk_sets = []
    for var in var_names:
        chunk_dir = os.path.join(zarr_path, var)
        if not os.path.isdir(chunk_dir):
            return set()
        s = set()
        for fname in os.listdir(chunk_dir):
            if fname.startswith('.'):   # .zarray, .zattrs, etc.
                continue
            try:
                s.add(int(fname.split('.')[0]))
            except (ValueError, IndexError):
                pass
        chunk_sets.append(s)
    return set.intersection(*chunk_sets) if chunk_sets else set()


def _check_or_init_monthly_zarr(zarr_path, cycles, members, leads_all,
                                 dst_lats, dst_lons, region_name):
    """Decide whether to reuse an existing zarr or wipe + recreate it.

    Reuse criteria: the existing zarr must already have init_time, member,
    and lead_hour coords that match `cycles`, `members`, `leads_all`
    exactly. If they do, return the set of cycle indices whose slab is
    already on disk for all four data variables. If anything mismatches
    (or the zarr is unreadable), the zarr is wiped via
    `create_empty_monthly_zarr` and the empty set is returned.
    """
    if not os.path.exists(zarr_path):
        create_empty_monthly_zarr(zarr_path, cycles, members, leads_all,
                                   dst_lats, dst_lons, region_name)
        return set()

    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        expected_init = np.asarray(cycles, dtype='datetime64[ns]')
        existing_init = ds['init_time'].values.astype('datetime64[ns]')
        existing_members = [str(m) for m in ds['member'].values]
        existing_leads = ds['lead_hour'].values.astype(int).tolist()
        ds.close()
        mismatch = (
            existing_init.shape != expected_init.shape
            or not np.array_equal(existing_init, expected_init)
            or existing_members != list(members)
            or existing_leads != list(map(int, leads_all))
        )
    except Exception as e:
        print(f"[warn] {zarr_path} unreadable ({e}); wiping")
        create_empty_monthly_zarr(zarr_path, cycles, members, leads_all,
                                   dst_lats, dst_lons, region_name)
        return set()

    if mismatch:
        print(f"[warn] {zarr_path} coords disagree with current config; "
              f"wiping")
        create_empty_monthly_zarr(zarr_path, cycles, members, leads_all,
                                   dst_lats, dst_lons, region_name)
        return set()

    return _scan_done_cycles_via_chunks(zarr_path)


def load_month_manifest(inventory_dir, month_str):
    path = os.path.join(inventory_dir, f"{month_str}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Inventory partition not found: {path}")
    return path, pd.read_parquet(path)


def update_manifest_status(parquet_path, df_full, processed_keys):
    """Mark processed (init_time, member, lead_hour, variable) rows as 'downloaded'."""
    if not processed_keys:
        return
    keys_set = set(processed_keys)
    mask = df_full.apply(
        lambda r: (r['init_time'], r['member'], int(r['lead_hour']),
                   r['variable']) in keys_set,
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
    print(f"[gefs-download] regions={[r['name'] for r in region_specs]}")
    for r in region_specs:
        print(f"  {r['name']}  grid={r['lats'].shape}  bbox={r['bbox']}")

    init_filter = cfg.get('INIT_TIME_FILTER')
    init_filter = ([pd.Timestamp(t) for t in init_filter]
                   if init_filter else None)

    lead_min = int(cfg['LEAD_MIN'])
    lead_max = int(cfg['LEAD_MAX'])
    lead_step = int(cfg.get('LEAD_STEP', 3))
    leads_all = list(range(lead_min, lead_max + 1, lead_step))
    n_lead = len(leads_all)

    members = list(cfg['MEMBERS'])
    n_member = len(members)
    member_to_idx = {m: i for i, m in enumerate(members)}

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
        # A region is "complete" only if its zarr has the
        # _DOWNLOAD_COMPLETE sentinel inside. Any zarr without the sentinel
        # is treated as partial and will be wiped + restarted by
        # create_empty_monthly_zarr below.
        def _sentinel_path(region_name):
            return os.path.join(cfg['OUT_BASE_DIR'], region_name,
                                f"{month_str}.zarr", '_DOWNLOAD_COMPLETE')
        pending = []
        for r in region_specs:
            if overwrite or not os.path.exists(_sentinel_path(r['name'])):
                zp = os.path.join(cfg['OUT_BASE_DIR'], r['name'],
                                  f"{month_str}.zarr")
                if os.path.exists(zp):
                    print(f"[warn] {month_str} {r['name']}: zarr exists "
                          f"without sentinel - treating as partial and "
                          f"restarting")
                pending.append(r)
        if not pending:
            print(f"[skip] {month_str}: all region zarrs complete")
            continue

        parquet_path, df_full = load_month_manifest(
            cfg['INVENTORY_DIR'], month_str)
        df = df_full[(df_full['lead_hour'].isin(leads_all))
                      & (df_full['member'].isin(members))
                      & (df_full['status'].isin(('planned', 'downloaded')))]
        if init_filter is not None:
            df = df[df['init_time'].isin(init_filter)]

        cycles = sorted(df['init_time'].unique())
        if not cycles:
            print(f"[skip] {month_str}: no rows match the filter")
            continue
        n_init = len(cycles)

        # For each pending region: reuse the existing zarr if its
        # init_time / member / lead_hour coords match the current config,
        # else wipe and recreate. Returns the set of cycle indices whose
        # slab is already on disk for that region.
        done_by_region = {}
        for r in pending:
            zp = os.path.join(cfg['OUT_BASE_DIR'], r['name'], f"{month_str}.zarr")
            done_by_region[r['name']] = _check_or_init_monthly_zarr(
                zp, cycles, members, leads_all,
                r['lats'], r['lons'], r['name'])

        # A cycle can only be skipped if EVERY pending region already has
        # its slab on disk. Otherwise we must reprocess to write it where
        # missing (the resulting overwrite of regions that already have
        # the data is harmless — same values land in the same chunk).
        fully_done = (set.intersection(*done_by_region.values())
                       if done_by_region else set())
        if fully_done:
            print(f"[resume] {month_str}: {len(fully_done)} of {n_init} "
                  f"cycle(s) already populated across all regions - "
                  f"resuming with {n_init - len(fully_done)} cycle(s) to fetch")

        processed_keys = []
        print(f"[gefs-download] {month_str}: {n_init} cycles x {n_member} members "
              f"x {n_lead} leads x 4 vars = "
              f"{n_init * n_member * n_lead * 4:,} GRIB messages  "
              f"-> {len(pending)} region(s): "
              f"{', '.join(r['name'] for r in pending)}")

        for i, ts in enumerate(tqdm(cycles, desc=month_str)):
            if i in fully_done:
                continue
            cyc_rows = df[df['init_time'] == ts]

            # Per-cycle slab accumulators (one set per pending region).
            slabs = {
                r['name']: {
                    'U': np.full((n_member, n_lead, *r['lats'].shape),
                                  np.nan, dtype=np.float32),
                    'V': np.full((n_member, n_lead, *r['lats'].shape),
                                  np.nan, dtype=np.float32),
                    'P': np.full((n_member, n_lead, *r['lats'].shape),
                                  np.nan, dtype=np.float32),
                    'M': np.full((n_member, n_lead, *r['lats'].shape),
                                  np.nan, dtype=np.float32),
                } for r in pending
            }

            # Per-member streaming: fetch this member's blobs, decode,
            # fan out to all regions, free, then next member. Bounds peak
            # memory to ~1.2 GB and isolates per-member decode failures.
            cycle_success_members = set()
            t_fetch = t_dec = t_int = 0.0
            for member in members:
                mem_rows = cyc_rows[cyc_rows['member'] == member]
                if mem_rows.empty:
                    continue  # member absent from manifest for this cycle

                t0 = time.time()
                try:
                    blob_by_lead = fetch_member_bytes(
                        session, mem_rows, workers, timeout)
                except Exception as e:
                    tqdm.write(f"  [warn] fetch failed for {ts} {member}: {e}")
                    continue
                t_fetch += time.time() - t0

                t0 = time.time()
                try:
                    leads_got, u_g, v_g, p_g, m_g, src_lat, src_lon = \
                        decode_member_to_global(blob_by_lead, leads_all)
                except Exception as e:
                    tqdm.write(f"  [warn] decode failed for {ts} {member}: {e}")
                    del blob_by_lead
                    continue
                t_dec += time.time() - t0
                del blob_by_lead

                lead_idx = {l: k for k, l in enumerate(leads_got)}
                m_slot = member_to_idx[member]

                t0 = time.time()
                for r in pending:
                    u_sub, v_sub, p_sub, m_sub, slat, slon = \
                        crop_global_to_region(u_g, v_g, p_g, m_g,
                                              src_lat, src_lon, r['bbox'])
                    u_r = interp_to_region(u_sub, slat, slon, r['lats'], r['lons'])
                    v_r = interp_to_region(v_sub, slat, slon, r['lats'], r['lons'])
                    p_r = interp_to_region(p_sub, slat, slon, r['lats'], r['lons']) / 100.0
                    m_r = interp_to_region(m_sub, slat, slon, r['lats'], r['lons']) / 100.0
                    s = slabs[r['name']]
                    for k, l in enumerate(leads_all):
                        if l not in lead_idx:
                            continue   # leaves slab NaN at that slot
                        src_k = lead_idx[l]
                        s['U'][m_slot, k] = u_r[src_k]
                        s['V'][m_slot, k] = v_r[src_k]
                        s['P'][m_slot, k] = p_r[src_k]
                        s['M'][m_slot, k] = m_r[src_k]
                t_int += time.time() - t0

                cycle_success_members.add(member)
                del u_g, v_g, p_g, m_g

            # Write cycle slab to each region's monthly zarr.
            t0 = time.time()
            for r in pending:
                zp = os.path.join(cfg['OUT_BASE_DIR'], r['name'],
                                  f"{month_str}.zarr")
                s = slabs[r['name']]
                write_cycle_slab(zp, i, ts, s['U'], s['V'], s['P'], s['M'])
            t_write = time.time() - t0

            # Only mark rows from successful members as downloaded.
            ok_rows = cyc_rows[cyc_rows['member'].isin(cycle_success_members)]
            for _, row in ok_rows.iterrows():
                processed_keys.append(
                    (row['init_time'], row['member'], int(row['lead_hour']),
                     row['variable']))

            sample = pending[0]['name']
            sample_s = slabs[sample]
            n_ok = len(cycle_success_members)
            tqdm.write(
                f"  {ts}  members ok={n_ok}/{n_member}  "
                f"fetch={t_fetch:.1f}s  decode={t_dec:.1f}s  "
                f"interp×{len(pending)}={t_int:.1f}s  write={t_write:.1f}s  "
                f"[{sample}] |U|max="
                f"{np.nanmax(np.abs(sample_s['U'])):.1f}  "
                f"PSFC[{np.nanmin(sample_s['P']):.1f},"
                f"{np.nanmax(sample_s['P']):.1f}] hPa  "
                f"MSLP[{np.nanmin(sample_s['M']):.1f},"
                f"{np.nanmax(sample_s['M']):.1f}] hPa"
            )

            del slabs

        # All cycles for this month written successfully — drop the
        # sentinel files so future runs treat these zarrs as complete.
        for r in pending:
            zp = os.path.join(cfg['OUT_BASE_DIR'], r['name'],
                              f"{month_str}.zarr")
            with open(os.path.join(zp, '_DOWNLOAD_COMPLETE'), 'w') as f:
                f.write(pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ\n'))

        update_manifest_status(parquet_path, df_full, processed_keys)
        print(f"[done] {month_str}  manifest updated -> {parquet_path}")

    print("[gefs-download] complete.")
