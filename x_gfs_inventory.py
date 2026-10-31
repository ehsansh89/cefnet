"""
x_gfs_inventory.py — Build the GFS forecast-atmosphere download plan.

Crawls GFS `.idx` files on `s3://noaa-gfs-bdp-pds` (HTTPS, no signing) for
every (init_date, cycle, lead_hour) in the configured window. Parses each
`.idx` to extract byte-ranges for the configured variables (default:
UGRD/VGRD at 10 m above ground + PRES:surface). Writes a per-month
parquet partition under

    {OUT_DIR}/download_plan/YYYYMM.parquet

plus a top-level summary csv at `{OUT_DIR}/inventory_summary.csv`.

This step downloads only ~30-50 KB of text per cycle-lead, never the GRIB
payload — it is the cheap source of truth for the heavier
`d_atmospheric_forecast_download.py` and is safe to rerun: existing
monthly partitions are skipped unless `FORCE_REBUILD` is set.

Schema of each row (one row per (init_time, lead_hour, variable)):

    init_time     datetime64[ns]   GFS cycle initialization time
    cycle         int              0 / 6 / 12 / 18
    lead_hour     int              0..LEAD_MAX
    variable      str              e.g. "UGRD:10 m above ground"
    idx_url       str              public .idx URL
    grib_url      str              public GRIB2 URL
    byte_start    int              inclusive byte-range start (-1 if missing)
    byte_end      int              inclusive byte-range end (-1 = open-ended
                                   or missing)
    n_bytes       int              byte_end - byte_start + 1 (0 if missing)
    status        str              'planned' | 'missing' | 'failed'
    error         str              short error message when status='failed'

See `docs/note_gfs_download_plan.md` §3.1.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


CONFIG_PATH = 'x_gfs_inventory_config.json'

# Public AWS S3 mirror of NOAA GFS — no credentials required.
GFS_HTTP_BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'

# .idx line:
#   "recno:byte_offset:d=YYYYMMDDCC:VAR:LEVEL:N hour fcst:" (or ":anl:")
# Capture (recno, offset, var, level); the trailing fcst-step token is
# ignored so f000 ('anl') and fNNN ('N hour fcst') parse the same.
IDX_LINE_RE = re.compile(r'^(\d+):(\d+):d=\d{10}:([^:]+):([^:]+):')


def build_grib_url(init_date, cycle, lead):
    return (f"{GFS_HTTP_BASE}/gfs.{init_date:%Y%m%d}/{cycle:02d}/atmos/"
            f"gfs.t{cycle:02d}z.pgrb2.0p25.f{lead:03d}")


def build_idx_url(init_date, cycle, lead):
    return build_grib_url(init_date, cycle, lead) + '.idx'


def make_session(pool_size, retries):
    """Requests session with HTTP keep-alive and exponential 5xx retry."""
    s = requests.Session()
    retry = Retry(
        total=retries, backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=('GET', 'HEAD'),
    )
    adapter = HTTPAdapter(pool_connections=pool_size,
                          pool_maxsize=pool_size, max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s


def parse_idx_records(idx_text):
    """Return list of (recno, byte_offset, var, level) tuples in file order."""
    out = []
    for line in idx_text.splitlines():
        m = IDX_LINE_RE.match(line)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)),
                    m.group(3).strip(), m.group(4).strip()))
    return out


def select_target_records(records, targets):
    """For each (var, level) target, return its byte_start and byte_end.

    `byte_end` is the next-record byte_offset minus 1 (inclusive). If the
    matching record is the last one in the file, `byte_end = -1` to signal
    'fetch to end of file' to the downstream HTTP range request.
    """
    sorted_recs = sorted(records, key=lambda r: r[1])
    offsets = [r[1] for r in sorted_recs]
    rows = []
    for var_name, var_level in targets:
        match = None
        for i, (_, off, var, lvl) in enumerate(sorted_recs):
            if var == var_name and lvl == var_level:
                end = offsets[i + 1] - 1 if i + 1 < len(sorted_recs) else -1
                match = (off, end)
                break
        if match is None:
            rows.append({'variable': f"{var_name}:{var_level}",
                         'byte_start': -1, 'byte_end': -1})
        else:
            rows.append({'variable': f"{var_name}:{var_level}",
                         'byte_start': match[0], 'byte_end': match[1]})
    return rows


def fetch_one_idx(session, init_date, cycle, lead, targets, timeout):
    """Fetch one `.idx` and return manifest rows for the configured targets."""
    idx_url = build_idx_url(init_date, cycle, lead)
    grib_url = build_grib_url(init_date, cycle, lead)
    init_time = pd.Timestamp(init_date) + pd.Timedelta(hours=cycle)
    base_row = dict(init_time=init_time, cycle=cycle, lead_hour=lead,
                    idx_url=idx_url, grib_url=grib_url)

    try:
        r = session.get(idx_url, timeout=timeout)
    except Exception as e:
        return [dict(base_row,
                     variable=f"{v}:{l}",
                     byte_start=-1, byte_end=-1, n_bytes=0,
                     status='failed', error=str(e)[:120])
                for v, l in targets]

    if r.status_code == 404:
        # Cycle simply absent from the archive (operational gap).
        return [dict(base_row,
                     variable=f"{v}:{l}",
                     byte_start=-1, byte_end=-1, n_bytes=0,
                     status='missing', error='')
                for v, l in targets]
    if not r.ok:
        return [dict(base_row,
                     variable=f"{v}:{l}",
                     byte_start=-1, byte_end=-1, n_bytes=0,
                     status='failed', error=f"HTTP {r.status_code}")
                for v, l in targets]

    records = parse_idx_records(r.text)
    target_rows = select_target_records(records, targets)
    out = []
    for tr in target_rows:
        bs, be = tr['byte_start'], tr['byte_end']
        if bs < 0:
            n_bytes, status = 0, 'missing'
        elif be < 0:
            # Last record in file — open-ended range; size unknown here.
            n_bytes, status = 0, 'planned'
        else:
            n_bytes, status = be - bs + 1, 'planned'
        out.append(dict(base_row, **tr, n_bytes=n_bytes,
                        status=status, error=''))
    return out


def iter_keys(start_date, end_date, cycles, leads):
    cur = start_date
    while cur <= end_date:
        for c in cycles:
            for l in leads:
                yield cur, c, l
        cur += timedelta(days=1)


def inventory_month(year, month, cfg, session, plan_dir):
    """Crawl all (date, cycle, lead) keys for one calendar month.

    Returns the per-month DataFrame, or None if the configured window
    has no overlap with this month.
    """
    g_start = pd.Timestamp(cfg['START_DATE']).date()
    g_end = pd.Timestamp(cfg['END_DATE']).date()

    month_start = datetime(year, month, 1).date()
    next_first = (datetime(year + 1, 1, 1).date() if month == 12
                  else datetime(year, month + 1, 1).date())
    month_end = next_first - timedelta(days=1)

    start_date = max(month_start, g_start)
    end_date = min(month_end, g_end)
    if start_date > end_date:
        return None

    targets = [tuple(t) for t in cfg['VARIABLES']]
    cycles = cfg['CYCLES']
    leads = list(range(cfg['LEAD_MIN'], cfg['LEAD_MAX'] + 1))
    timeout = cfg.get('REQUEST_TIMEOUT', 30)

    keys = list(iter_keys(start_date, end_date, cycles, leads))
    rows = []
    with ThreadPoolExecutor(max_workers=cfg['MAX_WORKERS']) as ex:
        futures = [ex.submit(fetch_one_idx, session, d, c, l, targets, timeout)
                   for d, c, l in keys]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"{year}-{month:02d}", leave=False):
            rows.extend(fut.result())

    df = pd.DataFrame(rows)
    df = df.sort_values(['init_time', 'lead_hour', 'variable']).reset_index(drop=True)
    out_path = os.path.join(plan_dir, f"{year}{month:02d}.parquet")
    df.to_parquet(out_path, index=False)
    return df


def summarize(plan_dir, summary_path):
    """Aggregate per-month parquets into one summary csv."""
    parts = sorted(p for p in os.listdir(plan_dir) if p.endswith('.parquet'))
    rows = []
    for p in parts:
        df = pd.read_parquet(os.path.join(plan_dir, p))
        rows.append(dict(
            month=p.replace('.parquet', ''),
            rows=len(df),
            planned=int((df['status'] == 'planned').sum()),
            missing=int((df['status'] == 'missing').sum()),
            failed=int((df['status'] == 'failed').sum()),
            unique_init_times=int(df['init_time'].nunique()),
        ))
    pd.DataFrame(rows).to_csv(summary_path, index=False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)

    out_dir = cfg['OUT_DIR']
    plan_dir = os.path.join(out_dir, 'download_plan')
    os.makedirs(plan_dir, exist_ok=True)

    session = make_session(pool_size=cfg['MAX_WORKERS'],
                           retries=cfg.get('HTTP_RETRIES', 3))

    g_start = pd.Timestamp(cfg['START_DATE']).date()
    g_end = pd.Timestamp(cfg['END_DATE']).date()
    months = pd.date_range(pd.Timestamp(g_start.replace(day=1)),
                           pd.Timestamp(g_end), freq='MS')
    months = [(d.year, d.month) for d in months]

    force = cfg.get('FORCE_REBUILD', False)

    print(f"[gfs-inventory] window={g_start}..{g_end}  months={len(months)}  "
          f"workers={cfg['MAX_WORKERS']}  out={plan_dir}")

    for y, m in tqdm(months, desc='months'):
        out_path = os.path.join(plan_dir, f"{y}{m:02d}.parquet")
        if os.path.exists(out_path) and not force:
            tqdm.write(f"[skip] {y}-{m:02d} already exists")
            continue
        t0 = time.time()
        df = inventory_month(y, m, cfg, session, plan_dir)
        if df is None:
            continue
        elapsed = time.time() - t0
        tqdm.write(
            f"[done] {y}-{m:02d}  rows={len(df):,}  "
            f"planned={(df['status']=='planned').sum():,}  "
            f"missing={(df['status']=='missing').sum():,}  "
            f"failed={(df['status']=='failed').sum():,}  "
            f"({elapsed:.0f}s)"
        )

    summary_path = os.path.join(out_dir, 'inventory_summary.csv')
    summarize(plan_dir, summary_path)
    print(f"[gfs-inventory] summary -> {summary_path}")
