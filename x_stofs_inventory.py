"""NOAA STOFS-2D-Global archive inventory on the public NODD S3 bucket.

Walks `s3://noaa-gestofs-pds/` (HTTPS REST, no AWS auth) for `stofs_2d_glo.YYYYMMDD/`
date prefixes and enumerates (date, cycle) pairs. STOFS-2D-Global has run 4 cycles
per day (00/06/12/18Z) consistently since the modern `stofs_2d_glo.*` naming began
on 2023-01-08. Earlier `estofs.*` cycles (2020-12-30 -> 2023-01-12) are out of scope
here because they predate the v2.1.4+ product layout we want to compare against.

Per the STOFS v2.1.7 service-change notice (effective 2024-05-14 12Z), bias-corrected
points files (`points.cwl.nc`) and a raw counterpart (`points.cwl.noanomaly.nc`) co-exist
after that date. Before it, only the raw run is available as `points.cwl.nc`. The
inventory flags each row with `stofs_version_period` so the download/skill scripts can
choose `noanomaly` vs `cwl` consistently across the upgrade boundary.

Outputs:
  data/stofs/inventory.csv         (date, cycle, stofs_version_period)
  data/stofs/inventory_summary.csv (archive boundaries + counts)
"""

import os
import re
import time
from datetime import datetime
from urllib.request import Request, urlopen

import pandas as pd

BUCKET_HOST = "noaa-gestofs-pds.s3.amazonaws.com"
PREFIX = "stofs_2d_glo."
CYCLES = ("t00z", "t06z", "t12z", "t18z")
# SCN 23-124 says v2.1.7 became effective with the 2024-05-14 12Z cycle, so
# 2024-05-14 t00z and t06z are still pre-v2.1.7. The boundary is cycle-level,
# not date-level.
V217_BOUNDARY = pd.Timestamp("2024-05-14T12:00")
DATE_RE = re.compile(r"^stofs_2d_glo\.(\d{8})/$")


def fetch(url, retries=3, timeout=60):
    last = None
    for _ in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "conus_ai_stofs_inventory/1.0"})
            with urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"fetch failed: {last} url={url}")


def list_dates():
    """List all stofs_2d_glo.YYYYMMDD/ common-prefixes, paginated."""
    dates = []
    token = None
    while True:
        url = (f"https://{BUCKET_HOST}/?list-type=2&prefix={PREFIX}"
               f"&delimiter=/&max-keys=1000")
        if token:
            url += f"&continuation-token={token}"
        xml = fetch(url)
        for m in re.finditer(r"<Prefix>([^<]+)</Prefix>", xml):
            mm = DATE_RE.match(m.group(1))
            if mm:
                dates.append(mm.group(1))
        tm = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if tm:
            # URL-encode the token (may contain '+', '=', etc.)
            from urllib.parse import quote
            token = quote(tm.group(1), safe="")
        else:
            break
    return sorted(set(dates))


def list_cycle_files(date_str, want_filenames):
    """List the keys under stofs_2d_glo.YYYYMMDD/ matching any of want_filenames.
    Returns set of base filenames present (not full keys)."""
    present = set()
    token = None
    while True:
        url = (f"https://{BUCKET_HOST}/?list-type=2&prefix={PREFIX}{date_str}/"
               f"&max-keys=1000")
        if token:
            url += f"&continuation-token={token}"
        xml = fetch(url)
        for m in re.finditer(r"<Key>([^<]+)</Key>", xml):
            base = m.group(1).rsplit("/", 1)[-1]
            if base in want_filenames:
                present.add(base)
        tm = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if tm:
            from urllib.parse import quote
            token = quote(tm.group(1), safe="")
        else:
            break
    return present


def verify_cycles_for_date(date_str):
    """Confirm which of the 4 cycles actually has a points.cwl.nc on this date.
    Returns list of cycle codes present (subset of CYCLES)."""
    want = {f"stofs_2d_glo.{c}.points.cwl.nc" for c in CYCLES}
    present = list_cycle_files(date_str, want)
    return sorted(c for c in CYCLES
                  if f"stofs_2d_glo.{c}.points.cwl.nc" in present)


def version_period(date_str, cycle):
    hh = int(cycle[1:3])  # 't00z' -> 0, 't12z' -> 12
    cycle_dt = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{hh:02d}:00")
    return "v217_biascorr" if cycle_dt >= V217_BOUNDARY else "pre_v217_raw"


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data/stofs", exist_ok=True)

    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] listing stofs_2d_glo.* date folders on {BUCKET_HOST} ...", flush=True)
    dates = list_dates()
    if not dates:
        raise SystemExit("No stofs_2d_glo.* dates returned from bucket listing")
    print(f"  {len(dates)} dates: {dates[0]} -> {dates[-1]}  ({time.time()-t0:.1f}s)", flush=True)

    # Verify cycles on the boundary dates only (start, v2.1.7 boundary, end).
    # Interior dates assume the standard 4-cycle daily schedule. The download
    # script handles missing cycles gracefully (404 -> skip + log).
    verify_dates = [dates[0], dates[-1]]
    bdy_str = V217_BOUNDARY.strftime("%Y%m%d")
    if bdy_str in dates:
        verify_dates.append(bdy_str)
    verify_dates = sorted(set(verify_dates))
    verified_cycles = {}
    for d in verify_dates:
        cs = verify_cycles_for_date(d)
        verified_cycles[d] = cs
        print(f"  verified {d}: cycles present = {cs}", flush=True)

    # Build inventory rows: 4 cycles per date, flagged with version period.
    rows = []
    for d in dates:
        for c in CYCLES:
            rows.append({
                "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                "cycle": c,
                "stofs_version_period": version_period(d, c),
            })
    inv = pd.DataFrame(rows)
    inv.to_csv("data/stofs/inventory.csv", index=False)

    # Summary
    inv["dt"] = pd.to_datetime(inv["date"])
    summary = pd.DataFrame([{
        "model": "stofs2d_global",
        "product": "points.cwl.nc",
        "archive_start": inv["dt"].min().strftime("%Y-%m-%d"),
        "archive_end":   inv["dt"].max().strftime("%Y-%m-%d"),
        "days":          inv["dt"].nunique(),
        "cycles_per_day_assumed": len(CYCLES),
        "total_cycles":  len(inv),
        "v217_boundary": V217_BOUNDARY.strftime("%Y-%m-%dT%H:%MZ"),
        "pre_v217_cycles":  int((inv["stofs_version_period"] == "pre_v217_raw").sum()),
        "v217_cycles":      int((inv["stofs_version_period"] == "v217_biascorr").sum()),
        "verified_dates":   ",".join(verify_dates),
        "verified_cycles":  ";".join(f"{d}:{','.join(verified_cycles[d])}" for d in verify_dates),
    }])
    summary.to_csv("data/stofs/inventory_summary.csv", index=False)

    print(f"\nTotal walltime: {time.time()-t0:.1f}s")
    print("\n=== STOFS-2D-Global Inventory summary ===")
    print(summary.to_string(index=False))
