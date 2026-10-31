"""NOAA NCEI OFS stations.forecast archive inventory.

Crawls https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-
hydrodynamic-model-systems-co-ops/access/ for each priority coastal OFS model,
walks YYYY/MM/ directory listings (HTML autoindex), and enumerates all
stations.forecast.nc files. NCEI has deeper archives than the NODD S3 bucket
(e.g. CBOFS starts 2022 vs 2024 on S3).

NGOFS2 has no stations product on NCEI either (only 2ds / fields); inventoried
via its 2ds.n000 files as the per-cycle anchor for the Gulf region.
"""

import os
import re
import time
from datetime import datetime
from urllib.request import Request, urlopen

import pandas as pd

BASE = ("https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-"
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

FORECAST_RE = re.compile(
    r"(?:nos\.)?(?P<model>\w+?)\.(?:stations\.forecast\.(?P<date1>\d{8})\.t(?P<cycle1>\d{2})z|"
    r"t(?P<cycle2>\d{2})z\.(?P<date2>\d{8})\.stations\.forecast)\.nc$"
)
NGOFS2_2DS_RE = re.compile(
    r"(?:nos\.)?ngofs2\.(?:2ds\.n000\.(?P<date1>\d{8})\.t(?P<cycle1>\d{2})z|"
    r"t(?P<cycle2>\d{2})z\.(?P<date2>\d{8})\.2ds\.f000)\.nc$"
)
HREF_RE = re.compile(r'href="([^"?/][^"]*)"', re.I)


def fetch(url, retries=3, timeout=60):
    last = None
    for _ in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "conus_ai_ofs_inventory_ncei/1.0"})
            with urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"fetch failed: {last} url={url}")


def list_entries(url):
    """Return list of href names (no ..) from NCEI autoindex HTML."""
    html = fetch(url)
    out = []
    for href in HREF_RE.findall(html):
        if href in ("/", "..", ".") or href.startswith(("http://", "https://", "mailto:", "?", "#")):
            continue
        out.append(href.rstrip("/"))
    return out


def walk_model(model, long_name):
    """Return list of (date_str, cycle_int) and pages fetched.

    All priority models have stations.forecast on NCEI (including NGOFS2 —
    confirmed July 2024). 2ds.n000 is available as a backup for NGOFS2 but
    not used here; stations product is preferred when present.
    """
    pat = FORECAST_RE

    found = []
    pages = 0
    root = f"{BASE}{long_name}/"
    entries = list_entries(root)
    pages += 1
    years = sorted(n for n in entries if re.match(r"^\d{4}$", n))

    for y in years:
        y_url = f"{root}{y}/"
        try:
            y_entries = list_entries(y_url)
        except Exception as e:
            print(f"    skip {y}: {e}", flush=True)
            continue
        pages += 1
        months = sorted(n for n in y_entries if re.match(r"^\d{2}$", n))
        for mo in months:
            m_url = f"{y_url}{mo}/"
            try:
                m_entries = list_entries(m_url)
            except Exception as e:
                print(f"    skip {y}/{mo}: {e}", flush=True)
                continue
            pages += 1
            for name in m_entries:
                m = pat.search(name)
                if m:
                    date = m.group("date1") or m.group("date2")
                    cyc = m.group("cycle1") or m.group("cycle2")
                    found.append((date, int(cyc)))
    return found, pages


def summarize_model(model, found):
    if not found:
        return None, None
    df = pd.DataFrame(found, columns=["date", "cycle"]).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values(["date", "cycle"]).reset_index(drop=True)
    start = df["date"].min()
    end = df["date"].max()
    days_covered = df["date"].nunique()
    span = (end - start).days + 1
    cycles = sorted(df["cycle"].unique().tolist())
    per_day = df.groupby("date")["cycle"].nunique()
    typical_cpd = int(per_day.mode().iloc[0]) if len(per_day) else 0
    full_days = int((per_day == typical_cpd).sum()) if typical_cpd else 0
    product = "stations.forecast"
    return {
        "model": model,
        "product": product,
        "archive_start": start.strftime("%Y-%m-%d"),
        "archive_end": end.strftime("%Y-%m-%d"),
        "days_with_any_cycle": days_covered,
        "span_days": span,
        "missing_days": span - days_covered,
        "cycles_observed": ",".join(f"t{c:02d}z" for c in cycles),
        "typical_cycles_per_day": typical_cpd,
        "full_days": full_days,
        "total_files": len(df),
    }, df


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs("data/ofs", exist_ok=True)
    all_rows = []
    summaries = []
    t0 = time.time()

    for model, long_name in MODEL_DIRS.items():
        t_model = time.time()
        print(f"[{datetime.now():%H:%M:%S}] walking {model} ({long_name}) ...", flush=True)
        try:
            found, pages = walk_model(model, long_name)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        if not found:
            print(f"  no files found ({pages} pages scanned)", flush=True)
            summaries.append({"model": model, "product": "-", "archive_start": None,
                              "archive_end": None, "days_with_any_cycle": 0,
                              "span_days": 0, "missing_days": 0,
                              "cycles_observed": "", "typical_cycles_per_day": 0,
                              "full_days": 0, "total_files": 0})
            continue
        summary, df = summarize_model(model, found)
        summaries.append(summary)
        for _, r in df.iterrows():
            all_rows.append({"model": model, "date": r["date"].strftime("%Y-%m-%d"),
                             "cycle": f"t{int(r['cycle']):02d}z"})
        dt = time.time() - t_model
        print(f"  {summary['archive_start']} -> {summary['archive_end']}: "
              f"{summary['days_with_any_cycle']}/{summary['span_days']} days, "
              f"cycles={summary['cycles_observed']}, files={summary['total_files']}, "
              f"pages={pages}, {dt:.1f}s", flush=True)

    if all_rows:
        pd.DataFrame(all_rows).to_csv("data/ofs/inventory_ncei.csv", index=False)
    pd.DataFrame(summaries).to_csv("data/ofs/inventory_ncei_summary.csv", index=False)

    print(f"\nTotal walltime: {time.time() - t0:.1f}s")
    print("\n=== NCEI Inventory summary ===")
    print(pd.DataFrame(summaries).to_string(index=False))
