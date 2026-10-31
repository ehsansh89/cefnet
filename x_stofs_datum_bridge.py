"""
x_stofs_datum_bridge.py — per-(gauge, STOFS source) datum offset.

Direct sibling of `x_ofs_datum_bridge.py`. STOFS `zeta`-like fields are stored
in the model's geodetic reference (`sea_surface_height_above_geoid`), not
NOAA-MSL. The free-surface vs NOAA-detrended series therefore differ by a
roughly constant scalar — the per-pair datum offset. Estimate it as

    offset = mean( stofs_value(lead_min=0) - obs.observed_detrended )

evaluated at every cycle inside a chosen baseline window, paired with the
hourly observation at the cycle's UTC time. We use the analysis snapshot at
lead_minutes = 0 (cycle init), not lead_step = 0 (which is at -354 min on
STOFS's grid) — finding #3 from the code review: index by VALUE on
`lead_minutes`, never by positional index.

For `swl` (sub-tidal), STOFS has already removed its own harmonic tide, so
the offset captures the geoid-vs-MSL shift plus any STOFS-tide-vs-NOAA-tide
residual at the analysis time. For `cwl` and `cwl_noanomaly`, tide remains in
both `stofs_value` and `observed_detrended`, so tide cancels out and the
offset is the pure datum delta.

Reads:
    data/stofs/per_station/{station_id}__stofs2d.nc
    data/water_levels_processed/processed_water_levels.h5
    data/stofs/crosswalk.csv

Writes:
    data/stofs/datum_offsets.csv  (configurable via OUTPUT_PATH)

Companion config: x_stofs_datum_bridge_config.json.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr


def load_config(path):
    with open(path) as f:
        return json.load(f)


def lead_zero_index(lead_minutes_arr):
    """Return the lead_step positional index where lead_minutes == 0.
    Raises if no exact match — STOFS native grid always contains 0."""
    lm = np.asarray(lead_minutes_arr).astype(np.int64)
    hits = np.where(lm == 0)[0]
    if hits.size == 0:
        raise ValueError(f"lead_minutes=0 not present; got range {lm.min()}..{lm.max()}")
    return int(hits[0])


def cycle_offset_for_pair(per_station_path, wl_store, station_id, source,
                          baseline_start, baseline_end, min_samples):
    """One scalar offset for one (station, source) pair, or None if the file
    is missing. Returns a dict with offset, n_samples, and diagnostics."""
    if not os.path.exists(per_station_path):
        return None

    stofs = xr.open_dataset(per_station_path)
    try:
        if source not in stofs.variables:
            return None
        ct = pd.DatetimeIndex(stofs.cycle_time.values)
        mask = (ct >= pd.Timestamp(baseline_start)) & (ct <= pd.Timestamp(baseline_end))
        if not mask.any():
            return None

        idx0 = lead_zero_index(stofs.lead_minutes.values)
        v = stofs[source].isel(lead_step=idx0).values  # (n_cycles,)
        v = v[mask]
        ct_baseline = ct[mask]
    finally:
        stofs.close()

    key = f"/station_{station_id}"
    if key not in wl_store:
        return None
    df = wl_store[key]
    obs_idx = df.index.tz_convert(None) if df.index.tz else df.index
    obs_at = df['observed_detrended'].copy()
    obs_at.index = obs_idx

    paired = pd.DataFrame({'val': v}, index=ct_baseline)
    paired = paired.join(obs_at.rename('obs_detrended'), how='inner')
    paired = paired.dropna()

    if len(paired) < min_samples:
        return {
            'station_id': station_id,
            'stofs_source': source,
            'n_samples': int(len(paired)),
            'offset': np.nan,
            'mean_stofs': float(paired['val'].mean()) if len(paired) else np.nan,
            'mean_obs_detrended': float(paired['obs_detrended'].mean()) if len(paired) else np.nan,
            'std_diff': np.nan,
            'note': f'insufficient samples (<{min_samples})',
        }

    diff = paired['val'] - paired['obs_detrended']
    return {
        'station_id': station_id,
        'stofs_source': source,
        'n_samples': int(len(paired)),
        'offset': float(diff.mean()),
        'mean_stofs': float(paired['val'].mean()),
        'mean_obs_detrended': float(paired['obs_detrended'].mean()),
        'std_diff': float(diff.std()),
        'note': '',
    }


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    cfg = load_config('x_stofs_datum_bridge_config.json')
    crosswalk = pd.read_csv('data/stofs/crosswalk.csv')
    crosswalk = crosswalk[crosswalk['matched']].copy()
    crosswalk['station_id'] = crosswalk['station_id'].astype(str)

    rows = []
    with pd.HDFStore('data/water_levels_processed/processed_water_levels.h5', 'r') as wl_store:
        for source in cfg['STOFS_SOURCES']:
            print(f'[{source}] {len(crosswalk)} matched pairs')
            for _, row in crosswalk.iterrows():
                sid = str(row['station_id'])
                pth = f'data/stofs/per_station/{sid}__stofs2d.nc'
                rec = cycle_offset_for_pair(
                    per_station_path=pth, wl_store=wl_store,
                    station_id=sid, source=source,
                    baseline_start=cfg['BASELINE_START'],
                    baseline_end=cfg['BASELINE_END'],
                    min_samples=cfg['MIN_SAMPLES'],
                )
                if rec is None:
                    continue
                rec['station_name'] = row.get('station_name', '')
                rec['region'] = int(row['region']) if pd.notna(row.get('region')) else -1
                rec['baseline_start'] = cfg['BASELINE_START']
                rec['baseline_end'] = cfg['BASELINE_END']
                rows.append(rec)
                if not np.isnan(rec.get('offset', np.nan)):
                    print(f"  {sid:>10s} {str(rec['station_name'])[:32]:<32s} "
                          f"n={rec['n_samples']:>4d} offset={rec['offset']:+.4f} m  "
                          f"std={rec['std_diff']:.4f}")

    out = pd.DataFrame(rows)
    cols = ['stofs_source', 'station_id', 'station_name', 'region',
            'offset', 'n_samples', 'mean_stofs', 'mean_obs_detrended',
            'std_diff', 'baseline_start', 'baseline_end', 'note']
    out = out[[c for c in cols if c in out.columns]]
    os.makedirs(os.path.dirname(cfg['OUTPUT_PATH']), exist_ok=True)
    out.to_csv(cfg['OUTPUT_PATH'], index=False)
    print(f"\nWrote {len(out)} rows to {cfg['OUTPUT_PATH']}")
