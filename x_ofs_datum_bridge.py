"""
x_ofs_datum_bridge.py — per-(gauge, OFS-model) datum offset.

OFS `zeta` is free-surface elevation in the model's vertical reference (a
post-spinup time-mean state, not NOAA MSL). NOAA `observed_detrended` is
hourly water level with the per-station SLR trend removed, in MSL.

Both series should differ by a roughly constant scalar — the per-pair datum
offset. Estimate it as

    offset = mean( ofs_zeta - obs.observed_detrended )

evaluated on the lead-0 OFS analysis snapshot at every cycle inside the
baseline window, paired with the hourly observation at the same UTC time.
The lead-0 OFS state is the closest to "observed sea state at cycle time",
so this isolates the datum bias from OFS forecast drift.

Reads:
    data/ofs/per_station/{station_id}__{model}.nc
    data/water_levels_processed/processed_water_levels.h5
    data/ofs/crosswalk.csv

Writes:
    data/ofs/datum_offsets.csv  (configurable via OUTPUT_PATH)

Companion config: x_ofs_datum_bridge_config.json.
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


def cycle_offset_for_pair(per_station_path, wl_store, station_id,
                          baseline_start, baseline_end, min_samples):
    """One scalar offset for one (station, model) pair.

    Returns dict with offset, n_samples, and basic diagnostics. Returns
    None if the pair has fewer than min_samples valid cycle-time matches.
    """
    if not os.path.exists(per_station_path):
        return None

    ofs = xr.open_dataset(per_station_path)
    try:
        # Lead-0 zeta at every cycle in the baseline window
        ct = pd.DatetimeIndex(ofs.cycle_time.values)
        mask = (ct >= pd.Timestamp(baseline_start)) & (ct <= pd.Timestamp(baseline_end))
        if not mask.any():
            return None

        zeta_lead0 = ofs.zeta.isel(lead_step=0).values[mask]  # (n_cycles,)
        ct_baseline = ct[mask]
    finally:
        ofs.close()

    # Pull obs at the matching cycle hours (round to hour, since OFS cycles
    # are on the hour — 00/06/12/18 for cbofs).
    key = f"/station_{station_id}"
    if key not in wl_store:
        return None
    df = wl_store[key]
    obs_idx = df.index.tz_convert(None) if df.index.tz else df.index
    obs_at = df['observed_detrended'].copy()
    obs_at.index = obs_idx

    # Hourly snap. ct_baseline is already on the hour.
    paired = pd.DataFrame({'zeta': zeta_lead0}, index=ct_baseline)
    paired = paired.join(obs_at.rename('obs_detrended'), how='inner')
    paired = paired.dropna()

    if len(paired) < min_samples:
        return {
            'station_id': station_id,
            'n_samples': int(len(paired)),
            'offset': np.nan,
            'mean_zeta': float(paired['zeta'].mean()) if len(paired) else np.nan,
            'mean_obs_detrended': float(paired['obs_detrended'].mean()) if len(paired) else np.nan,
            'std_diff': np.nan,
            'note': f'insufficient samples (<{min_samples})',
        }

    diff = paired['zeta'] - paired['obs_detrended']
    return {
        'station_id': station_id,
        'n_samples': int(len(paired)),
        'offset': float(diff.mean()),
        'mean_zeta': float(paired['zeta'].mean()),
        'mean_obs_detrended': float(paired['obs_detrended'].mean()),
        'std_diff': float(diff.std()),
        'note': '',
    }


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    cfg = load_config(os.environ.get('X_OFS_DATUM_CONFIG',
                                     'x_ofs_datum_bridge_config_v2.json'))
    crosswalk = pd.read_csv('data/ofs/crosswalk.csv')
    crosswalk = crosswalk[crosswalk['matched']].copy()

    rows = []
    # H5 store is config-driven so the v2 run measures offsets against the
    # v2-detrended gauges (the frame the v2 figure pairs in). Defaults to v1.
    wl_h5 = cfg.get('WATER_LEVELS_H5',
                    'data/water_levels_processed/processed_water_levels.h5')
    print(f"Observed gauges: {wl_h5}")
    with pd.HDFStore(wl_h5, 'r') as wl_store:
        for model in cfg['OFS_MODELS']:
            sub = crosswalk[crosswalk['ofs_model'] == model]
            print(f'[{model}] {len(sub)} matched pairs')
            for _, row in sub.iterrows():
                sid = str(row['station_id'])
                pth = f'data/ofs/per_station/{sid}__{model}.nc'
                rec = cycle_offset_for_pair(
                    per_station_path=pth,
                    wl_store=wl_store,
                    station_id=sid,
                    baseline_start=cfg['BASELINE_START'],
                    baseline_end=cfg['BASELINE_END'],
                    min_samples=cfg['MIN_SAMPLES'],
                )
                if rec is None:
                    continue
                rec['ofs_model'] = model
                rec['station_name'] = row['station_name']
                rec['region'] = int(row['region'])
                rec['baseline_start'] = cfg['BASELINE_START']
                rec['baseline_end'] = cfg['BASELINE_END']
                rows.append(rec)
                print(f"  {sid:>10s} {row['station_name']:<32s} "
                      f"n={rec['n_samples']:>4d} offset={rec['offset']:+.4f} m  "
                      f"std={rec['std_diff']:.4f}")

    out = pd.DataFrame(rows)
    cols = ['ofs_model', 'station_id', 'station_name', 'region',
            'offset', 'n_samples', 'mean_zeta', 'mean_obs_detrended',
            'std_diff', 'baseline_start', 'baseline_end', 'note']
    out = out[[c for c in cols if c in out.columns]]
    os.makedirs(os.path.dirname(cfg['OUTPUT_PATH']), exist_ok=True)
    out.to_csv(cfg['OUTPUT_PATH'], index=False)
    print(f"\nWrote {len(out)} rows to {cfg['OUTPUT_PATH']}")
