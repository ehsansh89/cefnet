"""
x_stofs_pair.py — build matched (AI, STOFS-2D-Global, obs) triples for one region.

Direct sibling of `x_ofs_pair.py`. The OFS layer compares the AI residual
forecast against a regional ROMS/FVCOM model; this layer compares it against
NOAA's CONUS-wide STOFS-2D-Global ADCIRC product. Both feed the same downstream
skill scripts (`o_stofs_skill.py` / `stofs_all_regions.py`).

Conventions
-----------
- AI cycle_time is on the hour at stride 48h. STOFS-2D-Global cycles at
  00/06/12/18Z. We keep only STOFS cycle_times that exactly match an AI
  cycle_time.
- AI lead_hour is integer 1..72. STOFS native cadence is 6 min on a
  `lead_step` axis whose `lead_minutes` coordinate runs -354..+10800. We
  always select STOFS by **lead_minutes value** (h * 60), never by
  positional index, because the 6-min grid is not perfectly aligned to t-360.
- AI predicts a residual (observed_detrended - harmonic_tide) directly.
- STOFS publishes three useful per-cycle products inside the per-station NC:
    `cwl`            — combined water level (bias-corrected against CO-OPS
                       observations on/after 2024-05-14 12Z)
    `cwl_noanomaly`  — raw model run, post-v2.1.7 only (NaN before)
    `swl`            — sub-tidal water level (STOFS's own surge product)
  Pick one as the comparand via STOFS_SOURCE ∈ {'cwl', 'cwl_noanomaly', 'swl'}.
  For an honest 'fair' comparison row that doesn't peek at NOAA observations,
  use 'cwl_noanomaly' (and accept that pre-2024-05-14 rows will be NaN).

- STOFS `zeta` is geodetic (`sea_surface_height_above_geoid`). Convert into
  the AI's residual space using one of two methods (parallel to OFS), set by
  STOFS_RESIDUAL_METHOD:

  * "datum_offset" (legacy / OFS-parallel):
        stofs_residual = stofs_zeta - datum_offset - harmonic_tide
    `datum_offset` comes from `x_stofs_datum_bridge.py` over a baseline window.

  * "slr_detrend" (symmetric, NOAA-aligned):
        stofs_residual = stofs_zeta - harmonic_tide - slr_component
    Subtract tide and gauge SLR component from both sides. Leaves any
    STOFS-vs-gauge mean-state bias visible in the reported skill.

  When STOFS_SOURCE == 'swl', the tide is already removed by STOFS, so:
    'datum_offset': stofs_residual = swl - datum_offset
    'slr_detrend' : stofs_residual = swl - slr_component

Reads
-----
    outputs/inference/{run}/region_{XX}/{span}/forecast.nc
    data/stofs/per_station/{station_id}__stofs2d.nc
    data/water_levels_processed/processed_water_levels.h5
    data/stofs/datum_offsets.csv   (only when METHOD == 'datum_offset')

Writes
------
    {OUTPUT_DIR}/pairs.parquet

Companion config: x_stofs_pair_config.json.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr


STOFS_MODEL_KEY = "stofs2d_global"


def load_config(path):
    with open(path) as f:
        return json.load(f)


def open_forecast(run_name, region_id, span):
    region_str = f'region_{region_id:02d}'
    fc_path = f'outputs/inference/{run_name}/{region_str}/{span}/forecast.nc'
    return xr.open_dataset(fc_path)


def load_obs_for_station(wl_store, station_id):
    """Return DataFrame with columns ['observed_detrended', 'predicted',
    'residual', 'slr_component'] indexed by tz-naive UTC hourly timestamps."""
    key = f'/station_{station_id}'
    if key not in wl_store:
        return None
    df = wl_store[key][['observed_detrended', 'predicted', 'residual',
                        'slr_component']].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


def stofs_lead_indices(lead_minutes_arr, ai_lead_hours):
    """Map AI lead hours (1..72) to the STOFS lead_step positional index
    by VALUE on `lead_minutes`, not by position. STOFS's grid is 6-min
    cadence starting at lead_minutes ≈ -354, so positional indexing would
    silently shift the lead by 6 min. Returns (idx_arr, ok_mask)."""
    target = ai_lead_hours.astype(np.int64) * 60
    lm = np.asarray(lead_minutes_arr).astype(np.int64)
    idx = np.searchsorted(lm, target)
    ok = np.zeros_like(target, dtype=bool)
    for i, t in enumerate(target):
        if idx[i] < lm.size and lm[idx[i]] == t:
            ok[i] = True
        elif idx[i] > 0 and lm[idx[i] - 1] == t:
            idx[i] -= 1
            ok[i] = True
    return idx, ok


def build_pairs_for_station(fc, stofs_path, obs_df, station_id,
                            source, method, datum_offset=None):
    """Return one DataFrame per (cycle_time, lead_hour) row for this station.

    `source` ∈ {'cwl', 'cwl_noanomaly', 'swl'} selects the STOFS variable.
    `method` selects how `stofs_residual` is computed (see module docstring).
    """
    stofs = xr.open_dataset(stofs_path)
    try:
        ai_ct = pd.DatetimeIndex(fc.cycle_time.values)
        stofs_ct = pd.DatetimeIndex(stofs.cycle_time.values)
        shared = ai_ct.intersection(stofs_ct)
        if len(shared) == 0:
            return None

        stations_str = fc.station.values.astype(str)
        if station_id not in stations_str:
            return None
        s_idx = int(np.where(stations_str == station_id)[0][0])

        ai_sub = fc.residual_mean.sel(cycle_time=shared).isel(station=s_idx)
        ai_arr = ai_sub.values
        lead_hours = fc.lead_hour.values.astype(np.int64)  # 1..72

        stofs_cycle_idx = pd.Index(stofs_ct).get_indexer(shared)
        lead_minutes = stofs.lead_minutes.values
        stofs_step_idx, ok_steps = stofs_lead_indices(lead_minutes, lead_hours)
        if not ok_steps.all():
            stofs_step_idx = stofs_step_idx[ok_steps]
            lead_hours = lead_hours[ok_steps]
            ai_arr = ai_arr[:, ok_steps]

        if source not in stofs.variables:
            raise KeyError(f"STOFS source '{source}' missing in {stofs_path}")
        stofs_z = stofs[source].values[np.ix_(stofs_cycle_idx, stofs_step_idx)]

        n_c, n_l = ai_arr.shape
        cyc_grid = np.repeat(shared.values, n_l)
        lead_grid = np.tile(lead_hours, n_c)
        ai_flat = ai_arr.reshape(-1)
        zeta_flat = stofs_z.reshape(-1)
        valid_time = cyc_grid + (lead_grid * np.timedelta64(1, 'h'))

        df = pd.DataFrame({
            'cycle_time': cyc_grid,
            'lead_hour': lead_grid,
            'valid_time': valid_time,
            'ai_residual': ai_flat,
            'stofs_zeta': zeta_flat,
        })

        # Join harmonic + obs + SLR signal by valid_time
        joined = df.join(
            obs_df[['predicted', 'residual', 'slr_component']].rename(
                columns={'predicted': 'harmonic_tide', 'residual': 'obs_residual'}),
            on='valid_time',
        )

        # Convert STOFS to residual space.
        is_swl = (source == 'swl')   # tide already removed by STOFS
        if method == 'datum_offset':
            if datum_offset is None:
                raise ValueError("method='datum_offset' requires a datum_offset value")
            if is_swl:
                joined['stofs_residual'] = joined['stofs_zeta'] - datum_offset
            else:
                joined['stofs_residual'] = (
                    joined['stofs_zeta'] - datum_offset - joined['harmonic_tide']
                )
            joined['datum_offset'] = datum_offset
        elif method == 'slr_detrend':
            if is_swl:
                joined['stofs_residual'] = (
                    joined['stofs_zeta'] - joined['slr_component']
                )
            else:
                joined['stofs_residual'] = (
                    joined['stofs_zeta'] - joined['harmonic_tide']
                    - joined['slr_component']
                )
        else:
            raise ValueError(f"unknown STOFS_RESIDUAL_METHOD={method!r} "
                             "(expected 'datum_offset' or 'slr_detrend')")

        joined['station_id'] = station_id
        joined['stofs_source'] = source
        joined['stofs_residual_method'] = method

        return joined
    finally:
        stofs.close()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    cfg = load_config('x_stofs_pair_config.json')
    out_dir = cfg['OUTPUT_DIR']
    os.makedirs(out_dir, exist_ok=True)

    fc = open_forecast(cfg['RUN_NAME'], cfg['REGION_ID'], cfg['INFERENCE_SPAN'])
    print(f"forecast.nc: {fc.sizes}")

    source = cfg.get('STOFS_SOURCE', 'cwl')
    method = cfg.get('STOFS_RESIDUAL_METHOD', 'slr_detrend')
    print(f"STOFS_SOURCE          : {source}")
    print(f"STOFS_RESIDUAL_METHOD : {method}")

    if method == 'datum_offset':
        datum = pd.read_csv(cfg['DATUM_OFFSETS_CSV'])
        datum = datum[(datum['stofs_source'] == source) &
                      datum['offset'].notna()].copy()
        datum['station_id'] = datum['station_id'].astype(str)
        print(f"datum offsets: {len(datum)} stations for source={source}")
    elif method == 'slr_detrend':
        datum = None
        print("slr_detrend mode: stofs_residual = stofs_zeta - harmonic_tide - "
              "slr_component (no datum offsets file used)")
    else:
        raise ValueError(f"STOFS_RESIDUAL_METHOD={method!r} not in "
                         "{'datum_offset', 'slr_detrend'}")

    fc_stations = set(fc.station.values.astype(str).tolist())

    all_pairs = []
    n_skipped_no_stofs = 0
    n_skipped_no_offset = 0
    n_skipped_no_obs = 0

    with pd.HDFStore('data/water_levels_processed/processed_water_levels.h5', 'r') as wl:
        for sid in sorted(fc_stations):
            stofs_path = f"data/stofs/per_station/{sid}__stofs2d.nc"
            if not os.path.exists(stofs_path):
                n_skipped_no_stofs += 1
                continue

            offset = None
            if method == 'datum_offset':
                row = datum[datum['station_id'] == sid]
                if row.empty:
                    n_skipped_no_offset += 1
                    continue
                offset = float(row['offset'].iloc[0])

            obs_df = load_obs_for_station(wl, sid)
            if obs_df is None:
                n_skipped_no_obs += 1
                continue

            df = build_pairs_for_station(fc, stofs_path, obs_df, sid,
                                         source=source, method=method,
                                         datum_offset=offset)
            if df is None:
                n_skipped_no_stofs += 1
                continue

            all_pairs.append(df)
            print(f"  {sid}: {len(df):>6d} rows  "
                  f"(ai_nan={df['ai_residual'].isna().sum()}, "
                  f"stofs_nan={df['stofs_zeta'].isna().sum()}, "
                  f"obs_nan={df['obs_residual'].isna().sum()})")

    if not all_pairs:
        print("\nNo pairs produced; nothing to write.")
        raise SystemExit(1)

    pairs = pd.concat(all_pairs, ignore_index=True)
    pairs['region_id'] = cfg['REGION_ID']
    pairs['stofs_model'] = STOFS_MODEL_KEY
    pairs['run_name'] = cfg['RUN_NAME']

    print(f"\nTotal rows before drop: {len(pairs)}")
    before = len(pairs)
    pairs_clean = pairs.dropna(subset=['ai_residual', 'stofs_residual', 'obs_residual'])
    print(f"Rows after drop NaN:    {len(pairs_clean)}  ({before - len(pairs_clean)} dropped)")
    print(f"Stations:               {pairs_clean['station_id'].nunique()}")
    print(f"Cycles:                 {pairs_clean['cycle_time'].nunique()}")
    print(f"Skipped: no STOFS file={n_skipped_no_stofs}, no offset={n_skipped_no_offset}, no obs={n_skipped_no_obs}")

    out_path = os.path.join(out_dir, 'pairs.parquet')
    pairs_clean.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Quick sanity: lead-bin pooled metrics
    pairs_clean = pairs_clean.copy()
    pairs_clean['lead_bin'] = pd.cut(pairs_clean['lead_hour'],
                                     bins=[0, 12, 24, 36, 48, 72],
                                     labels=['1-12', '13-24', '25-36', '37-48', '49-72'])
    summary = pairs_clean.groupby('lead_bin', observed=True).agg(
        n=('ai_residual', 'size'),
        ai_mae_cm=('ai_residual',
                   lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).abs().mean()),
        stofs_mae_cm=('stofs_residual',
                      lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).abs().mean()),
        ai_bias_cm=('ai_residual',
                    lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).mean()),
        stofs_bias_cm=('stofs_residual',
                       lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).mean()),
    )
    print("\nQuick lead-bin summary:")
    print(summary.round(2).to_string())

    fc.close()
