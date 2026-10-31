"""
x_ofs_pair.py — build matched (AI, OFS, obs) triples for one region/OFS pair.

Given a finished AI inference archive (`forecast.nc`) and a downloaded
OFS per-station archive, emit a tidy parquet of every (cycle_time, gauge,
lead_hour) triple where AI, OFS, and observed values are all available.

Conventions
-----------
- AI cycle_time is on the hour at stride 48h. OFS cbofs cycles at 00/06/12/18Z.
  We keep only OFS cycle_times that exactly match an AI cycle_time.
- AI lead_hour is integer 1..48. OFS cadence is 6 minutes (lead_step), so
  AI lead h corresponds to OFS lead_step = h * 10 (lead_minutes = 60*h).
- AI predicts a residual (observed_detrended − harmonic_tide) directly.
- OFS predicts free-surface elevation in its own vertical reference. We
  convert it into the same residual space using one of two methods, picked
  by `OFS_RESIDUAL_METHOD` in the config:

  * "datum_offset" (legacy):
        ofs_residual = ofs_zeta - datum_offset - harmonic_tide
    where `datum_offset` is produced by `x_ofs_datum_bridge.py` over a
    Q1-2022 baseline window. The scalar absorbs CBOFS's geoid shift
    against NTDE-MSL **plus** the cumulative SLR component at the
    calibration epoch.

  * "slr_detrend" (symmetric, NOAA-aligned):
        ofs_residual = ofs_zeta - harmonic_tide - slr_component(valid_time)
    where `slr_component` comes from the gauge h5 (the same per-hour SLR
    signal that detrended the observed water level). This treats both
    sides identically — subtract tide and gauge SLR from each — and
    leaves any CBOFS-vs-gauge mean-state bias visible in the reported
    skill.

Reads
-----
    outputs/inference/{run}/region_{XX}/{span}/forecast.nc
    data/ofs/per_station/{station_id}__{model}.nc
    data/water_levels_processed/processed_water_levels.h5
    data/ofs/datum_offsets.csv   (only when METHOD == 'datum_offset')

Writes
------
    {OUTPUT_DIR}/pairs.parquet

Companion config: x_ofs_pair_config.json.
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


def build_pairs_for_station(fc, ofs_path, obs_df, station_id,
                            method, datum_offset=None):
    """Return a DataFrame with one row per (cycle_time, lead_hour) for this
    station, holding ai/ofs/obs residuals plus the OFS cycle_time match flag.

    `method` selects how `ofs_residual` is computed:
        'datum_offset' — `ofs_zeta - datum_offset - harmonic_tide` (legacy).
        'slr_detrend'  — `ofs_zeta - harmonic_tide - slr_component`
                         (symmetric with the gauge detrend).
    """
    ofs = xr.open_dataset(ofs_path)
    try:
        ai_ct = pd.DatetimeIndex(fc.cycle_time.values)
        ofs_ct = pd.DatetimeIndex(ofs.cycle_time.values)
        # cycles where both AI and OFS have a run
        shared = ai_ct.intersection(ofs_ct)
        if len(shared) == 0:
            return None

        # AI residual at this station, all leads, on shared cycles
        # forecast.nc 'station' is a string array — match by string equality
        stations_str = fc.station.values.astype(str)
        if station_id not in stations_str:
            return None
        s_idx = int(np.where(stations_str == station_id)[0][0])

        ai_sub = fc.residual_mean.sel(cycle_time=shared).isel(station=s_idx)
        # shape (cycle, lead_hour)
        ai_arr = ai_sub.values
        lead_hours = fc.lead_hour.values.astype(int)  # 1..48

        # Map OFS cycles via integer index
        ofs_idx = pd.Index(ofs_ct).get_indexer(shared)
        # AI lead h -> OFS step idx = h*10
        ofs_step_idx = lead_hours * 10
        # Bound check (some OFS files may have shorter horizons)
        n_steps = ofs.sizes['lead_step']
        ok_steps = ofs_step_idx < n_steps
        if not ok_steps.all():
            ofs_step_idx = ofs_step_idx[ok_steps]
            lead_hours = lead_hours[ok_steps]
            ai_arr = ai_arr[:, ok_steps]

        ofs_zeta = ofs.zeta.values[np.ix_(ofs_idx, ofs_step_idx)]  # (cycle, lead)

        # Build long-form dataframe
        n_c, n_l = ai_arr.shape
        cyc_grid = np.repeat(shared.values, n_l)
        lead_grid = np.tile(lead_hours, n_c)
        ai_flat = ai_arr.reshape(-1)
        zeta_flat = ofs_zeta.reshape(-1)
        valid_time = cyc_grid + (lead_grid * np.timedelta64(1, 'h'))

        df = pd.DataFrame({
            'cycle_time': cyc_grid,
            'lead_hour': lead_grid,
            'valid_time': valid_time,
            'ai_residual': ai_flat,
            'ofs_zeta': zeta_flat,
        })

        # Join harmonic + obs + SLR signal by valid_time
        joined = df.join(
            obs_df[['predicted', 'residual', 'slr_component']].rename(
                columns={'predicted': 'harmonic_tide', 'residual': 'obs_residual'}),
            on='valid_time',
        )

        # Convert OFS to residual space — see module docstring for the two methods.
        if method == 'datum_offset':
            if datum_offset is None:
                raise ValueError("method='datum_offset' requires a datum_offset value")
            joined['ofs_residual'] = (
                joined['ofs_zeta'] - datum_offset - joined['harmonic_tide']
            )
            joined['datum_offset'] = datum_offset
        elif method == 'slr_detrend':
            joined['ofs_residual'] = (
                joined['ofs_zeta'] - joined['harmonic_tide'] - joined['slr_component']
            )
        else:
            raise ValueError(f"unknown OFS_RESIDUAL_METHOD={method!r} "
                             "(expected 'datum_offset' or 'slr_detrend')")

        joined['station_id'] = station_id
        joined['ofs_residual_method'] = method

        return joined
    finally:
        ofs.close()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    cfg = load_config('x_ofs_pair_config.json')
    out_dir = cfg['OUTPUT_DIR']
    os.makedirs(out_dir, exist_ok=True)

    fc = open_forecast(cfg['RUN_NAME'], cfg['REGION_ID'], cfg['INFERENCE_SPAN'])
    print(f"forecast.nc: {fc.sizes}")

    method = cfg.get('OFS_RESIDUAL_METHOD', 'datum_offset')
    print(f"OFS_RESIDUAL_METHOD: {method}")

    if method == 'datum_offset':
        datum = pd.read_csv(cfg['DATUM_OFFSETS_CSV'])
        datum = datum[(datum['ofs_model'] == cfg['OFS_MODEL']) &
                      datum['offset'].notna()].copy()
        datum['station_id'] = datum['station_id'].astype(str)
        print(f"datum offsets: {len(datum)} stations for model={cfg['OFS_MODEL']}")
    elif method == 'slr_detrend':
        datum = None
        print(f"slr_detrend mode: ofs_residual = ofs_zeta - harmonic_tide - "
              f"slr_component (no datum offsets file used)")
    else:
        raise ValueError(f"OFS_RESIDUAL_METHOD={method!r} not in "
                         "{'datum_offset', 'slr_detrend'}")

    fc_stations = set(fc.station.values.astype(str).tolist())

    all_pairs = []
    n_skipped_no_ofs = 0
    n_skipped_no_offset = 0
    n_skipped_no_obs = 0

    with pd.HDFStore('data/water_levels_processed/processed_water_levels.h5', 'r') as wl:
        for sid in sorted(fc_stations):
            ofs_path = f"data/ofs/per_station/{sid}__{cfg['OFS_MODEL']}.nc"
            if not os.path.exists(ofs_path):
                n_skipped_no_ofs += 1
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

            df = build_pairs_for_station(fc, ofs_path, obs_df, sid,
                                         method=method, datum_offset=offset)
            if df is None:
                n_skipped_no_ofs += 1
                continue

            all_pairs.append(df)
            print(f"  {sid}: {len(df):>6d} rows  "
                  f"(ai_nan={df['ai_residual'].isna().sum()}, "
                  f"ofs_nan={df['ofs_zeta'].isna().sum()}, "
                  f"obs_nan={df['obs_residual'].isna().sum()})")

    pairs = pd.concat(all_pairs, ignore_index=True)
    pairs['region_id'] = cfg['REGION_ID']
    pairs['ofs_model'] = cfg['OFS_MODEL']
    pairs['run_name'] = cfg['RUN_NAME']

    print(f"\nTotal rows before drop: {len(pairs)}")
    before = len(pairs)
    pairs_clean = pairs.dropna(subset=['ai_residual', 'ofs_residual', 'obs_residual'])
    print(f"Rows after drop NaN:    {len(pairs_clean)}  ({before - len(pairs_clean)} dropped)")
    print(f"Stations:               {pairs_clean['station_id'].nunique()}")
    print(f"Cycles:                 {pairs_clean['cycle_time'].nunique()}")
    print(f"Skipped: no OFS file={n_skipped_no_ofs}, no offset={n_skipped_no_offset}, no obs={n_skipped_no_obs}")

    out_path = os.path.join(out_dir, 'pairs.parquet')
    pairs_clean.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Quick sanity: bias by lead bin (rough check)
    pairs_clean = pairs_clean.copy()
    pairs_clean['lead_bin'] = pd.cut(pairs_clean['lead_hour'],
                                     bins=[0, 12, 24, 36, 48],
                                     labels=['1-12', '13-24', '25-36', '37-48'])
    summary = pairs_clean.groupby('lead_bin', observed=True).agg(
        n=('ai_residual', 'size'),
        ai_mae_cm=('ai_residual', lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).abs().mean()),
        ofs_mae_cm=('ofs_residual', lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).abs().mean()),
        ai_bias_cm=('ai_residual', lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).mean()),
        ofs_bias_cm=('ofs_residual', lambda x: 100 * (x - pairs_clean.loc[x.index, 'obs_residual']).mean()),
    )
    print("\nQuick lead-bin summary:")
    print(summary.round(2).to_string())

    fc.close()
