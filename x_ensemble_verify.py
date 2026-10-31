"""
x_ensemble_verify.py — Align AI ensemble + observations + flood thresholds +
STOFS + regional OFS into a single canonical `verify_arrays.nc` per region.

Reads:
    outputs/inference_ensemble/{run}/{exp_tag}/{span}/region_XX/ensemble_forecast.nc
    data/stofs/per_station/{station_id}__stofs2d.nc       (where present)
    data/ofs/per_station/{station_id}__{model}.nc          (where present)
    data/water_levels_processed/processed_water_levels.h5  (obs + tide + slr)
    data/stations_processed/stations_processed.csv         (flood thresholds)

Writes:
    outputs/uq_postprocessing/{run}/{exp_tag}/{span}/region_XX/
        verify_arrays.nc
        verify_manifest.json
        station_thresholds.csv

Schema (verify_arrays.nc):
    dims:   cycle_time, atm_member, model_seed, lead_hour, station
    5-D AI vars:  ai_residual_member, ai_water_level_member, member_valid
    3-D vars (broadcast over member/seed at metric time):
        obs_residual, obs_water_level
        stofs_residual, stofs_water_level      (NaN where no STOFS file/match)
        regional_ofs_residual, regional_ofs_water_level
                                                (NaN where no OFS file/match)
        tide, slr_component
    1-D station vars:
        minor_threshold, moderate_threshold, major_threshold  (m above NTDE-MSL)
        ofs_model_name                                         (str)

Convention: residuals are in the AI's space (observed_detrended − tide). For
STOFS/OFS we use slr_detrend:  baseline_residual = baseline_zeta − tide − slr.
This leaves any mean-state bias visible in metrics. `*_water_level` is the
reconstructed total: `residual + tide + slr_component`.
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import xarray as xr


CONFIG_PATH = os.environ.get('X_ENSEMBLE_VERIFY_CONFIG',
                             'x_ensemble_verify_config_v2.json')

OFS_PRIORITY = ['cbofs', 'ngofs2', 'tbofs', 'nyofs', 'sfbofs',
                'dbofs', 'gomofs', 'wcofs']


# =============================================================================
# HELPERS
# =============================================================================
def find_ofs_path(station_id, ofs_dir='data/ofs/per_station'):
    """Return the most-preferred OFS path for a station, or (None, None)."""
    for model in OFS_PRIORITY:
        p = os.path.join(ofs_dir, f'{station_id}__{model}.nc')
        if os.path.exists(p):
            return p, model
    return None, None


def stofs_path(station_id, stofs_dir='data/stofs/per_station'):
    p = os.path.join(stofs_dir, f'{station_id}__stofs2d.nc')
    return p if os.path.exists(p) else None


def _baseline_lead_indices(lead_minutes_arr, ai_lead_hours):
    """Map AI lead hours -> nearest baseline `lead_step` positional index by
    `lead_minutes` value. Returns (idx_arr_int64, ok_mask)."""
    target = ai_lead_hours.astype(np.int64) * 60
    valid = ~np.isnan(lead_minutes_arr.astype(np.float64))
    lm = lead_minutes_arr.astype(np.int64)
    # searchsorted then prefer exact match.
    idx = np.searchsorted(lm, target, side='left')
    idx = np.clip(idx, 0, len(lm) - 1)
    ok = np.zeros(len(target), dtype=bool)
    for i, t in enumerate(target):
        j = idx[i]
        if j < len(lm) and lm[j] == t and valid[j]:
            ok[i] = True
        elif j - 1 >= 0 and lm[j - 1] == t and valid[j - 1]:
            idx[i] = j - 1
            ok[i] = True
    return idx, ok


def slr_detrend_baseline(zeta_arr, tide_arr, slr_arr, source_kind):
    """Convert a baseline forecast (STOFS cwl or OFS zeta) into the AI's
    residual space using slr_detrend.

    For STOFS swl (already tide-removed), pass source_kind='swl' to use
    `residual = swl - slr`. Otherwise (CWL or OFS zeta) we subtract both.
    """
    if source_kind == 'swl':
        return zeta_arr - slr_arr
    return zeta_arr - tide_arr - slr_arr


def load_h5_aligned(h5_path, station_ids, cycle_times, forecast_hours):
    """Reuse the same loader as inference_ensemble but expose tide/slr/obs.
    Returns dict of (n_cycles, lead, n_sta) arrays.
    """
    n_c = len(cycle_times)
    n_l = forecast_hours
    n_s = len(station_ids)
    tide = np.full((n_c, n_l, n_s), np.nan, dtype=np.float32)
    obs_wl = np.full_like(tide, np.nan)
    obs_res = np.full_like(tide, np.nan)
    slr = np.full_like(tide, np.nan)
    obs_detrended = np.full_like(tide, np.nan)

    cycle_ts = [pd.Timestamp(c) for c in cycle_times]
    all_valid_times = pd.DatetimeIndex(sorted({
        ts + pd.Timedelta(hours=h)
        for ts in cycle_ts for h in range(1, n_l + 1)
    }))
    t2i = pd.Series(np.arange(len(all_valid_times)), index=all_valid_times)

    with pd.HDFStore(h5_path, mode='r') as store:
        keys = set(store.keys())
        for s_idx, sid in enumerate(station_ids):
            key = f'/station_{sid}'
            if key not in keys:
                continue
            df = store.get(key)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep='first')]
            sub = df.reindex(all_valid_times)
            for c_idx, ts in enumerate(cycle_ts):
                valid_times = pd.date_range(ts + pd.Timedelta(hours=1),
                                             periods=n_l, freq='h')
                idx = t2i.reindex(valid_times).values.astype(int)
                if 'predicted' in df.columns:
                    tide[c_idx, :, s_idx] = sub['predicted'].values[idx]
                if 'observed_detrended' in df.columns:
                    obs_detrended[c_idx, :, s_idx] = sub['observed_detrended'].values[idx]
                if 'residual' in df.columns:
                    obs_res[c_idx, :, s_idx] = sub['residual'].values[idx]
                if 'slr_component' in df.columns:
                    slr[c_idx, :, s_idx] = sub['slr_component'].values[idx]
    # Reconstruct total observed water level (relative to NTDE-MSL +
    # current-epoch SLR):  obs_wl = observed_detrended + slr_component.
    # observed_detrended already has the SLR trend removed.
    obs_wl = obs_detrended + np.where(np.isnan(slr), 0.0, slr)
    return dict(tide=tide, obs_water_level=obs_wl, obs_residual=obs_res,
                slr_component=slr)


def load_stofs_for_region(station_ids, cycle_times, forecast_hours,
                           tide, slr, source='cwl'):
    """Return (stofs_residual, stofs_water_level), each (n_cycles, lead, n_sta).
    Uses slr_detrend to convert STOFS into the AI's residual space.

    For STOFS `swl` source: residual = swl - slr (already tide-removed).
    For STOFS `cwl` (default): residual = cwl - tide - slr.
    """
    n_c = len(cycle_times)
    n_l = forecast_hours
    n_s = len(station_ids)
    res = np.full((n_c, n_l, n_s), np.nan, dtype=np.float32)
    wl = np.full_like(res, np.nan)

    cycle_arr = pd.DatetimeIndex([pd.Timestamp(c) for c in cycle_times])
    ai_leads = np.arange(1, n_l + 1, dtype=np.int64)

    for s_idx, sid in enumerate(station_ids):
        p = stofs_path(sid)
        if p is None:
            continue
        try:
            ds = xr.open_dataset(p)
        except Exception:
            continue
        try:
            stofs_cyc = pd.DatetimeIndex(ds.cycle_time.values)
            ai_to_stofs = pd.Series(np.arange(len(stofs_cyc)),
                                     index=stofs_cyc).reindex(cycle_arr)
            mask = ~np.isnan(ai_to_stofs.values)
            if not mask.any():
                continue
            cyc_idx = ai_to_stofs.values.astype('float64')
            lm = ds.lead_minutes.values
            l_idx, l_ok = _baseline_lead_indices(lm, ai_leads)

            for c_pos, ci in enumerate(cyc_idx):
                if np.isnan(ci):
                    continue
                ci = int(ci)
                if source == 'swl':
                    series = ds['swl'].values[ci]
                else:
                    series = ds[source].values[ci]
                vals = np.where(l_ok, series[l_idx], np.nan).astype(np.float32)
                # slr_detrend conversion
                tide_v = tide[c_pos, :, s_idx]
                slr_v = np.where(np.isnan(slr[c_pos, :, s_idx]), 0.0,
                                  slr[c_pos, :, s_idx])
                if source == 'swl':
                    r = vals - slr_v.astype(np.float32)
                else:
                    r = vals - tide_v.astype(np.float32) - slr_v.astype(np.float32)
                res[c_pos, :, s_idx] = r
                wl[c_pos, :, s_idx] = (r + tide_v.astype(np.float32)
                                          + slr_v.astype(np.float32))
        finally:
            ds.close()
    return res, wl


def load_ofs_for_region(station_ids, cycle_times, forecast_hours, tide, slr):
    """Return (ofs_residual, ofs_water_level, ofs_model_per_station)."""
    n_c = len(cycle_times)
    n_l = forecast_hours
    n_s = len(station_ids)
    res = np.full((n_c, n_l, n_s), np.nan, dtype=np.float32)
    wl = np.full_like(res, np.nan)
    model_per = ['' for _ in station_ids]

    cycle_arr = pd.DatetimeIndex([pd.Timestamp(c) for c in cycle_times])
    ai_leads = np.arange(1, n_l + 1, dtype=np.int64)

    for s_idx, sid in enumerate(station_ids):
        p, model = find_ofs_path(sid)
        if p is None:
            continue
        try:
            ds = xr.open_dataset(p)
        except Exception:
            continue
        try:
            model_per[s_idx] = model
            ofs_cyc = pd.DatetimeIndex(ds.cycle_time.values)
            ai_to_ofs = pd.Series(np.arange(len(ofs_cyc)),
                                   index=ofs_cyc).reindex(cycle_arr)
            cyc_idx = ai_to_ofs.values.astype('float64')
            lm = ds.lead_minutes.values
            l_idx, l_ok = _baseline_lead_indices(lm, ai_leads)
            zeta_arr = ds['zeta'].values

            for c_pos, ci in enumerate(cyc_idx):
                if np.isnan(ci):
                    continue
                ci = int(ci)
                vals = np.where(l_ok, zeta_arr[ci, l_idx], np.nan).astype(np.float32)
                tide_v = tide[c_pos, :, s_idx].astype(np.float32)
                slr_v = np.where(np.isnan(slr[c_pos, :, s_idx]), 0.0,
                                  slr[c_pos, :, s_idx]).astype(np.float32)
                r = vals - tide_v - slr_v
                res[c_pos, :, s_idx] = r
                wl[c_pos, :, s_idx] = r + tide_v + slr_v
        finally:
            ds.close()
    return res, wl, model_per


def load_thresholds(stations_csv, station_ids):
    df = pd.read_csv(stations_csv)
    df['station_id'] = df['station_id'].astype(str)
    minor = []
    moderate = []
    major = []
    for sid in station_ids:
        row = df[df['station_id'] == str(sid)]
        if row.empty:
            minor.append(np.nan); moderate.append(np.nan); major.append(np.nan)
        else:
            minor.append(float(row['f_min_ms_m'].values[0]))
            moderate.append(float(row['f_mod_ms_m'].values[0]))
            major.append(float(row['f_maj_ms_m'].values[0]))
    return (np.asarray(minor, dtype=np.float32),
            np.asarray(moderate, dtype=np.float32),
            np.asarray(major, dtype=np.float32))


# =============================================================================
# MAIN
# =============================================================================
def verify_region(region_dir, out_dir, cfg):
    """Run verification on one region archive. Returns out paths."""
    fc_path = os.path.join(region_dir, 'ensemble_forecast.nc')
    if not os.path.exists(fc_path):
        return None

    # Chunked open: the 5-D AI arrays pass through to verify_arrays.nc
    # unchanged, so they are kept as dask arrays and STREAMED at write time
    # (a full .values load is ~2.2 GiB per variable on the largest region and
    # exhausts RAM). Everything else is small and loaded eagerly.
    ds = xr.open_dataset(fc_path, chunks={'cycle_time': 128})
    attrs = dict(ds.attrs)
    cycle_times = pd.to_datetime(ds.cycle_time.values)
    station_ids = [str(s) for s in ds.station.values]
    station_names = [str(s) for s in ds.station_name.values]
    st_lats = ds.station_lat.values
    st_lons = ds.station_lon.values
    forecast_hours = int(attrs.get('forecast_hours', 72))
    members = list(ds.atm_member.values)
    seeds = list(ds.model_seed.values)

    ai_res = ds.residual.astype(np.float32).data       # dask, lazy
    ai_wl = ds.water_level.astype(np.float32).data     # dask, lazy
    member_valid = ds.member_valid.values.astype(np.uint8)

    # ---- Obs + tide + slr ----
    h5 = load_h5_aligned(cfg['WATER_LEVELS_H5'], station_ids, cycle_times,
                         forecast_hours)
    tide = h5['tide']
    slr = h5['slr_component']
    obs_wl = h5['obs_water_level']
    obs_res = h5['obs_residual']

    # ---- STOFS ----
    stofs_source = cfg.get('STOFS_SOURCE', 'cwl')
    stofs_res, stofs_wl = load_stofs_for_region(
        station_ids, cycle_times, forecast_hours, tide, slr, source=stofs_source,
    )

    # ---- OFS ----
    ofs_res, ofs_wl, ofs_models = load_ofs_for_region(
        station_ids, cycle_times, forecast_hours, tide, slr,
    )

    # ---- Thresholds ----
    minor_thr, moderate_thr, major_thr = load_thresholds(
        cfg['STATIONS_CSV'], station_ids,
    )

    # ---- Write verify_arrays.nc ----
    os.makedirs(out_dir, exist_ok=True)
    cycle_dt = np.array([np.datetime64(pd.Timestamp(c).to_datetime64(), 'ns')
                         for c in cycle_times])
    ds_out = xr.Dataset(
        data_vars=dict(
            ai_residual_member=(
                ('cycle_time', 'atm_member', 'model_seed', 'lead_hour', 'station'),
                ai_res,
            ),
            ai_water_level_member=(
                ('cycle_time', 'atm_member', 'model_seed', 'lead_hour', 'station'),
                ai_wl,
            ),
            member_valid=(
                ('cycle_time', 'atm_member', 'model_seed', 'station'),
                member_valid,
            ),
            obs_residual=(('cycle_time', 'lead_hour', 'station'),
                            obs_res.astype(np.float32)),
            obs_water_level=(('cycle_time', 'lead_hour', 'station'),
                               obs_wl.astype(np.float32)),
            stofs_residual=(('cycle_time', 'lead_hour', 'station'),
                              stofs_res.astype(np.float32)),
            stofs_water_level=(('cycle_time', 'lead_hour', 'station'),
                                 stofs_wl.astype(np.float32)),
            regional_ofs_residual=(('cycle_time', 'lead_hour', 'station'),
                                       ofs_res.astype(np.float32)),
            regional_ofs_water_level=(('cycle_time', 'lead_hour', 'station'),
                                          ofs_wl.astype(np.float32)),
            tide=(('cycle_time', 'lead_hour', 'station'),
                    tide.astype(np.float32)),
            slr_component=(('cycle_time', 'lead_hour', 'station'),
                              np.where(np.isnan(slr), 0.0, slr).astype(np.float32)),
            minor_threshold=(('station',), minor_thr),
            moderate_threshold=(('station',), moderate_thr),
            major_threshold=(('station',), major_thr),
            ofs_model_name=(('station',),
                              np.asarray(ofs_models, dtype=object)),
            station_name=(('station',),
                            np.asarray(station_names, dtype=object)),
            station_lat=(('station',), np.asarray(st_lats, dtype=np.float32)),
            station_lon=(('station',), np.asarray(st_lons, dtype=np.float32)),
        ),
        coords=dict(
            cycle_time=cycle_dt,
            atm_member=np.asarray(members, dtype=object),
            model_seed=np.asarray(seeds, dtype=np.int16),
            lead_hour=np.arange(1, forecast_hours + 1, dtype=np.int16),
            station=np.asarray(station_ids, dtype=object),
        ),
        attrs={
            **{k: v for k, v in attrs.items() if k != 'created_utc'},
            'verify_stofs_source': stofs_source,
            'verify_stofs_residual_method': 'slr_detrend',
            'verify_ofs_residual_method': 'slr_detrend',
            'threshold_units': 'm above NTDE-MSL (per f_min_ms_m etc.)',
            'calibration_method': 'raw',
            'source_created_utc': attrs.get('created_utc', ''),
            'verify_created_utc':
                pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
    )
    comp = dict(zlib=True, complevel=4)
    encoding = {v: comp for v in ds_out.data_vars
                  if ds_out[v].dtype.kind in ('f', 'u', 'i')}
    out_path = os.path.join(out_dir, 'verify_arrays.nc')
    ds_out.to_netcdf(out_path, encoding=encoding)   # streams the dask chunks
    ds.close()

    # Station thresholds + OFS-model table.
    pd.DataFrame(dict(
        station_id=station_ids,
        station_name=station_names,
        ofs_model=ofs_models,
        minor_threshold_m=minor_thr,
        moderate_threshold_m=moderate_thr,
        major_threshold_m=major_thr,
        has_stofs=[1 if not np.all(np.isnan(stofs_res[:, :, i])) else 0
                     for i in range(len(station_ids))],
        has_ofs=[1 if not np.all(np.isnan(ofs_res[:, :, i])) else 0
                   for i in range(len(station_ids))],
    )).to_csv(os.path.join(out_dir, 'station_thresholds.csv'), index=False)

    manifest = dict(
        source_archive=fc_path,
        out_path=out_path,
        stofs_source=stofs_source,
        stofs_pair_count=int((~np.isnan(stofs_res)).sum()),
        ofs_pair_count=int((~np.isnan(ofs_res)).sum()),
        ofs_models_per_station={str(s): m
                                  for s, m in zip(station_ids, ofs_models)},
        n_cycles=len(cycle_times),
        n_atm_members=len(members),
        n_model_seeds=len(seeds),
        n_stations=len(station_ids),
        created_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    with open(os.path.join(out_dir, 'verify_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return out_path


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg.setdefault('STATIONS_CSV', 'data/stations_processed/stations_processed.csv')
    cfg.setdefault('WATER_LEVELS_H5',
                   'data/water_levels_processed/processed_water_levels.h5')
    cfg.setdefault('IN_BASE',
                   'outputs/inference_ensemble/nn_24lb_72fh/gefs_psfc_v1_raw')
    cfg.setdefault('OUT_BASE',
                   'outputs/uq_postprocessing/nn_24lb_72fh/gefs_psfc_v1_raw')
    cfg.setdefault('STOFS_SOURCE', 'cwl')

    spans = [d for d in glob.glob(os.path.join(cfg['IN_BASE'], '*'))
              if os.path.isdir(d)]
    if not spans:
        spans = [cfg['IN_BASE']]

    for span_dir in spans:
        regs = sorted(d for d in glob.glob(os.path.join(span_dir, 'region_*'))
                       if os.path.isdir(d))
        for rd in regs:
            span_tag = os.path.basename(span_dir)
            reg_name = os.path.basename(rd)
            out_dir = os.path.join(cfg['OUT_BASE'], span_tag, reg_name)
            if os.path.exists(os.path.join(out_dir, 'verify_arrays.nc')):
                print(f"Verify -> {out_dir}  [exists, skip]")
                continue
            print(f"Verify -> {out_dir}")
            verify_region(rd, out_dir, cfg)
