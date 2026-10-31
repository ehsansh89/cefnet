"""
inference_ensemble.py — Member-aware AI surge inference driver.

For each (cycle_time, GEFS member, model seed), forwards the DeepSurge model
and persists the predicted residual + derived water level into a single
member-level NetCDF archive. Sibling of `inference.py` but with two extra
ensemble dimensions: `atm_member` (31 GEFS members) and `model_seed` (the
trained seeds for the run, typically 1, 2, 3).

The archive shape preserves the native (atm_member × model_seed) factorisation
so variance decomposition can read it directly without reshaping.

Outputs (per region, per experiment tag, per time span):

    outputs/inference_ensemble/{run}/{experiment_tag}/{span}/region_XX/
        parts/forecast_YYYYMM.nc   — per-month partial archives (resume units)
        parts/summary_YYYYMM.nc
        ensemble_forecast.nc       — full 5-D archive (consolidated)
        ensemble_summary.nc        — percentile rollup (cycle, lead, station)
        meta.json
        _COMPLETE                  — sentinel

Memory model
------------
Inference streams one *month* of cycles at a time. Each month's outputs are
written to a compressed part file and freed before the next month, so peak RAM
is ~one month of forecasts (hundreds of MB) regardless of how long the span is.
Part files are written atomically (.tmp -> rename); an existing part is treated
as a complete month and skipped on resume. The final single-file archive is
built by streaming-concatenating the parts with `open_mfdataset` + lazy
`to_netcdf`, which never holds the whole span in RAM.

This script does NOT make plots or QC reports. Plot/diagnostic responsibilities
live in `x_ensemble_archive_qc.py`, `o_ensemble_uq.py`, etc.
"""

import os
import glob
import json
import time
import shutil

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

from nn import DeepSurgeForecaster
from s_sample_batch import (
    DatasetCache, get_region_stations, sample_batch,
)
from s_scalers import normalize_data
from train import load_train_config
from s_gefs_archive import (
    GEFSArchiveCache, available_gefs_cycles, fetch_lr_wind_gefs_cycle,
)


# =============================================================================
# CONFIG
# =============================================================================
CONFIG_PATH = 'inference_ensemble_config.json'

QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def generate_cycle_timestamps(start_time, end_time, cadence_hours, forecast_hours):
    """Cycle anchors spaced `cadence_hours` apart that fit in [start, end].
    Mirrors `inference.generate_cycle_timestamps`.
    """
    stamps = []
    current = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    step = pd.Timedelta(hours=cadence_hours)
    while current + pd.Timedelta(hours=forecast_hours) <= end:
        stamps.append(current)
        current = current + step
    return stamps


def load_model(ckpt_path, model_config, num_stations, train_cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DeepSurgeForecaster(
        num_stations=num_stations,
        lookback_hours=model_config['lookback_hours'],
        forecast_hours=model_config['forecast_hours'],
        d_model=train_cfg['D_MODEL'],
        hidden_dim=train_cfg['HIDDEN_DIM'],
        token_grid=train_cfg['TOKEN_GRID'],
        num_fuse_tokens=train_cfg.get('NUM_FUSE_TOKENS', 2),
        dropout=train_cfg['DROPOUT'],
        head_type=train_cfg.get('HEAD_TYPE', 'cross_attn'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def forward_one_seed_one_cycle(model, cycle_time, members, lr_member_batch,
                               model_config, station_ids, st_lats, st_lons,
                               cache, device, member_chunk):
    """Forward all GEFS members for one cycle through one loaded model seed.

    `lr_member_batch` is (n_members, T_total, 3, 64, 64) raw (unscaled) float16.
    The station tensor is identical across members; we broadcast it.

    Returns:
        residual_pred  (n_members, forecast_hours, n_stations)  float32, m
        target_mask    (n_stations,)                            bool
    """
    forecast_hours = model_config['forecast_hours']
    n_members = lr_member_batch.shape[0]
    n_stations = len(station_ids)

    # Build station tensor once for this single cycle.
    # `sample_batch` expects timestamp_ids list and an LR override of shape
    # (B, T, 3, 64, 64). We feed all members as the batch dimension.
    out_mean = np.full((n_members, forecast_hours, n_stations), np.nan,
                       dtype=np.float32)
    target_mask = np.zeros((n_stations,), dtype=bool)

    # Process members in chunks to limit GPU memory.
    if member_chunk is None or member_chunk >= n_members:
        member_chunk = n_members

    for c_start in range(0, n_members, member_chunk):
        c_end = min(c_start + member_chunk, n_members)
        ts_batch = [cycle_time] * (c_end - c_start)
        lr_chunk = lr_member_batch[c_start:c_end]
        lr_w, st_t, _res, tm = sample_batch(
            ts_batch, model_config, station_ids, st_lats, st_lons,
            cache, scale=True, lr_wind_override=lr_chunk,
        )
        inputs = {
            'low_res':  torch.from_numpy(lr_w.astype(np.float32))
                           .to(device, non_blocking=True),
            'stations': torch.from_numpy(st_t.astype(np.float32))
                           .to(device, non_blocking=True),
        }
        out = model(inputs).cpu().numpy()              # (B, F, S, 2)
        out_mean[c_start:c_end] = out[..., 0]
        if c_start == 0:
            target_mask = tm[0].astype(bool)            # same across members
        del inputs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return out_mean, target_mask


def descale_residual(mean_scaled, residual_scaler_path):
    """Descale residual predictions back to meters."""
    return normalize_data(mean_scaled, 'descale', residual_scaler_path).astype(np.float32)


def load_tide_obs_for_cycles(h5_path, station_ids, cycle_times, forecast_hours):
    """For each cycle, read tide / observed / residual / slr at the 72 valid
    times. Returns four arrays each (n_cycles, forecast_hours, n_stations).

    NaN where the gauge file is missing or doesn't cover the valid time.
    """
    n_c = len(cycle_times)
    n_l = forecast_hours
    n_s = len(station_ids)
    tide = np.full((n_c, n_l, n_s), np.nan, dtype=np.float32)
    obs_wl = np.full_like(tide, np.nan)
    obs_res = np.full_like(tide, np.nan)
    slr = np.full_like(tide, np.nan)

    cycle_ts = [pd.Timestamp(c) for c in cycle_times]
    # Build the union of all valid times we need.
    all_valid_times = pd.DatetimeIndex(sorted({
        ts + pd.Timedelta(hours=h)
        for ts in cycle_ts for h in range(1, n_l + 1)
    }))

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
            cols = [c for c in ['predicted', 'observed_detrended', 'residual',
                                 'slr_component'] if c in df.columns]
            sub = df.reindex(all_valid_times)[cols]
            t2i = pd.Series(np.arange(len(all_valid_times)),
                             index=all_valid_times)
            for c_idx, ts in enumerate(cycle_ts):
                valid_times = pd.date_range(ts + pd.Timedelta(hours=1),
                                             periods=n_l, freq='h')
                idx = t2i.reindex(valid_times).values.astype(int)
                if 'predicted' in cols:
                    tide[c_idx, :, s_idx] = sub['predicted'].values[idx]
                if 'observed_detrended' in cols:
                    obs_wl[c_idx, :, s_idx] = sub['observed_detrended'].values[idx]
                if 'residual' in cols:
                    obs_res[c_idx, :, s_idx] = sub['residual'].values[idx]
                if 'slr_component' in cols:
                    slr[c_idx, :, s_idx] = sub['slr_component'].values[idx]
    return tide, obs_wl, obs_res, slr


# =============================================================================
# DATASET BUILDERS (one block of cycles -> forecast + summary datasets)
# =============================================================================
def _base_attrs(run_name, region_id, span_tag, cfg, members, seeds,
                n_cycles, forecast_hours, lookback_hours, pressure_var):
    return dict(
        title='DeepSurge member-level ensemble forecast archive',
        run_name=run_name,
        region_id=int(region_id),
        experiment_tag=cfg['EXPERIMENT_TAG'],
        span=span_tag,
        start_time=cfg['START_TIME'],
        end_time=cfg['END_TIME'],
        cycle_cadence_hours=cfg['CYCLE_CADENCE_HOURS'],
        forecast_hours=forecast_hours,
        lookback_hours=lookback_hours,
        nwp_source='gefs',
        pressure_var=pressure_var,
        pressure_contract=cfg.get('PRESSURE_CONTRACT', 'v1_psfc'),
        lookback_source='gfs_archive',
        lead_interpolation='linear_3h_to_1h',
        n_atm_members=len(members),
        n_model_seeds=len(seeds),
        n_cycles=int(n_cycles),
        units_residual='m',
        units_water_level='m relative to detrended observation baseline',
        created_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    )


def build_block_datasets(run_name, region_id, span_tag, cfg,
                         cycle_times, station_ids, station_names, st_lats, st_lons,
                         members, seeds, member_idx,
                         residual_pred, member_valid,
                         tide, obs_wl, obs_res, slr,
                         forecast_hours, lookback_hours, pressure_var):
    """Build the (forecast, summary) xarray datasets for one block of cycles.

    `residual_pred` is (n_cycles, n_members, n_seeds, forecast_hours, n_stations).
    Returns two in-memory datasets; the caller writes them to disk.
    """
    cycle_dt = np.array([np.datetime64(pd.Timestamp(c).to_datetime64(), 'ns')
                          for c in cycle_times])
    lead_hour = np.arange(1, forecast_hours + 1, dtype=np.int16)

    # Per-trace water level. tide / slr broadcast over (member, seed).
    water_level = (residual_pred
                    + tide[:, None, None, :, :].astype(np.float32)
                    + np.where(np.isnan(slr[:, None, None, :, :]), 0.0,
                                slr[:, None, None, :, :]).astype(np.float32))

    attrs = _base_attrs(run_name, region_id, span_tag, cfg, members, seeds,
                        len(cycle_times), forecast_hours, lookback_hours,
                        pressure_var)

    ds = xr.Dataset(
        data_vars=dict(
            residual=(
                ('cycle_time', 'atm_member', 'model_seed', 'lead_hour', 'station'),
                residual_pred,
            ),
            water_level=(
                ('cycle_time', 'atm_member', 'model_seed', 'lead_hour', 'station'),
                water_level.astype(np.float32),
            ),
            member_valid=(
                ('cycle_time', 'atm_member', 'model_seed', 'station'),
                member_valid.astype(np.uint8),
            ),
            atm_member_idx=(('atm_member',), member_idx.astype(np.int16)),
            atm_member_kind=(('atm_member',),
                              np.array(['control' if m == 'gec00' else 'perturbation'
                                          for m in members], dtype=object)),
            tide=(('cycle_time', 'lead_hour', 'station'), tide.astype(np.float32)),
            slr_component=(('cycle_time', 'lead_hour', 'station'),
                              np.where(np.isnan(slr), 0.0, slr).astype(np.float32)),
            obs_residual=(('cycle_time', 'lead_hour', 'station'),
                            obs_res.astype(np.float32)),
            obs_water_level=(('cycle_time', 'lead_hour', 'station'),
                               obs_wl.astype(np.float32)),
            station_name=(('station',), np.asarray(station_names, dtype=object)),
            station_lat=(('station',), np.asarray(st_lats, dtype=np.float32)),
            station_lon=(('station',), np.asarray(st_lons, dtype=np.float32)),
        ),
        coords=dict(
            cycle_time=cycle_dt,
            atm_member=np.asarray(members, dtype=object),
            model_seed=np.asarray(seeds, dtype=np.int16),
            lead_hour=lead_hour,
            station=np.asarray(station_ids, dtype=object),
        ),
        attrs=attrs,
    )

    # ---------- Summary percentiles (flattened over atm_member × model_seed)
    flat = residual_pred.reshape(
        residual_pred.shape[0], -1,
        residual_pred.shape[3], residual_pred.shape[4],
    )  # (cycle, member_flat, lead, station)
    flat_wl = water_level.reshape(flat.shape)

    def _pct(arr, q):
        return np.nanpercentile(arr, q, axis=1).astype(np.float32)

    res_qs = {f'residual_p{q:02d}': _pct(flat, q) for q in QUANTILES}
    wl_qs = {f'water_level_p{q:02d}': _pct(flat_wl, q) for q in QUANTILES}
    n_members_total = flat.shape[1]
    n_members_valid = (~np.isnan(flat)).sum(axis=1).astype(np.int16)

    summary = xr.Dataset(
        data_vars=dict(
            residual_mean=(('cycle_time', 'lead_hour', 'station'),
                            np.nanmean(flat, axis=1).astype(np.float32)),
            residual_median=(('cycle_time', 'lead_hour', 'station'),
                              res_qs['residual_p50']),
            residual_spread_p90=(('cycle_time', 'lead_hour', 'station'),
                                  (res_qs['residual_p95']
                                    - res_qs['residual_p05']).astype(np.float32)),
            residual_spread_iqr=(('cycle_time', 'lead_hour', 'station'),
                                  (res_qs['residual_p75']
                                    - res_qs['residual_p25']).astype(np.float32)),
            water_level_mean=(('cycle_time', 'lead_hour', 'station'),
                                np.nanmean(flat_wl, axis=1).astype(np.float32)),
            water_level_median=(('cycle_time', 'lead_hour', 'station'),
                                  wl_qs['water_level_p50']),
            water_level_spread_p90=(('cycle_time', 'lead_hour', 'station'),
                                      (wl_qs['water_level_p95']
                                        - wl_qs['water_level_p05']).astype(np.float32)),
            water_level_spread_iqr=(('cycle_time', 'lead_hour', 'station'),
                                      (wl_qs['water_level_p75']
                                        - wl_qs['water_level_p25']).astype(np.float32)),
            n_members_total=(('cycle_time', 'lead_hour', 'station'),
                              np.full(flat.shape[0:1] + flat.shape[2:],
                                       n_members_total, dtype=np.int16)),
            n_members_valid=(('cycle_time', 'lead_hour', 'station'),
                              n_members_valid),
            obs_residual=(('cycle_time', 'lead_hour', 'station'),
                            obs_res.astype(np.float32)),
            obs_water_level=(('cycle_time', 'lead_hour', 'station'),
                               obs_wl.astype(np.float32)),
            tide=(('cycle_time', 'lead_hour', 'station'), tide.astype(np.float32)),
            **{k: (('cycle_time', 'lead_hour', 'station'), v)
                  for k, v in res_qs.items()},
            **{k: (('cycle_time', 'lead_hour', 'station'), v)
                  for k, v in wl_qs.items()},
            station_name=(('station',), np.asarray(station_names, dtype=object)),
            station_lat=(('station',), np.asarray(st_lats, dtype=np.float32)),
            station_lon=(('station',), np.asarray(st_lons, dtype=np.float32)),
        ),
        coords=dict(
            cycle_time=cycle_dt,
            lead_hour=lead_hour,
            station=np.asarray(station_ids, dtype=object),
        ),
        attrs=attrs.copy(),
    )
    summary.attrs['title'] = 'DeepSurge ensemble summary (percentiles)'
    return ds, summary


def _comp_encoding(ds):
    comp = dict(zlib=True, complevel=4)
    return {v: comp for v in ds.data_vars if ds[v].dtype.kind in ('f', 'u', 'i')}


def write_netcdf_atomic(ds, path):
    """Write `ds` to `path` via a temp file + atomic rename, so a half-written
    file never looks complete to the resume logic."""
    tmp = path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    ds.to_netcdf(tmp, encoding=_comp_encoding(ds))
    ds.close()
    os.replace(tmp, path)


def consolidate_parts(parts_dir, prefix, final_path, total_attrs):
    """Stream-concatenate per-month part files into one NetCDF along
    `cycle_time` without holding the whole span in RAM.

    `open_mfdataset` opens the parts lazily (dask-backed, chunked per month);
    `to_netcdf` then writes chunk-by-chunk. Peak RAM ~ one month.
    """
    part_files = sorted(glob.glob(os.path.join(parts_dir, f'{prefix}_*.nc')))
    if not part_files:
        raise RuntimeError(f"No {prefix} part files to consolidate in {parts_dir}")

    ds = xr.open_mfdataset(
        part_files, combine='nested', concat_dim='cycle_time',
        data_vars='minimal', coords='minimal', compat='override',
        join='exact',
    )
    # Span-level attrs (per-month parts carry per-month n_cycles / created_utc).
    ds.attrs.update(total_attrs)

    tmp = final_path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    ds.to_netcdf(tmp, encoding=_comp_encoding(ds))
    ds.close()
    os.replace(tmp, final_path)
    return final_path, len(part_files)


def find_checkpoint(run_name, region_id, seed):
    return os.path.join('models', run_name, f'region_{region_id:02d}',
                         f'seed_{int(seed)}', 'best_phase1.pt')


def run_region(cfg, region_id, device):
    run_name = cfg['RUN_NAME']
    region_name = f'region_{region_id:02d}'
    experiment_tag = cfg['EXPERIMENT_TAG']
    start_tag = pd.Timestamp(cfg['START_TIME']).strftime('%Y%m%d')
    end_tag = pd.Timestamp(cfg['END_TIME']).strftime('%Y%m%d')
    span_tag = f"{start_tag}_{end_tag}"

    out_dir = os.path.join(cfg['OUTPUT_DIR'], run_name, experiment_tag,
                             span_tag, region_name)
    parts_dir = os.path.join(out_dir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    sentinel = os.path.join(out_dir, '_COMPLETE')

    if os.path.exists(sentinel) and not cfg.get('OVERWRITE', False):
        print(f"[skip] {region_name}: _COMPLETE sentinel present")
        return out_dir

    # ── Configs ──
    with open(cfg['MODEL_CONFIG'], 'r') as f:
        model_config = json.load(f)
    train_cfg = load_train_config(cfg['TRAIN_CONFIG'])
    lookback_hours = model_config['lookback_hours']
    forecast_hours = model_config['forecast_hours']

    # ── Stations ──
    station_ids, station_names, st_lats, st_lons = get_region_stations(
        cfg['STATIONS_CSV'], region_name,
    )
    num_stations = len(station_ids)
    print(f"[{region_name}] stations: {num_stations}")

    # ── GEFS archive cache ──
    pressure_var = cfg.get('PRESSURE_VAR', 'PSFC')
    gefs_cache = GEFSArchiveCache(
        region_name=region_name,
        gefs_base_dir=cfg.get('GEFS_BASE_DIR', 'data/atmospheric_forecasts/gefs'),
        gfs_base_dir=cfg.get('GFS_BASE_DIR', 'data/atmospheric_forecasts/gfs'),
        pressure_var=pressure_var,
    )

    # ── Cycle list (subset of GEFS inits inside [start, end]) ──
    cadence = int(cfg['CYCLE_CADENCE_HOURS'])
    available = available_gefs_cycles(gefs_cache, cfg['START_TIME'], cfg['END_TIME'])
    cycle_times = [c for c in available
                    if int(pd.Timestamp(c).hour) % cadence == 0
                    and pd.Timestamp(c) + pd.Timedelta(hours=forecast_hours)
                        <= pd.Timestamp(cfg['END_TIME'])]
    if not cycle_times:
        raise RuntimeError(f"No GEFS cycles available for {region_name} in window.")
    print(f"[{region_name}] cycles: {len(cycle_times)} "
          f"({cycle_times[0]} .. {cycle_times[-1]}) cadence={cadence}h")

    # ── Members (take from first cycle's manifest, assume stable across cycles) ──
    members_all = gefs_cache.gefs_members_for_cycle(cycle_times[0])
    member_filter = cfg.get('MEMBER_FILTER')
    if member_filter:
        members = [m for m in members_all if m in member_filter]
    else:
        members = members_all
    member_idx = np.arange(len(members), dtype=np.int16)
    print(f"[{region_name}] members: {len(members)} (control + {len(members) - 1} perturbations)")

    seeds = list(cfg['SEEDS'])
    n_members = len(members)
    n_seeds = len(seeds)
    print(f"[{region_name}] seeds: {seeds}")
    print(f"[{region_name}] expected traces per (cycle, lead, station): "
          f"{n_members * n_seeds}")

    # ── Group cycles into calendar months (resume + RAM units) ──
    months = {}
    for c in cycle_times:
        months.setdefault(pd.Timestamp(c).strftime('%Y%m'), []).append(c)
    month_keys = sorted(months)
    print(f"[{region_name}] months: {len(month_keys)} "
          f"({month_keys[0]} .. {month_keys[-1]})")

    # ── Caches / scalers ──
    h5_path = cfg['WATER_LEVELS_H5']
    member_chunk = int(cfg.get('MEMBER_CHUNK', 16))
    residual_scaler_path = cfg['RESIDUAL_SCALER']
    cache = DatasetCache(region_name, h5_path)

    try:
        # Pre-load model seeds once for the whole region; small models.
        models = []
        for s in seeds:
            ckpt_path = find_checkpoint(run_name, region_id, s)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            models.append(load_model(ckpt_path, model_config, num_stations,
                                     train_cfg, device))
        print(f"[{region_name}] loaded {len(models)} model seeds")

        region_t0 = time.time()
        done_cycles = 0
        for mi, ym in enumerate(month_keys):
            block_cycles = months[ym]
            n_cyc_m = len(block_cycles)
            part_fc = os.path.join(parts_dir, f'forecast_{ym}.nc')
            part_sm = os.path.join(parts_dir, f'summary_{ym}.nc')

            if (os.path.exists(part_fc) and os.path.exists(part_sm)
                    and not cfg.get('OVERWRITE', False)):
                print(f"[{region_name}] {ym}: parts present, skip "
                      f"({n_cyc_m} cycles)")
                done_cycles += n_cyc_m
                continue

            # Per-month output buffers (freed at end of iteration).
            residual_pred = np.full(
                (n_cyc_m, n_members, n_seeds, forecast_hours, num_stations),
                np.nan, dtype=np.float32,
            )
            member_valid = np.zeros(
                (n_cyc_m, n_members, n_seeds, num_stations), dtype=np.uint8,
            )

            m_t0 = time.time()
            for j, cycle in enumerate(tqdm(block_cycles,
                                           desc=f'{region_name} {ym}',
                                           leave=False)):
                lr_batch = fetch_lr_wind_gefs_cycle(
                    gefs_cache, cycle, members, lookback_hours, forecast_hours,
                )
                for s_idx, model in enumerate(models):
                    pred_scaled, tm = forward_one_seed_one_cycle(
                        model, cycle, members, lr_batch,
                        model_config, station_ids, st_lats, st_lons,
                        cache, device, member_chunk,
                    )
                    residual_pred[j, :, s_idx, :, :] = descale_residual(
                        pred_scaled, residual_scaler_path)
                    member_valid[j, :, s_idx, :] = tm[None, :].astype(np.uint8)

            # Tide / observation context for this month's cycles only.
            tide, obs_wl, obs_res, slr = load_tide_obs_for_cycles(
                h5_path, station_ids, block_cycles, forecast_hours,
            )

            fc_ds, sm_ds = build_block_datasets(
                run_name, region_id, span_tag, cfg,
                block_cycles, station_ids, station_names, st_lats, st_lons,
                members, seeds, member_idx,
                residual_pred, member_valid,
                tide, obs_wl, obs_res, slr,
                forecast_hours, lookback_hours, pressure_var,
            )
            write_netcdf_atomic(fc_ds, part_fc)
            write_netcdf_atomic(sm_ds, part_sm)

            done_cycles += n_cyc_m
            dt = time.time() - m_t0
            rate = dt / max(n_cyc_m, 1)
            elapsed = time.time() - region_t0
            eta = elapsed / max(done_cycles, 1) * (len(cycle_times) - done_cycles)
            print(f"[{region_name}] {ym}: {n_cyc_m} cycles in {dt:.0f}s "
                  f"({rate:.1f}s/cycle) | {done_cycles}/{len(cycle_times)} done "
                  f"| ETA {eta/60:.0f} min")

            # Free month buffers before the next month.
            del residual_pred, member_valid, fc_ds, sm_ds
            del tide, obs_wl, obs_res, slr
    finally:
        cache.close()

    # ── Consolidate per-month parts into single-file archives ──
    print(f"[{region_name}] consolidating {len(month_keys)} months -> single files...")
    total_attrs = dict(
        n_cycles=int(len(cycle_times)),
        span=span_tag,
        start_time=cfg['START_TIME'],
        end_time=cfg['END_TIME'],
        consolidated_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    fc_path, n_fc = consolidate_parts(parts_dir, 'forecast',
                                      os.path.join(out_dir, 'ensemble_forecast.nc'),
                                      total_attrs)
    sm_path, _ = consolidate_parts(parts_dir, 'summary',
                                   os.path.join(out_dir, 'ensemble_summary.nc'),
                                   {**total_attrs,
                                    'title': 'DeepSurge ensemble summary (percentiles)'})
    print(f"[{region_name}] wrote {fc_path}")
    print(f"[{region_name}] wrote {sm_path}")

    # ── Meta + sentinel ──
    meta = dict(
        region_name=region_name,
        run_name=run_name,
        experiment_tag=experiment_tag,
        span=span_tag,
        start_time=cfg['START_TIME'],
        end_time=cfg['END_TIME'],
        cycle_cadence_hours=cadence,
        forecast_hours=forecast_hours,
        lookback_hours=lookback_hours,
        n_cycles=int(len(cycle_times)),
        n_months=int(len(month_keys)),
        n_atm_members=int(n_members),
        n_model_seeds=int(n_seeds),
        n_stations=int(num_stations),
        pressure_var=pressure_var,
        pressure_contract=cfg.get('PRESSURE_CONTRACT', 'v1_psfc'),
        nwp_source='gefs',
        lookback_source='gfs_archive',
        lead_interpolation='linear_3h_to_1h',
        files=dict(
            ensemble_forecast=os.path.basename(fc_path),
            ensemble_summary=os.path.basename(sm_path),
        ),
        generated_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    with open(sentinel, 'w') as f:
        f.write(pd.Timestamp.utcnow().isoformat() + 'Z\n')

    # ── Optionally drop parts to save disk (default: keep for safety) ──
    if not cfg.get('KEEP_PARTS', True):
        shutil.rmtree(parts_dir, ignore_errors=True)
        print(f"[{region_name}] removed parts/ (KEEP_PARTS=false)")

    return out_dir


def write_manifest(cfg, region_dirs):
    """Top-level manifest pointing into per-region archives."""
    run_name = cfg['RUN_NAME']
    experiment_tag = cfg['EXPERIMENT_TAG']
    start_tag = pd.Timestamp(cfg['START_TIME']).strftime('%Y%m%d')
    end_tag = pd.Timestamp(cfg['END_TIME']).strftime('%Y%m%d')
    span_tag = f"{start_tag}_{end_tag}"
    base = os.path.join(cfg['OUTPUT_DIR'], run_name, experiment_tag, span_tag)
    manifest = dict(
        run_name=run_name,
        experiment_tag=experiment_tag,
        span=span_tag,
        region_dirs=region_dirs,
        config=cfg,
        generated_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    with open(os.path.join(base, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    region_ids = cfg['REGION_IDS']
    region_dirs = []
    for rid in region_ids:
        d = run_region(cfg, rid, device)
        region_dirs.append(d)

    write_manifest(cfg, region_dirs)
    print('\n' + '=' * 56)
    print(f"DONE: {len(region_dirs)} region(s) under "
          f"{cfg['OUTPUT_DIR']}/{cfg['RUN_NAME']}/{cfg['EXPERIMENT_TAG']}/")
    print('=' * 56)
