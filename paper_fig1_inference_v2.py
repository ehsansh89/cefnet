"""
paper_fig1_inference_v2.py — Figure 1 (AI beats physics), v2 model archive.

Identical grind to paper_fig1_inference.py, but for the corrected-detrend
retrain `nn_24lb_72fh_v2` (secular-only detrend, 122 gauges, train->2021 split).
The ONLY substantive difference is the import surface: building blocks come from
`inference_v2` (which wires `s_sample_batch_v2` -> samples/scalers_v2 for input
scaling) and `s_sample_batch_v2` for the dataset cache / station list, so the
forecast archives are scaled with the v2 scalers the v2 models were trained on.
Mixing v1 scalers here would silently corrupt the inputs.

Produces the STOFS-aligned (00/06/12/18Z) deterministic GFS-forced forecast
archive for every region x cycle-hour x seed, plus the OFS-aligned extra hours
(03/09/15/21Z for regions 6/7, NGOFS2). Writes ONLY `forecast.nc` (no plots),
and is idempotent: a (region, cycle-hour, seed) whose archive already exists is
skipped unless FORCE is set.

Writes:
    outputs/paper/inference/nn_24lb_72fh_v2/region_{XX}/gfs_{HH}z/seed_{S}/forecast.nc
    outputs/paper/inference/nn_24lb_72fh_v2/manifest_fig1.json

Config: paper_fig1_config_v2.json   (override via X_PAPER_FIG1_CONFIG)
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch

# v2 building blocks: inference_v2.run_forecast -> sample_batch from
# s_sample_batch_v2 -> samples/scalers_v2. One inference code path, v2-wired.
from inference_v2 import (
    generate_cycle_timestamps,
    load_model_from_checkpoint,
    run_forecast,
    save_forecast_archive,
)
from s_sample_batch_v2 import DatasetCache, get_region_stations
from train import load_train_config


CONFIG_PATH = os.environ.get('X_PAPER_FIG1_CONFIG', 'paper_fig1_config_v2.json')


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def cycle_hours_for_region(cfg, region_id):
    """Cycle hours to generate for a region: the STOFS synoptic hours
    (00/06/12/18Z), plus any OFS-aligned extra hours declared in
    OFS_EXTRA_HOURS for this region (int or list, e.g. NGOFS2 @ 3/9/15/21Z)."""
    hours = list(cfg.get('STOFS_CYCLE_HOURS', [0]))
    extra = cfg.get('OFS_EXTRA_HOURS', {}) or {}
    h = extra.get(str(region_id))
    if h is not None:
        for hh in (h if isinstance(h, (list, tuple)) else [h]):
            if int(hh) not in hours:
                hours.append(int(hh))
    return sorted(hours)


def archive_path(cfg, region_id, hour, seed):
    return os.path.join(
        cfg['PAPER_INFER_DIR'], cfg['RUN_NAME'],
        f"region_{region_id:02d}", f"gfs_{hour:02d}z", f"seed_{seed}",
        'forecast.nc',
    )


def run_one(cfg, region_id, hour, seed, device, model_config):
    """Produce one forecast.nc for (region, cycle-hour, seed). Returns a manifest
    row dict. Idempotent: skips if the archive already exists and FORCE is off."""
    region_name = f"region_{region_id:02d}"
    out_path = archive_path(cfg, region_id, hour, seed)
    row = {'region_id': region_id, 'cycle_hour': hour, 'seed': seed,
           'path': out_path}

    if os.path.exists(out_path) and not cfg.get('FORCE', False):
        row['status'] = 'skipped_exists'
        print(f"  [skip] {out_path} (exists)")
        return row

    ckpt = cfg['CKPT_TEMPLATE'].format(region=region_id, seed=seed)
    train_cfg_path = cfg['TRAIN_CFG_TEMPLATE'].format(region=region_id, seed=seed)
    if not os.path.exists(ckpt):
        row['status'] = 'missing_checkpoint'
        print(f"  [warn] checkpoint missing: {ckpt}")
        return row

    train_cfg = load_train_config(train_cfg_path)

    # Stations for the region.
    station_ids, station_names, st_lats, st_lons = get_region_stations(
        cfg['STATIONS_CSV'], region_name)
    num_stations = len(station_ids)

    # Cycle anchors at the requested hour-of-day. The hour is set purely by the
    # time-of-day of START_TIME; STRIDE_HOURS keeps every cycle on that hour.
    start_time = f"{cfg['SPAN_START']} {hour:02d}:00"
    end_time = cfg['SPAN_END']
    forecast_hours = model_config['forecast_hours']
    stride = int(cfg.get('STRIDE_HOURS', forecast_hours))
    cycle_times = generate_cycle_timestamps(start_time, end_time, stride,
                                            forecast_hours)
    if cfg.get('SMOKE', False):
        cycle_times = cycle_times[:int(cfg.get('SMOKE_MAX_CYCLES', 6))]
    if not cycle_times:
        row['status'] = 'no_cycles'
        print(f"  [warn] no cycles for {region_name} @ {hour:02d}Z")
        return row

    # Model.
    model, _ = load_model_from_checkpoint(ckpt, model_config, num_stations,
                                          train_cfg, device)

    # GFS-archive atmospheric forcing (the fair operational atmosphere).
    from s_gfs_archive import GFSArchiveCache, fetch_lr_wind_gfs
    gfs_cache = GFSArchiveCache(region_name=region_name,
                                gfs_base_dir=cfg['GFS_BASE_DIR'])
    lookback = model_config['lookback_hours']

    def lr_wind_fetcher(batch_ts):
        return np.stack([fetch_lr_wind_gfs(gfs_cache, ts, lookback, forecast_hours)
                         for ts in batch_ts], axis=0)

    cache = DatasetCache(region_name, cfg['WATER_LEVELS_H5'])
    try:
        t0 = time.time()
        mean_m, var_m, valid = run_forecast(
            model, cycle_times, model_config, station_ids, st_lats, st_lons,
            cache, device, cfg['RESIDUAL_SCALER'], int(cfg.get('BATCH_SIZE', 8)),
            lr_wind_fetcher=lr_wind_fetcher,
        )
        dt = time.time() - t0
    finally:
        cache.close()

    # Persist ONLY the archive. Build the minimal cfg dict save_forecast_archive
    # records into the NetCDF attrs.
    arch_cfg = {
        'RUN_NAME': cfg['RUN_NAME'], 'REGION_ID': region_id,
        'CHECKPOINT': ckpt, 'START_TIME': start_time, 'END_TIME': end_time,
        'STRIDE_HOURS': stride, 'ATMOSPHERIC_SOURCE': 'gfs_archive',
    }
    save_forecast_archive(out_path, cycle_times, station_ids, station_names,
                          st_lats, st_lons, mean_m, var_m, valid,
                          forecast_hours, model_config, arch_cfg)
    row.update(status='written', n_cycles=len(cycle_times),
               n_stations=num_stations, seconds=round(dt, 1))
    print(f"  [ok] {out_path}  ({len(cycle_times)} cycles, {dt:.1f}s)")
    return row


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()

    with open(cfg['MODEL_CONFIG']) as f:
        model_config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"STOFS-aligned GFS inference (v2) -> {cfg['PAPER_INFER_DIR']}/{cfg['RUN_NAME']}/")

    regions = ([int(cfg.get('SMOKE_REGION', 6))] if cfg.get('SMOKE', False)
               else cfg['REGIONS'])
    seeds = ([cfg['SEEDS'][0]] if cfg.get('SMOKE', False) else cfg['SEEDS'])
    if cfg.get('SMOKE', False):
        print(f"*** SMOKE MODE: region {regions[0]}, seed {seeds[0]}, "
              f"{cfg.get('SMOKE_MAX_CYCLES', 6)} cycles ***")

    manifest = []
    for region_id in regions:
        for hour in cycle_hours_for_region(cfg, region_id):
            for seed in seeds:
                print(f"region_{region_id:02d} @ {hour:02d}Z seed_{seed}")
                try:
                    manifest.append(run_one(cfg, region_id, hour, seed,
                                            device, model_config))
                except Exception as e:
                    print(f"  [ERROR] region {region_id} {hour:02d}Z seed {seed}: {e}")
                    manifest.append({'region_id': region_id, 'cycle_hour': hour,
                                     'seed': seed, 'status': f'error: {e}'})

    man_path = os.path.join(cfg['PAPER_INFER_DIR'], cfg['RUN_NAME'],
                            'manifest_fig1.json')
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, 'w') as f:
        json.dump({'generated_utc': pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                   'runs': manifest}, f, indent=2, default=str)
    n_ok = sum(1 for r in manifest if r.get('status') == 'written')
    n_skip = sum(1 for r in manifest if r.get('status') == 'skipped_exists')
    print(f"\nDone. written={n_ok} skipped={n_skip} total={len(manifest)}")
    print(f"Manifest: {man_path}")
