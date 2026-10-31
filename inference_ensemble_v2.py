"""
inference_ensemble_v2.py — Member-aware ensemble inference on the v2
(corrected-detrend) retrain `nn_24lb_72fh_v2`.

Thin v2 driver over `inference_ensemble.py`: identical cycle grid (GEFS-init
00/06/12/18Z synoptic anchors — the STOFS-matched grid), identical stitched
GFS lookback, archive schema, and monthly resume logic. The ONLY substantive
difference is the import surface, rebound to the v2 twins so input scaling
uses `samples/scalers_v2` and the observed/tide context comes from the
v2-detrended gauges — mixing v1 scalers here would silently corrupt the inputs
(same rule as paper_fig1_inference_v2.py):

    s_sample_batch -> s_sample_batch_v2  (DatasetCache, get_region_stations,
                                          sample_batch; scalers_v2 paths)
    s_scalers      -> s_scalers_v2       (normalize_data, residual descale)

The rebinding works because inference_ensemble's functions resolve these names
as module globals at call time. Everything else (checkpoint discovery under
models/{RUN_NAME}, output layout, percentile summary) is driven by the config.

Pressure contract is PSFC — "v2" here is the detrend retrain, NOT the shelved
MSLP track.

Writes:
    outputs/inference_ensemble/nn_24lb_72fh_v2/{EXPERIMENT_TAG}/{span}/region_XX/
        parts/forecast_YYYYMM.nc, parts/summary_YYYYMM.nc,
        ensemble_forecast.nc, ensemble_summary.nc, meta.json, _COMPLETE

Config: inference_ensemble_config_v2.json   (override via X_INFER_ENSEMBLE_CONFIG)
"""

import json
import os

import torch

import inference_ensemble as ie
import s_sample_batch_v2 as sb2
import s_scalers_v2 as sc2


CONFIG_PATH = os.environ.get('X_INFER_ENSEMBLE_CONFIG',
                             'inference_ensemble_config_v2.json')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Rebind the v1 data-path names inside inference_ensemble to their v2 twins.
    ie.DatasetCache = sb2.DatasetCache
    ie.get_region_stations = sb2.get_region_stations
    ie.sample_batch = sb2.sample_batch
    ie.normalize_data = sc2.normalize_data

    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Run: {cfg['RUN_NAME']}  tag: {cfg['EXPERIMENT_TAG']}  "
          f"regions: {cfg['REGION_IDS']}")

    region_dirs = []
    for rid in cfg['REGION_IDS']:
        region_dirs.append(ie.run_region(cfg, rid, device))

    ie.write_manifest(cfg, region_dirs)
    print('\n' + '=' * 56)
    print(f"DONE: {len(region_dirs)} region(s) under "
          f"{cfg['OUTPUT_DIR']}/{cfg['RUN_NAME']}/{cfg['EXPERIMENT_TAG']}/")
    print('=' * 56)
