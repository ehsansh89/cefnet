"""
s_generate_samples_v2.py — precompute scaled training samples to a single zarr.  [v2]

v2: reads scalers_v2 (via s_sample_batch_v2), metadata_v2 catalogs, *_v2 processed
WL/stations; writes to a FRESH data/precomputed_v2/ (no rmtree of v1's precomputed —
the path doesn't exist yet, so build_store skips the slow Lustre delete). Logic
identical to v1.

For a given region, walks the sampling catalog (sample_metadata_{region}.csv)
and writes every sample's already-scaled tensors into one zarr store:

    data/precomputed_v2/{region_name}/samples.zarr
        lr_wind      (N, T_total, 3, 64, 64)  float16
        station      (N, T_total, S, 6)        float16
        residual     (N, T_forecast, S)        float16
        target_mask  (N, S)                    uint8

A companion catalog (data/precomputed/{region_name}/catalog.parquet) carries
per-sample metadata (timestamp_id, sample_category, train/val/test period, ...).

Running once, train.py then opens this store read-only and streams batches
via a plain torch DataLoader — no on-the-fly HDF5 / zarr month fetchers.

Set LIMIT at the top of the file to cap samples for a quick smoke test.
"""

import os
import time

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from s_sample_batch_v2 import (
    DatasetCache,
    _apply_scaling,
    get_region_stations,
    load_config,
    sample_batch,
)


# ==========================================
# CONFIGURATION (edit here; no CLI args)
# ==========================================
REGION_NAME = 'region_10'
LIMIT = None  # int to cap samples, None for full catalog
MODEL_CONFIG_PATH = 'nn_config.json'
STATIONS_CSV = 'data/stations_processed_v2/stations_processed.csv'
WATER_LEVELS_H5 = 'data/water_levels_processed_v2/processed_water_levels.h5'
OUT_DIR = 'data/precomputed_v2'


def build_store(out_path, n_samples, t_total, t_forecast, num_stations):
    """Create an empty zarr group sized for n_samples, overwriting any prior store."""
    if os.path.exists(out_path):
        # Lustre/NFS can fail rmtree with "Directory not empty" because unlinks
        # haven't settled before the parent rmdir runs — retry with small backoff.
        import shutil
        for attempt in range(5):
            try:
                shutil.rmtree(out_path)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.3 * (attempt + 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=True)

    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

    root.create_dataset(
        'lr_wind', shape=(n_samples, t_total, 3, 64, 64),
        chunks=(1, t_total, 3, 64, 64), dtype='float16', compressor=compressor,
    )
    root.create_dataset(
        'station', shape=(n_samples, t_total, num_stations, 6),
        chunks=(1, t_total, num_stations, 6), dtype='float16', compressor=compressor,
    )
    root.create_dataset(
        'residual', shape=(n_samples, t_forecast, num_stations),
        chunks=(1, t_forecast, num_stations), dtype='float16', compressor=compressor,
    )
    root.create_dataset(
        'target_mask', shape=(n_samples, num_stations),
        chunks=(n_samples, num_stations), dtype='uint8', compressor=compressor,
    )

    root.attrs['t_total'] = t_total
    root.attrs['t_forecast'] = t_forecast
    root.attrs['num_stations'] = num_stations
    return root


def main(region_name, limit, model_config_path, stations_csv, water_levels_h5,
         sampling_csv, out_dir):
    config = load_config(model_config_path)
    lookback = config['lookback_hours']
    forecast = config['forecast_hours']
    t_total = lookback + forecast

    station_ids, station_names, st_lats, st_lons = get_region_stations(stations_csv, region_name)
    num_stations = len(station_ids)
    print(f"[{region_name}] {num_stations} stations, lookback={lookback}h, forecast={forecast}h")

    catalog = pd.read_csv(sampling_csv)
    catalog['dt'] = pd.to_datetime(catalog['timestamp_id'], format='%Y%m%d%H')
    catalog = catalog.sort_values('dt').reset_index(drop=True)
    if limit is not None:
        catalog = catalog.head(limit).copy()
    n_samples = len(catalog)
    print(f"[{region_name}] writing {n_samples} samples")

    out_path = os.path.join(out_dir, region_name, 'samples.zarr')
    root = build_store(out_path, n_samples, t_total, forecast, num_stations)

    cache = DatasetCache(region_name, water_levels_h5)
    t0 = time.time()
    try:
        for i, row in catalog.iterrows():
            dt = row['dt']
            lr, st, res, tm = sample_batch(
                [dt], config, station_ids, st_lats, st_lons, cache, scale=False,
            )
            lr_s, st_s, res_s = _apply_scaling(lr.copy(), st.copy(), res.copy())

            root['lr_wind'][i] = lr_s[0].astype(np.float16)
            root['station'][i] = st_s[0].astype(np.float16)
            root['residual'][i] = res_s[0].astype(np.float16)
            root['target_mask'][i] = tm[0].astype(np.uint8)

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{n_samples}]  {rate:.2f} samples/s  ETA {eta:.0f}s")
    finally:
        cache.close()

    # Catalog parquet next to the zarr.
    catalog_out = os.path.join(out_dir, region_name, 'catalog.parquet')
    slim = catalog[['timestamp_id', 'sample_category', 'dt']].copy()
    slim['sample_index'] = np.arange(n_samples, dtype=np.int64)
    slim.to_parquet(catalog_out, index=False)

    total = time.time() - t0
    print(f"Done. {n_samples} samples in {total:.1f}s ({n_samples/total:.2f}/s)")
    print(f"  store:   {out_path}")
    print(f"  catalog: {catalog_out}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    regions_env = os.environ.get('REGIONS')
    regions = [r.strip() for r in regions_env.split(',')] if regions_env else [REGION_NAME]

    for region_name in regions:
        sampling_csv = f'samples/metadata_v2/{region_name}/sample_metadata_{region_name}.csv'
        main(
            region_name=region_name,
            limit=LIMIT,
            model_config_path=MODEL_CONFIG_PATH,
            stations_csv=STATIONS_CSV,
            water_levels_h5=WATER_LEVELS_H5,
            sampling_csv=sampling_csv,
            out_dir=OUT_DIR,
        )
