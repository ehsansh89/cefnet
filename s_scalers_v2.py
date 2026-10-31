"""
Fit, cache, and apply global scalers for the CONUS storm-surge model.  [v2]

v2: water-level scalers fit from data/water_levels_processed_v2 (122-gauge set,
revised detrend); all scalers written to samples/scalers_v2/. Logic identical to v1.

Scope
-----
Single global (not per-region, not per-station) scaler per variable. The
quality impact of per-region fits is negligible given similar CONUS-wide
distributions, and a single pickle keeps the batch path simple.

Variables
---------
- wind_velocity.pkl          : ERA5 U10/V10 jointly, robust (median/IQR)
- surface_pressure.pkl       : ERA5 PSFC, z-score (mean/std)
- water_levels.pkl           : NOAA detrended observed, z-score
- water_level_residuals.pkl  : NOAA residuals (model target), z-score

HYCOM ocean state and CONUS404 high-resolution wind are out of scope for
the current training pipeline and have been removed from this file.

API
---
- normalize_data(data, mode, scaler_path, method, axis) : fit/scale/descale
- _load_scaler(scaler_path)                             : cached pickle load
- plot_and_save_distribution(...)                       : optional sanity plot
"""

import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Module-level cache so repeated scale/descale calls don't re-read the pickle
# from disk on every batch. Keyed by absolute path.
_SCALER_CACHE = {}


def _load_scaler(scaler_path):
    """Load a scaler pickle once and cache the params dict in memory."""
    key = os.path.abspath(scaler_path)
    if key not in _SCALER_CACHE:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as f:
            _SCALER_CACHE[key] = pickle.load(f)
    return _SCALER_CACHE[key]


def normalize_data(data, mode, scaler_path, method='zscore', axis=None):
    """
    Unified fit / scale / descale against a single pickle scaler file.

    Parameters
    ----------
    data : np.ndarray of any shape
    mode : 'fit', 'scale', or 'descale'
    scaler_path : path to the variable's pickle file
    method : 'zscore' or 'robust' (only used when mode == 'fit')
    axis : axis along which to compute stats (None for global)
    """

    # Upcast float16 to float32 so nan-aware reductions don't overflow
    if data.dtype == np.float16:
        data = data.astype(np.float32)

    if mode == 'fit':
        params = {'method': method}

        with np.errstate(invalid='ignore', all='ignore'):
            if method == 'zscore':
                mean = np.nanmean(data, axis=axis)
                std = np.nanstd(data, axis=axis)
                std = np.where(std == 0, 1e-6, std)
                params['mean'] = mean
                params['std'] = std
                scaled_data = (data - mean) / std

            elif method == 'robust':
                median = np.nanmedian(data, axis=axis)
                q75 = np.nanpercentile(data, 75, axis=axis)
                q25 = np.nanpercentile(data, 25, axis=axis)
                iqr = q75 - q25
                iqr = np.where(iqr == 0, 1e-6, iqr)
                params['median'] = median
                params['iqr'] = iqr
                scaled_data = (data - median) / iqr

            else:
                raise ValueError("Method must be 'zscore' or 'robust'")

        os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(params, f)

        # Invalidate cache in case this path was loaded earlier in the process
        _SCALER_CACHE.pop(os.path.abspath(scaler_path), None)

        print(f"Saved scaler to {scaler_path}")
        return params, scaled_data

    if mode in ('scale', 'descale'):
        params = _load_scaler(scaler_path)

        if mode == 'scale':
            if params['method'] == 'zscore':
                return (data - params['mean']) / params['std']
            if params['method'] == 'robust':
                return (data - params['median']) / params['iqr']

        # descale
        if params['method'] == 'zscore':
            return data * params['std'] + params['mean']
        if params['method'] == 'robust':
            return data * params['iqr'] + params['median']

    raise ValueError("Mode must be 'fit', 'scale', or 'descale'")


def plot_and_save_distribution(original, scaled, title, method='zscore', save_dir='scalers'):
    """Sanity-check plot: before vs. after distribution with key stats."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    if original.dtype == np.float16:
        original = original.astype(np.float32)

    orig_flat = original.flatten()
    orig_flat = orig_flat[~np.isnan(orig_flat)]
    scaled_flat = scaled.flatten()
    scaled_flat = scaled_flat[~np.isnan(scaled_flat)]

    if method == 'zscore':
        orig_s1, orig_s2 = np.nanmean(orig_flat), np.nanstd(orig_flat)
        scaled_s1, scaled_s2 = np.nanmean(scaled_flat), np.nanstd(scaled_flat)
        stat_labels = ('Mean', 'Std')
    else:
        orig_s1 = np.nanmedian(orig_flat)
        orig_s2 = np.nanpercentile(orig_flat, 75) - np.nanpercentile(orig_flat, 25)
        scaled_s1 = np.nanmedian(scaled_flat)
        scaled_s2 = np.nanpercentile(scaled_flat, 75) - np.nanpercentile(scaled_flat, 25)
        stat_labels = ('Median', 'IQR')

    text_props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(orig_flat, bins=50, color='royalblue', alpha=0.7)
    ax1.set_title(f'{title} - Original')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.text(0.95, 0.95, f'{stat_labels[0]}: {orig_s1:.2f}\n{stat_labels[1]}: {orig_s2:.2f}',
             transform=ax1.transAxes, ha='right', va='top', bbox=text_props)

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(scaled_flat, bins=50, color='seagreen', alpha=0.7)
    ax2.set_title(f'{title} - Normalized')
    ax2.set_xlabel('Scaled Value')
    ax2.text(0.95, 0.95, f'{stat_labels[0]}: {scaled_s1:.2f}\n{stat_labels[1]}: {scaled_s2:.2f}',
             transform=ax2.transAxes, ha='right', va='top', bbox=text_props)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title}_distribution.png'))
    plt.close()


def _fit_atmospheric_var_from_files(files, var_extract_fn, method, desc):
    """
    Chunk-wise fit for variables where holding the full archive in memory is
    infeasible. Mean/variance are combined exactly; median/IQR are combined
    by sample-weighted averaging across chunks — an approximation that is
    fine for scaling purposes but not for reporting precise population stats.
    """
    from tqdm import tqdm

    chunk_stats = []
    total_samples = 0
    last_chunk_data = None

    for path in tqdm(files, desc=desc):
        ds = xr.open_zarr(path)
        chunk_data = var_extract_fn(ds)
        chunk_data = chunk_data[~np.isnan(chunk_data)]
        n = len(chunk_data)
        if n == 0:
            continue

        entry = {
            'n': n,
            'mean': np.mean(chunk_data),
            'var': np.var(chunk_data),
        }
        if method == 'robust':
            entry['median'] = np.median(chunk_data)
            entry['q25'] = np.percentile(chunk_data, 25)
            entry['q75'] = np.percentile(chunk_data, 75)

        chunk_stats.append(entry)
        total_samples += n
        last_chunk_data = chunk_data

    if total_samples == 0:
        return None, None

    global_mean = sum(c['n'] * c['mean'] for c in chunk_stats) / total_samples
    global_var = sum(
        c['n'] * (c['var'] + (c['mean'] - global_mean) ** 2) for c in chunk_stats
    ) / total_samples
    global_std = max(np.sqrt(global_var), 1e-6)

    if method == 'zscore':
        params = {'method': 'zscore', 'mean': global_mean, 'std': global_std}
    else:
        # Sample-weighted approximation across chunks
        global_median = sum(c['n'] * c['median'] for c in chunk_stats) / total_samples
        global_q25 = sum(c['n'] * c['q25'] for c in chunk_stats) / total_samples
        global_q75 = sum(c['n'] * c['q75'] for c in chunk_stats) / total_samples
        global_iqr = global_q75 - global_q25
        global_iqr = 1e-6 if global_iqr == 0 else global_iqr
        params = {'method': 'robust', 'median': global_median, 'iqr': global_iqr}

    return params, last_chunk_data


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import glob
    import random

    SCALER_DIR = 'samples/scalers_v2'
    SAMPLE_FRAC = 0.05  # fraction of monthly files to sample per variable
    os.makedirs(SCALER_DIR, exist_ok=True)
    print("Fitting Scalers...")

    regions = [f'region_{i:02d}' for i in range(1, 11)]

    # ==========================================
    # 1. ERA5 Low-Res Wind (U10, V10) — robust
    # ==========================================
    all_wind_files = []
    for region in regions:
        search_pattern = f'data/atmospheric/{region}/lr_grid/*.zarr'
        all_wind_files.extend(glob.glob(search_pattern))

    random.seed(42)
    sample_size = max(1, int(len(all_wind_files) * SAMPLE_FRAC))
    wind_files = random.sample(all_wind_files, sample_size)
    print(f"Wind: found {len(all_wind_files)} files, sampling "
          f"{int(SAMPLE_FRAC * 100)}% ({len(wind_files)}).")

    def _extract_wind(ds):
        u = ds['U10'].values.astype(np.float32)
        v = ds['V10'].values.astype(np.float32)
        return np.concatenate([u.flatten(), v.flatten()])

    wind_params, last_wind = _fit_atmospheric_var_from_files(
        wind_files, _extract_wind, method='robust', desc='Wind chunks'
    )

    if wind_params is not None:
        wind_scaler_path = f'{SCALER_DIR}/wind_velocity.pkl'
        with open(wind_scaler_path, 'wb') as f:
            pickle.dump(wind_params, f)
        _SCALER_CACHE.pop(os.path.abspath(wind_scaler_path), None)
        print(f"Saved global wind scaler to {wind_scaler_path}")

        if last_wind is not None:
            wind_scaled_sample = (last_wind - wind_params['median']) / wind_params['iqr']
            plot_and_save_distribution(
                last_wind, wind_scaled_sample,
                'Wind_Velocity_Sample', method='robust', save_dir=SCALER_DIR,
            )

    # ==========================================
    # 2. ERA5 Surface Pressure (PSFC) — z-score
    # ==========================================
    # Same glob-and-sample pattern as wind so the fit is representative
    # across all regions, not tied to a single month.
    random.seed(43)
    psfc_sample_size = max(1, int(len(all_wind_files) * SAMPLE_FRAC))
    psfc_files = random.sample(all_wind_files, psfc_sample_size)
    print(f"PSFC: sampling {int(SAMPLE_FRAC * 100)}% ({len(psfc_files)}) "
          f"of atmospheric files.")

    def _extract_psfc(ds):
        return ds['PSFC'].values.astype(np.float32).flatten()

    psfc_params, last_psfc = _fit_atmospheric_var_from_files(
        psfc_files, _extract_psfc, method='zscore', desc='PSFC chunks'
    )

    if psfc_params is not None:
        psfc_scaler_path = f'{SCALER_DIR}/surface_pressure.pkl'
        with open(psfc_scaler_path, 'wb') as f:
            pickle.dump(psfc_params, f)
        _SCALER_CACHE.pop(os.path.abspath(psfc_scaler_path), None)
        print(f"Saved global surface pressure scaler to {psfc_scaler_path}")

        if last_psfc is not None:
            psfc_scaled_sample = (last_psfc - psfc_params['mean']) / psfc_params['std']
            plot_and_save_distribution(
                last_psfc, psfc_scaled_sample,
                'Surface_Pressure_Sample', method='zscore', save_dir=SCALER_DIR,
            )

    # ==========================================
    # 3. Water Levels & Residuals (NOAA, HDF5)
    # ==========================================
    with pd.HDFStore('data/water_levels_processed_v2/processed_water_levels.h5', mode='r') as store:
        station_keys = store.keys()
        water_levels = []
        residuals = []
        for key in station_keys:
            df = store.get(key)
            water_levels.append(df['observed_detrended'].values)
            residuals.append(df['residual'].values)

    water_levels = np.column_stack(water_levels)
    residuals = np.column_stack(residuals)

    # Water Levels (Global Z-Score)
    water_levels_scaler_path = f'{SCALER_DIR}/water_levels.pkl'
    water_levels_params, water_levels_scaled = normalize_data(
        water_levels, mode='fit', scaler_path=water_levels_scaler_path, method='zscore',
    )
    plot_and_save_distribution(
        water_levels, water_levels_scaled,
        'Water_Levels', method='zscore', save_dir=SCALER_DIR,
    )

    # Residuals / model target (Global Z-Score across all stations)
    residuals_scaler_path = f'{SCALER_DIR}/water_level_residuals.pkl'
    res_params, residuals_scaled = normalize_data(
        residuals, mode='fit', scaler_path=residuals_scaler_path, method='zscore',
    )
    plot_and_save_distribution(
        residuals, residuals_scaled,
        'Water_Level_Residuals', method='zscore', save_dir=SCALER_DIR,
    )

    print("All scalers saved to individual files in the 'scalers' directory.")
