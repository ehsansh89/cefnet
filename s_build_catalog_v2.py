"""
s_build_catalog_v2.py — Build per-region sampling catalogs for DeepSurge training.

v2: reads *_v2 processed WL/stations, writes catalogs to samples/metadata_v2/.
Logic identical to v1.

For each region in data/stations_processed_v2/stations_processed.csv this script:
  1. Loads hourly observed detrended water levels and residuals for the region's
     stations from the HDF5 archive.
  2. Labels every hour with a flood category (0/1/2/3) based on observed level
     vs each station's (minor/moderate/major) threshold, SLR-corrected to 2025.
  3. Computes forecast-horizon roll-ups (category max, |residual| max/avg/
     avg-of-max) over the next `forecast_hours`.
  4. Walks the timeline with an adaptive stride: dense sampling during
     moderate/major flood windows OR when forecast-horizon residuals are in
     the top 10–25% of the region's distribution.
  5. Subsamples "boring" rows (calm AND low residuals across all three metrics)
     down to the configured fraction to keep training balanced.
  6. Computes a diagnostic LR-wind-max at station locations (used only by the
     category-vs-wind violin plot).
  7. Saves per region:
       - hour_metadata_{region}.csv    (one row per valid hour; diagnostic)
       - sample_metadata_{region}.csv  (the training catalog)
     plus six PNG figures.

HYCOM and HR-wind availability checks are intentionally absent — those modalities
are out of scope per the project roadmap.
"""

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================
# Helper Functions
# ==========================================

def load_config(config_path: str) -> dict:
    """Loads the model configuration JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_region_stations(stations_csv: str, region_name: str) -> pd.DataFrame:
    """Loads station metadata and filters by region."""
    df = pd.read_csv(stations_csv)
    region_id = int(region_name.split('_')[-1])
    return df[df['region'] == region_id].copy()


def get_nearest_grid_indices(lat_grid: np.ndarray, lon_grid: np.ndarray,
                             st_lats: np.ndarray, st_lons: np.ndarray) -> list:
    """Finds the nearest (y, x) indices on a 2D grid for a list of coordinates."""
    indices = []
    for lat, lon in zip(st_lats, st_lons):
        dist = (lat_grid - lat) ** 2 + (lon_grid - lon) ** 2
        y, x = np.unravel_index(dist.argmin(), dist.shape)
        indices.append((y, x))
    return indices


def analyze_residual_percentiles(df: pd.DataFrame, output_dir: Path):
    """Violin plots: residual_max and wind_max stratified by hour_category."""
    if df.empty:
        print("Error: The DataFrame is empty. Skipping visualization.")
        return

    print("Generating residual and wind analysis visualization...")

    percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]

    res_summary = df.groupby('hour_category')['residual_max'].quantile(percentiles).unstack()
    if not res_summary.empty and res_summary.shape[1] == len(percentiles):
        res_summary.columns = [f"{int(p*100)}th_Percentile" for p in percentiles]

    wind_summary = df.groupby('hour_category')['wind_max'].quantile(percentiles).unstack()
    if not wind_summary.empty and wind_summary.shape[1] == len(percentiles):
        wind_summary.columns = [f"{int(p*100)}th_Percentile" for p in percentiles]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Plot 1: Residual Max (Top)
    sns.violinplot(
        x='hour_category', y='residual_max', data=df,
        inner="quartile", palette="Blues",
        hue='hour_category', legend=False,
        density_norm="count", ax=axes[0]
    )
    axes[0].set_title("Maximum Residuals by Flood Category (Scaled by Count)", fontsize=14)
    axes[0].set_xlabel("Flood Category (0: Calm, 1: Minor, 2: Moderate, 3: Major)", fontsize=12)
    axes[0].set_ylabel("Maximum Residual (m)", fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Wind Max (Bottom)
    sns.violinplot(
        x='hour_category', y='wind_max', data=df,
        inner="quartile", palette="Oranges",
        hue='hour_category', legend=False,
        density_norm="count", ax=axes[1]
    )
    axes[1].set_title("Maximum Wind Speed by Flood Category (Scaled by Count)", fontsize=14)
    axes[1].set_xlabel("Flood Category (0: Calm, 1: Minor, 2: Moderate, 3: Major)", fontsize=12)
    axes[1].set_ylabel("Maximum Wind Speed (m/s)", fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    plot_path = output_dir / "residual_and_wind_vs_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Visualization saved to {plot_path}")


def plot_residual_distribution(data_series: pd.Series, output_dir: Path, filename: str,
                               title: str, station_matrix: pd.DataFrame = None):
    """Plots the distribution of a residual series, optionally with per-station overlays."""
    if data_series.empty:
        print(f"Error: Data series is empty. Skipping {filename}.")
        return

    print(f"Generating {filename}...")

    plt.figure(figsize=(12, 6))

    # 1. Per-station background lines
    if station_matrix is not None and not station_matrix.empty:
        for col in station_matrix.columns:
            station_data = station_matrix[col].dropna()
            if len(station_data) > 0:
                sns.kdeplot(station_data, color='gray', alpha=0.2, linewidth=0.5)

    # 2. Main aggregate distribution
    valid_data = data_series.dropna().values
    if len(valid_data) == 0:
        print(f"No valid data to plot for {filename}.")
        plt.close()
        return

    sns.kdeplot(valid_data, color='blue', linewidth=2.5, label='Aggregate Distribution')

    # 3. Percentile markers
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    perc_values = np.percentile(valid_data, percentiles)

    line_styles = {1: 'dotted', 5: 'dashed', 25: 'dashdot', 50: 'solid',
                   75: 'dashdot', 95: 'dashed', 99: 'dotted'}
    colors = {1: '#d73027', 5: '#fc8d59', 25: '#fee08b', 50: '#1a9850',
              75: '#fee08b', 95: '#fc8d59', 99: '#d73027'}

    for p, val in zip(percentiles, perc_values):
        plt.axvline(val, color=colors[p], linestyle=line_styles[p],
                    linewidth=1.5, label=f'{p}th Perc: {val:.2f}m')

    plt.title(title, fontsize=14)
    plt.xlabel("Residual (m)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(np.percentile(valid_data, 0.1), np.percentile(valid_data, 99.9))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_category_distribution(df, output_dir, col_name, filename, title):
    """Pie chart of flood-category counts (works for both hours and samples)."""
    if df.empty or col_name not in df.columns:
        print(f"Skipping pie chart: {col_name} not found.")
        return

    counts = df[col_name].value_counts().sort_index()
    labels_map = {0: '0: Calm', 1: '1: Minor', 2: '2: Moderate', 3: '3: Major'}
    colors_map = {0: '#6baed6', 1: '#fd8d3c', 2: '#e6550d', 3: '#a63603'}

    labels = [labels_map.get(cat, f"Unknown ({cat})") for cat in counts.index]
    plot_colors = [colors_map.get(cat, '#cccccc') for cat in counts.index]

    plt.figure(figsize=(8, 8))
    explode = [0.05] * len(counts)
    plt.pie(counts, labels=labels, autopct='%1.3f%%', startangle=90,
            colors=plot_colors, explode=explode, shadow=False)
    plt.title(title, fontsize=14)

    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Pie chart saved to {plot_path}")


def plot_sample_residual_distribution(samples_df, output_dir, col_name, filename,
                                      title, color='purple'):
    """KDE of a residual column on the samples catalog."""
    if samples_df.empty or col_name not in samples_df.columns:
        print(f"Skipping distribution: {col_name} not found.")
        return

    print(f"Generating {filename}...")
    plt.figure(figsize=(12, 6))
    data = samples_df[col_name].dropna()

    sns.kdeplot(data, color=color, linewidth=2.5, fill=True, alpha=0.1,
                label='Sample Distribution')

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    perc_values = np.percentile(data, percentiles)

    line_styles = {1: 'dotted', 5: 'dashed', 25: 'dashdot', 50: 'solid',
                   75: 'dashdot', 95: 'dashed', 99: 'dotted'}
    colors = {1: '#d73027', 5: '#fc8d59', 25: '#fee08b', 50: '#1a9850',
              75: '#fee08b', 95: '#fc8d59', 99: '#d73027'}

    for p, val in zip(percentiles, perc_values):
        plt.axvline(val, color=colors[p], linestyle=line_styles[p],
                    linewidth=1.5, label=f'{p}th Perc: {val:.2f}m')

    plt.title(title, fontsize=14)
    plt.xlabel("Residual (m)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(np.percentile(data, 0.001), np.percentile(data, 99.9))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right',
               fontsize='small', ncol=2)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()


# ==========================================
# Main Execution Block
# ==========================================

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ── Config ──
    config_path = './nn_config.json'
    config = load_config(config_path)

    forecast_hours   = config.get('forecast_hours', 48)
    lookback_hours   = config.get('lookback_hours', 24)
    norm_stride      = config.get('normal_sample_stride_hours', 48)
    mod_stride       = config.get('moderate_sample_stride_hours', 12)
    maj_stride       = config.get('major_sample_stride_hours', 3)
    start_date       = config.get('catalog_start_date', '1980-01-01')
    end_date         = config.get('catalog_end_date', '2025-12-31 23:00:00')
    boring_keep_frac = config.get('boring_sample_keep_fraction', 0.10)
    random_seed      = config.get('boring_sample_seed', 42)

    stations_csv = Path('data/stations_processed_v2/stations_processed.csv')
    wl_h5        = Path('data/water_levels_processed_v2/processed_water_levels.h5')

    all_stations_df = pd.read_csv(stations_csv)
    unique_regions = sorted([r for r in all_stations_df['region'].unique() if r >= 0])

    for region_id in unique_regions:
        region_name = f"region_{region_id:02d}"
        print(f"\n{'='*50}")
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Processing {region_name}")
        print(f"{'='*50}")

        atmos_dir  = Path(f'data/atmospheric/{region_name}/lr_grid')
        output_dir = Path(f'samples/metadata_v2/{region_name}')
        output_dir.mkdir(parents=True, exist_ok=True)
        hours_csv   = output_dir / f'hour_metadata_{region_name}.csv'
        samples_csv = output_dir / f'sample_metadata_{region_name}.csv'

        stations    = get_region_stations(stations_csv, region_name)
        station_ids = stations['station_id'].astype(str).tolist()

        time_index = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')
        df_meta = pd.DataFrame(index=time_index)
        df_meta['timestamp_id'] = df_meta.index.strftime('%Y%m%d%H').astype(int)

        # ── Load observed + residual matrices ──
        print(f"Loading and aligning water level data for {len(station_ids)} stations...")
        obs_matrix = pd.DataFrame(index=time_index, columns=station_ids)
        res_matrix = pd.DataFrame(index=time_index, columns=station_ids)

        with pd.HDFStore(wl_h5, mode='r') as store:
            for sid in station_ids:
                key = f'station_{sid}'
                if key in store:
                    df_station = store.select(key)
                    df_station = df_station.reindex(time_index)
                    obs_matrix[sid] = df_station['observed_detrended']
                    res_matrix[sid] = df_station['residual']

        # ── Per-hour flood category (worst across stations) ──
        print("Computing flood categories and metrics...")
        # v2: stations_processed_v2 renamed slr_2025_m -> msl_2025_m (= trend line value
        # at 2025 on the station datum). Identical flood labels to v1: the t->2025 projection
        # depends only on the slope, so the baseline/intercept difference cancels.
        thresh_min = (stations['f_min_ms_m'] - stations['msl_2025_m']).values
        thresh_mod = (stations['f_mod_ms_m'] - stations['msl_2025_m']).values
        thresh_maj = (stations['f_maj_ms_m'] - stations['msl_2025_m']).values

        is_major = obs_matrix.values >= thresh_maj
        is_mod   = (obs_matrix.values >= thresh_mod) & ~is_major
        is_min   = (obs_matrix.values >= thresh_min) & ~is_major & ~is_mod

        is_valid = ~np.isnan(obs_matrix.values)
        is_calm  = is_valid & ~(is_major | is_mod | is_min)

        num_valid_stations = is_valid.sum(axis=1)
        num_valid_stations = np.where(num_valid_stations == 0, 1, num_valid_stations)

        df_meta['calm_pct']     = is_calm.sum(axis=1)  / num_valid_stations
        df_meta['minor_pct']    = is_min.sum(axis=1)   / num_valid_stations
        df_meta['moderate_pct'] = is_mod.sum(axis=1)   / num_valid_stations
        df_meta['major_pct']    = is_major.sum(axis=1) / num_valid_stations

        cat_matrix = (is_major * 3) + (is_mod * 2) + (is_min * 1)
        cat_matrix = np.where(is_valid, cat_matrix, -1)

        df_meta['hour_category'] = cat_matrix.max(axis=1)
        df_meta.loc[df_meta['hour_category'] == -1, 'hour_category'] = np.nan

        df_meta['residual_max'] = res_matrix.abs().max(axis=1)

        # ── Forecast-horizon roll-ups ──
        print(f"Computing forecast horizons for the next {forecast_hours} hours...")
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=forecast_hours)

        df_meta['forecast_horizon_category']         = df_meta['hour_category'].rolling(window=indexer, min_periods=1).max()
        df_meta['forecast_horizon_residual_max']     = df_meta['residual_max'].rolling(window=indexer, min_periods=1).max()
        df_meta['forecast_horizon_residual_avg']     = res_matrix.abs().rolling(window=indexer, min_periods=1).mean().mean(axis=1)
        df_meta['forecast_horizon_residual_avg_max'] = res_matrix.abs().rolling(window=indexer, min_periods=1).max().mean(axis=1)

        forecast_abs_max_matrix = res_matrix.abs().rolling(window=indexer, min_periods=1).max()
        forecast_abs_avg_matrix = res_matrix.abs().rolling(window=indexer, min_periods=1).mean()

        # ── LR-wind diagnostic (used only by the violin plot) ──
        print("Processing atmospheric (LR wind) data month by month...")
        st_lats = stations['lat'].values
        st_lons = stations['lon'].values
        grid_indices = None
        df_meta['wind_max'] = np.nan

        month_groups = df_meta.groupby([df_meta.index.year, df_meta.index.month])

        for (year, month), _ in month_groups:
            zarr_path = atmos_dir / f"{year}{month:02d}.zarr"
            if not zarr_path.exists():
                continue
            try:
                ds = xr.open_zarr(zarr_path)
                if grid_indices is None:
                    lat_grid = ds['lat'].values
                    lon_grid = ds['lon'].values
                    grid_indices = get_nearest_grid_indices(lat_grid, lon_grid, st_lats, st_lons)
                    y_idx, x_idx = zip(*grid_indices)
                    y_idx, x_idx = list(y_idx), list(x_idx)

                u10 = ds['U10'].values[:, y_idx, x_idx]
                v10 = ds['V10'].values[:, y_idx, x_idx]
                ws = np.sqrt(u10 ** 2 + v10 ** 2)
                ws_max = np.nanmax(ws, axis=1)

                zarr_times = pd.DatetimeIndex(ds['time'].values)
                if zarr_times.tz is None:
                    zarr_times = zarr_times.tz_localize('UTC')

                ws_series = pd.Series(ws_max, index=zarr_times)
                common_times = df_meta.index.intersection(zarr_times)
                df_meta.loc[common_times, 'wind_max'] = ws_series[common_times]
                ds.close()
            except Exception as e:
                print(f"Error processing {zarr_path.name}: {e}")

        # ── Drop hours where no station was valid ──
        print(f"Cleaning up and saving {region_name}...")
        df_save = df_meta.dropna(subset=['hour_category']).copy()

        if df_save.empty:
            print(f"No valid hours for {region_name}. Skipping visualization.")
            continue

        df_save['hour_category']             = df_save['hour_category'].astype(int)
        df_save['forecast_horizon_category'] = df_save['forecast_horizon_category'].astype(int)

        # ── Adaptive stride: min(category-path, percentile-path) ──
        print("Precomputing availability masks for fast sample generation...")

        stride_cat = pd.Series(norm_stride, index=df_save.index)
        stride_cat[df_save['forecast_horizon_category'] == 2] = mod_stride
        stride_cat[df_save['forecast_horizon_category'] == 3] = maj_stride

        res_abs_max = df_save['forecast_horizon_residual_avg_max']
        p75 = res_abs_max.quantile(0.75)
        p90 = res_abs_max.quantile(0.90)

        stride_perc = pd.Series(norm_stride, index=df_save.index)
        stride_perc[(res_abs_max >= p75) & (res_abs_max < p90)] = mod_stride
        stride_perc[res_abs_max >= p90] = maj_stride

        df_save['stride'] = np.minimum(stride_cat, stride_perc)

        # ── Per-hour WL availability (fraction of stations reporting) ──
        wl_avail_per_hour = res_matrix.notna().sum(axis=1) / len(res_matrix.columns)

        # ── Walk the timeline with the adaptive stride ──
        print("Generating temporal samples...")
        samples = []
        current_time = df_save.index[0]
        end_time = df_save.index[-1]

        while current_time <= end_time:
            if current_time not in df_save.index:
                current_time += pd.Timedelta(hours=1)
                continue

            row = df_save.loc[current_time]
            stride = int(row['stride'])

            lb_start = current_time - pd.Timedelta(hours=lookback_hours)
            lb_end   = current_time - pd.Timedelta(hours=1)
            fc_start = current_time
            fc_end   = current_time + pd.Timedelta(hours=forecast_hours - 1)

            lb_wl_avail = wl_avail_per_hour.loc[lb_start:lb_end].mean() if lookback_hours > 0 else 0.0
            fc_wl_avail = wl_avail_per_hour.loc[fc_start:fc_end].mean() if forecast_hours > 0 else 0.0

            lb_wl_avail = 0.0 if pd.isna(lb_wl_avail) else float(lb_wl_avail)
            fc_wl_avail = 0.0 if pd.isna(fc_wl_avail) else float(fc_wl_avail)

            samples.append({
                'timestamp_id':            int(row['timestamp_id']),
                'sample_category':         int(row['forecast_horizon_category']),
                'sample_residual_avg':     row['forecast_horizon_residual_avg'],
                'sample_residual_max':     row['forecast_horizon_residual_max'],
                'sample_residual_avg_max': row['forecast_horizon_residual_avg_max'],
                'pred_wl_avail':           round(fc_wl_avail, 4),
                'lookback_wl_avail':       round(lb_wl_avail, 4),
            })

            current_time += pd.Timedelta(hours=max(1, stride))

        samples_df = pd.DataFrame(samples)

        # ── Subsample "boring" rows (calm + low on all three residual metrics) ──
        mask_calm       = samples_df['sample_category'] == 0
        mask_low_avg    = samples_df['sample_residual_avg']     < samples_df['sample_residual_avg'].quantile(0.50)
        mask_low_max    = samples_df['sample_residual_max']     < samples_df['sample_residual_max'].quantile(0.25)
        mask_low_avgmax = samples_df['sample_residual_avg_max'] < samples_df['sample_residual_avg_max'].quantile(0.25)
        mask_boring = mask_calm & mask_low_avg & mask_low_max & mask_low_avgmax

        df_boring      = samples_df[mask_boring]
        df_interesting = samples_df[~mask_boring]
        df_boring_kept = df_boring.sample(frac=boring_keep_frac, random_state=random_seed)

        print(f"Original samples: {len(samples_df)}")
        print(f"Boring samples identified: {len(df_boring)}")
        samples_df = pd.concat([df_interesting, df_boring_kept]).sort_index()
        print(f"Samples after keeping {int(boring_keep_frac*100)}% of boring: {len(samples_df)}")

        # ── Write catalogs ──
        df_save.to_csv(hours_csv, index=False, float_format='%.3f')
        samples_df.to_csv(samples_csv, index=False, float_format='%.3f')

        # ==========================================
        # SUMMARY REPORT
        # ==========================================
        cat_counts = samples_df['sample_category'].value_counts().to_dict()
        pct_pred_wl_complete = (samples_df['pred_wl_avail']    == 1.0).sum() / len(samples_df) * 100
        pct_lb_wl_complete   = (samples_df['lookback_wl_avail'] == 1.0).sum() / len(samples_df) * 100
        avg_pred_wl          = samples_df['pred_wl_avail'].mean()    * 100
        avg_lb_wl            = samples_df['lookback_wl_avail'].mean() * 100

        print(f"\n--- {region_name} Execution Summary ---")
        print(f"Total Valid Hours Processed: {len(df_save):,}")
        print(f"Total Model Samples Yielded: {len(samples_df):,}")
        print("\nSample Target Classes:")
        print(f"  - Calm (0):     {cat_counts.get(0, 0):,}")
        print(f"  - Minor (1):    {cat_counts.get(1, 0):,}")
        print(f"  - Moderate (2): {cat_counts.get(2, 0):,}")
        print(f"  - Major (3):    {cat_counts.get(3, 0):,}")
        print("\nAverage Water-Level Availability Inside Sample Windows:")
        print(f"  - Forecast WL (100% Complete): {pct_pred_wl_complete:.1f}%")
        print(f"  - Lookback WL (100% Complete): {pct_lb_wl_complete:.1f}%")
        print(f"  - Forecast WL (Avg Density):   {avg_pred_wl:.1f}%")
        print(f"  - Lookback WL (Avg Density):   {avg_lb_wl:.1f}%")
        print("-" * 40 + "\n")

        # ── Visualizations ──
        analyze_residual_percentiles(df_save, output_dir)

        plot_category_distribution(
            df=df_save, output_dir=output_dir,
            col_name='hour_category',
            filename='hours_category_pie.png',
            title='Percentage of Hours by Flood Category\n(1980 - 2025)',
        )

        all_residuals_series = pd.Series(res_matrix.values.flatten())
        plot_residual_distribution(
            data_series=all_residuals_series, output_dir=output_dir,
            filename='residual_distribution.png',
            title='Distribution of Hourly Water Level Residuals Across Stations',
            station_matrix=res_matrix,
        )

        all_forecast_abs_max_series = pd.Series(forecast_abs_max_matrix.values.flatten())
        plot_residual_distribution(
            data_series=all_forecast_abs_max_series, output_dir=output_dir,
            filename='forecast_horizon_residual_abs_max_distribution.png',
            title=f'Distribution of Forecast Horizon Abs Max Residuals ({forecast_hours}h)',
            station_matrix=forecast_abs_max_matrix,
        )

        all_forecast_abs_avg_series = pd.Series(forecast_abs_avg_matrix.values.flatten())
        plot_residual_distribution(
            data_series=all_forecast_abs_avg_series, output_dir=output_dir,
            filename='forecast_horizon_residual_abs_avg_distribution.png',
            title=f'Distribution of Forecast Horizon Abs Avg Residuals ({forecast_hours}h)',
            station_matrix=forecast_abs_avg_matrix,
        )

        print(f"Generating sample-level distributions for {region_name}...")

        plot_category_distribution(
            df=samples_df, output_dir=output_dir,
            col_name='sample_category',
            filename='sample_category_distribution_pie.png',
            title=f'Distribution of Final Model Samples by Category\n(Strided Sampling - {region_name})',
        )

        plot_sample_residual_distribution(
            samples_df=samples_df, output_dir=output_dir,
            col_name='sample_residual_avg',
            filename='sample_residual_avg_distribution.png',
            title=f'Distribution of Sample Average Residuals (Target Variable)\nRegion: {region_name}',
            color='blue',
        )

        plot_sample_residual_distribution(
            samples_df=samples_df, output_dir=output_dir,
            col_name='sample_residual_avg_max',
            filename='sample_residual_avg_max_distribution.png',
            title=f'Distribution of Sample Avg Max Residuals (Target Variable)\nRegion: {region_name}',
            color='blue',
        )

        plot_sample_residual_distribution(
            samples_df=samples_df, output_dir=output_dir,
            col_name='sample_residual_max',
            filename='sample_residual_max_distribution.png',
            title=f'Distribution of Sample Max Residuals (Target Variable)\nRegion: {region_name}',
            color='blue',
        )
