"""
s_sample_batch.py — Build training batches from the s_build_catalog sampling catalog.

For each sample timestamp this module:
  1. Loads the ERA5 LR wind / surface-pressure sequence spanning lookback +
     forecast hours (cached per month in RAM).
  2. Loads per-station observed water levels, tide predictions, and residual
     targets from the HDF5 archive (cached per station in RAM).
  3. Returns scaled or raw NumPy arrays ready to stack into a batch.

The 4-element `missing_mask` output and the HR-wind / HYCOM ocean fetchers
have been removed — those modalities are out of scope per the project
roadmap. Per-station, per-hour `tide_avail` and `obs_avail` flags are kept
as channels 2 and 3 of `station_data` because real tide-gauge gaps (sensor
outages, late-installation stations) still need to be signalled to the model.
"""

import os
import json
import re
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from s_scalers import normalize_data


# ==========================================
# CONFIGURATION
# ==========================================
REGION_NAME = 'region_02'
SAMPLING_METADATA_CSV = f'samples/metadata/{REGION_NAME}/sample_metadata_{REGION_NAME}.csv'
MODEL_CONFIG_DIR = 'nn_config.json'
STATIONS_CSV = 'data/stations_processed/stations_processed.csv'
WATER_LEVELS_H5 = 'data/water_levels_processed/processed_water_levels.h5'
SCALER_PATHS = 'samples/scalers/'

# Dashboard-only knobs (None = no preference / random pick).
VISUALIZED_SAMPLE_CATEGORY = None
VISUALIZATION_TIMESTAMP = None   # e.g. '2011-08-28'
BATCH_SIZE = 16


# ==========================================
# HELPERS
# ==========================================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def to_json_safe(arr):
    arr = np.array(arr, dtype=float)
    return np.where(np.isnan(arr), None, arr).tolist()


def get_region_stations(stations_csv, region_name):
    """Return (station_ids, names, lats, lons) for the requested region."""
    df = pd.read_csv(stations_csv)
    match = re.search(r'\d+', str(region_name))
    if not match:
        raise ValueError(f"Cannot parse region number from '{region_name}'")
    target_region_num = int(match.group())
    df['region_num'] = pd.to_numeric(df['region'], errors='coerce')
    region_stations = df[df['region_num'] == target_region_num]
    if region_stations.empty:
        raise ValueError(
            f"No stations found for region '{region_name}' in {stations_csv}. "
            f"Check REGION_NAME and the stations CSV."
        )
    return (
        region_stations['station_id'].astype(str).tolist(),
        region_stations['name'].tolist(),
        region_stations['lat'].tolist(),
        region_stations['lon'].tolist(),
    )


def get_sample_timestamps(csv_path, target_dt, category, batch_size):
    """Pick `batch_size` timestamps from the catalog, biased toward target_dt + category."""
    df = pd.read_csv(csv_path)
    df['dt'] = pd.to_datetime(df['timestamp_id'], format='%Y%m%d%H')

    covers_timestamp = (df['dt'] == target_dt)
    is_category = df['sample_category'] == category

    ideal = df[covers_timestamp & is_category]
    if not ideal.empty:
        sampled_ideal = ideal.sample(1)
    else:
        fallback = df[covers_timestamp]
        if not fallback.empty:
            print(f"[WARNING]: Category {category} not found. Prioritizing timestamp {target_dt}.")
            sampled_ideal = fallback.sample(1)
        else:
            print(f"[WARNING]: Timestamp {target_dt} missing. Picking randomly.")
            sampled_ideal = df.sample(1)

    pool = df.drop(sampled_ideal.index)
    sampled_rest = pool.sample(batch_size - 1)
    batch_metadata = pd.concat([sampled_ideal, sampled_rest]).reset_index(drop=True)
    return batch_metadata['dt'].tolist()


# ==========================================
# DATASET CACHE
# ==========================================
class DatasetCache:
    """Persistent file handles + month/station memoization for batch generation."""

    def __init__(self, region_name, h5_path):
        self.region_name = region_name
        self.h5_path = h5_path

        # HDF5 is not thread-safe — keep the store open and serialize reads.
        self.h5_store = pd.HDFStore(h5_path, mode='r')
        self._h5_lock = threading.Lock()

        # Atmospheric zarrs: open lazily per month, compute to numpy once.
        self.atmos_cache = {}           # month_str -> lazy xarray.Dataset or None
        self.atmos_computed_cache = {}  # month_str -> dict of numpy arrays or None
        self._atmos_lock = threading.Lock()

        # Per-station dataframes: read from HDF5 once then held in memory.
        self.station_cache = {}

    def get_atmos_ds(self, month_str):
        if month_str not in self.atmos_cache:
            zarr_path = f'data/atmospheric/{self.region_name}/lr_grid/{month_str}.zarr'
            if os.path.exists(zarr_path):
                self.atmos_cache[month_str] = xr.open_zarr(zarr_path)
            else:
                self.atmos_cache[month_str] = None
        return self.atmos_cache[month_str]

    def get_atmos_computed(self, month_str):
        """Return fully computed numpy arrays for a month (cached after first read)."""
        with self._atmos_lock:
            if month_str in self.atmos_computed_cache:
                return self.atmos_computed_cache[month_str]

        ds = self.get_atmos_ds(month_str)
        if ds is None:
            with self._atmos_lock:
                self.atmos_computed_cache[month_str] = None
            return None

        computed = ds.compute()
        result = {
            'time': pd.DatetimeIndex(computed['time'].values),
            'U10': computed['U10'].values,
            'V10': computed['V10'].values,
            'PSFC': computed['PSFC'].values,
        }
        with self._atmos_lock:
            self.atmos_computed_cache[month_str] = result
        return result

    def get_station_df(self, station_id):
        if station_id not in self.station_cache:
            with self._h5_lock:
                key = f'/station_{station_id}'
                if key in self.h5_store.keys():
                    df = self.h5_store.get(key)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    self.station_cache[station_id] = df[~df.index.duplicated(keep='first')]
                else:
                    self.station_cache[station_id] = None
        return self.station_cache[station_id]

    def close(self):
        self.h5_store.close()
        self.atmos_computed_cache.clear()


# ==========================================
# FETCHERS
# ==========================================
def fetch_lr_wind(cache, time_range):
    """Fetch LR wind (U10, V10) + surface pressure for `time_range`.

    Returns (T, 3, 64, 64) float16 array. Hours outside the zarr archive
    coverage are NaN-filled rather than silently mapped to boundary values.
    """
    y_dim, x_dim = 64, 64
    nan_arr = np.full((len(time_range), 3, y_dim, x_dim), np.nan, dtype=np.float16)

    months = pd.Series(time_range).dt.strftime('%Y%m').unique()
    month_data = [cache.get_atmos_computed(m) for m in months]
    month_data = [md for md in month_data if md is not None]
    if not month_data:
        return nan_arr

    try:
        if len(month_data) == 1:
            all_times = month_data[0]['time']
            all_u10 = month_data[0]['U10']
            all_v10 = month_data[0]['V10']
            all_psfc = month_data[0]['PSFC']
        else:
            all_times = month_data[0]['time'].append([md['time'] for md in month_data[1:]])
            all_u10 = np.concatenate([md['U10'] for md in month_data], axis=0)
            all_v10 = np.concatenate([md['V10'] for md in month_data], axis=0)
            all_psfc = np.concatenate([md['PSFC'] for md in month_data], axis=0)

        # Out-of-range guard (same spirit as the ocean fetcher used to do):
        # hours before the earliest cached month or after the latest get NaN
        # instead of being silently mapped to the first/last available hour.
        t_values = pd.DatetimeIndex(time_range).values
        t_min, t_max = all_times.values[0], all_times.values[-1]
        in_range = (t_values >= t_min) & (t_values <= t_max)
        if not in_range.any():
            return nan_arr

        indices = all_times.searchsorted(time_range, side='left')
        indices = np.clip(indices, 0, len(all_times) - 1)

        arr = np.stack(
            [all_u10[indices], all_v10[indices], all_psfc[indices]],
            axis=1,
        ).astype(np.float16)

        out_of_range = ~in_range
        if out_of_range.any():
            arr[out_of_range] = np.nan
        return arr
    except (KeyError, IndexError, ValueError) as e:
        # Narrow catch + explicit logging — we no longer swallow silently.
        print(f"[WARN] fetch_lr_wind failed for {time_range[0]}..{time_range[-1]}: {e}")
        traceback.print_exc()
        return nan_arr


def fetch_water_levels(cache, station_ids, st_lats, st_lons, time_range, lookback_hours):
    """Fetch per-station inputs and residual targets.

    station_data layout  (T, N, 6):
        ch 0 = tide prediction   (NaN where missing)
        ch 1 = observed detrended water level, lookback hours only
        ch 2 = tide_avail flag   (1 / 0)
        ch 3 = obs_avail flag    (1 / 0, lookback hours only)
        ch 4 = station lat       (raw degrees)
        ch 5 = station lon       (raw degrees)

    residual layout      (forecast_hours, N): NaN where missing.
    """
    total_hours = len(time_range)
    num_stations = len(station_ids)

    station_data = np.zeros((total_hours, num_stations, 6), dtype=np.float32)
    residual = np.full((total_hours - lookback_hours, num_stations), np.nan, dtype=np.float32)

    for s_idx, sid in enumerate(station_ids):
        station_data[:, s_idx, 4] = st_lats[s_idx]
        station_data[:, s_idx, 5] = st_lons[s_idx]

        df = cache.get_station_df(sid)
        if df is None:
            continue

        # Single reindex, slice columns — avoids three separate reindex passes.
        full = df.reindex(time_range)
        tide = full['predicted'].values
        obs = full['observed_detrended'].values[:lookback_hours]
        res = full['residual'].values[lookback_hours:]

        tide_valid = ~np.isnan(tide)
        station_data[:, s_idx, 0] = np.where(tide_valid, tide, np.nan)
        station_data[:, s_idx, 2] = tide_valid.astype(np.float32)

        obs_valid = ~np.isnan(obs)
        station_data[:lookback_hours, s_idx, 1] = np.where(obs_valid, obs, np.nan)
        station_data[:lookback_hours, s_idx, 3] = obs_valid.astype(np.float32)

        residual[:, s_idx] = res

    return station_data, residual


# ==========================================
# BATCH SAMPLER
# ==========================================
def _apply_scaling(batch_lr_wind, batch_station, batch_res):
    """Scale inputs in-place and replace NaNs with 0. Returns the same arrays."""
    batch_lr_wind[:, :, 0:2] = normalize_data(
        batch_lr_wind[:, :, 0:2], mode='scale',
        scaler_path=f"{SCALER_PATHS}wind_velocity.pkl",
    )
    batch_lr_wind[:, :, 2:3] = normalize_data(
        batch_lr_wind[:, :, 2:3], mode='scale',
        scaler_path=f"{SCALER_PATHS}surface_pressure.pkl",
    )
    batch_station[:, :, :, 0:2] = normalize_data(
        batch_station[:, :, :, 0:2], mode='scale',
        scaler_path=f"{SCALER_PATHS}water_levels.pkl",
    )
    batch_res = normalize_data(
        batch_res, mode='scale',
        scaler_path=f"{SCALER_PATHS}water_level_residuals.pkl",
    )

    batch_station[:, :, :, 4] = batch_station[:, :, :, 4] / 90.0   # lat -> [-1, 1]
    batch_station[:, :, :, 5] = batch_station[:, :, :, 5] / 180.0  # lon -> [-1, 1]

    batch_lr_wind = np.nan_to_num(batch_lr_wind, nan=0.0)
    batch_station = np.nan_to_num(batch_station, nan=0.0)
    batch_res = np.nan_to_num(batch_res, nan=0.0)
    return batch_lr_wind, batch_station, batch_res


def sample_batch(timestamp_ids, config, station_ids, st_lats, st_lons, cache,
                 scale=False, lr_wind_override=None):
    """Build a batch.

    Returns:
        batch_lr_wind (B, T, 3, 64, 64)     float16
        batch_station (B, T, N, 6)          float32
        batch_res     (B, F, N)             float32
        target_mask   (B, N)                float32  — 1 where the residual
                                             is complete for all F forecast
                                             hours at that station.

    `lr_wind_override`, when given, is a (B, T, 3, 64, 64) float16 array of
    raw (m/s, hPa, NaN-where-missing) low-res atmospheric data — used by the
    inference GFS-archive path so that this function can keep its single
    return contract while the upstream caller swaps the data source.
    """
    lookback = config['lookback_hours']
    forecast = config['forecast_hours']
    seq_len = lookback + forecast
    num_stations = len(station_ids)
    b_size = len(timestamp_ids)

    if lr_wind_override is not None:
        batch_lr_wind = lr_wind_override.astype(np.float16, copy=True)
    else:
        batch_lr_wind = np.zeros((b_size, seq_len, 3, 64, 64), dtype=np.float16)
    batch_station = np.zeros((b_size, seq_len, num_stations, 6), dtype=np.float32)
    batch_res = np.zeros((b_size, forecast, num_stations), dtype=np.float32)
    target_mask = np.zeros((b_size, num_stations), dtype=np.float32)

    # Time ranges + unique months needed across the whole batch.
    all_time_ranges = []
    all_months = set()
    for dt in timestamp_ids:
        time_range = pd.date_range(
            end=dt + pd.Timedelta(hours=forecast), periods=seq_len, freq='h',
        )
        all_time_ranges.append(time_range)
        all_months.update(pd.Series(time_range).dt.strftime('%Y%m').unique())

    # Pre-warm atmospheric months in parallel only when we'll actually read
    # them (override path supplies the LR wind directly).
    if all_months and lr_wind_override is None:
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(cache.get_atmos_computed, all_months))

    # Pre-load station DataFrames once (HDF5 access is lock-serialized anyway).
    for sid in station_ids:
        cache.get_station_df(sid)

    # Per-sample fetch is serial — after caches are warm, this is pure NumPy
    # slicing / stacking; threading here adds complexity with ~zero speedup.
    for i, time_range in enumerate(all_time_ranges):
        if lr_wind_override is None:
            batch_lr_wind[i] = fetch_lr_wind(cache, time_range)
        st, res = fetch_water_levels(
            cache, station_ids, st_lats, st_lons, time_range, lookback,
        )
        batch_station[i] = st
        batch_res[i] = res
        target_mask[i] = (~np.isnan(res)).all(axis=0).astype(np.float32)

    if scale:
        batch_lr_wind, batch_station, batch_res = _apply_scaling(
            batch_lr_wind, batch_station, batch_res,
        )

    return batch_lr_wind, batch_station, batch_res, target_mask


# ==========================================
# DASHBOARD (personal-check visualization only)
# ==========================================
def get_grid_coords(region_name, shape):
    y_dim, x_dim = shape[0], shape[1]
    try:
        base = f'data/atmospheric/{region_name}/lr_grid'
        valid = [x for x in os.listdir(base) if x.endswith('.zarr')][0]
        ds = xr.open_zarr(os.path.join(base, valid))
        return ds['lat'].values, ds['lon'].values
    except Exception:
        lon_g, lat_g = np.meshgrid(
            np.linspace(-95, -75, x_dim), np.linspace(20, 35, y_dim),
        )
        return lat_g, lon_g


def get_station_indices(st_lats, st_lons, lats, lons):
    """Closest (x, y) grid index for each station lat/lon."""
    indices = []
    if lats.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)
    for lat, lon in zip(st_lats, st_lons):
        dist = (lats - lat) ** 2 + (lons - lon) ** 2
        y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
        indices.append({"x": int(x_idx), "y": int(y_idx)})
    return indices


def process_map_data(data, lats, lons, q_step_override=None):
    """Prepare LR wind data for Plotly: pressure heatmap + downsampled U/V vectors."""
    seq_len = data.shape[0]
    y_dim = data.shape[2]

    q_step = q_step_override if q_step_override is not None else max(1, y_dim // 20)

    if lats.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    q_lats = lats[::q_step, ::q_step].flatten()
    q_lons = lons[::q_step, ::q_step].flatten()
    q_valid_mask = ~np.isnan(data[0, 1, ::q_step, ::q_step].flatten())

    frames_hm_2d = []
    frames_q_u, frames_q_v = [], []
    z_min, z_max = np.inf, -np.inf

    for t in range(seq_len):
        d_t = data[t]
        hm_val = d_t[2]   # PSFC
        u = d_t[0]        # U10
        v = d_t[1]        # V10

        u_sub = u[::q_step, ::q_step].flatten()[q_valid_mask]
        v_sub = v[::q_step, ::q_step].flatten()[q_valid_mask]

        valid = ~np.isnan(hm_val)
        hm_clean = hm_val[valid]
        if len(hm_clean) > 0:
            z_min = min(z_min, np.nanpercentile(hm_clean, 2))
            z_max = max(z_max, np.nanpercentile(hm_clean, 98))

        hm_2d_list = [
            [None if np.isnan(val) else round(float(val), 2) for val in row]
            for row in hm_val
        ]
        frames_hm_2d.append(hm_2d_list)
        frames_q_u.append(u_sub.round(2).tolist())
        frames_q_v.append(v_sub.round(2).tolist())

    return {
        "hm_2d": frames_hm_2d,
        "q_u": frames_q_u,
        "q_v": frames_q_v,
        "q_lat": q_lats[q_valid_mask].round(4).tolist(),
        "q_lon": q_lons[q_valid_mask].round(4).tolist(),
        "zmin": float(z_min) if z_min != np.inf else 0,
        "zmax": float(z_max) if z_max != -np.inf else 1,
    }


def export_html_dashboard(sample_data_raw, sample_data_scaled, station_names,
                          st_lats, st_lons, issue_time, config,
                          target_mask=None, tide_avail=None, obs_avail=None,
                          filename="index.html"):
    lr_raw, tide_raw, obs_raw, res_raw = sample_data_raw
    lr_sc, tide_sc, obs_sc, res_sc = sample_data_scaled
    lookback = config['lookback_hours']
    forecast = config['forecast_hours']
    seq_len = lookback + forecast

    time_labels = [
        (issue_time + pd.Timedelta(hours=i)).strftime('%Y-%m-%d %H:00')
        for i in range(-lookback + 1, forecast + 1)
    ]

    print("Preparing LR grid for HTML rendering...")
    lr_lats, lr_lons = get_grid_coords(REGION_NAME, lr_raw.shape[2:4])
    st_indices_lr = get_station_indices(st_lats, st_lons, lr_lats, lr_lons)

    print("Processing raw data...")
    raw_payload = {
        "lr": process_map_data(lr_raw, lr_lats, lr_lons),
        "tide": to_json_safe(tide_raw),
        "obs": to_json_safe(obs_raw),
        "res": to_json_safe(res_raw),
    }
    print("Processing scaled data...")
    scaled_payload = {
        "lr": process_map_data(lr_sc, lr_lats, lr_lons),
        "tide": to_json_safe(tide_sc),
        "obs": to_json_safe(obs_sc),
        "res": to_json_safe(res_sc),
    }

    data_payload = {
        "time_axis": time_labels,
        "lookback_idx": lookback,
        "station_names": station_names,
        "station_lats": st_lats,
        "station_lons": st_lons,
        "center_lat": float(np.mean(st_lats)) if st_lats else 30.0,
        "center_lon": float(np.mean(st_lons)) if st_lons else -90.0,
        "st_indices": {"lr": st_indices_lr},
        "raw": raw_payload,
        "scaled": scaled_payload,
        "target_mask": to_json_safe(target_mask) if target_mask is not None else [],
        "tide_avail": to_json_safe(tide_avail) if tide_avail is not None else [],
        "obs_avail": to_json_safe(obs_avail) if obs_avail is not None else [],
    }

    json_filename = "dashboard_data.json"
    with open(json_filename, "w") as f:
        json.dump(data_payload, f)
    print(f"Data payload saved to {json_filename}")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Storm Surge - Batch Sample Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 20px; }}
            .plot-box {{ background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 10px; margin-bottom: 15px; }}
            .controls {{ display: flex; gap: 20px; align-items: center; background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
            input[type=range] {{ flex-grow: 1; }}
            .info-panel {{ display: flex; gap: 20px; background: white; padding: 12px 15px; border-radius: 8px; margin-bottom: 15px; align-items: center; flex-wrap: wrap; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; color: white; }}
            .badge-target {{ background: #2980b9; }}
            .badge-no-target {{ background: #bdc3c7; color: #555; }}
        </style>
    </head>
    <body>
        <h2>Sample Dashboard</h2>

        <div class="controls">
            <label><b>Station:</b>
                <select id="stationSelect"></select>
            </label>
            <label style="flex-grow: 1; display: flex; align-items: center; gap: 10px;">
                <b>Time Step:</b> <span id="timeLabel">Loading data...</span>
                <input type="range" id="timeSlider" min="0" max="{seq_len-1}" value="{lookback-1}" step="1" disabled>
            </label>
            <label style="background:#e0e0e0; padding: 5px 10px; border-radius: 5px; cursor: pointer;">
                <input type="checkbox" id="heatmapToggle" disabled> <b>Show 2D Heatmap Grid</b>
            </label>
            <label style="background:#d5e8d4; padding: 5px 10px; border-radius: 5px; cursor: pointer;">
                <input type="checkbox" id="scaleToggle" disabled> <b>Scaled View</b>
            </label>
        </div>

        <div class="info-panel">
            <div><b>Target Mask:</b> <span id="targetMaskInfo"></span></div>
        </div>

        <div class="plot-box" id="map_lr" style="height: 500px;"></div>
        <div class="plot-box" id="ts_chart" style="height: 350px;"></div>
        <div class="plot-box" id="avail_chart" style="height: 160px;"></div>

        <script>
            fetch('{json_filename}')
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('timeSlider').disabled = false;
                    document.getElementById('heatmapToggle').disabled = false;
                    document.getElementById('scaleToggle').disabled = false;

                    let currentTimeIdx = data.lookback_idx - 1;
                    let currentStationIdx = 0;
                    let isScaled = false;
                    function getView() {{ return isScaled ? data.scaled : data.raw; }}

                    function updateTargetMaskDisplay(stationIdx) {{
                        const tmDiv = document.getElementById('targetMaskInfo');
                        if (!data.target_mask || !data.target_mask.length) {{ tmDiv.textContent = 'N/A'; return; }}
                        let html = '';
                        data.station_names.forEach((name, idx) => {{
                            let v = data.target_mask[idx];
                            let cls = v ? 'badge badge-target' : 'badge badge-no-target';
                            let border = idx === stationIdx ? 'outline: 3px solid #e67e22;' : '';
                            html += '<span class="' + cls + '" style="margin: 2px; cursor: default; ' + border + '" title="' + name + '">' + (v ? '\u2713' : '\u2717') + ' ' + name + '</span> ';
                        }});
                        tmDiv.innerHTML = html;
                    }}
                    updateTargetMaskDisplay(0);

                    const select = document.getElementById("stationSelect");
                    data.station_names.forEach((name, idx) => {{
                        let opt = document.createElement("option");
                        opt.value = idx;
                        opt.innerHTML = name;
                        select.appendChild(opt);
                    }});

                    const layoutMap = (title) => ({{
                        title: title, margin: {{t: 30, b: 10, l: 10, r: 10}},
                        mapbox: {{ style: 'carto-positron', center: {{lat: data.center_lat, lon: data.center_lon}}, zoom: 4.5 }},
                        uirevision: 'constant', showlegend: false
                    }});

                    const layoutHeatmap = (title) => ({{
                        title: title, margin: {{t: 30, b: 30, l: 30, r: 30}},
                        xaxis: {{ visible: false }}, yaxis: {{ visible: false, autorange: 'reversed' }},
                        uirevision: 'constant', showlegend: false
                    }});

                    function getMapTraces(mapData, timeIdx, isHeatmap, gridKey) {{
                        if (isHeatmap) {{
                            let st = data.st_indices[gridKey][currentStationIdx];
                            return [
                                {{
                                    type: 'heatmap',
                                    z: mapData.hm_2d[timeIdx],
                                    colorscale: 'Viridis',
                                    zmin: mapData.zmin, zmax: mapData.zmax,
                                    hoverinfo: 'z', name: 'Pressure'
                                }},
                                {{
                                    type: 'scatter', mode: 'markers',
                                    x: [st.x], y: [st.y],
                                    marker: {{size: 14, color: 'red', symbol: 'circle', line: {{color: 'white', width: 2}} }},
                                    hoverinfo: 'none', showlegend: false
                                }}
                            ];
                        }} else {{
                            let traces = [];
                            let lons = [], lats = [];
                            let q_u = mapData.q_u[timeIdx], q_v = mapData.q_v[timeIdx];
                            let scale = 0.02;

                            for(let i=0; i < mapData.q_lon.length; i++) {{
                                lons.push(mapData.q_lon[i], mapData.q_lon[i] + q_u[i]*scale, null);
                                lats.push(mapData.q_lat[i], mapData.q_lat[i] + q_v[i]*scale, null);
                            }}

                            traces.push({{
                                type: 'scattermapbox', mode: 'lines', lon: lons, lat: lats,
                                line: {{color: 'rgba(0,0,0,0.7)', width: 1.5}}, hoverinfo: 'none'
                            }});
                            traces.push({{
                                type: 'scattermapbox', mode: 'markers',
                                lon: [data.station_lons[currentStationIdx]], lat: [data.station_lats[currentStationIdx]],
                                marker: {{size: 14, color: 'red', symbol: 'circle'}}, hoverinfo: 'none'
                            }});
                            return traces;
                        }}
                    }}

                    function getTimeseriesData(stationIdx) {{
                        const view = getView();
                        const tideData = view.tide.map(row => row[stationIdx]);
                        const obsData = Array({seq_len}).fill(null);
                        const targetData = Array({seq_len}).fill(null);

                        view.obs.forEach((row, i) => obsData[i] = row[stationIdx] !== null ? row[stationIdx] : null);
                        view.res.forEach((row, i) => {{
                            let resVal = row[stationIdx];
                            let tideVal = view.tide[i + data.lookback_idx][stationIdx];
                            if (resVal !== null && tideVal !== null) {{
                                targetData[i + data.lookback_idx] = resVal + tideVal;
                            }}
                        }});

                        return [
                            {{ x: data.time_axis, y: tideData, name: 'Tide Predicted', type: 'scatter', line: {{color: '#1f77b4'}} }},
                            {{ x: data.time_axis, y: obsData, name: 'Observed Water Level', type: 'scatter', line: {{color: '#2ca02c'}} }},
                            {{ x: data.time_axis, y: targetData, name: 'Target Water Level (Forecast)', type: 'scatter', line: {{color: '#d62728'}} }}
                        ];
                    }}

                    function getAvailTraces(stationIdx) {{
                        let traces = [];
                        if (data.tide_avail && data.tide_avail.length) {{
                            const tideA = data.tide_avail.map(row => row[stationIdx]);
                            traces.push({{
                                x: data.time_axis, y: tideA, name: 'Tide Avail',
                                type: 'scatter', fill: 'tozeroy',
                                fillcolor: 'rgba(31,119,180,0.25)', line: {{color: 'rgba(31,119,180,0.6)', width: 1, shape: 'hv'}}
                            }});
                        }}
                        if (data.obs_avail && data.obs_avail.length) {{
                            const obsA = data.obs_avail.map(row => row[stationIdx]);
                            traces.push({{
                                x: data.time_axis, y: obsA, name: 'Obs Avail',
                                type: 'scatter', fill: 'tozeroy',
                                fillcolor: 'rgba(44,160,44,0.25)', line: {{color: 'rgba(44,160,44,0.6)', width: 1, shape: 'hv'}}
                            }});
                        }}
                        return traces;
                    }}

                    const availLayout = {{
                        title: 'Station Data Availability Flags', height: 160,
                        margin: {{t: 30, b: 30, l: 40, r: 20}},
                        xaxis: {{ title: '' }},
                        yaxis: {{ title: 'Flag', range: [-0.1, 1.3], dtick: 1 }},
                        legend: {{ orientation: 'h', y: 1.25 }},
                        shapes: [
                            {{ type: 'line', x0: data.time_axis[data.lookback_idx - 1], x1: data.time_axis[data.lookback_idx - 1], y0: 0, y1: 1, yref: 'paper', line: {{color: 'black', dash: 'dash'}} }}
                        ]
                    }};
                    Plotly.newPlot('avail_chart', getAvailTraces(currentStationIdx), availLayout);

                    const tsLayout = {{
                        title: "Water Levels & Forecast Target", margin: {{t: 40, b: 40, l: 40, r: 20}},
                        xaxis: {{ title: "Timestamp" }}, yaxis: {{ title: "Water Level (m)" }},
                        shapes: [
                            {{ type: 'line', x0: data.time_axis[data.lookback_idx - 1], x1: data.time_axis[data.lookback_idx - 1], y0: 0, y1: 1, yref: 'paper', line: {{color: 'black', dash: 'dash'}} }},
                            {{ type: 'line', x0: data.time_axis[currentTimeIdx], x1: data.time_axis[currentTimeIdx], y0: 0, y1: 1, yref: 'paper', line: {{color: 'gray', width: 4, opacity: 0.3}} }}
                        ]
                    }};
                    Plotly.newPlot('ts_chart', getTimeseriesData(currentStationIdx), tsLayout);

                    function updateMaps() {{
                        const isHeatmap = document.getElementById('heatmapToggle').checked;
                        const getLayout = (title) => isHeatmap ? layoutHeatmap(title) : layoutMap(title);
                        const view = getView();
                        Plotly.react('map_lr', getMapTraces(view.lr, currentTimeIdx, isHeatmap, 'lr'), getLayout("LR Wind & Surface Pressure"));
                    }}
                    updateMaps();

                    document.getElementById('timeSlider').addEventListener('input', function(e) {{
                        currentTimeIdx = parseInt(e.target.value);
                        document.getElementById('timeLabel').innerText = data.time_axis[currentTimeIdx];
                        updateMaps();
                        tsLayout.shapes[1].x0 = data.time_axis[currentTimeIdx];
                        tsLayout.shapes[1].x1 = data.time_axis[currentTimeIdx];
                        Plotly.relayout('ts_chart', {{ shapes: tsLayout.shapes }});
                    }});

                    document.getElementById('stationSelect').addEventListener('change', function(e) {{
                        currentStationIdx = parseInt(e.target.value);
                        Plotly.react('ts_chart', getTimeseriesData(currentStationIdx), tsLayout);
                        Plotly.react('avail_chart', getAvailTraces(currentStationIdx), availLayout);
                        updateTargetMaskDisplay(currentStationIdx);
                        updateMaps();
                    }});

                    document.getElementById('heatmapToggle').addEventListener('change', updateMaps);
                    document.getElementById('scaleToggle').addEventListener('change', function(e) {{
                        isScaled = e.target.checked;
                        updateMaps();
                        Plotly.react('ts_chart', getTimeseriesData(currentStationIdx), tsLayout);
                    }});

                    document.getElementById('timeLabel').innerText = data.time_axis[currentTimeIdx];
                }})
                .catch(error => {{
                    console.error("Error loading JSON:", error);
                    document.getElementById('timeLabel').innerText = "Failed to load data. Ensure you are using a local server.";
                }});
        </script>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nDashboard generated! Open '{filename}' with Live Server.")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading configuration and metadata...")
    config = load_config(MODEL_CONFIG_DIR)

    station_ids, station_names, st_lats, st_lons = get_region_stations(STATIONS_CSV, REGION_NAME)
    print(f"Loaded {len(station_ids)} stations for {REGION_NAME}.")

    target_dt = pd.to_datetime(VISUALIZATION_TIMESTAMP)
    timestamp_ids = get_sample_timestamps(
        SAMPLING_METADATA_CSV, target_dt, VISUALIZED_SAMPLE_CATEGORY, BATCH_SIZE,
    )
    print(f"Sampled {BATCH_SIZE} rows. Generating batch data...")

    data_cache = DatasetCache(REGION_NAME, WATER_LEVELS_H5)

    try:
        start_time = time.time()
        # Fetch raw once; produce scaled in-memory from a copy (single I/O pass).
        batch_lr_raw, batch_st_raw, batch_res_raw, batch_target_mask = sample_batch(
            timestamp_ids, config, station_ids, st_lats, st_lons, data_cache, scale=False,
        )
        batch_lr_sc, batch_st_sc, batch_res_sc = _apply_scaling(
            batch_lr_raw.copy(), batch_st_raw.copy(), batch_res_raw.copy(),
        )
        end_time = time.time()
        print(f"[INFO] Batch generation (raw + scaled) of size {BATCH_SIZE} "
              f"complete in {end_time - start_time:.2f} seconds.")
    finally:
        data_cache.close()

    lookback = config['lookback_hours']

    idx = 13  # batch member to visualize (kept simple — dashboard is a user-check tool)
    sample_raw = (
        batch_lr_raw[idx],
        batch_st_raw[idx, :, :, 0],             # tide predicted
        batch_st_raw[idx, :lookback, :, 1],     # observed (lookback only)
        batch_res_raw[idx],                     # residual target
    )
    sample_scaled = (
        batch_lr_sc[idx],
        batch_st_sc[idx, :, :, 0],
        batch_st_sc[idx, :lookback, :, 1],
        batch_res_sc[idx],
    )
    export_html_dashboard(
        sample_raw, sample_scaled, station_names, st_lats, st_lons,
        timestamp_ids[idx], config,
        target_mask=batch_target_mask[idx],
        tide_avail=batch_st_raw[idx, :, :, 2],   # (seq_len, num_stations)
        obs_avail=batch_st_raw[idx, :, :, 3],    # (seq_len, num_stations)
        filename="index.html",
    )
