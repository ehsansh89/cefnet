"""
s_gfs_archive.py — GFS forecast-atmosphere adapter for inference.

Implements the §5 stitched-lookback fetcher: for each (cycle_time c,
lookback_hours, forecast_hours), build a (T_total, 3, 64, 64) tensor of
LR wind/pressure where every frame comes from a GFS forecast initialized
*before* the frame's valid hour.

  * Forecast hours h = c+1 .. c+forecast_hours: read from one fixed init,
    `atm_init = max(GFS init <= c)`. nwp_lead = h - atm_init (>= 1).
  * Lookback hours h = c-23 .. c: stitched across GFS inits — for each h,
    `atm_init(h) = max(GFS init <= h)`, nwp_lead = h - atm_init(h) (0..5).

Output channel order, units, and dtype match the ERA5 path
(`fetch_lr_wind` in s_sample_batch.py): [U10 m/s, V10 m/s, PSFC hPa],
float16. Frames whose source slice is missing from the archive are
NaN-filled — `_apply_scaling` then replaces NaN with 0 inside the
existing scaler pipeline.

See docs/note_gfs_download_plan.md §5.
"""

import os
import threading

import numpy as np
import pandas as pd
import xarray as xr


GFS_CYCLE_HOURS = 6   # GFS deterministic runs at 00 / 06 / 12 / 18 UTC.


def _atm_init_for(hour):
    """Most recent GFS init at or before `hour` (hourly resolution)."""
    h = pd.Timestamp(hour).floor('h')
    return h - pd.Timedelta(hours=h.hour % GFS_CYCLE_HOURS)


class GFSArchiveCache:
    """Monthly-zarr-backed cache for the GFS forecast atmosphere.

    Each access returns NumPy arrays (held in memory after first load)
    keyed by the calendar month containing the requested init_time. Hourly
    leads are read directly with no temporal interpolation.
    """

    def __init__(self, region_name, gfs_base_dir):
        self.region_name = region_name
        self.gfs_base_dir = gfs_base_dir
        self._cache = {}
        self._lock = threading.Lock()

    def _zarr_path(self, month_str):
        return os.path.join(self.gfs_base_dir, self.region_name,
                            f'{month_str}.zarr')

    def _load_month(self, month_str):
        with self._lock:
            if month_str in self._cache:
                return self._cache[month_str]
        path = self._zarr_path(month_str)
        if not os.path.exists(path):
            with self._lock:
                self._cache[month_str] = None
            return None
        ds = xr.open_zarr(path, consolidated=True)
        computed = ds.compute()
        result = {
            'init_time': pd.DatetimeIndex(computed['init_time'].values),
            'lead_hour': computed['lead_hour'].values.astype(int),
            'U10':  computed['U10'].values,
            'V10':  computed['V10'].values,
            'PSFC': computed['PSFC'].values,
        }
        ds.close()
        with self._lock:
            self._cache[month_str] = result
        return result

    def get_field(self, init_time, lead_hour):
        """Return (U10, V10, PSFC) at (init_time, lead_hour) as 2D arrays.

        Returns None if the month is absent or the (init, lead) pair isn't
        present in the zarr.
        """
        init_time = pd.Timestamp(init_time)
        month_str = init_time.strftime('%Y%m')
        month = self._load_month(month_str)
        if month is None:
            return None
        i_idx = int(month['init_time'].searchsorted(init_time))
        if (i_idx == len(month['init_time'])
                or month['init_time'][i_idx] != init_time):
            return None
        l_idx = int(np.searchsorted(month['lead_hour'], lead_hour))
        if (l_idx == len(month['lead_hour'])
                or month['lead_hour'][l_idx] != lead_hour):
            return None
        return (month['U10'][i_idx, l_idx],
                month['V10'][i_idx, l_idx],
                month['PSFC'][i_idx, l_idx])


def fetch_lr_wind_gfs(gfs_cache, cycle_time, lookback_hours, forecast_hours):
    """Build a (T_total, 3, 64, 64) float16 LR-wind tensor for one cycle.

    The hour timeline matches the ERA5 path's `time_range` convention:
        time_range = pd.date_range(end=c + forecast_hours,
                                    periods=lookback + forecast, freq='h')
    so frame i ↔ hour `c + forecast_hours - (T_total - 1 - i)`.
    """
    total = lookback_hours + forecast_hours
    cycle_time = pd.Timestamp(cycle_time)
    hours = pd.date_range(
        end=cycle_time + pd.Timedelta(hours=forecast_hours),
        periods=total, freq='h',
    )

    fc_atm_init = _atm_init_for(cycle_time)

    out = np.full((total, 3, 64, 64), np.nan, dtype=np.float16)
    for i, h in enumerate(hours):
        if h <= cycle_time:
            atm_init = _atm_init_for(h)            # stitched
        else:
            atm_init = fc_atm_init                 # fixed forecast init
        lead = int((h - atm_init).total_seconds() // 3600)
        field = gfs_cache.get_field(atm_init, lead)
        if field is None:
            continue
        u, v, p = field
        out[i, 0] = u
        out[i, 1] = v
        out[i, 2] = p
    return out
