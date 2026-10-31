"""
s_gefs_archive.py — Member-aware GEFS forecast-atmosphere adapter for offline
ensemble inference.

Sibling of `s_gfs_archive.py`. For each (cycle_time c, GEFS member m,
lookback_hours, forecast_hours) it builds a (T_total, 3, 64, 64) tensor where:

  * Forecast hours h = c+1 .. c+forecast_hours come from the GEFS member m
    initialized at `atm_init = max(GEFS init <= c)`. GEFS native lead cadence
    is 3-hourly out to +84h, so we linearly interpolate to hourly leads.
  * Lookback hours h = c-23 .. c come from the deterministic GFS archive
    stitched across inits (same logic as `s_gfs_archive.py`).

This mirrors the operational website pipeline: deterministic GFS lookback +
ensemble GEFS forecast.

Output channel order matches the ERA5/GFS path: [U10 m/s, V10 m/s, P hPa]
where P = PSFC (v1 contract) or MSLP (v2 contract), float16 with NaN where the
source is missing — `_apply_scaling` in `s_sample_batch.py` replaces NaN with 0.
"""

import os
import threading

import numpy as np
import pandas as pd
import xarray as xr


GFS_CYCLE_HOURS = 6     # GFS deterministic: 00 / 06 / 12 / 18 UTC.
GEFS_CYCLE_HOURS = 6    # GEFS:              00 / 06 / 12 / 18 UTC.
GEFS_LEAD_STEP = 3      # GEFS native lead cadence (hours).


def _gfs_init_for(hour):
    """Most recent GFS init at or before `hour` (hourly resolution)."""
    h = pd.Timestamp(hour).floor('h')
    return h - pd.Timedelta(hours=h.hour % GFS_CYCLE_HOURS)


def _gefs_init_for(hour):
    """Most recent GEFS init at or before `hour` (hourly resolution)."""
    h = pd.Timestamp(hour).floor('h')
    return h - pd.Timedelta(hours=h.hour % GEFS_CYCLE_HOURS)


class GEFSArchiveCache:
    """Monthly-zarr-backed cache for the GEFS forecast atmosphere.

    Pairs with an internal GFS deterministic cache for lookback frames.

    `pressure_var` is "PSFC" (v1 contract) or "MSLP" (v2 contract). The same
    name is used to slice both the GEFS and GFS zarrs.
    """

    def __init__(self, region_name, gefs_base_dir, gfs_base_dir,
                 pressure_var='PSFC', max_months=2):
        self.region_name = region_name
        self.gefs_base_dir = gefs_base_dir
        self.gfs_base_dir = gfs_base_dir
        self.pressure_var = pressure_var
        self.max_months = max_months

        # Per-month dense numpy caches (LRU-evicted to `max_months` each, so a
        # multi-year span never holds more than a couple of months resident).
        self._gefs_cache = {}
        self._gefs_order = []
        self._gfs_cache = {}
        self._gfs_order = []
        self._lock = threading.Lock()

    @staticmethod
    def _lru_insert(cache, order, key, value, max_months):
        """Insert `key`->`value`, evicting the oldest entries past `max_months`."""
        cache[key] = value
        if key in order:
            order.remove(key)
        order.append(key)
        while len(order) > max_months:
            old = order.pop(0)
            cache.pop(old, None)

    # ------------------------------------------------------------------ GEFS
    def _gefs_path(self, month_str):
        return os.path.join(self.gefs_base_dir, self.region_name,
                            f'{month_str}.zarr')

    def _load_gefs_month(self, month_str):
        with self._lock:
            if month_str in self._gefs_cache:
                return self._gefs_cache[month_str]
        path = self._gefs_path(month_str)
        if not os.path.exists(path):
            with self._lock:
                self._lru_insert(self._gefs_cache, self._gefs_order,
                                 month_str, None, self.max_months)
            return None
        ds = xr.open_zarr(path, consolidated=True)
        members = [str(m) for m in ds['member'].values]
        member_to_idx = {m: i for i, m in enumerate(members)}
        # Load each needed variable individually (avoids ds.compute() pulling
        # all 4 vars — incl. the unused MSLP — into RAM simultaneously).
        result = {
            'init_time': pd.DatetimeIndex(ds['init_time'].values),
            'lead_hour': ds['lead_hour'].values.astype(int),
            'members':   members,
            'member_to_idx': member_to_idx,
            'U10':  ds['U10'].values,
            'V10':  ds['V10'].values,
            self.pressure_var: ds[self.pressure_var].values,
        }
        ds.close()
        with self._lock:
            self._lru_insert(self._gefs_cache, self._gefs_order,
                             month_str, result, self.max_months)
        return result

    def gefs_get(self, init_time, member, lead_hour):
        """Return raw (U10, V10, P) at GEFS (init, member, lead_hour) as 2D
        arrays. Returns None if the slice is absent.
        """
        init_time = pd.Timestamp(init_time)
        month_str = init_time.strftime('%Y%m')
        month = self._load_gefs_month(month_str)
        if month is None:
            return None
        i_idx = int(month['init_time'].searchsorted(init_time))
        if (i_idx == len(month['init_time'])
                or month['init_time'][i_idx] != init_time):
            return None
        m_idx = month['member_to_idx'].get(str(member))
        if m_idx is None:
            return None
        l_idx = int(np.searchsorted(month['lead_hour'], lead_hour))
        if (l_idx == len(month['lead_hour'])
                or month['lead_hour'][l_idx] != lead_hour):
            return None
        return (month['U10'][i_idx, m_idx, l_idx],
                month['V10'][i_idx, m_idx, l_idx],
                month[self.pressure_var][i_idx, m_idx, l_idx])

    def gefs_members_for_cycle(self, init_time):
        """Return the list of available member labels for a GEFS init."""
        init_time = pd.Timestamp(init_time)
        month = self._load_gefs_month(init_time.strftime('%Y%m'))
        if month is None:
            return []
        if init_time not in month['init_time']:
            return []
        return list(month['members'])

    def available_gefs_inits(self, start_time, end_time):
        """List GEFS init times in [start, end] (inclusive) across months."""
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        months = pd.date_range(start.to_period('M').to_timestamp(),
                               end.to_period('M').to_timestamp(),
                               freq='MS').strftime('%Y%m').tolist()
        out = []
        for m in months:
            path = self._gefs_path(m)
            if not os.path.exists(path):
                continue
            # Read only the init_time coordinate — never the heavy data vars.
            ds = xr.open_zarr(path, consolidated=True)
            inits = pd.DatetimeIndex(ds['init_time'].values)
            ds.close()
            mask = (inits >= start) & (inits <= end)
            out.extend(inits[mask].tolist())
        return sorted(set(out))

    # ------------------------------------------------------------------- GFS
    def _gfs_path(self, month_str):
        return os.path.join(self.gfs_base_dir, self.region_name,
                            f'{month_str}.zarr')

    def _load_gfs_month(self, month_str):
        with self._lock:
            if month_str in self._gfs_cache:
                return self._gfs_cache[month_str]
        path = self._gfs_path(month_str)
        if not os.path.exists(path):
            with self._lock:
                self._lru_insert(self._gfs_cache, self._gfs_order,
                                 month_str, None, self.max_months)
            return None
        ds = xr.open_zarr(path, consolidated=True)
        result = {
            'init_time': pd.DatetimeIndex(ds['init_time'].values),
            'lead_hour': ds['lead_hour'].values.astype(int),
            'U10':  ds['U10'].values,
            'V10':  ds['V10'].values,
            self.pressure_var: ds[self.pressure_var].values,
        }
        ds.close()
        with self._lock:
            self._lru_insert(self._gfs_cache, self._gfs_order,
                             month_str, result, self.max_months)
        return result

    def gfs_get(self, init_time, lead_hour):
        """Deterministic GFS slice at (init, lead). Returns None if absent."""
        init_time = pd.Timestamp(init_time)
        month_str = init_time.strftime('%Y%m')
        month = self._load_gfs_month(month_str)
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
                month[self.pressure_var][i_idx, l_idx])


def _gefs_interp_member(cache, atm_init, member, target_lead_h):
    """Linearly interpolate a single GEFS member to an hourly lead.

    GEFS native leads are 0, 3, 6, ..., 84. For a target lead `target_lead_h`,
    find the bracketing leads `l0 <= target_lead_h <= l1` (l1 - l0 == 3) and
    return a weighted average of the two fields. If target_lead_h is exactly
    on a native lead, the lookup is a direct slice with no interpolation.
    """
    if target_lead_h < 0 or target_lead_h > 84:
        return None
    # Bracketing 3-hour leads.
    l0 = (target_lead_h // GEFS_LEAD_STEP) * GEFS_LEAD_STEP
    l1 = l0 + GEFS_LEAD_STEP
    w1 = (target_lead_h - l0) / float(GEFS_LEAD_STEP)
    f0 = cache.gefs_get(atm_init, member, int(l0))
    if f0 is None:
        return None
    if w1 == 0.0:
        return f0
    f1 = cache.gefs_get(atm_init, member, int(l1))
    if f1 is None:
        # Fall back to nearest available lead rather than NaN.
        return f0
    u = (1.0 - w1) * f0[0].astype(np.float32) + w1 * f1[0].astype(np.float32)
    v = (1.0 - w1) * f0[1].astype(np.float32) + w1 * f1[1].astype(np.float32)
    p = (1.0 - w1) * f0[2].astype(np.float32) + w1 * f1[2].astype(np.float32)
    return (u, v, p)


def fetch_lr_wind_gefs_member(cache, cycle_time, member,
                              lookback_hours, forecast_hours):
    """Build a (T_total, 3, 64, 64) float16 tensor for one (cycle, member).

    Hour layout matches the ERA5/GFS path:
        time_range = pd.date_range(end=c + forecast_hours,
                                    periods=lookback + forecast, freq='h')
    Forecast frames come from the GEFS member; lookback frames come from
    stitched deterministic GFS (member-independent).
    """
    total = lookback_hours + forecast_hours
    cycle_time = pd.Timestamp(cycle_time)
    hours = pd.date_range(
        end=cycle_time + pd.Timedelta(hours=forecast_hours),
        periods=total, freq='h',
    )

    gefs_init = _gefs_init_for(cycle_time)

    out = np.full((total, 3, 64, 64), np.nan, dtype=np.float16)
    for i, h in enumerate(hours):
        if h <= cycle_time:
            # Lookback: stitched deterministic GFS.
            atm_init = _gfs_init_for(h)
            lead = int((h - atm_init).total_seconds() // 3600)
            field = cache.gfs_get(atm_init, lead)
        else:
            # Forecast: GEFS member, 3h→1h interpolated.
            lead_h = int((h - gefs_init).total_seconds() // 3600)
            field = _gefs_interp_member(cache, gefs_init, member, lead_h)
        if field is None:
            continue
        u, v, p = field
        out[i, 0] = u
        out[i, 1] = v
        out[i, 2] = p
    return out


def fetch_lr_wind_gefs_cycle(cache, cycle_time, members,
                             lookback_hours, forecast_hours):
    """Stack `fetch_lr_wind_gefs_member` over `members`.

    Returns (n_members, T_total, 3, 64, 64) float16. The lookback portion is
    identical across members (deterministic GFS), so we build it once and
    broadcast.
    """
    total = lookback_hours + forecast_hours
    cycle_time = pd.Timestamp(cycle_time)
    hours = pd.date_range(
        end=cycle_time + pd.Timedelta(hours=forecast_hours),
        periods=total, freq='h',
    )

    # Lookback once.
    look_arr = np.full((lookback_hours, 3, 64, 64), np.nan, dtype=np.float16)
    for i, h in enumerate(hours[:lookback_hours]):
        atm_init = _gfs_init_for(h)
        lead = int((h - atm_init).total_seconds() // 3600)
        field = cache.gfs_get(atm_init, lead)
        if field is None:
            continue
        look_arr[i, 0] = field[0]
        look_arr[i, 1] = field[1]
        look_arr[i, 2] = field[2]

    gefs_init = _gefs_init_for(cycle_time)

    out = np.full((len(members), total, 3, 64, 64), np.nan, dtype=np.float16)
    for m_idx, m in enumerate(members):
        out[m_idx, :lookback_hours] = look_arr
        for j, h in enumerate(hours[lookback_hours:]):
            i = lookback_hours + j
            lead_h = int((h - gefs_init).total_seconds() // 3600)
            field = _gefs_interp_member(cache, gefs_init, m, lead_h)
            if field is None:
                continue
            out[m_idx, i, 0] = field[0]
            out[m_idx, i, 1] = field[1]
            out[m_idx, i, 2] = field[2]
    return out


def available_gefs_cycles(cache, start_time, end_time):
    """Return GEFS init times in [start, end] that are present in the archive."""
    return cache.available_gefs_inits(start_time, end_time)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Lightweight self-test: open Oct 2024 region_05 archive and fetch one
    # (cycle, member) tensor. No model invocation.
    cache = GEFSArchiveCache(
        region_name='region_05',
        gefs_base_dir='data/atmospheric_forecasts/gefs',
        gfs_base_dir='data/atmospheric_forecasts/gfs',
        pressure_var='PSFC',
    )
    cycles = available_gefs_cycles(cache, '2024-10-10', '2024-10-12')
    print(f"GEFS inits between 2024-10-10..12: {len(cycles)}")
    if cycles:
        c = cycles[len(cycles) // 2]
        members = cache.gefs_members_for_cycle(c)
        print(f"  picked cycle {c}, members available: {len(members)}")
        tens = fetch_lr_wind_gefs_member(cache, c, members[0], 24, 72)
        print(f"  single-member tensor: {tens.shape}, dtype={tens.dtype}, "
              f"NaN frac={np.isnan(tens).mean():.3f}")
        tens_all = fetch_lr_wind_gefs_cycle(cache, c, members[:3], 24, 72)
        print(f"  3-member tensor: {tens_all.shape}, "
              f"NaN frac={np.isnan(tens_all).mean():.3f}")
