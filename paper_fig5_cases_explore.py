"""
paper_fig5_cases_explore.py — internal browsing plots for the Fig 5/6 round.

In plain English: for each named storm that hit our gauges in 2024-2025
(Beryl, Debby, Francine, Helene, Milton, Erin, Imelda), this script makes
one big internal figure: the storm's best track on a map, plus one row per
hard-hit tide gauge showing three forecast "snapshots" — what the system
predicted when the forecast was issued about 60, 36, and 12 hours before
the water peaked at that gauge. Each snapshot shows the observed water
level, the STOFS-2D physics forecast, all 93 AI ensemble members, the AI
deterministic forecast (GFS-driven, 3-seed mean), the AI hindcast driven
by ERA5 reanalysis (i.e. "perfect" weather input — stitched from the
stride-72 h continuous archive, so it is a reference curve, NOT a forecast
issued at that cycle), and the gauge's official NOAA flood thresholds.
These are for our own discussion (what should Figs 5-6 say?), not for
the paper.

Reads:  outputs/uq_postprocessing/.../region_*/verify_arrays.nc  (93-member
        archive w/ obs, STOFS, thresholds),
        outputs/paper/inference/.../region_*/gfs_*z/seed_*/forecast.nc
        (deterministic GFS runs, via paper_fig2_warn_process.det_forecast),
        outputs/inference/nn_24lb_72fh_v2/region_*/20240101_20251130/
        forecast.nc  (ERA5-driven stride-72h runs, stitched hourly),
        data/storm_tracks/ibtracs_last3years.csv
Gives:  outputs/storms/fan_v2/fan_<storm>.png   (one per storm)
        outputs/storms/fan_v2/fan_station_table.csv  (per-storm gauge stats)
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from paper_fig2_warn_process import det_forecast

BASE = r'outputs/uq_postprocessing/nn_24lb_72fh_v2/gefs_psfc_v2_raw/20240101_20251201'
IBTRACS = r'data/storm_tracks/ibtracs_last3years.csv'
OUT_DIR = r'outputs/storms/fan_v2'

# storm windows come from the archive exceedance scan (2026-07-23), padded
STORMS = [
    dict(name='Beryl',    season=2024, t0='2024-07-05', t1='2024-07-12'),
    dict(name='Debby',    season=2024, t0='2024-08-02', t1='2024-08-10'),
    dict(name='Francine', season=2024, t0='2024-09-09', t1='2024-09-15'),
    dict(name='Helene',   season=2024, t0='2024-09-24', t1='2024-09-30'),
    dict(name='Milton',   season=2024, t0='2024-10-06', t1='2024-10-13'),
    dict(name='Erin',     season=2025, t0='2025-08-18', t1='2025-08-26'),
    dict(name='Imelda',   season=2025, t0='2025-09-28', t1='2025-10-07'),
]

MIN_PEAK_RES = 0.40      # gauge qualifies if observed surge peaks >= 40 cm
MAX_STATIONS = 12        # rows per storm figure, top peaks first
LEADS = (60, 36, 12)     # hours before the gauge's observed peak

C_OBS, C_ST, C_ENS = '#3d3d3d', '#AA3377', '#DDAA33'
C_AI = '#4477AA'
C_ERA = '#228833'
ERA_BASE = r'outputs/inference/nn_24lb_72fh_v2'
ERA_SPAN = '20240101_20251130'
DET_CACHE = {}     # region name -> (cycle DatetimeIndex, residual (C,72,S))
ERA_CACHE = {}     # region name -> pd.DataFrame hourly residual, cols = sids
THR_STYLE = {'minor': ('#4477AA', 'minor'), 'moderate': ('#e08214', 'moderate'),
             'major': ('#c22222', 'major')}


def load_track(name, season):
    """Reads the IBTrACS best-track file for one storm.
    Gets: storm name + season. Gives: DataFrame time/lat/lon/wind (kt)."""
    tr = pd.read_csv(IBTRACS, skiprows=[1], low_memory=False,
                     keep_default_na=False)
    tr = tr[(tr['BASIN'] == 'NA') & (tr['NAME'] == name.upper())
            & (tr['SEASON'].astype(str).str.startswith(str(season)))]
    out = pd.DataFrame(dict(
        time=pd.to_datetime(tr['ISO_TIME']),
        lat=pd.to_numeric(tr['LAT'], errors='coerce'),
        lon=pd.to_numeric(tr['LON'], errors='coerce'),
        wind=pd.to_numeric(tr['USA_WIND'], errors='coerce')))
    return out.dropna(subset=['lat', 'lon']).reset_index(drop=True)


def hourly_obs(ds, cyc_idx, var='obs_residual'):
    """Rebuilds an hourly observed series from a set of 6-hourly cycles:
    leads 1-6 of each cycle tile the timeline with no gaps.
    Gets: open dataset + cycle indices. Gives: (DatetimeIndex, (T,S) array)."""
    ct = pd.DatetimeIndex(ds.cycle_time.values[cyc_idx])
    arr = ds[var].isel(cycle_time=cyc_idx, lead_hour=slice(0, 6)).values
    times, rows = [], []
    for i, t in enumerate(ct):
        for lh in range(6):
            times.append(t + pd.Timedelta(hours=lh + 1))
            rows.append(arr[i, lh, :])
    return pd.DatetimeIndex(times), np.asarray(rows)


def pick_stations(storm):
    """Finds the gauges a storm hit hard: scans every region's observed
    surge inside the storm window. Gets: storm dict. Gives: DataFrame with
    one row per qualifying gauge (region dir, index, peak, coords, ...)."""
    t0, t1 = pd.Timestamp(storm['t0']), pd.Timestamp(storm['t1'])
    rows = []
    for rdir in sorted(glob.glob(os.path.join(BASE, 'region_*'))):
        ds = xr.open_dataset(os.path.join(rdir, 'verify_arrays.nc'))
        ct = pd.DatetimeIndex(ds.cycle_time.values)
        cyc = np.where((ct >= t0 - pd.Timedelta(hours=6)) & (ct <= t1))[0]
        if len(cyc) == 0:
            ds.close()
            continue
        times, res = hourly_obs(ds, cyc, 'obs_residual')
        _, wl = hourly_obs(ds, cyc, 'obs_water_level')
        sids = ds.station.values.astype(str)
        for s in range(len(sids)):
            v = res[:, s]
            if not np.isfinite(v).any():
                continue
            k = int(np.nanargmax(v))
            if v[k] < MIN_PEAK_RES:
                continue
            rows.append(dict(
                storm=storm['name'], region=os.path.basename(rdir),
                s_idx=s, sid=sids[s],
                sname=str(ds.station_name.values[s]),
                lat=float(ds.station_lat.values[s]),
                lon=float(ds.station_lon.values[s]),
                peak_res_m=float(v[k]), peak_time=times[k],
                peak_wl_m=float(wl[k, s]) if np.isfinite(wl[k, s]) else np.nan,
                minor=float(ds.minor_threshold.values[s]),
                moderate=float(ds.moderate_threshold.values[s]),
                major=float(ds.major_threshold.values[s])))
        ds.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values('peak_res_m', ascending=False)
              .head(MAX_STATIONS).reset_index(drop=True))


def era5_series(region):
    """Stitches the region's stride-72h ERA5-driven runs into one continuous
    hourly residual table ("perfect weather input" hindcast). Gets: region
    name. Gives: DataFrame indexed by valid hour, one column per gauge id,
    or None if the archive is missing."""
    p = os.path.join(ERA_BASE, region, ERA_SPAN, 'forecast.nc')
    if not os.path.exists(p):
        return None
    ds = xr.open_dataset(p)
    ct = pd.DatetimeIndex(ds.cycle_time.values)
    res = ds.residual_mean.values.astype(np.float32)          # (C, 72, S)
    if 'valid' in ds.variables:
        bad = ~ds['valid'].values.astype(bool)                # (C, S)
        res[bad[:, None, :].repeat(72, axis=1)] = np.nan
    sids = [str(s) for s in ds.station.values]
    ds.close()
    times, rows = [], []
    for i, t in enumerate(ct):
        for lh in range(72):
            times.append(t + pd.Timedelta(hours=lh + 1))
            rows.append(res[i, lh, :])
    return pd.DataFrame(np.asarray(rows), index=pd.DatetimeIndex(times),
                        columns=sids)


def era5_wl_line(ds, region, c_idx, s_idx):
    """Looks up the stitched ERA5 hindcast over one panel's 72 valid hours
    at one gauge, converted to water level with the cycle's tide + SLR.
    Gives: (72,) array, or None if unavailable."""
    if region not in ERA_CACHE:
        ERA_CACHE[region] = era5_series(region)
    era = ERA_CACHE[region]
    sid = str(ds.station.values[s_idx])
    if era is None or sid not in era.columns:
        return None
    ct = pd.Timestamp(ds.cycle_time.values[c_idx])
    tt = pd.DatetimeIndex([ct + pd.Timedelta(hours=int(h))
                           for h in ds.lead_hour.values])
    res = era[sid].reindex(tt).values
    if not np.isfinite(res).any():
        return None
    tide_slr = (ds.tide.isel(cycle_time=c_idx, station=s_idx).values
                + ds.slr_component.isel(cycle_time=c_idx, station=s_idx).values)
    return res + tide_slr


def det_wl_line(ds, region, c_idx, s_idx):
    """Looks up the AI deterministic (GFS 3-seed-mean) forecast for one
    cycle at one gauge and converts it to water level (residual + tide +
    sea-level-rise component). Gets: open dataset, region name, cycle and
    station index. Gives: (72,) water-level array, or None if missing."""
    if region not in DET_CACHE:
        sids = [str(s) for s in ds.station.values]
        DET_CACHE[region] = det_forecast(region, sids)
    det = DET_CACHE[region]
    if det is None:
        return None
    det_t, det_a = det
    ct = pd.Timestamp(ds.cycle_time.values[c_idx])
    hits = np.where(det_t == ct)[0]
    if len(hits) == 0:
        return None
    res = det_a[int(hits[0]), :, s_idx]
    tide_slr = (ds.tide.isel(cycle_time=c_idx, station=s_idx).values
                + ds.slr_component.isel(cycle_time=c_idx, station=s_idx).values)
    return res + tide_slr


def draw_fan(ax, ds, region, c_idx, s_idx, row):
    """Draws one forecast snapshot: obs, STOFS, 93 members, AI det (GFS),
    thresholds, all in water-level space for one cycle at one gauge.
    Gets: axis, open dataset, region name, cycle index, station index,
    gauge row."""
    ct = pd.Timestamp(ds.cycle_time.values[c_idx])
    tt = pd.DatetimeIndex([ct + pd.Timedelta(hours=int(h))
                           for h in ds.lead_hour.values])
    mem = ds.ai_water_level_member.isel(
        cycle_time=c_idx, station=s_idx).values.reshape(-1, 72)   # (93, 72)
    ok = ds.member_valid.isel(
        cycle_time=c_idx, station=s_idx).values.reshape(-1).astype(bool)
    mem = mem[ok] if ok.any() else mem
    obs = ds.obs_water_level.isel(cycle_time=c_idx, station=s_idx).values
    st = ds.stofs_water_level.isel(cycle_time=c_idx, station=s_idx).values
    det = det_wl_line(ds, region, c_idx, s_idx)
    era = era5_wl_line(ds, region, c_idx, s_idx)

    for m in mem:
        ax.plot(tt, m, color=C_ENS, lw=0.4, alpha=0.16, zorder=2)
    if det is not None:
        ax.plot(tt, det, color=C_AI, lw=1.2, zorder=4)
    if era is not None:
        ax.plot(tt, era, color=C_ERA, lw=1.1, ls=(0, (4, 2)), zorder=4)
    ax.plot(tt, st, color=C_ST, lw=1.1, zorder=3)
    ax.plot(tt, obs, color=C_OBS, lw=1.2, zorder=5)
    ax.axvline(row['peak_time'], color='#999999', lw=0.6, ls=':', zorder=1)

    dmin = np.nanmin(det) if det is not None else np.nan
    dmax = np.nanmax(det) if det is not None else np.nan
    emin = np.nanmin(era) if era is not None else np.nan
    emax = np.nanmax(era) if era is not None else np.nan
    lo = np.nanmin([np.nanmin(obs), np.nanmin(mem) if len(mem) else np.nan,
                    np.nanmin(st), dmin, emin])
    hi = np.nanmax([np.nanmax(obs), np.nanmax(mem) if len(mem) else np.nan,
                    np.nanmax(st), dmax, emax])
    pad = 0.12 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + 3.5 * pad)
    for key, (col, lab) in THR_STYLE.items():
        thr = row[key]
        if np.isfinite(thr) and thr < hi + 3.5 * pad:
            ax.axhline(thr, color=col, lw=0.7, ls=(0, (5, 3)), zorder=1,
                       alpha=0.8)
    ax.set_xlim(tt[0], tt[-1])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.tick_params(labelsize=6)
    ax.grid(color='#f2f2f2', lw=0.4)
    ax.set_axisbelow(True)
    return ct


def draw_map(ax, track, sel, storm):
    """Draws the best track (dots colored by wind speed) and the selected
    gauges (colored by observed surge peak) on a coastline map."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lats = list(sel['lat']) + track['lat'].tolist()
    lons = list(sel['lon']) + track['lon'].tolist()
    lat0, lat1 = min(lats), max(lats)
    lon0, lon1 = max(min(lons), -100), min(max(lons), -60)
    ax.set_extent([lon0 - 2, lon1 + 2,
                   max(lat0 - 2, 15), min(lat1 + 2, 50)], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f2efe9')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#eaf2f8')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#8a8a8a',
                   lw=0.5)
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#cccccc',
                   lw=0.3)
    ax.plot(track['lon'], track['lat'], color='#666666', lw=0.8,
            transform=ccrs.PlateCarree(), zorder=4)
    sc1 = ax.scatter(track['lon'], track['lat'], c=track['wind'],
                     cmap='YlOrRd', vmin=20, vmax=140, s=14,
                     edgecolors='#555555', linewidths=0.3,
                     transform=ccrs.PlateCarree(), zorder=5)
    sc2 = ax.scatter(sel['lon'], sel['lat'], c=sel['peak_res_m'],
                     cmap='viridis', vmin=0, s=55, marker='^',
                     edgecolors='white', linewidths=0.6,
                     transform=ccrs.PlateCarree(), zorder=6)
    for _, r in sel.iterrows():
        ax.annotate(r['sname'].split(',')[0], (r['lon'], r['lat']),
                    xytext=(4, 3), textcoords='offset points', fontsize=5.5,
                    color='#333333', zorder=7)
    plt.colorbar(sc1, ax=ax, fraction=0.035, pad=0.01).set_label(
        'track wind (kt)', fontsize=6)
    plt.colorbar(sc2, ax=ax, fraction=0.035, pad=0.04).set_label(
        'observed surge peak (m)', fontsize=6)
    ax.set_title(f"{storm['name']} ({storm['season']})  ·  window "
                 f"{storm['t0']} .. {storm['t1']}", fontsize=9,
                 fontweight='bold')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    import cartopy.crs as ccrs
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    for storm in STORMS:
        sel = pick_stations(storm)
        if sel.empty:
            print(f"[{storm['name']}] no gauge above {MIN_PEAK_RES} m - skip")
            continue
        all_rows.append(sel)
        track = load_track(storm['name'], storm['season'])
        n = len(sel)
        print(f"\n=== {storm['name']} - {n} gauges, peak "
              f"{sel['peak_res_m'].max():.2f} m @ {sel.iloc[0]['sname']} ===")

        fig = plt.figure(figsize=(13, 3.8 + 1.55 * n))
        gs = fig.add_gridspec(n + 1, 3, height_ratios=[2.6] + [1] * n,
                              hspace=0.55, wspace=0.16,
                              left=0.055, right=0.985, top=0.97, bottom=0.02)
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        draw_map(ax_map, track, sel, storm)

        # keep one dataset open per region used by this storm
        open_ds = {r: xr.open_dataset(os.path.join(BASE, r, 'verify_arrays.nc'))
                   for r in sel['region'].unique()}
        for i, (_, row) in enumerate(sel.iterrows()):
            ds = open_ds[row['region']]
            ct_all = pd.DatetimeIndex(ds.cycle_time.values)
            for j, lead in enumerate(LEADS):
                ax = fig.add_subplot(gs[i + 1, j])
                target = row['peak_time'] - pd.Timedelta(hours=lead)
                cand = np.where(ct_all <= target)[0]
                if len(cand) == 0:
                    ax.set_visible(False)
                    continue
                c_idx = int(cand[-1])
                ct = draw_fan(ax, ds, row['region'], c_idx, row['s_idx'], row)
                real_lead = (row['peak_time'] - ct).total_seconds() / 3600
                ax.set_title(f"issued {ct:%m/%d %HZ}  ({real_lead:.0f} h "
                             f"before peak)", fontsize=6.5, pad=2)
                if j == 0:
                    ax.set_ylabel(f"{row['sname'].split(',')[0]}\n"
                                  f"({row['sid']})  wl (m)", fontsize=6.5)
        for ds in open_ds.values():
            ds.close()

        # one shared legend at the very top right
        from matplotlib.lines import Line2D
        fig.legend(handles=[
            Line2D([], [], color=C_OBS, lw=1.2, label='observed'),
            Line2D([], [], color=C_ST, lw=1.1, label='STOFS-2D'),
            Line2D([], [], color=C_ENS, lw=1.0, alpha=0.5,
                   label='AI members (93)'),
            Line2D([], [], color=C_AI, lw=1.2,
                   label='AI deterministic (GFS)'),
            Line2D([], [], color=C_ERA, lw=1.1, ls=(0, (4, 2)),
                   label='AI w/ ERA5 reanalysis (stitched hindcast)'),
            Line2D([], [], color='#4477AA', lw=0.7, ls=(0, (5, 3)),
                   label='minor / moderate / major thresholds')],
            loc='upper right', fontsize=6.5, frameon=False,
            bbox_to_anchor=(0.995, 0.998))

        out = os.path.join(OUT_DIR, f"fan_{storm['name'].lower()}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    tab = pd.concat(all_rows, ignore_index=True)
    tab.to_csv(os.path.join(OUT_DIR, 'fan_station_table.csv'), index=False)
    print(f"\nwrote {os.path.join(OUT_DIR, 'fan_station_table.csv')}")
    print(tab[['storm', 'sname', 'peak_res_m', 'peak_wl_m', 'minor',
               'moderate', 'major']].round(2).to_string(index=False))
