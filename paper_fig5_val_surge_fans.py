"""
paper_fig5_val_surge_fans.py — dense surge fan sheets for the VALIDATION-
era storms (2022-2023: Ian, Nicole, Idalia, Ophelia), extending
outputs/storms/fan_v2_surge/.

In plain English: same browsing sheets as the 2024-25 storms (one figure
per storm, one row per hard-hit gauge, 11 snapshots issued 66..6 h before
that gauge's peak) — but these years have no STOFS or GEFS archives, so
each panel shows what legitimately exists: the observed surge and the AI
model driven by TRUE (ERA5) winds, run fresh here for every 6-hourly
cycle (3 seeds: thin lines; mean: green). These storms were in the
model's VALIDATION years (used for early stopping only, never for weight
updates) — extrapolation browsing, not operational verification.

Reads:  ERA5 zarrs + v2 water levels + checkpoints, IBTrACS full file,
        fig5_extrap_support{.csv,_levels.parquet}
Gives:  outputs/storms/fan_v2_surge/fan_<storm>.png (ian, nicole,
        idalia, ophelia)
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from s_sample_batch_v2 import DatasetCache, get_region_stations, sample_batch
from s_scalers import normalize_data
from train import load_train_config
from inference_v2 import load_model_from_checkpoint
from paper_fig5_cases_explore import draw_map

STATIONS_CSV = r'data/stations_processed_v2/stations_processed.csv'
H5 = r'data/water_levels_processed_v2/processed_water_levels.h5'
SCALER = r'samples/scalers_v2/water_level_residuals.pkl'
SUP = r'outputs/paper/v2/fig5_extrap_support.csv'
LEVELS = r'outputs/paper/v2/fig5_extrap_support_levels.parquet'
IBTRACS_FULL = r'data/storm_tracks/ibtracs_NA_full.csv'
OUT_DIR = r'outputs/storms/fan_v2_surge'

VAL_STORMS = [
    dict(name='Ian', season=2022, t0='2022-09-26', t1='2022-10-04'),
    dict(name='Nicole', season=2022, t0='2022-11-07', t1='2022-11-12'),
    dict(name='Idalia', season=2023, t0='2023-08-28', t1='2023-09-02'),
    dict(name='Ophelia', season=2023, t0='2023-09-21', t1='2023-09-26'),
]
LEADS = tuple(range(66, 0, -6))
MIN_PEAK_RES, MAX_STATIONS = 0.40, 12
SEEDS = (1, 2, 3)
C_OBS, C_ERA = '#3d3d3d', '#228833'


def load_track_full(name, season):
    """Best track from the full IBTrACS NA file (covers 2022-23).
    Gets storm name + season; gives DataFrame time/lat/lon/wind."""
    tr = pd.read_csv(IBTRACS_FULL, skiprows=[1], low_memory=False,
                     keep_default_na=False)
    tr = tr[(tr['BASIN'] == 'NA') & (tr['NAME'] == name.upper())
            & (tr['SEASON'].astype(str).str.startswith(str(season)))]
    out = pd.DataFrame(dict(
        time=pd.to_datetime(tr['ISO_TIME'], errors='coerce'),
        lat=pd.to_numeric(tr['LAT'], errors='coerce'),
        lon=pd.to_numeric(tr['LON'], errors='coerce'),
        wind=pd.to_numeric(tr['USA_WIND'], errors='coerce')))
    return out.dropna(subset=['lat', 'lon']).reset_index(drop=True)


def lessons_fn():
    """(sid, level) -> independent training events at/above level."""
    lv = pd.read_parquet(LEVELS)
    tab = {sid: (g['level'].values, g['events'].values)
           for sid, g in lv.groupby('sid')}

    def fn(sid, obs):
        if sid not in tab:
            return None
        levels, events = tab[sid]
        if obs > levels[-1] + 0.05:
            return 0
        idx = np.searchsorted(levels, obs, side='right') - 1
        return int(events[min(max(idx, 0), len(events) - 1)])
    return fn


def pick_stations_h5(storm, st):
    """Gauges the storm hit hard, from the h5 observations alone.
    Gets storm dict + stations table; gives DataFrame like the 2024
    browse (sid, sname, region, lat, lon, peak_res_m, peak_time...)."""
    t0, t1 = pd.Timestamp(storm['t0']), pd.Timestamp(storm['t1'])
    rows = []
    with pd.HDFStore(H5, mode='r') as store:
        for _, r in st.iterrows():
            key = f"/station_{r['station_id']}"
            if key not in store:
                continue
            df = store.get(key)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            res = df['residual'].loc[t0:t1].dropna()
            if res.empty or res.max() < MIN_PEAK_RES:
                continue
            rows.append(dict(
                sid=r['station_id'], sname=r['name'],
                region=f"region_{int(r['region']):02d}",
                lat=float(r['lat']), lon=float(r['lon']),
                peak_res_m=float(res.max()), peak_time=res.idxmax()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values('peak_res_m', ascending=False)
              .head(MAX_STATIONS).reset_index(drop=True))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    import cartopy.crs as ccrs
    device = torch.device('cpu')
    train_cfg = load_train_config('train_config_v2.json')
    with open('nn_config.json') as f:
        model_config = json.load(f)

    st = pd.read_csv(STATIONS_CSV, dtype={'station_id': str})
    sup = pd.read_csv(SUP, dtype={'sid': str}).set_index('sid')
    les = lessons_fn()

    models, caches, meta = {}, {}, {}

    def region_setup(reg):
        if reg not in caches:
            sids, snames, lats, lons = get_region_stations(STATIONS_CSV,
                                                           reg)
            caches[reg] = DatasetCache(reg, H5)
            meta[reg] = (sids, lats, lons,
                         {str(s): i for i, s in enumerate(sids)})
            for seed in SEEDS:
                models[(reg, seed)], _ = load_model_from_checkpoint(
                    f'models/nn_24lb_72fh_v2/{reg}/seed_{seed}/'
                    f'best_phase1.pt', model_config, len(sids), train_cfg,
                    device)

    run_cache = {}

    def era_forecast(reg, cyc):
        """ERA5-driven det forecast for one cycle (all region stations,
        all seeds). Gives (seed, 72, S) residual array, cached."""
        keyc = (reg, cyc)
        if keyc in run_cache:
            return run_cache[keyc]
        sids, lats, lons, _ = meta[reg]
        raw, _, _, _ = sample_batch([pd.Timestamp(cyc)], model_config,
                                    sids, lats, lons, caches[reg],
                                    scale=False)
        lr_w, st_t, _, _ = sample_batch([pd.Timestamp(cyc)], model_config,
                                        sids, lats, lons, caches[reg],
                                        scale=True,
                                        lr_wind_override=raw.copy())
        inp = {'low_res': torch.from_numpy(lr_w.astype(np.float32)),
               'stations': torch.from_numpy(st_t.astype(np.float32))}
        outs = []
        for seed in SEEDS:
            with torch.no_grad():
                o = models[(reg, seed)](inp).numpy()[..., 0]
            outs.append(normalize_data(o, 'descale', SCALER)[0])
        run_cache[keyc] = np.stack(outs)          # (seeds, 72, S)
        return run_cache[keyc]

    with pd.HDFStore(H5, mode='r') as store:
        obs_all = {}

        def obs_series(sid):
            if sid not in obs_all:
                df = store.get(f'/station_{sid}')
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                obs_all[sid] = df['residual']
            return obs_all[sid]

        for storm in VAL_STORMS:
            sel = pick_stations_h5(storm, st)
            if sel.empty:
                print(f"[{storm['name']}] nothing above threshold")
                continue
            for reg in sel['region'].unique():
                region_setup(reg)
            track = load_track_full(storm['name'], storm['season'])
            n = len(sel)
            print(f"=== {storm['name']} ({storm['season']}) - {n} gauges, "
                  f"peak {sel['peak_res_m'].max():.2f} m ===")

            fig = plt.figure(figsize=(13, 3.8 + 1.55 * n))
            gs = fig.add_gridspec(n + 1, len(LEADS),
                                  height_ratios=[2.6] + [1] * n,
                                  hspace=0.55, wspace=0.10,
                                  left=0.05, right=0.995, top=0.97,
                                  bottom=0.02)
            ax_map = fig.add_subplot(gs[0, :],
                                     projection=ccrs.PlateCarree())
            draw_map(ax_map, track, sel, storm)

            for i, (_, row) in enumerate(sel.iterrows()):
                reg = row['region']
                sids, lats, lons, sid_pos = meta[reg]
                s = sid_pos.get(row['sid'])
                if s is None:
                    continue
                tmax = (sup.loc[row['sid'], 'train_max']
                        if row['sid'] in sup.index else np.nan)
                n_les = les(row['sid'], row['peak_res_m'])
                obs_r = obs_series(row['sid'])
                for jcol, lead in enumerate(LEADS):
                    cyc = (row['peak_time']
                           - pd.Timedelta(hours=lead)).floor('6h')
                    ax = fig.add_subplot(gs[i + 1, jcol])
                    pred = era_forecast(reg, cyc)         # (seeds, 72, S)
                    tt = pd.DatetimeIndex(
                        [cyc + pd.Timedelta(hours=h)
                         for h in range(1, 73)])
                    for si in range(len(SEEDS)):
                        ax.plot(tt, pred[si, :, s], color=C_ERA, lw=0.5,
                                alpha=0.4, zorder=3)
                    ax.plot(tt, pred[:, :, s].mean(axis=0), color=C_ERA,
                            lw=1.2, zorder=4)
                    ax.plot(tt, obs_r.reindex(tt).values, color=C_OBS,
                            lw=1.2, zorder=5)
                    ax.axvline(row['peak_time'], color='#999999', lw=0.6,
                               ls=':', zorder=1)
                    if np.isfinite(tmax):
                        ax.axhline(tmax, color='#c22222', lw=0.8,
                                   ls=(0, (4, 2)), zorder=1)
                    if jcol == 0 and n_les is not None:
                        ax.text(0.03, 0.94, f'{n_les} lessons ≥ obs peak',
                                transform=ax.transAxes, fontsize=5.0,
                                va='top',
                                color='#c22222' if n_les < 3
                                else '#333333')
                    real_lead = ((row['peak_time'] - cyc)
                                 .total_seconds() / 3600)
                    ax.set_title(f'−{real_lead:.0f} h', fontsize=5.5,
                                 pad=1.5)
                    lo = np.nanmin([np.nanmin(obs_r.reindex(tt).values),
                                    np.nanmin(pred[:, :, s])])
                    hi = np.nanmax([np.nanmax(obs_r.reindex(tt).values),
                                    np.nanmax(pred[:, :, s]),
                                    tmax if np.isfinite(tmax)
                                    else np.nan])
                    pad = 0.10 * (hi - lo + 1e-6)
                    ax.set_ylim(lo - pad, hi + 2.5 * pad)
                    ax.set_xlim(tt[0], tt[-1])
                    ax.xaxis.set_major_locator(
                        mdates.AutoDateLocator(minticks=2, maxticks=2))
                    ax.xaxis.set_major_formatter(
                        mdates.DateFormatter('%m/%d'))
                    ax.tick_params(labelsize=4.2, pad=1)
                    ax.grid(color='#f2f2f2', lw=0.4)
                    ax.set_axisbelow(True)
                    if jcol == 0:
                        ax.set_ylabel(f"{row['sname'].split(',')[0]}\n"
                                      f"({row['sid']})  surge (m)",
                                      fontsize=6.5)
                    else:
                        plt.setp(ax.get_yticklabels(), visible=False)

            from matplotlib.lines import Line2D
            fig.legend(handles=[
                Line2D([], [], color=C_OBS, lw=1.2,
                       label='observed surge'),
                Line2D([], [], color=C_ERA, lw=1.2,
                       label='AI w/ ERA5 (true) winds — 3-seed mean'),
                Line2D([], [], color=C_ERA, lw=0.5, alpha=0.4,
                       label='individual seeds'),
                Line2D([], [], color='#c22222', lw=0.8, ls=(0, (4, 2)),
                       label='largest training surge (1980–2021)')],
                loc='upper right', fontsize=6.5, frameon=False,
                bbox_to_anchor=(0.995, 0.998))
            fig.text(0.055, 0.985,
                     'VALIDATION-ERA storm (2022–23: model selection '
                     'only, never trained on) · no STOFS/GEFS archives '
                     'exist for these years', fontsize=7, color='#555555')

            out = os.path.join(OUT_DIR,
                               f"fan_{storm['name'].lower()}.png")
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f'wrote {out}')

    for c in caches.values():
        c.close()
