"""
paper_fig6_ladder_draft.py — v4.1 Figure 4: "the sample-starvation law in
real hurricanes." (Script keeps its v3-era name until the renaming sweep;
it writes docs/paper/figures/v4/fig4.png.)

In plain English: for the five biggest-surge storms of 2022-2025 (Helene,
Idalia, Ian, Beryl, Debby), each row opens with a locator map — coastline,
state borders, the hurricane best track (red dashed), and the six tide
gauges (teal, numbered) — followed by those six gauges as time-series
panels ordered left-to-right by training experience (how many independent
training events reached that surge level). CEFNet (reanalysis/ERA5-forced,
blue) under-calls the peak where experience is missing (left panels) and
tracks the observations where it is plentiful (right panels); observed
surge is thin black, and each panel's dashed peach line marks that gauge's
largest 1980-2021 training surge. 2022-23 storms are validation-era (model
selection only, never trained on). The attention-capture helpers
(storm_map_pack / attn_trace / CBAMCapture) are retained for the deferred
mechanism strip but are not drawn in this figure.

Reads:  ERA5 zarrs + v2 water levels + checkpoints, IBTrACS full file,
        fig5_extrap_support{.csv,_levels.parquet}, Natural Earth coastline
Gives:  docs/paper/figures/v4/fig4.png
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree

from o_style import setup, savefig, PAPER_FONT as F
from s_sample_batch_v2 import (DatasetCache, get_region_stations,
                               sample_batch, get_grid_coords,
                               get_station_indices)
from s_scalers import normalize_data
from train import load_train_config
from inference_v2 import load_model_from_checkpoint
from paper_fig6_tier3 import CBAMCapture
import paper_fig5_val_surge_fans as vsf
from paper_fig5_val_surge_fans import (load_track_full, lessons_fn,
                                       pick_stations_h5)

STATIONS_CSV = r'data/stations_processed_v2/stations_processed.csv'
H5 = r'data/water_levels_processed_v2/processed_water_levels.h5'
SCALER = r'samples/scalers_v2/water_level_residuals.pkl'
SUP = r'outputs/paper/v2/fig5_extrap_support.csv'
OUT_PNG = r'docs/paper/figures/v4/fig4.png'
MM = 1 / 25.4

STORMS5 = [
    dict(name='Helene', season=2024, t0='2024-09-24', t1='2024-09-30'),
    dict(name='Idalia', season=2023, t0='2023-08-28', t1='2023-09-02'),
    dict(name='Ian', season=2022, t0='2022-09-26', t1='2022-10-04'),
    dict(name='Beryl', season=2024, t0='2024-07-05', t1='2024-07-12'),
    dict(name='Debby', season=2024, t0='2024-08-02', t1='2024-08-10'),
]
LEAD = 36
SEEDS = (1, 2, 3)
STRATA = [(0, 1), (1, 3), (3, 10), (10, 30), (30, 100), (100, np.inf)]
C_OBS, C_ERA = '#1a1a1a', '#208eb7'    # observed = black ink; CEFNet blue
C_BND = '#fcb790'                      # largest-training-surge dash (peach)
C_GAUGE = '#4fbbbc'                    # gauge markers on the locator maps
C_TRK = '#fc2c44'                      # hurricane best track (red)
# manual label placement on crowded maps: (storm, label#) -> (dlon, dlat, ha, va)
_UL = (-0.5, 0.35, 'right', 'bottom')   # upper-left of the point
_LL = (-0.5, -0.35, 'right', 'top')     # lower-left
_L = (-0.45, 0.0, 'right', 'center')    # left
_R = (0.45, 0.0, 'left', 'center')      # right
LABEL_POS = {('Helene', 5): (-0.32, 0.22, 'right', 'bottom'),  # UL, closer
             ('Ian', 4): _UL,
             ('Ian', 5): (0.32, -0.22, 'left', 'top'),   # lower-right, close
             ('Beryl', 3): (-0.32, -0.26, 'right', 'top'),   # lower-left
             ('Beryl', 4): (0.0, -0.30, 'center', 'top'),    # lower-center
             ('Debby', 4): _LL, ('Debby', 6): _R, ('Debby', 3): _L}


# curation: tidal-river gauges whose oscillatory response makes the
# panel unreadable are excluded and replaced by the backfill rule
EXCLUDE = {('Debby', '8548989')}          # Newbold, Delaware River


def pick_ladder(storm, st, les, n_want=6):
    """Gauges for one storm: one per lesson stratum (peak >= 0.55 m,
    highest peak within stratum), backfilled by peak; sorted by lessons."""
    vsf.MAX_STATIONS = 40
    cand = pick_stations_h5({**storm}, st)
    cand = cand[cand['peak_res_m'] >= 0.55].copy()
    cand = cand[~cand['sid'].isin(
        {s for (nm, s) in EXCLUDE if nm == storm['name']})]
    if cand.empty:
        return []
    cand['lessons'] = [les(r['sid'], r['peak_res_m'])
                       for _, r in cand.iterrows()]
    cand = cand.dropna(subset=['lessons'])
    picks = []
    for lo, hi in STRATA:
        g = cand[(cand['lessons'] >= lo) & (cand['lessons'] < hi)
                 & (~cand['sid'].isin([p['sid'] for p in picks]))]
        if not g.empty:
            picks.append(g.sort_values('peak_res_m',
                                       ascending=False).iloc[0].to_dict())
    rest = cand[~cand['sid'].isin([p['sid'] for p in picks])]
    for _, r in rest.sort_values('peak_res_m', ascending=False).iterrows():
        if len(picks) >= n_want:
            break
        picks.append(r.to_dict())
    return sorted(picks, key=lambda p: p['lessons'])[:n_want]


def coast_grid(glat, glon, tree):
    """Natural-Earth coastline mapped into fractional grid coordinates
    (NaN-broken polyline; long jumps split)."""
    from scipy.interpolate import griddata
    from cartopy.io import shapereader
    pts = np.column_stack([glat.ravel(), glon.ravel()])
    Cc = np.tile(np.arange(64), (64, 1)).ravel().astype(float)
    Rr = np.repeat(np.arange(64), 64).astype(float)
    lat0, lat1 = glat.min() - 0.5, glat.max() + 0.5
    lon0, lon1 = glon.min() - 0.5, glon.max() + 0.5
    xs, ys = [], []
    shp = shapereader.natural_earth('50m', 'physical', 'coastline')
    for geom in shapereader.Reader(shp).geometries():
        for ln in getattr(geom, 'geoms', [geom]):
            lo, la = np.asarray(ln.xy[0]), np.asarray(ln.xy[1])
            m = (la > lat0) & (la < lat1) & (lo > lon0) & (lo < lon1)
            if not m.any():
                continue
            idx = np.where(m)[0]
            q = np.column_stack([la[m], lo[m]])
            c = griddata(pts, Cc, q, method='linear')
            r = griddata(pts, Rr, q, method='linear')
            ok = np.isfinite(c) & np.isfinite(r)
            if ok.sum() < 2:
                continue
            c, r, idx = c[ok], r[ok], idx[ok]
            for i in range(len(c)):
                if i > 0 and (idx[i] - idx[i - 1] > 1
                              or abs(c[i] - c[i - 1]) > 5
                              or abs(r[i] - r[i - 1]) > 5):
                    xs.append(np.nan)
                    ys.append(np.nan)
                xs.append(c[i])
                ys.append(r[i])
            xs.append(np.nan)
            ys.append(np.nan)
    return np.array(xs), np.array(ys)


def attn_trace(a, n_frames):
    """Attention-eye trace: time-mean removed, top-8% cells, largest
    connected blob's weighted centroid, light smoothing. Gives (x, y)."""
    sc = 64 / a.shape[-1]
    seq = a[:n_frames]
    anom = seq - seq.mean(axis=0, keepdims=True)
    tx, ty = [], []
    for t in range(0, len(anom), 2):
        m = anom[t]
        hot = m >= np.percentile(m, 92)
        lab, nlab = ndimage.label(hot)
        if nlab == 0:
            continue
        sums = ndimage.sum_labels(np.maximum(m, 0), lab,
                                  range(1, nlab + 1))
        yy, xx = np.where(lab == (1 + int(np.argmax(sums))))
        w = np.maximum(m[yy, xx], 1e-6)
        tx.append(np.average(xx, weights=w) * sc + sc / 2 - 0.5)
        ty.append(np.average(yy, weights=w) * sc + sc / 2 - 0.5)
    tx = pd.Series(tx).rolling(3, center=True, min_periods=1).mean()
    ty = pd.Series(ty).rolling(3, center=True, min_periods=1).mean()
    return tx.values, ty.values


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    setup()
    device = torch.device('cpu')
    train_cfg = load_train_config('train_config_v2.json')
    with open('nn_config.json') as f:
        model_config = json.load(f)

    st = pd.read_csv(STATIONS_CSV, dtype={'station_id': str})
    sup = pd.read_csv(SUP, dtype={'sid': str}).set_index('sid')
    les = lessons_fn()

    ladders = {s['name']: pick_ladder(s, st, les) for s in STORMS5}
    for nm, picks in ladders.items():
        print(nm + ':', ', '.join(
            f"{p['sname'].split(',')[0][:14]}({p['lessons']:.0f})"
            for p in picks))

    models, caches, meta = {}, {}, {}

    def region_setup(reg):
        if reg in caches:
            return
        sids, snames, lats, lons = get_region_stations(STATIONS_CSV, reg)
        caches[reg] = DatasetCache(reg, H5)
        meta[reg] = (sids, lats, lons,
                     {str(s): i for i, s in enumerate(sids)})
        for seed in SEEDS:
            models[(reg, seed)], _ = load_model_from_checkpoint(
                f'models/nn_24lb_72fh_v2/{reg}/seed_{seed}/best_phase1.pt',
                model_config, len(sids), train_cfg, device)

    run_cache = {}

    def era_forecast(reg, cyc):
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
        run_cache[keyc] = np.stack(outs)
        return run_cache[keyc]

    def storm_map_pack(storm, picks):
        """One capture forward at the top gauge's ~41h cycle in the home
        region. Gives dict with attention fill, winds, trace, coast,
        track and in-domain gauge markers (all grid coords)."""
        top = picks[int(np.argmax([p['peak_res_m'] for p in picks]))]
        reg = top['region']
        region_setup(reg)
        sids, lats, lons, sid_pos = meta[reg]
        j = 40                                  # ~41 h cycle, peak at idx 40
        cyc = (top['peak_time'] - pd.Timedelta(hours=41)).floor('6h')
        jj = int((top['peak_time'] - cyc).total_seconds() // 3600) - 1
        raw, _, _, _ = sample_batch([cyc], model_config, sids, lats, lons,
                                    caches[reg], scale=False)
        cap = CBAMCapture(models[(reg, 1)], layer_names=('layer3',))
        lr_w, st_t, _, _ = sample_batch([cyc], model_config, sids, lats,
                                        lons, caches[reg], scale=True,
                                        lr_wind_override=raw.copy())
        with torch.no_grad():
            models[(reg, 1)]({
                'low_res': torch.from_numpy(lr_w.astype(np.float32)),
                'stations': torch.from_numpy(st_t.astype(np.float32))})
        cap.close()
        a = np.concatenate(cap.spatial[0], axis=0)[:, 0]
        fill = a[24 + jj - 12:24 + jj - 5].mean(axis=0)
        tx, ty = attn_trace(a, 24 + jj + 1)
        glat, glon = get_grid_coords(reg, (64, 64))
        if np.ndim(glat) == 1:
            glon, glat = np.meshgrid(glon, glat)
        tree = cKDTree(np.column_stack([glat.ravel(), glon.ravel()]))
        trk = load_track_full(storm['name'], storm['season'])
        d, i = tree.query(np.column_stack([trk['lat'], trk['lon']]))
        keep = d < 0.5
        rows, cols = np.divmod(i[keep], 64)
        cx, cy = coast_grid(glat, glon, tree)
        gpts = []
        for p in picks:
            if p['region'] == reg:
                gi = get_station_indices([p['lat']], [p['lon']],
                                         glat, glon)[0]
                gpts.append((gi['x'], gi['y'], p['lessons']))
        return dict(fill=fill, u=raw[0, 24 + jj - 6, 0],
                    v=raw[0, 24 + jj - 6, 1], tx=tx, ty=ty,
                    trk=(cols, rows), coast=(cx, cy), gpts=gpts)

    # ------- figure: 5 rows x (track map + 6 numbered panels) -------
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    n_row = len(STORMS5)
    fig = plt.figure(figsize=(180 * MM, 21 * MM * n_row + 8 * MM))
    gs = fig.add_gridspec(n_row, 8,
                          width_ratios=[1.12, 0.18] + [1] * 6,
                          left=0.0235, right=0.9765, top=0.94, bottom=0.03,
                          hspace=0.16, wspace=0.16)

    with pd.HDFStore(H5, mode='r') as store:
        for irow, storm in enumerate(STORMS5):
            picks = ladders[storm['name']]
            if not picks:
                continue
            for p in picks:
                region_setup(p['region'])
            trk_geo = load_track_full(storm['name'], storm['season'])

            # -- map 1: geographic track + stations --
            axg = fig.add_subplot(gs[irow, 0],
                                  projection=ccrs.PlateCarree())
            la = [p['lat'] for p in picks]
            lo_ = [p['lon'] for p in picks]
            axg.set_extent([min(lo_) - 2.5, min(max(lo_) + 2.5, -70),
                            max(min(la) - 2.5, 20),
                            min(max(la) + 2.5, 40)],
                           crs=ccrs.PlateCarree())
            axg.add_feature(cfeature.LAND.with_scale('50m'),
                            facecolor='#f2efe9')
            axg.add_feature(cfeature.STATES.with_scale('50m'),
                            edgecolor='#999999', lw=0.1)
            axg.add_feature(cfeature.COASTLINE.with_scale('50m'),
                            edgecolor='black', lw=0.2)
            axg.plot(trk_geo['lon'], trk_geo['lat'], color=C_TRK,
                     lw=0.8, ls=(0, (3, 2)),
                     transform=ccrs.PlateCarree())
            placed = []
            for k, p in enumerate(picks):
                axg.plot(p['lon'], p['lat'], marker='o', ms=3.4,
                         color=C_GAUGE, mec='black', mew=0.1,
                         transform=ccrs.PlateCarree(), zorder=5)
                key = (storm['name'], k + 1)
                if key in LABEL_POS:
                    dlon, dlat, ha_l, va_l = LABEL_POS[key]
                    tx, ty = p['lon'] + dlon, p['lat'] + dlat
                else:
                    # small nudge to keep the number near its gauge while
                    # avoiding a previous label
                    lx, ly = p['lon'] + 0.4, p['lat'] + 0.15
                    for off in ((0, 0), (0, -0.6), (-0.6, 0.3), (0.5, 0.5),
                                (-0.6, -0.6)):
                        tx, ty = lx + off[0], ly + off[1]
                        if all(abs(tx - qx) > 0.7 or abs(ty - qy) > 0.5
                               for qx, qy in placed):
                            break
                    ha_l, va_l = 'left', 'baseline'
                placed.append((tx, ty))
                axg.text(tx, ty, f'{k + 1}', fontsize=F['label_sm'],
                         fontweight='bold', ha=ha_l, va=va_l,
                         transform=ccrs.PlateCarree(), zorder=6,
                         clip_on=True)
            axg.set_aspect('auto')
            axg.set_title(f"{storm['name']} ({storm['season']})",
                          fontsize=F['label_sm'], pad=2)

            ymax = max((sup.loc[p['sid'], 'train_max']
                        for p in picks if p['sid'] in sup.index),
                       default=0.0)
            series = []
            for p in picks:
                reg = p['region']
                sids, lats, lons, sid_pos = meta[reg]
                s = sid_pos[p['sid']]
                cyc = (p['peak_time']
                       - pd.Timedelta(hours=LEAD)).floor('6h')
                pred = era_forecast(reg, cyc)[:, :, s]
                tt = pd.DatetimeIndex([cyc + pd.Timedelta(hours=h)
                                       for h in range(1, 73)])
                df = store.get(f"/station_{p['sid']}")
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                obs = df['residual'].reindex(tt).values
                series.append((p, tt, pred, obs))
                ymax = max(ymax, np.nanmax(obs), np.nanmax(pred))

            for k, (p, tt, pred, obs) in enumerate(series):
                ax = fig.add_subplot(gs[irow, k + 2])
                ax.plot(tt, pred.mean(axis=0), color=C_ERA, lw=1.1,
                        zorder=4)
                ax.plot(tt, obs, color=C_OBS, lw=0.5, zorder=5)
                tmax = (sup.loc[p['sid'], 'train_max']
                        if p['sid'] in sup.index else np.nan)
                if np.isfinite(tmax):
                    ax.axhline(tmax, color=C_BND, lw=0.7,
                               ls=(0, (4, 2)), zorder=1)
                nl = int(p['lessons'])
                col = '#1a1a1a'
                box = dict(facecolor='white', edgecolor='none',
                           alpha=0.85, pad=0.6)
                ax.text(0.05, 0.95, f'{k + 1}', transform=ax.transAxes,
                        fontsize=F['label'], fontweight='bold',
                        color=col, va='top', bbox=box, zorder=8)
                ax.text(0.95, 0.95,
                        f"{nl} event{'s' if nl != 1 else ''}",
                        transform=ax.transAxes, fontsize=F['label_sm'],
                        color=col, va='top', ha='right', bbox=box,
                        zorder=8)
                ax.set_ylim(-0.45, ymax + 0.35)
                ax.set_xlim(tt[0], tt[-1])
                ax.set_xticks([])
                ax.tick_params(labelsize=F['tick'] - 1, pad=1)
                if k == 0:
                    ax.set_ylabel('surge (m)', fontsize=F['label_sm'])
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.grid(False)
                ax.set_axisbelow(True)

    for c in caches.values():
        c.close()

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=C_OBS, lw=1.2, label='observed surge'),
        Line2D([], [], color=C_ERA, lw=1.2,
               label='CEFNet (reanalysis-forced)'),
        Line2D([], [], color=C_BND, lw=0.8, ls=(0, (4, 2)),
               label='largest training surge (1980–2021)'),
        Line2D([], [], color=C_TRK, lw=0.9, ls=(0, (3, 2)),
               label='best track')],
        loc='upper center', ncol=4, fontsize=F['small'], frameon=False,
        bbox_to_anchor=(0.5, 0.997), columnspacing=1.2)
    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, OUT_PNG, dpi=450)
    plt.close(fig)
    print('wrote', OUT_PNG)
