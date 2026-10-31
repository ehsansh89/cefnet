"""
paper_fig4_panel_a.py — draft of Figure 4: the warning dial + its receipts,
stacked one-column like Figure 2.

In plain English: on Figure 2's performance diagram (up = detects more flood
windows, right = fewer false alarms, gray curves = equal combined score), we
draw the AI ensemble's full trade-off curve per severity by sweeping the
warning cutoff over all 93 member fractions. The deterministic AI and STOFS
dots from Figure 2 are re-drawn
on the same axes: a deterministic model is one point, the ensemble is the
whole curve. With DRESSED = True, the dressed-STOFS curves (STOFS plus its
own post-seam error history — the best probabilistic system buildable from
the physics model, as introduced in Figure 3) are added as a thin background
family. Draft for the Figure 4 panel round — protocol matches the main
Fig 2 (post-seam, 112 gauges, common-valid windows).

Panel (b) is the receipts table of the frozen dial (from
paper_fig4_receipts_process.py): a cutoff chosen on post-seam 2024 to match
STOFS's false-alert workload, frozen, then scored on 2025. The MAIN figure
counts per 72-h window (same currency as panel a); the SI companion
(RECEIPTS='episodes') counts distinct flood episodes. Either way: caught,
missed, unique catches (caught by only that system), and realized 2025
false-alert days per gauge-year; bold marks the better catch count where the
paired 95% interval excludes zero.

Output: docs/paper/figures/v4/drafts/fig4_draft.png              (main: dial + windows)
        docs/paper/figures/v4/SI/fig4_si_receipts_episodes.png (SI: episodes)
"""

import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from o_style import setup, savefig, PAPER_FONT as F
from paper_common import eligible_station_ids


SEAM = pd.Timestamp('2024-05-14T12:00')
THRESHOLDS = ['minor', 'moderate', 'major']
UQ_BASE = r'outputs/uq_postprocessing/nn_24lb_72fh_v2/gefs_psfc_v2_raw'
SPAN = '20240101_20251201'
OUT_PNG = 'docs/paper/figures/v4/drafts/fig4_draft.png'
DSTOFS_CACHE = 'outputs/paper/v2/fig4_dstofs_probs.parquet'
DRESSED = True             # add the dressed-STOFS background curves
RECEIPTS = 'windows'       # 'windows' = main figure (a+b); 'episodes' = SI table
MM = 1 / 25.4
COL = {'ai': '#4477AA', 'st': '#AA3377', 'cut': '#EE7733'}
LS = {'minor': '-', 'moderate': '--', 'major': ':'}
MARK = {'minor': 'o', 'moderate': 's', 'major': '^'}


def build_dstofs_probs():
    """Gets nothing; gives (and caches) dressed-STOFS window-peak
    probabilities per cycle x gauge x threshold, with the error quantiles
    fitted on post-seam 2024 only (the fair, current-product baseline)."""
    if os.path.exists(DSTOFS_CACHE):
        return pd.read_parquet(DSTOFS_CACHE)
    from paper_fig3_process import (fit_error_quantiles, dressed_members,
                                    _lead_bin_index)
    rows = []
    for rd in sorted(glob.glob(os.path.join(UQ_BASE, SPAN, 'region_*'))):
        ds = xr.open_dataset(os.path.join(rd, 'verify_arrays.nc'))
        cycles = pd.to_datetime(ds.cycle_time.values)
        sids = [str(s) for s in ds.station.values]
        obs_wl = ds.obs_water_level.values.astype(np.float32)
        stofs_wl = ds.stofs_water_level.values.astype(np.float32)
        thr = {t: ds[f'{t}_threshold'].values.astype(np.float32)
               for t in THRESHOLDS}
        n_l = ds.sizes['lead_hour']
        ds.close()
        fit_mask = np.asarray((cycles.year == 2024) & (cycles >= SEAM))
        lbi = _lead_bin_index(n_l)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN')
            eq = fit_error_quantiles(stofs_wl, obs_wl, fit_mask, lbi)
            dm = dressed_members(stofs_wl, eq, lbi)
            dmax = np.nanmax(dm, axis=2)
        valid = np.all(np.isfinite(dmax), axis=1)
        for t in THRESHOLDS:
            okt = np.isfinite(thr[t])
            p = np.mean(dmax >= thr[t][None, None, :], axis=1)
            p = np.where(valid, p, np.nan).astype(np.float32)
            for s in range(len(sids)):
                if okt[s]:
                    rows.append(pd.DataFrame(dict(
                        cycle_time=cycles, station_id=sids[s],
                        threshold=t, dstofs_prob=p[:, s])))
        print('dressed ->', os.path.basename(rd))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(DSTOFS_CACHE, index=False)
    return out


def load_windows():
    """Gets nothing; gives the common-valid post-seam window frame (112
    gauges) with raw ensemble probability + det/STOFS/dressed flags."""
    with open('paper_fig1_config_v2.json') as f:
        cfg = json.load(f)
    elig = set(eligible_station_ids(cfg))
    det = pd.read_parquet('outputs/paper/v2/fig2_warn_detgfs.parquet')
    common = set.intersection(*[
        set(det.loc[det['threshold'] == t, 'station_id'])
        for t in THRESHOLDS])
    keep = elig & common
    evs = []
    for rd in sorted(glob.glob(os.path.join(UQ_BASE, SPAN, 'region_*'))):
        ev = pd.read_parquet(
            os.path.join(rd, 'decision_events.parquet'),
            columns=['cycle_time', 'station_id', 'threshold', 'obs_event',
                     'obs_valid', 'ai_prob_raw', 'stofs_event',
                     'stofs_valid'])
        ev['station_id'] = ev['station_id'].astype(str)
        evs.append(ev[ev['station_id'].isin(keep)])
    ev = pd.concat(evs, ignore_index=True)
    ev = ev.merge(det, on=['cycle_time', 'station_id', 'threshold'],
                  how='left')
    if DRESSED:
        ev = ev.merge(build_dstofs_probs(),
                      on=['cycle_time', 'station_id', 'threshold'],
                      how='left')
    ev = ev[(ev['cycle_time'] >= SEAM) & ev['obs_valid']
            & ev['stofs_valid'] & ev['det_gfs_valid'].fillna(False)
            & ev['ai_prob_raw'].notna()]
    return ev


def point(warn, obs):
    """Gets warn/observed flags; gives (success ratio, POD), or None if the
    rule never warns."""
    if warn.sum() == 0:
        return None
    h = (warn & obs).sum()
    f = (warn & ~obs).sum()
    return 1 - f / (h + f), h / obs.sum()


def draw_panel(ax, ev):
    """Gets the window frame; draws the dial panel: CSI contours, one sweep
    curve per severity (AI ensemble, all raw cutoffs), the dressed-STOFS
    background curves, and the deterministic AI/STOFS dots. The frozen
    2024-selected policy whose receipts are in panel b is identified in the
    caption, not marked, to keep the plane uncluttered."""
    sr = np.linspace(0.02, 1, 400)
    SR, PD = np.meshgrid(sr, sr)
    with np.errstate(divide='ignore'):
        CSI = 1.0 / (1.0 / SR + 1.0 / PD - 1.0)
    CSI[(1.0 / SR + 1.0 / PD - 1.0) <= 0] = np.nan
    levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ax.contour(SR, PD, CSI, levels=levels, colors='#e6e6e6',
               linewidths=0.4)
    for lv in levels:
        y = 1.0 / (1.0 / lv - 1.0 / 0.985 + 1.0)
        if 0.17 < y < 0.99:
            ax.text(0.985, y, f'{lv:.1f}', fontsize=F['small'],
                    color='#bbbbbb', ha='right', va='bottom')
    ax.text(0.985, 0.995, 'CSI', fontsize=F['small'], color='#bbbbbb',
            ha='right', va='top')

    sev_off = {'minor': (-7, -5, 'right', 'top'),
               'moderate': (-5, 0, 'right', 'center'),
               'major': (3, 8, 'left', 'bottom')}
    cuts = (np.arange(1, 94) - 0.5) / 93.0
    for thr in THRESHOLDS:
        g = ev[ev['threshold'] == thr]
        obs = g['obs_event'].to_numpy(bool)
        p_raw = g['ai_prob_raw'].to_numpy(float)
        xy = np.array([point(p_raw >= c, obs) for c in cuts])
        ax.plot(xy[:, 0], xy[:, 1], LS[thr], color=COL['ai'], lw=1.4,
                zorder=3)
        if DRESSED:
            m = g['dstofs_prob'].notna().to_numpy()
            xy_d = np.array([point(g['dstofs_prob'].to_numpy(float)[m] >= c,
                                   obs[m])
                             for c in (np.arange(1, 20) - 0.5) / 19.0])
            ax.plot(xy_d[:, 0], xy_d[:, 1], LS[thr], color=COL['st'],
                    lw=0.9, alpha=0.75, zorder=2)
        for col, colr in [('det_gfs_event', COL['ai']),
                          ('stofs_event', COL['st'])]:
            w = g[col].to_numpy(bool)
            ax.plot(*point(w, obs), MARK[thr], ms=5.6, mfc=colr,
                    mec='#1a1a1a', mew=0.6, zorder=7)
        dx, dy, ha, va = sev_off[thr]
        ax.annotate(thr, (xy[2, 0], xy[2, 1]), textcoords='offset points',
                    xytext=(dx, dy), fontsize=F['label_sm'],
                    color='#1a1a1a', ha=ha, va=va)

    handles = [
        Line2D([], [], color=COL['ai'], lw=1.4, label='AI ensemble'),
        Line2D([], [], marker='o', ls='none', ms=5.2, mfc=COL['ai'],
               mec='#1a1a1a', mew=0.5, label='AI deterministic'),
        Line2D([], [], marker='o', ls='none', ms=5.2, mfc=COL['st'],
               mec='#1a1a1a', mew=0.5, label='STOFS-2D'),
    ]
    if DRESSED:
        handles.insert(1, Line2D([], [], color=COL['st'], lw=0.9,
                                 alpha=0.75, label='dressed STOFS'))
    ax.legend(handles=handles, loc='lower left', fontsize=F['legend'],
              handletextpad=0.5, borderaxespad=0.4, labelspacing=0.35)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_xlabel('success ratio (1 − FAR)', fontsize=F['label'])
    ax.set_ylabel('probability of detection (POD)', fontsize=F['label'])
    ax.tick_params(labelsize=F['tick'])
    ax.grid(False)


def receipt_rows(r):
    """Gets one receipts CSV row (window- or episode-currency); gives the
    common display block: four forecast systems, each a tuple
    (name, caught, delta, lo, hi, sig, fa-days). delta is caught minus
    STOFS-2D (None for STOFS itself, the reference). Order: the two AI
    products, then the two physics products, STOFS last as reference."""
    tot = int(r['flood_windows'] if 'flood_windows' in r else r['episodes'])
    ai_c, st_c = int(r['ai_caught']), int(r['st_caught'])
    det_c, ds_c = int(r['det_caught']), int(r['ds_caught'])
    return tot, [
        ('AI ensemble', ai_c, int(r['net']), int(r['net_lo']),
         int(r['net_hi']), bool(r['excludes_zero']), r['fa_days_ai']),
        ('AI deterministic', det_c, int(r['det_net']), int(r['det_net_lo']),
         int(r['det_net_hi']), bool(r['det_excludes_zero']),
         r['fa_days_det']),
        ('dressed STOFS', ds_c, int(r['ds_net']), int(r['ds_net_lo']),
         int(r['ds_net_hi']), bool(r['ds_excludes_zero']), r['fa_days_ds']),
        ('STOFS-2D', st_c, None, None, None, False, r['fa_days_st'])]


def draw_receipts(ax, rc, caught_label='windows\ncaught'):
    """Gets the receipts frame (window- or episode-currency); draws the
    frozen-dial table in Figure 2's style — four rows per severity
    (AI ensemble / AI deterministic / dressed STOFS / STOFS-2D): flood
    events caught, the paired difference in catches versus STOFS-2D with its
    95% interval, and the realized 2025 false-alert days. Bold marks a catch
    count whose paired interval versus STOFS excludes zero. The per-severity
    total and which rows are workload-matched are caption obligations."""
    ax.axis('off')
    cols = ['severity', 'forecast system', caught_label,
            'Δ vs STOFS-2D\n[95% CI]', 'false-alert\ndays/st-yr']
    x = [0.00, 0.155, 0.525, 0.85, 1.00]
    group_y = [0.80, 0.53, 0.26]
    dy_in = 0.066
    ax.axhline(0.99, color='#1a1a1a', lw=0.9)
    ax.axhline(0.885, color='#1a1a1a', lw=0.5)
    ax.axhline(group_y[-1] - 3 * dy_in - 0.035, color='#1a1a1a', lw=0.9)
    for j, c in enumerate(cols):
        ax.text(x[j], 0.965, c, fontsize=F['label_sm'], fontweight='bold',
                ha='left' if j < 2 else 'right', va='top', linespacing=1.0)
    for i, thr in enumerate(THRESHOLDS):
        r = rc[rc['threshold'] == thr].iloc[0]
        _, rows = receipt_rows(r)
        st_caught = rows[-1][1]           # STOFS-2D is the last row
        for k, (name, caught, net, lo, hi, sig, fa) in enumerate(rows):
            yy = group_y[i] - k * dy_in
            bold = sig and caught > st_caught
            delta = '—' if net is None else f'{net:+,} [{lo:+,}, {hi:+,}]'
            vals = [thr if k == 0 else '', name, f'{caught:,}', delta,
                    f'{fa:.2f}']
            for j, v in enumerate(vals):
                fw = 'bold' if (j == 2 and bold) else 'normal'
                ax.text(x[j], yy, v, fontsize=F['label_sm'], fontweight=fw,
                        ha='left' if j < 2 else 'right', va='top',
                        color='#1a1a1a')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    setup()
    if RECEIPTS == 'windows':
        # main figure: dial (a) + per-window receipts (b), stacked
        ev = load_windows()
        rc = pd.read_csv('outputs/paper/v2/fig4_receipts_windows.csv')
        fig = plt.figure(figsize=(88 * MM, 166 * MM))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.56, 0.44],
                              left=0.125, right=0.972, top=0.965,
                              bottom=0.04, hspace=0.20)
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])
        draw_panel(ax_a, ev)
        draw_receipts(ax_b, rc, caught_label='windows\ncaught')
        ax_a.text(-0.145, 1.025, 'a', transform=ax_a.transAxes,
                  fontsize=F['panel'], fontweight='bold')
        ax_b.text(-0.145, 1.025, 'b', transform=ax_b.transAxes,
                  fontsize=F['panel'], fontweight='bold')
        out = OUT_PNG
    else:
        # SI companion: per-episode receipts table alone
        rc = pd.read_csv('outputs/paper/v2/fig4_receipts.csv')
        fig = plt.figure(figsize=(88 * MM, 74 * MM))
        ax_b = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        draw_receipts(ax_b, rc, caught_label='episodes\ncaught')
        out = 'docs/paper/figures/v4/SI/fig4_si_receipts_episodes.png'
    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, out, dpi=600)
    plt.close(fig)
    print('Wrote', out)
