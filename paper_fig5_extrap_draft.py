"""
paper_fig5_extrap_draft.py — v4.1 Figure 3: "the extreme errors are sample
starvation, not a general failure." (Script keeps its v3-era name until the
renaming sweep; it writes docs/paper/figures/v4/fig3.png.)

In plain English: how well CEFNet does at a gauge depends on how often the
training years (1980-2021) actually showed that gauge water this high.
Panel (a): the peak-error difference (CEFNet minus STOFS-2D, event-locked
within +/-3 h) binned by the number of independent training events that
reached the observed surge level, with a log window-count strip beneath —
the gap is ~0 while experience is plentiful and blows up in the rare bins.
Panel (b): per flood severity, a dumbbell from the in-support median
(>=3 prior events) to the sample-starved median (0-2 prior events) for
each model, showing the starvation penalty is what carries the big errors.
Panel (c, added 2026-08-02): the payoff view — signed peak-error boxes for
the IN-SUPPORT windows only (>=3 prior events), per severity and model:
within its experience the learned model is healthy at every severity,
including major. No named hurricanes anywhere — storms are just the events
that occupy the sample-starved tail.

Reads:  outputs/paper/v2/fig5_extremes_windows.parquet (Fig 2 cache),
        data/stations_processed_v2/stations_processed.csv,
        data/water_levels_processed_v2/processed_water_levels.h5
Gives:  outputs/paper/v2/fig5_extrap_support.csv   (per-gauge training max)
        outputs/paper/v2/fig5_extrap_bins.csv      (panel-a bins + CIs)
        outputs/paper/v2/fig5_extrap_burden.csv    (panel-b numbers)
        docs/paper/figures/v4/fig3.png             (3-panel, approved 2026-08-02)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from o_style import setup, savefig, PAPER_FONT as F

WINDOWS = r'outputs/paper/v2/fig5_extremes_windows.parquet'
STATIONS = r'data/stations_processed_v2/stations_processed.csv'
H5 = r'data/water_levels_processed_v2/processed_water_levels.h5'
OUT_SUP = r'outputs/paper/v2/fig5_extrap_support.csv'
OUT_BIN = r'outputs/paper/v2/fig5_extrap_bins.csv'
OUT_BUR = r'outputs/paper/v2/fig5_extrap_burden.csv'
OUT_PNG = r'docs/paper/figures/v4/fig3.png'
MM = 1 / 25.4

SEAM = pd.Timestamp('2024-05-14 12:00')
TRAIN = ('1980-01-01', '2021-12-31')
EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]      # + open >= 1.0
MIN_N, MIN_G, MIN_B = 20, 3, 3                    # Fig 2 bin-support rule
N_BOOT = 1000
C_AI, C_ST = '#208eb7', '#c46fd8'      # paper palette 2026-07-24
C_INK, C_BND = '#1a1a1a', '#666666'    # ink line; boundary dash (gray)


def gauge_support():
    """Per-gauge training-era support, 1980-2021: max, 99.99th percentile,
    AND the full level -> count curves (hours at/above each level, and
    independent events at/above each level — exceedance clusters separated
    by > 72 h count as one event, so one hurricane = one lesson).
    Cached. Gives (summary DataFrame, level-curve DataFrame)."""
    lvl_path = OUT_SUP.replace('.csv', '_levels.parquet')
    if os.path.exists(OUT_SUP) and os.path.exists(lvl_path):
        return (pd.read_csv(OUT_SUP, dtype={'sid': str}),
                pd.read_parquet(lvl_path))
    st = pd.read_csv(STATIONS, dtype={'station_id': str})
    rows, lvl_rows = [], []
    with pd.HDFStore(H5, mode='r') as store:
        for sid in st['station_id']:
            key = f'/station_{sid}'
            if key not in store:
                continue
            df = store.get(key)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            res = df['residual'].loc[TRAIN[0]:TRAIN[1]].dropna()
            if len(res) < 24 * 365 * 5:      # need a real training record
                continue
            tmax = float(res.max())
            rows.append(dict(sid=sid, train_max=tmax,
                             q9999=float(res.quantile(0.9999)),
                             train_hours=len(res)))
            t = res.index.values
            v = res.values
            for lev in np.arange(0.1, tmax + 0.06, 0.05):
                m = v >= lev
                n_h = int(m.sum())
                if n_h == 0:
                    n_ev = 0
                else:
                    tt = t[m]
                    gaps = np.diff(tt).astype('timedelta64[h]').astype(int)
                    n_ev = int(1 + (gaps > 72).sum())
                lvl_rows.append(dict(sid=sid, level=round(float(lev), 2),
                                     hours=n_h, events=n_ev))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUP, index=False)
    lv = pd.DataFrame(lvl_rows)
    lv.to_parquet(lvl_path, index=False)
    return out, lv


def events_at_level(lv):
    """Builds a fast lookup: (sid, observed level) -> number of independent
    training events at/above that level (0 if beyond the curve)."""
    tab = {sid: (g['level'].values, g['events'].values)
           for sid, g in lv.groupby('sid')}

    def fn(sid, obs):
        if sid not in tab:
            return np.nan
        levels, events = tab[sid]
        idx = np.searchsorted(levels, obs, side='right') - 1
        if idx < 0:
            return float(events[0])          # below the lowest tracked level
        if obs > levels[-1] + 0.05:
            return 0.0                       # beyond anything in training
        return float(events[min(idx, len(events) - 1)])
    return fn


def block_boot_median(vals, blocks, n=N_BOOT):
    """Median with a 10-day-block bootstrap CI. Gets value+block arrays;
    gives (median, lo, hi)."""
    med = float(np.median(vals))
    uniq = np.unique(blocks)
    rng = np.random.default_rng(11)
    bs = []
    idx = {b: np.where(blocks == b)[0] for b in uniq}
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = np.concatenate([vals[idx[b]] for b in pick])
        bs.append(np.median(v))
    return med, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    setup()

    sup, lv = gauge_support()
    ev_fn = events_at_level(lv)
    st = pd.read_csv(STATIONS, dtype={'station_id': str})
    three = st[np.isfinite(st['f_min_ms_m']) & np.isfinite(st['f_mod_ms_m'])
               & np.isfinite(st['f_maj_ms_m'])]['station_id']

    w = pd.read_parquet(WINDOWS)
    w = w[(w['cycle_time'] >= SEAM)
          & w['station_id'].isin(set(three))
          & np.isfinite(w['det_res_np3']) & np.isfinite(w['stofs_res_np3'])
          & np.isfinite(w['obs_res_max'])]
    w = w.merge(sup.rename(columns={'sid': 'station_id'}), on='station_id')
    w['ratio'] = w['obs_res_max'] / w['train_max']
    w['ratio2'] = w['obs_res_max'] / w['q9999']
    w['n_ev'] = [ev_fn(s, o) for s, o in
                 zip(w['station_id'], w['obs_res_max'])]
    w['ai_err'] = (w['det_res_np3'] - w['obs_res_max']).abs() * 100
    w['st_err'] = (w['stofs_res_np3'] - w['obs_res_max']).abs() * 100
    w['ai_serr'] = (w['det_res_np3'] - w['obs_res_max']) * 100   # signed, panel c
    w['st_serr'] = (w['stofs_res_np3'] - w['obs_res_max']) * 100
    w['d'] = w['ai_err'] - w['st_err']
    w['block'] = ((w['cycle_time'] - SEAM).dt.days // 10).astype(int)
    w['band'] = np.where(w['fh_cnt_major'] > 0, 'major',
                np.where(w['fh_cnt_moderate'] > 0, 'moderate',
                np.where(w['fh_cnt_minor'] > 0, 'minor', 'none')))
    print(f'windows: {len(w):,}   gauges: {w.station_id.nunique()}   '
          f'beyond support: {(w.ratio >= 1).sum():,} at '
          f'{w[w.ratio >= 1].station_id.nunique()} gauges')

    # ------- panel (a): bins by TRAINING EXEMPLAR COUNT, merge to rare end
    EV_BINS = [(100, np.inf, '≥100'), (30, 100, '30–99'), (10, 30, '10–29'),
               (3, 10, '3–9'), (1, 3, '1–2'), (0, 1, 'never')]
    bins = []
    for lo_e, hi_e, lab in EV_BINS:
        g = w[(w['n_ev'] >= lo_e) & (w['n_ev'] < hi_e)]
        bins.append(dict(lab=lab, n=len(g),
                         n_gauges=g['station_id'].nunique(),
                         n_blocks=g['block'].nunique(), rows=g))
    merged, hold = [], None
    for b in bins[::-1]:          # merge from the rare end upward
        if hold is not None:
            b = dict(lab=f"{b['lab']} + {hold['lab']}",
                     rows=pd.concat([b['rows'], hold['rows']]))
            b['n'] = len(b['rows'])
            b['n_gauges'] = b['rows']['station_id'].nunique()
            b['n_blocks'] = b['rows']['block'].nunique()
            hold = None
        if (b['n'] >= MIN_N and b['n_gauges'] >= MIN_G
                and b['n_blocks'] >= MIN_B):
            merged.append(b)
        else:
            hold = b
    merged = merged[::-1]

    bin_rows = []
    for xi, b in enumerate(merged):
        g = b['rows']
        med, lo, hi = block_boot_median(g['d'].values, g['block'].values)
        bin_rows.append(dict(
            lab=b['lab'], x=xi, n=b['n'], n_gauges=b['n_gauges'],
            n_blocks=b['n_blocks'], d_med=med, d_lo=lo, d_hi=hi,
            ai_med=float(g['ai_err'].median()),
            st_med=float(g['st_err'].median())))
    bt = pd.DataFrame(bin_rows)
    bt.to_csv(OUT_BIN, index=False)
    print(bt.round(2).to_string(index=False)
          .replace('\u2265', '>=').replace('\u2013', '-'))   # ASCII-safe console copy

    # ------- panel (b) severity split, SAME currency as panel (a):
    # "starved" = the observed level had fewer than 3 independent
    # training events at that gauge (where panel a shows the blow-up)
    bur_rows = []
    for band in ('minor', 'moderate', 'major'):
        g = w[w['band'] == band]
        out = g[g['n_ev'] < 3]
        ins = g[g['n_ev'] >= 3]
        bur_rows.append(dict(
            band=band, n=len(g), n_out=len(out),
            pct_windows_out=100 * len(out) / max(len(g), 1),
            pct_ai_err_out=100 * out['ai_err'].sum()
            / max(g['ai_err'].sum(), 1e-9),
            pct_st_err_out=100 * out['st_err'].sum()
            / max(g['st_err'].sum(), 1e-9),
            ai_med_in=float(ins['ai_err'].median()) if len(ins) else np.nan,
            ai_med_out=float(out['ai_err'].median()) if len(out) else np.nan,
            st_med_in=float(ins['st_err'].median()) if len(ins) else np.nan,
            st_med_out=float(out['st_err'].median()) if len(out) else np.nan))
    bu = pd.DataFrame(bur_rows)
    bu.to_csv(OUT_BUR, index=False)
    print(bu.round(1).to_string(index=False))

    # ---------------- figure (one column, 88 mm) ----------------
    # v2 layout: the original 94.9 mm two-block geometry is preserved in
    # absolute mm (panel sizes, gaps, margins identical); panel (c) is a
    # third block with exactly panel (b)'s box height, so the page grows to
    # 131.1 mm and every fraction below is the old mm position / new height.
    fig = plt.figure(figsize=(88 * MM, 131.1 * MM))
    gs_a = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.17],
                            left=0.216, right=0.858, top=0.9614, bottom=0.6335,
                            hspace=0.07)
    gs_b = fig.add_gridspec(1, 1, left=0.216, right=0.858,
                            top=0.5487, bottom=0.3573)
    gs_c = fig.add_gridspec(1, 1, left=0.216, right=0.858,
                            top=0.2725, bottom=0.0812)

    ax = fig.add_subplot(gs_a[0, 0])
    ax_n = fig.add_subplot(gs_a[1, 0], sharex=ax)
    ax.fill_between(bt['x'], bt['d_lo'], bt['d_hi'], color='#1f3e9e',
                    alpha=0.15, lw=0)
    ax.plot(bt['x'], bt['d_med'], 'o-', color='#1f3e9e', lw=1.3, ms=3.4,
            mec='black', mew=0.3)
    from matplotlib.lines import Line2D as _L2D
    from matplotlib.patches import Patch as _Patch
    ax.legend(handles=[
        _L2D([], [], color='#1f3e9e', lw=1.3, marker='o', ms=3.4,
             mec='black', mew=0.3, label='median'),
        _Patch(facecolor='#1f3e9e', alpha=0.15, label='95% CI')],
        loc='upper left', fontsize=F['small'], frameon=False,
        handlelength=1.4, borderaxespad=0.3, labelspacing=0.25)
    ax.set_ylabel('peak-error difference (cm)\nCEFNet − STOFS-2D',
                  fontsize=F['label'])
    ax.tick_params(axis='y', labelsize=F['tick'])
    ax.tick_params(axis='x', labelbottom=False)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.text(-0.205, 1.0, 'a', transform=ax.transAxes, fontsize=F['panel'],
            fontweight='bold', va='bottom')

    # window-count strip (log), fig-5 style — carries the shared x-axis
    ax_n.bar(bt['x'], bt['n'], width=0.25, color='#a386b4',
             edgecolor='#7c5e91', lw=0.3, zorder=3)
    ax_n.set_yscale('log')
    ax_n.set_ylim(50, bt['n'].max() * 4)
    ax_n.set_yticks([1e2, 1e4])
    ax_n.set_ylabel('windows', fontsize=F['label_sm'])
    ax_n.set_xticks(bt['x'])
    ax_n.set_xticklabels(['0–2' if 'never' in l else l for l in bt['lab']],
                         fontsize=F['tick'])
    ax_n.set_xlim(-0.35, len(bt) - 0.6)
    ax_n.set_xlabel('training events reaching this surge level',
                    fontsize=F['label'])
    ax_n.tick_params(labelsize=F['tick'])
    ax_n.grid(False)

    # panel (b): dumbbells — in-support circle -> starved triangle per model;
    # the connector length is the "starvation penalty". Opaque connector with a
    # thin black edge (path-effect stroke) so it matches the marker edges.
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    axc = fig.add_subplot(gs_b[0, 0])
    dx = 0.09
    line_edge = [pe.Stroke(linewidth=2.0, foreground='black'), pe.Normal()]
    for i, band in enumerate(('minor', 'moderate', 'major')):
        r = bu[bu['band'] == band].iloc[0]
        for xo, col, cin, cout in ((-dx, C_AI, 'ai_med_in', 'ai_med_out'),
                                   (dx, C_ST, 'st_med_in', 'st_med_out')):
            x = i + xo
            if r['n_out'] >= MIN_N:      # starved point only where support allows
                axc.plot([x, x], [r[cin], r[cout]], color=col, lw=1.4,
                         solid_capstyle='round', zorder=2,
                         path_effects=line_edge)
                axc.scatter([x], [r[cout]], s=30, color=col, marker='^',
                            edgecolor='black', linewidths=0.3, zorder=4)
            axc.scatter([x], [r[cin]], s=22, color=col, marker='o',
                        edgecolor='black', linewidths=0.3, zorder=4)
    axc.set_xticks(range(3))
    axc.set_xticklabels(('minor', 'moderate', 'major'), fontsize=F['tick'])
    axc.set_xlim(-0.5, 2.5)
    axc.set_ylim(0, 99)
    axc.set_ylabel('median |peak error| (cm)', fontsize=F['label'])
    axc.text(-0.205, 1.02, 'b', transform=axc.transAxes, fontsize=F['panel'],
             fontweight='bold', va='bottom')
    axc.tick_params(labelsize=F['tick'])
    axc.grid(False)
    axc.set_axisbelow(True)

    axc.legend(handles=[
        Line2D([], [], color=C_AI, lw=3, label='CEFNet'),
        Line2D([], [], color=C_ST, lw=3, label='STOFS-2D'),
        Line2D([], [], color='#555555', marker='o', ls='none', ms=4.5,
               mec='black', mew=0.3, label='≥3 prior events'),
        Line2D([], [], color='#555555', marker='^', ls='none', ms=5.5,
               mec='black', mew=0.3, label='0–2 prior events')],
        loc='upper left', ncol=1, fontsize=F['small'], frameon=False,
        handlelength=1.3, labelspacing=0.35, borderaxespad=0.3)

    # panel (c): in-support windows only (>=3 prior events) — signed peak
    # error boxes per severity and model, same currency as panels a/b
    # (event-locked residual peaks). No fliers; whiskers at 1.5 IQR.
    axd = fig.add_subplot(gs_c[0, 0])
    dxc = 0.19
    data, positions, colors = [], [], []
    for i, band in enumerate(('minor', 'moderate', 'major')):
        ins = w[(w['band'] == band) & (w['n_ev'] >= 3)]
        data += [ins['ai_serr'].values, ins['st_serr'].values]
        positions += [i - dxc, i + dxc]
        colors += [C_AI, C_ST]
        print(f'panel c {band}: n_in={len(ins)}')
    bp = axd.boxplot(data, positions=positions, widths=0.30,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color=C_INK, lw=1.0),
                     whiskerprops=dict(color=C_INK, lw=0.6),
                     capprops=dict(color=C_INK, lw=0.6))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_edgecolor(C_INK)
        patch.set_linewidth(0.6)
    axd.axhline(0, color='#888888', lw=0.6, zorder=1)
    axd.set_xticks(range(3))
    axd.set_xticklabels(('minor', 'moderate', 'major'), fontsize=F['tick'])
    axd.set_xlim(-0.5, 2.5)
    axd.set_ylabel('signed peak error (cm)\nwithin support (≥3 events)',
                   fontsize=F['label'])
    axd.text(-0.205, 1.02, 'c', transform=axd.transAxes, fontsize=F['panel'],
             fontweight='bold', va='bottom')
    axd.tick_params(labelsize=F['tick'])
    axd.grid(False)
    axd.set_axisbelow(True)

    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, OUT_PNG, dpi=450)
    plt.close(fig)
    print('wrote', OUT_PNG)
