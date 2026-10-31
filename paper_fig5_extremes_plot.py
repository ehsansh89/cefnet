"""
paper_fig5_extremes_plot.py — v4.1 Figure 2 ("skill distribution — where the
skill lives"), rendered from the fig5_extremes_* tables. (Script keeps its
v3-era fig5 name until the renaming sweep; it writes
docs/paper/figures/v4/fig2.png.)

In plain English: the top row shows, gauge by gauge, whether CEFNet beats
STOFS-2D on all forecasts (x-axis) and on just the flood windows of each
severity (y-axis) — upper-right means better on both. Bottom left shows,
for the same flood windows, how far each model's predicted 72-h peak landed
from the observed peak (negative = under-called; paired boxes, CEFNet left,
STOFS-2D right — swapped from split violins 2026-08-02 to match Fig 3c).
Bottom right shows the same peak error as
a curve against observed surge size over ALL windows: both systems
under-call bigger events in near-lockstep, separating only at the very
largest.

Output: docs/paper/figures/v4/fig2.png   (180 x 114 mm, 600 dpi)
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from o_style import setup, savefig, PAPER_FONT as F


IN_DIR = 'outputs/paper/v2'
OUT_PNG = 'docs/paper/figures/v4/fig2.png'
MM = 1 / 25.4
COL = {'ai': '#208eb7', 'st': '#c46fd8'}   # paper palette 2026-07-24
NAME = {'ai': 'CEFNet', 'st': 'STOFS-2D'}
THRESHOLDS = ['minor', 'moderate', 'major']
LIM = (-1.0, 1.0)
MIN_N_KEY = 'min_flood_windows'


def draw_quadrant(ax, quad, thr, summ, show_ylabel, n_max):
    """Gets one severity's slice of the quadrant table; draws the per-gauge
    scatter (x = all-window skill, y = flood-window skill) with the
    upper-right fraction printed. Point sizes share one scale across all
    three facets (n_max = the global flood-window maximum), so equal circles
    mean equal support everywhere — major's thin evidence stays visible."""
    min_n = summ['protocol'][MIN_N_KEY]
    d = quad.dropna(subset=[f'skill_flood_{thr}']).copy()
    d = d[d[f'n_flood_{thr}'] >= min_n]
    x = d['skill_all'].clip(*LIM)
    y = d[f'skill_flood_{thr}'].clip(*LIM)
    n_fl = d[f'n_flood_{thr}'].to_numpy(float)
    size = 6.0 + 40.0 * np.sqrt(n_fl / n_max)

    ax.axhline(0, color='#999999', lw=0.6, zorder=1)
    ax.axvline(0, color='#999999', lw=0.6, zorder=1)
    ax.scatter(x, y, s=size, c='#1f3e9e', alpha=0.55, edgecolors='#1a1a1a',
               linewidths=0.15, zorder=3)

    # headline fraction + medians move to the caption (Nature cleanliness);
    # the panel keeps only its title and axes.
    ax.set_title(f'{thr} flood windows', fontsize=F['label'], pad=3)
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect('equal')
    ax.set_xlabel('MAE skill, all windows', fontsize=F['label_sm'])
    if show_ylabel:
        ax.set_ylabel('MAE skill, flood windows', fontsize=F['label_sm'])
    ax.tick_params(labelsize=F['tick'])
    ax.grid(False)


def draw_peaks(ax, peaks, summ):
    """Gets the flood-window peak-error table; draws paired boxes (CEFNet
    left, STOFS-2D right) per severity, Fig-3c style (2026-08-02 swap from
    split violins for cross-figure consistency): no fliers, whiskers at
    1.5x IQR, median = box line. The major tail's deep under-calls are the
    named hurricanes — stated in the manuscript text, not annotated here."""
    xpos = {t: i * 1.1 for i, t in enumerate(THRESHOLDS)}
    ax.axhline(0, color='#999999', lw=0.6, zorder=1)
    data, positions, colors = [], [], []
    for thr in THRESHOLDS:
        f = peaks[peaks['threshold'] == thr]
        s = summ['peaks'][thr]
        for sysk, colk, xo in [('ai', 'peak_err_ai_m', -0.19),
                               ('st', 'peak_err_stofs_m', 0.19)]:
            v = f[colk].dropna().to_numpy() * 100
            data.append(v)
            positions.append(xpos[thr] + xo)
            colors.append(COL[sysk])
        # per-severity window count returns to the panel (black)
        ax.annotate(f"n = {s['n_windows']:,}",
                    xy=(xpos[thr], 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -2), textcoords='offset points',
                    ha='center', va='top', fontsize=F['label_sm'],
                    color='#1a1a1a')
    bp = ax.boxplot(data, positions=positions, widths=0.30,
                    showfliers=False, patch_artist=True, zorder=3,
                    medianprops=dict(color='#1a1a1a', lw=1.0),
                    whiskerprops=dict(color='#1a1a1a', lw=0.6),
                    capprops=dict(color='#1a1a1a', lw=0.6))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_edgecolor('#1a1a1a')
        patch.set_linewidth(0.6)
    # headroom for the n = ... annotations above the whiskers
    ax.set_ylim(ax.get_ylim()[0], 50)
    ax.set_xticks([xpos[t] for t in THRESHOLDS])
    ax.set_xticklabels(THRESHOLDS, fontsize=F['label'])
    ax.set_xlim(-0.62, xpos['major'] + 0.62)
    ax.set_ylabel('signed peak error (cm)', fontsize=F['label_sm'])
    ax.tick_params(labelsize=F['tick'])
    # no gridlines (the severity verticals existed to anchor the old split
    # violins; boxes need none, matching Fig 3c)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.legend(handles=[Line2D([], [], marker='s', ls='none', ms=5,
                              mfc=COL[s], mec='none', label=NAME[s])
                       for s in ('ai', 'st')],
              loc='lower left', fontsize=F['legend'], frameon=False,
              handletextpad=0.4, borderaxespad=0.3)


def draw_bins(ax, bins_df):
    """Gets the binned peak-error table; draws the surge stress test: median
    signed SURGE-peak error (residual space — tide removed from both sides)
    vs observed surge peak, 10-cm bins over ALL common-valid windows, IQR
    bands, per-bin counts. Event-locked: the forecast max is taken within
    +/-3 h of the hour the observed surge peaked — our declared tolerance,
    justified by the 72-h horizon and robust across the +/-1/2/6 h and
    timing-free sensitivities (+/-1 h ~ NOS's 1.5-h water-level timing
    acceptance on the hourly grid); the timing-free own-max variant is the
    SI sensitivity."""
    b = bins_df[bins_df['bin_type'] == 'surge_nearpeak_pm3h'].sort_values('lo')
    # bin centers; the open ">= X" overflow group sits half a bin past its lo
    xc = np.where(b['open_end'], b['lo'] + 7.0, (b['lo'] + b['hi']) / 2)
    ax.axhline(0, color='#999999', lw=0.6, zorder=1)
    for sysk, med, q25, q75 in [('ai', 'med_ai_cm', 'q25_ai', 'q75_ai'),
                                ('st', 'med_st_cm', 'q25_st', 'q75_st')]:
        ax.fill_between(xc, b[q25], b[q75], color=COL[sysk], alpha=0.12,
                        lw=0)
        ax.plot(xc, b[med], '-o', ms=2.4, lw=1.1, color=COL[sysk],
                mec='#1a1a1a', mew=0.3, zorder=4)
    for x, (_, r) in zip(xc, b.iterrows()):
        ax.annotate(f"{int(r['n']):,}", xy=(x, 1.0),
                    xycoords=('data', 'axes fraction'), xytext=(0, -1),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=5.4, color='#1a1a1a', rotation=90)
    last = b.iloc[-1]
    if last['open_end']:
        ax.annotate(f"≥{int(last['lo'])}", xy=(xc[-1], last['med_ai_cm']),
                    xytext=(4, -6), textcoords='offset points',
                    ha='left', va='top', fontsize=F['small'],
                    color='#1a1a1a')
    # headroom above the data so the rotated bin counts don't sit on the
    # zero line / low-bin curves
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.16 * (hi_y - lo_y))
    from matplotlib.patches import Patch
    handles = [Line2D([], [], marker='o', ls='-', color=COL[s],
                      mec='#1a1a1a', mew=0.3, ms=3, label=NAME[s])
               for s in ('ai', 'st')]
    handles.append(Patch(facecolor='#9db4c8', alpha=0.35, lw=0,
                         label='interquartile range'))
    ax.legend(handles=handles, loc='lower left', fontsize=F['legend'],
              handletextpad=0.4, borderaxespad=0.4)
    ax.set_xlabel('observed surge peak (cm)', fontsize=F['label_sm'])
    ax.set_ylabel('signed peak error (cm)', fontsize=F['label_sm'])
    ax.tick_params(labelsize=F['tick'])
    ax.grid(False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    mpl.use('Agg')
    setup()
    quad = pd.read_csv(os.path.join(IN_DIR, 'fig5_extremes_quadrant.csv'),
                       dtype={'station_id': str})
    peaks = pd.read_parquet(os.path.join(IN_DIR,
                                         'fig5_extremes_peaks.parquet'))
    bins_df = pd.read_csv(os.path.join(IN_DIR, 'fig5_extremes_peakbins.csv'))
    with open(os.path.join(IN_DIR, 'fig5_extremes_summary.json')) as f:
        summ = json.load(f)

    fig = plt.figure(figsize=(180 * MM, 114.0 * MM))
    # top row (panel a) sits in a narrower horizontal band so the three
    # square scatters are ~25% smaller, with the freed width becoming
    # left/right margin. The bottom row (b, c) shares panel a's exact
    # left/right margins (0.165 / 0.865) and is scaled down in height by the
    # same width-shrink factor so each subplot keeps its aspect ratio. The
    # freed height is taken out of the canvas (132 -> 114.0 mm) rather than
    # left as empty band; the inter-row gap is ~11 mm, so panel a stays put
    # and the figure just gets shorter.
    gs_top = fig.add_gridspec(1, 3, left=0.165, right=0.865,
                              top=0.9421, bottom=0.5542, wspace=0.30)
    gs_bot = fig.add_gridspec(1, 2, width_ratios=[0.44, 0.56],
                              left=0.165, right=0.865, top=0.4567,
                              bottom=0.0902, wspace=0.26)

    n_max = max(float(quad[f'n_flood_{t}'].max()) for t in THRESHOLDS)
    for i, thr in enumerate(THRESHOLDS):
        ax = fig.add_subplot(gs_top[0, i])
        draw_quadrant(ax, quad, thr, summ, show_ylabel=(i == 0), n_max=n_max)
        if i == 0:
            ax.text(-0.24, 1.06, 'a', transform=ax.transAxes,
                    fontsize=F['panel'], fontweight='bold')
    from matplotlib.transforms import ScaledTranslation
    down6 = ScaledTranslation(0, -6 / 72, fig.dpi_scale_trans)  # 6 pt down
    ax_b = fig.add_subplot(gs_bot[0, 0])
    draw_peaks(ax_b, peaks, summ)
    ax_b.text(-0.155, 1.06, 'b', transform=ax_b.transAxes + down6,
              fontsize=F['panel'], fontweight='bold')
    ax_c = fig.add_subplot(gs_bot[0, 1])
    draw_bins(ax_c, bins_df)
    ax_c.text(-0.13, 1.06, 'c', transform=ax_c.transAxes + down6,
              fontsize=F['panel'], fontweight='bold')

    # uniform closed frame: all four spines, one line width, every panel
    for axx in fig.get_axes():
        for sp in axx.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_color('#333333')
    # panel c (the line stress-test) reads as an open plot: drop top/right
    for sp in ('top', 'right'):
        ax_c.spines[sp].set_visible(False)

    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, OUT_PNG, dpi=600)
    plt.close(fig)
    print('Wrote', OUT_PNG)
