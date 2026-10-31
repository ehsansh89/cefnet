"""
paper_fig4_assembly_draft.py — v4.1 Figure 6: "from prediction to warning —
the free dial." (Script keeps its v3-era name until the renaming sweep; it
writes docs/paper/figures/v4/fig6.png.) A 2 x 3 grid, one flood severity per
column. Row (a): the warning dial (probability of detection vs success ratio
on CSI contours) — the CEFNet 93-cutoff ensemble sweep (orange curve) plus
the two deterministic operating points, CEFNet (blue) and STOFS-2D (magenta);
the severity retreat reads left to right. Row (b): the outcome counts for the
same severity — observed flood windows (the target), then caught / missed /
false warnings for CEFNet, STOFS-2D, and the nine-cutoff ensemble ladder,
honest linear axis. Dressed STOFS lives in the SI graphics.

Colors: blue = CEFNet, orange = CEFNet ensemble, magenta = STOFS-2D,
dark gray = observed.

Output: docs/paper/figures/v4/fig6.png   (180 x 103 mm)
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from o_style import setup, savefig, PAPER_FONT as F
from paper_fig4_panel_a import load_windows, point


IN_DIR = 'outputs/paper/v2'
OUT_PNG = 'docs/paper/figures/v4/fig6.png'
MM = 1 / 25.4
THRESHOLDS = ['minor', 'moderate', 'major']
CUTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
C_AI = '#208eb7'           # CEFNet deterministic (palette 2026-07-24)
C_ENS = '#ffa500'          # CEFNet ensemble (curve + ladder)
C_ST = '#c46fd8'           # STOFS-2D
C_OBS = '#3d3d3d'          # observed flood windows (the target)
METRICS = [('caught', 'caught'), ('missed', 'missed'),
           ('false_warns', 'false\nwarns')]
W_BIG = 0.85               # wide bar width
W_LAD = 0.20               # ladder bar width (contiguous, no spacing)
GAP = 0.25                 # one uniform gap between all bar blocks
EDGE = dict(edgecolor='none')


def draw_dial_sev(ax, ev, thr, show_ylabel, show_legend):
    """Gets the window frame + one severity; draws that severity's dial:
    CSI contours, the AI-ensemble sweep (93 raw cutoffs, solid) and the
    two deterministic dots (dressed STOFS lives in the SI graphics)."""
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

    g = ev[ev['threshold'] == thr]
    obs = g['obs_event'].to_numpy(bool)
    cuts = (np.arange(1, 94) - 0.5) / 93.0
    p_raw = g['ai_prob_raw'].to_numpy(float)
    xy = np.array([point(p_raw >= c, obs) for c in cuts])
    ax.plot(xy[:, 0], xy[:, 1], '-', color=C_ENS, lw=1.2, zorder=3)
    for col, colr in [('det_gfs_event', C_AI), ('stofs_event', C_ST)]:
        w = g[col].to_numpy(bool)
        ax.plot(*point(w, obs), 'o', ms=5.4, mfc=colr, mec='#1a1a1a',
                mew=0.6, zorder=7)

    ax.set_title(f'{thr} flooding', fontsize=F['label'], pad=3)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_xlabel('success ratio (1 − FAR)', fontsize=F['label_sm'])
    if show_ylabel:
        ax.set_ylabel('probability of detection', fontsize=F['label_sm'])
    ax.tick_params(labelsize=F['tick'])
    ax.grid(False)
    if show_legend:
        handles = [
            Line2D([], [], color=C_ENS, lw=1.2, label='CEFNet (ensemble)'),
            Line2D([], [], marker='o', ls='none', ms=5, mfc=C_AI,
                   mec='#1a1a1a', mew=0.5, label='CEFNet'),
            Line2D([], [], marker='o', ls='none', ms=5, mfc=C_ST,
                   mec='#1a1a1a', mew=0.5, label='STOFS-2D')]
        ax.legend(handles=handles, loc='lower left', fontsize=F['legend'],
                  handletextpad=0.5, borderaxespad=0.35, labelspacing=0.35)


def draw_counts(ax, st, thr, show_ylabel, show_legend):
    """Gets the table stats + one severity; draws the counts bar panel:
    the observed flood-window total (gray, caught group only — the
    target), AI and STOFS-2D bars, and the nine-cutoff ensemble ladder in
    one flat teal, all with hairline black edges, linear axis."""
    def val(sysk, col):
        return float(st[(st['threshold'] == thr)
                        & (st['system'] == sysk)].iloc[0][col])
    events = float(st[(st['threshold'] == thr)
                      & (st['system'] == 'ai_det')].iloc[0]['events'])
    e, ticks = 0.0, []           # e = running left edge; uniform GAP
    for mcol, _ in METRICS:
        left = e
        wides = ([(C_OBS, events)] if mcol == 'caught' else []) \
            + [(C_AI, val('ai_det', mcol)), (C_ST, val('stofs', mcol))]
        for color, v in wides:
            ax.bar(e + W_BIG / 2, v, width=W_BIG, color=color, zorder=3,
                   **EDGE)
            e += W_BIG + GAP
        for k, c in enumerate(CUTS):
            ax.bar(e + W_LAD / 2 + k * W_LAD, val(f'ens{c}', mcol),
                   width=W_LAD, color=C_ENS, zorder=3, **EDGE)
        e += len(CUTS) * W_LAD
        ticks.append((left + e) / 2)
        e += 1.7
    ax.set_xticks(ticks)
    ax.set_xticklabels([lab for _, lab in METRICS], fontsize=F['label_sm'],
                       linespacing=0.9)
    ax.tick_params(labelsize=F['tick'])
    if show_ylabel:
        ax.set_ylabel('72-h windows', fontsize=F['label_sm'])
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_legend:
        handles = [Patch(facecolor=C_OBS, label='observed'),
                   Patch(facecolor=C_AI, label='CEFNet'),
                   Patch(facecolor=C_ST, label='STOFS-2D'),
                   Patch(facecolor=C_ENS, label='CEFNet (ensemble),'),
                   Patch(facecolor='none', edgecolor='none', label='10→90%')]
        ax.legend(handles=handles, loc='upper right', fontsize=F['legend'],
                  handlelength=1.0, handletextpad=0.35, borderaxespad=0.2,
                  labelspacing=0.3)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    mpl.use('Agg')
    setup()
    st = pd.read_csv(os.path.join(IN_DIR, 'fig4_table_stats.csv'))
    ev = load_windows()

    fig = plt.figure(figsize=(180 * MM, 103 * MM))
    # equal-height rows + the fixed panel width (Fig 2 margins) make each
    # panel ~square; a real top margin leaves room above the titles
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          left=0.165, right=0.865, top=0.9126, bottom=0.0835,
                          hspace=0.359, wspace=0.24)
    for i, thr in enumerate(THRESHOLDS):
        ax = fig.add_subplot(gs[0, i])
        draw_dial_sev(ax, ev, thr, show_ylabel=(i == 0),
                      show_legend=(i == 0))
        if i == 0:
            ax.text(-0.24, 1.05, 'a', transform=ax.transAxes,
                    fontsize=F['panel'], fontweight='bold')
    for i, thr in enumerate(THRESHOLDS):
        ax = fig.add_subplot(gs[1, i])
        draw_counts(ax, st, thr, show_ylabel=(i == 0),
                    show_legend=(i == 0))
        if i == 0:
            ax.text(-0.24, 1.05, 'b', transform=ax.transAxes,
                    fontsize=F['panel'], fontweight='bold')

    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, OUT_PNG, dpi=600)
    plt.close(fig)
    print('Wrote', OUT_PNG)
