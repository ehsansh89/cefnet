"""
paper_fig3_rework_draft.py — v4.1 Figure 5: "the ensemble probabilities are
honest." (Script keeps its v3-era name until the renaming sweep; it writes
docs/paper/figures/v4/fig5.png.) The main figure is the reliability diagram
+ window-count strip alone (isotonic fit on 2024, evaluated forward on
2025), with the raw ensemble as a light-gray reference and 95% block-
bootstrap CI bands on the calibrated curves. Decision value / usefulness is
Figure 6's job.

The spread-skill panel is rendered separately as a PRE-AUDIT SI candidate
only: its smooth pooled relation may be partly station-mixing (gauges with
big surge climates have both larger spread and larger errors), so before
it can appear in the SI it owes a within-station re-binning + block-
bootstrap audit; the text sentence "the ensemble anticipates its own
error" is suspended until that audit passes.

Reads:  outputs/paper/v2/fig2_reliability.csv (via paper_fig2_plot)
        per-region raw/spread_skill_bins.csv (v2 UQ archive)
Writes: outputs/paper/v2/fig3_spread_skill.csv       (pooled, pre-audit)
        docs/paper/figures/v4/fig5.png               (main, 88 x 80 mm)
        docs/paper/figures/v4/SI/fig3_si_spread_skill_preaudit.png
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from o_style import setup, savefig, PAPER_FONT as F
from paper_fig2_plot import draw_reliability, draw_counts, load_all


UQ_BASE = r'outputs/uq_postprocessing/nn_24lb_72fh_v2/gefs_psfc_v2_raw'
SPAN = '20240101_20251201'
OUT_CSV = 'outputs/paper/v2/fig3_spread_skill.csv'
OUT_PNG = 'docs/paper/figures/v4/fig5.png'
OUT_SS = 'docs/paper/figures/v4/SI/fig3_si_spread_skill_preaudit.png'
TEXT = '#1a1a1a'


def pool_spread_skill():
    """Reads the ten regions' raw spread_skill_bins.csv and pools them per
    (lead bin, spread decile), n-weighted. PRE-AUDIT: deciles are within
    region, not within station — station mixing may inflate the relation.
    Gives the pooled frame."""
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(
        UQ_BASE, SPAN, 'region_*', 'raw', 'spread_skill_bins.csv')))]
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby(['lead_bin', 'spread_bin']).apply(
        lambda x: pd.Series(dict(
            n=x['n'].sum(),
            mean_spread_cm=np.average(x['mean_spread_cm'], weights=x['n']),
            mae_cm=np.average(x['mae_cm'], weights=x['n'])))).reset_index()
    g['lead_lo'] = g['lead_bin'].str.split('-').str[0].astype(int)
    g = g.sort_values(['lead_lo', 'spread_bin']).reset_index(drop=True)
    g.to_csv(OUT_CSV, index=False)
    return g


def draw_spread_skill(ax, ss):
    """Draws the pre-audit SI candidate: MAE of the ensemble median vs the
    ensemble's own spread, one line per lead bin."""
    leads = ss.drop_duplicates('lead_bin').sort_values('lead_lo')
    shades = plt.get_cmap('Blues')(np.linspace(0.45, 0.95, len(leads)))
    for shade, (_, row) in zip(shades, leads.iterrows()):
        g = ss[ss['lead_bin'] == row['lead_bin']]
        ax.plot(g['mean_spread_cm'], g['mae_cm'], 'o-', ms=2.6, lw=1.0,
                color=shade, mec='black', mew=0.25, zorder=3,
                label=f"{row['lead_bin']} h")
    ax.set_xlabel('ensemble spread (cm)', fontsize=F['label'], labelpad=1.5,
                  color=TEXT)
    ax.set_ylabel('MAE of ensemble median (cm)', fontsize=F['label'],
                  labelpad=2, color=TEXT)
    ax.tick_params(labelsize=F['tick'], colors=TEXT)
    ax.legend(fontsize=F['legend'], loc='lower right', frameon=False,
              borderaxespad=0.4, handlelength=1.5, labelcolor=TEXT,
              title='lead time', title_fontsize=F['legend'],
              labelspacing=0.3)
    ax.grid(False)
    ax.set_axisbelow(True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    setup()
    cfg, rel, _, summary = load_all()

    # ---------------- main figure: reliability only ----------------
    fig = plt.figure(figsize=(3.46, 3.16))       # 88 x 80 mm
    # margins match Fig 3 (24 pt left / 35 pt right); reliability panel kept
    # ~square and the count strip sized to Fig 3's strip height (~6 mm)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.103],
                          left=0.189, right=0.858, top=0.9502, bottom=0.1245,
                          hspace=0.046)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[1, 0], sharex=ax_a)
    draw_reliability(ax_a, rel)
    draw_counts(ax_n, rel)
    for axx in (ax_a, ax_n):
        for sp in axx.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('black')
        axx.tick_params(width=0.5, color='black')
    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, OUT_PNG, dpi=600)
    plt.close(fig)
    print('Wrote', OUT_PNG)

    # -------- SI candidate (pre-audit; not shipped until audited) --------
    ss = pool_spread_skill()
    fig, ax = plt.subplots(figsize=(3.46, 2.55))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.955, bottom=0.155)
    draw_spread_skill(ax, ss)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color('black')
    ax.tick_params(width=0.5, color='black')
    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, OUT_SS, dpi=600)
    plt.close(fig)
    print('Wrote', OUT_SS, '(pre-audit SI candidate)')
