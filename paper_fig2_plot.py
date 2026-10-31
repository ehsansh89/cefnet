"""
paper_fig2_plot.py — Figure 2: forward-calibrated flood probabilities, and
probabilistic skill vs post-processed physics (Rounds 3-4 in
docs/paper/panel_discussion_on_figures.md).

Plain summary: renders the one-column Figure 2 from the CSVs written by
paper_fig2_process.py. Panel (a): reliability of flood probabilities under
forward out-of-sample evaluation (isotonic fit on 2024, applied to 2025) —
calibration improves reliability, with a residual mid-range overforecast at
the minor threshold; raw ensemble drawn as a gray reference; window-count
strip beneath. Panel (b): CRPSS vs lead against deterministic and dressed
STOFS — the claim it carries is "the complete AI probabilistic system beats
post-processed physics." The dressed-AI same-center control is NOT in the
main panel (Round 4 decision): it renders as the SI figure so the negative
control is published, not buried.

Reads (PROCESS_OUT_DIR = outputs/paper/v2):
    fig2_reliability.csv, fig2_crpss_lead.csv, fig2_summary.json
Writes:
    {FIG_OUT_DIR}/fig2.png                  (88 mm x ~130 mm, one column)
    {FIG_OUT_DIR}/fig2_si_dressed_ai.png    (SI: panel b + dressed-AI control)

Everything renders from the CSVs with zero manual touch-up so the figure
regenerates when the causally clean run lands. Headline caption numbers are
printed to the console.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from o_style import setup, savefig, PAPER_FONT as F


DIRECTION = 'fit2024_eval2025'      # headline: fit on 2024, evaluate on 2025
MIN_BIN_N = 100                     # bins below this are suppressed in (a)
PHYS_COLOR = '#AA3377'
CTRL_COLOR = '#555555'              # SI-only dressed-AI control line
RAW_COLOR = '#c4c4c4'               # light gray = raw-ensemble reference
TEXT = '#1a1a1a'
# Round-4 review (Ehsan): thresholds need real hue separation — warmth =
# severity (NOAA's own convention): minor blue, moderate red (Tol bright,
# CVD-safe). Raw is gray so reference vs product cannot be confused.
THR_STYLE = {'minor': dict(marker='o', color='#cb2c09'),
             'moderate': dict(marker='s', color='#781486')}
# paper palette 2026-07-24: both curves are the AI ensemble product;
# minor = light warm sibling of the ensemble orange, moderate = deep
# purple. Peach is light — keep the black marker edges for legibility.


def load_all():
    """Reads the two configs' worth of paths and the three fig2 backing
    files; gives (fig config, reliability frame, crpss frame, summary dict)."""
    with open('paper_fig1_config_v2.json') as f:
        cfg = json.load(f)
    proc = cfg['PROCESS_OUT_DIR']
    rel = pd.read_csv(os.path.join(proc, 'fig2_reliability.csv'))
    crpss = pd.read_csv(os.path.join(proc, 'fig2_crpss_lead.csv'))
    with open(os.path.join(proc, 'fig2_summary.json')) as f:
        summary = json.load(f)
    return cfg, rel, crpss, summary


# =============================================================================
# Panel (a): reliability + count strip
# =============================================================================
def _prob_ticks(ax, axis='both'):
    """0.2-labeled major ticks + unlabeled 0.1 minor ticks (Round-4 choice)."""
    major = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    minor = [0.1, 0.3, 0.5, 0.7, 0.9]
    if axis in ('x', 'both'):
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)
    if axis in ('y', 'both'):
        ax.set_yticks(major)
        ax.set_yticks(minor, minor=True)
    ax.tick_params(which='minor', length=1.6, width=0.5, color='black')


def draw_reliability(ax, rel):
    """Draws panel (a): forecast probability vs observed frequency, calibrated
    minor/moderate prominent with block-bootstrap CIs, raw as gray reference.
    Gets the axes and the reliability frame; gives nothing (draws in place)."""
    d = rel[rel['direction'] == DIRECTION]
    ax.plot([0, 1], [0, 1], color='black', lw=0.5, ls=(0, (4, 2)), zorder=1)
    for thr, st in THR_STYLE.items():
        raw = d[(d['threshold'] == thr) & (d['product'] == 'raw')
                & (d['n'] >= MIN_BIN_N)]
        # Reference layer: very light gray, same marker shape as its
        # calibrated sibling (circle=minor, square=moderate) so shape reads
        # as severity and the light color keeps raw subordinate.
        ax.plot(raw['pred_mean'], raw['obs_freq'], st['marker'] + '-',
                color=RAW_COLOR, lw=0.7, ms=2.4, mec='none', zorder=3)
        iso = d[(d['threshold'] == thr) & (d['product'] == 'isotonic')
                & (d['n'] >= MIN_BIN_N)].sort_values('pred_mean')
        # 95% block-bootstrap CI as a shaded band (consistent with the band
        # style in Figs 1 and 3), drawn under the markers.
        ax.fill_between(iso['pred_mean'], iso['obs_freq_lo'],
                        iso['obs_freq_hi'], color=st['color'], alpha=0.15,
                        lw=0, zorder=2)
        ax.plot(iso['pred_mean'], iso['obs_freq'], st['marker'] + '-',
                color=st['color'], lw=0.9, ms=3.0, mec='black', mew=0.3,
                zorder=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _prob_ticks(ax, 'both')
    ax.set_ylabel('observed frequency', fontsize=F['label'], labelpad=2, color=TEXT)
    ax.tick_params(labelsize=F['tick'], colors=TEXT, labelbottom=False)
    ax.grid(False)
    # Legend matches the plot exactly: marker-less dotted gray = the raw
    # reference lines (the Round-4 review's literal form, coherent now that the
    # plotted raw curves are marker-less too).
    handles = [
        mpl.lines.Line2D([], [], marker='o', ms=3.0, lw=0.9, mec='black',
                         mew=0.3, color=THR_STYLE['minor']['color'],
                         label='minor, calibrated'),
        mpl.lines.Line2D([], [], marker='s', ms=2.8, lw=0.9, mec='black',
                         mew=0.3, color=THR_STYLE['moderate']['color'],
                         label='moderate, calibrated'),
        mpl.lines.Line2D([], [], marker='o', ms=2.6, lw=0.7, color=RAW_COLOR,
                         mec='none', label='raw ensemble'),
    ]
    ax.legend(handles=handles, fontsize=F['legend'], loc='upper left',
              frameon=False, borderaxespad=0.3, handlelength=1.7,
              labelcolor=TEXT)


def draw_counts(ax, rel):
    """Draws the thin strip under (a): forecasts per probability bin (log
    scale), colored like the curves above. Empty ranges are real — isotonic
    is a step function and can emit no value in a bin.
    Gets the axes and the reliability frame; gives nothing."""
    d = rel[(rel['direction'] == DIRECTION) & (rel['product'] == 'isotonic')]
    for thr, off in [('minor', -0.011), ('moderate', 0.011)]:
        g = d[(d['threshold'] == thr) & (d['n'] > 0)]
        ctr = (g['bin_left'] + g['bin_right']) / 2
        ax.bar(ctr + off, g['n'], width=0.020,
               color=THR_STYLE[thr]['color'], edgecolor='black',
               linewidth=0.2)
    ax.set_yscale('log')
    ax.set_ylim(1, 3e5)
    ax.set_yticks([1e2, 1e4])
    ax.set_xlim(-0.02, 1.02)
    _prob_ticks(ax, 'x')
    ax.set_ylabel('windows', fontsize=F['label_sm'], labelpad=2, color=TEXT)
    ax.set_xlabel('forecast probability', fontsize=F['label'], labelpad=1.5,
                  color=TEXT)
    ax.tick_params(labelsize=F['tick'], colors=TEXT)
    ax.grid(False)


# =============================================================================
# Panel (b): CRPSS vs lead
# =============================================================================
def draw_crpss(ax, cr, include_control=False):
    """Draws CRPSS vs lead. Main figure: vs deterministic and dressed STOFS
    only. With include_control=True (SI figure), adds the dressed-AI
    same-center control — the published negative control (CRPSS −0.083).
    Gets the axes, the per-lead frame, and the SI flag; gives nothing."""
    ax.axvline(12, color='black', lw=0.5, ls=(0, (4, 2)), zorder=1)
    ax.axhline(0, color='black', lw=0.5, ls=(0, (4, 2)), zorder=1)
    series = [
        ('stofs', PHYS_COLOR, '-', 'vs STOFS-2D'),
        ('dressed_stofs', PHYS_COLOR, (0, (4, 2)), 'vs dressed STOFS-2D'),
    ]
    if include_control:
        series.append(('dressed_ai', CTRL_COLOR, (0, (4, 2)),
                       'vs dressed AI (same center)'))
    for key, color, ls, label in series:
        ax.fill_between(cr['lead_hour'], cr[f'crpss_vs_{key}_lo'],
                        cr[f'crpss_vs_{key}_hi'], color=color, alpha=0.18,
                        lw=0, zorder=2)
        ax.plot(cr['lead_hour'], cr[f'crpss_vs_{key}'], color=color, lw=1.0,
                ls=ls, label=label, zorder=3)
    hi = max(float(cr[f'crpss_vs_{k}_hi'].max()) for k, _, _, _ in series)
    lo = min(0.0, min(float(cr[f'crpss_vs_{k}_lo'].min())
                      for k, _, _, _ in series))
    ax.set_xlim(1, 72)
    ax.set_ylim(lo - (0.03 if lo < 0 else 0.0), hi * 1.12)
    ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax.set_xlabel('lead time (h)', fontsize=F['label'], labelpad=1.5, color=TEXT)
    ax.set_ylabel('CRPSS', fontsize=F['label'], labelpad=2, color=TEXT)
    ax.tick_params(labelsize=F['tick'], colors=TEXT)
    ax.legend(fontsize=F['legend'], loc='lower right', frameon=False,
              borderaxespad=0.4, handlelength=1.7, labelcolor=TEXT)
    ax.grid(False)


# =============================================================================
def build_si(cfg, crpss):
    """Renders the SI figure: panel b plus the dressed-AI same-center control
    line, so the negative control is published (figure review, Round 4).
    Gets the fig config and the per-lead frame; gives the output path."""
    fig, ax = plt.subplots(figsize=(3.46, 2.45))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.96, bottom=0.155)
    draw_crpss(ax, crpss, include_control=True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color('black')
    ax.tick_params(width=0.5, color='black')
    out = os.path.join(cfg.get('FIG_OUT_DIR', 'docs/paper/figures'),
                       'fig2_si_dressed_ai.png')
    savefig(fig, out)
    plt.close(fig)
    return out


def build():
    """Renders the main Figure 2 (and the SI control figure) from the backing
    CSVs and prints the caption numbers. Gets nothing; gives nothing."""
    cfg, rel, crpss, summary = load_all()
    fig = plt.figure(figsize=(3.46, 5.12))       # 88 x 130 mm
    # Margins are designed into the canvas (tight-crop disabled at save):
    # visible air left/right, minimal top/bottom (Ehsan, Round 4 polish).
    gs = fig.add_gridspec(4, 1, height_ratios=[1.00, 0.155, 0.10, 0.62],
                          left=0.185, right=0.945, top=0.962, bottom=0.075,
                          hspace=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_b = fig.add_subplot(gs[3, 0])

    draw_reliability(ax_a, rel)
    draw_counts(ax_n, rel)
    draw_crpss(ax_b, crpss)

    for axx in (ax_a, ax_n, ax_b):
        for sp in axx.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('black')
        axx.tick_params(width=0.5, color='black')

    pa, pb = ax_a.get_position(), ax_b.get_position()
    fig.text(0.03, pa.y1 - 0.002, 'a', fontsize=F['panel'], fontweight='bold',
             va='bottom', ha='left')
    fig.text(0.03, pb.y1 + 0.004, 'b', fontsize=F['panel'], fontweight='bold',
             va='bottom', ha='left')

    fig_dir = cfg.get('FIG_OUT_DIR', 'docs/paper/figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, 'fig2.png')
    # Keep the full designed canvas — the o_style default tight-crop would
    # trim the side margins right back off.
    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, out, dpi=600)
    plt.close(fig)
    print(f'  wrote {out}')
    print(f'  wrote {build_si(cfg, crpss)}')

    # Caption numbers (single source: fig2_summary.json).
    b = summary['reliability_scoreboard'][DIRECTION]
    p = summary['crpss_pooled']
    print('  caption numbers [fig2.png]:')
    for thr in ('minor', 'moderate'):
        r = b[thr]
        print(f"    {thr}: Brier raw {r['raw']['brier']:.5f} -> isotonic "
              f"{r['isotonic']['brier']:.5f} (BSS vs clim "
              f"{r['isotonic']['bss_vs_climatology']:+.3f}); ECE "
              f"{r['raw']['ece']:.4f} -> {r['isotonic']['ece']:.4f}; "
              f"n={r['isotonic']['n']:,}")
    for k in ('stofs', 'dressed_stofs', 'dressed_ai'):
        v = p[f'crpss_vs_{k}']
        print(f"    CRPSS vs {k}: {v['value']:+.3f} "
              f"[{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print(f"    episodes: {summary['episodes']}; gauges: "
          f"{summary['n_gauges_reliability']} (a) / "
          f"{summary['n_gauges_crpss']} (b); matched pairs (b): "
          f"{summary['matched_pairs_crpss']:,}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    setup()
    print('===== Figure 2: honest probabilities vs post-processed physics =====')
    build()
