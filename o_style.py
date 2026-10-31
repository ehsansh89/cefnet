"""
o_style.py — Publication-grade matplotlib styling for CONUS_AI figures.

All `eval.py`, `inference.py`, and future `o_*.py` plot scripts call
`setup()` once before plotting. Gives every output the same typography,
palette, and figure geometry so the paper reads as one document.

Design targets: clean, data-dense, colorblind-safe. Modelled after
Nature / Science / Weather and Forecasting conventions — small sans-serif
labels, no heavy frames, no shadows, whitespace around elements, thin
axis lines, subtle grid.

Usage:
    from o_style import setup, COLORS, FIG, savefig
    setup()
    fig, ax = plt.subplots(figsize=FIG['single'])
    ax.plot(x, y, color=COLORS['obs'])
    savefig(fig, 'outputs/.../panel.pdf')
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# PALETTE — Paul Tol's colorblind-safe bright scheme, plus semantic aliases
# =============================================================================
# Tol bright (distinguishable under all common CVD types)
_TOL_BRIGHT = {
    'blue':   '#4477AA',
    'red':    '#EE6677',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'cyan':   '#66CCEE',
    'purple': '#AA3377',
    'grey':   '#BBBBBB',
}

COLORS = {
    # Semantic roles used across eval / inference / comparison plots.
    'obs':       '#1a1a1a',              # black = ground truth
    'pred':      _TOL_BRIGHT['blue'],    # blue  = model mean
    'envelope':  _TOL_BRIGHT['blue'],    # same blue, alpha'd for uncertainty
    'baseline':  _TOL_BRIGHT['grey'],    # grey  = reference (persistence, tide)
    'stofs':     _TOL_BRIGHT['purple'],  # purple = physics comparison
    'ofs':       _TOL_BRIGHT['green'],   # green  = regional-OFS comparison
    'peak':      _TOL_BRIGHT['red'],     # red    = peak / extreme highlight
    'accent':    _TOL_BRIGHT['yellow'],

    # Category palette for 0/1/2/3 flood-severity buckets.
    'cat_calm':      '#CCCCCC',
    'cat_mild':      _TOL_BRIGHT['cyan'],
    'cat_moderate':  _TOL_BRIGHT['yellow'],
    'cat_major':     _TOL_BRIGHT['red'],
}

# Category index -> color
CATEGORY_COLORS = [
    COLORS['cat_calm'], COLORS['cat_mild'],
    COLORS['cat_moderate'], COLORS['cat_major'],
]
CATEGORY_LABELS = ['calm', 'mild', 'moderate', 'major']


# =============================================================================
# FIGURE SIZES (inches) — W&F / Nature column conventions
# =============================================================================
FIG = {
    'single':  (3.5, 2.6),     # single-column
    'double':  (7.2, 3.2),     # double-column wide
    'hero':    (7.2, 4.6),     # double-column tall (hero figure)
    'tall':    (3.5, 5.0),     # single-column tall
    'square':  (4.0, 4.0),
    'video':   (14.0, 8.0),    # sliding-window frame layout (wide)
}

# Paper-figure font ladder (pt). The rcParams in setup() suit standalone
# diagnostic plots; the dense 88 / 180 mm manuscript panels use this smaller
# shared ladder so every paper_fig*_plot.py stays consistent by construction
# instead of by copied literals (panel decision, Fig 2 Round 4).
PAPER_FONT = {
    'panel':    11,     # bold panel letters a/b/c
    'label':    6.5,    # axis labels
    'label_sm': 6.0,    # compact axis labels (thin strips)
    'tick':     5.8,    # tick labels
    'legend':   6.0,    # legends
    'small':    5.6,    # registers, colorbar ticks/titles, fine annotations
}


# =============================================================================
# STYLE SETUP
# =============================================================================
def setup():
    """Install the CONUS_AI rcParams. Idempotent — safe to call more than once."""
    mpl.rcParams.update({
        # Typography: prefer Helvetica/Arial, fall back to Matplotlib's
        # bundled DejaVu Sans. Nature-style is sans-serif throughout.
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':          8.5,
        'axes.titlesize':     9.5,
        'axes.labelsize':     8.5,
        'xtick.labelsize':    7.5,
        'ytick.labelsize':    7.5,
        'legend.fontsize':    7.5,
        'figure.titlesize':   10.5,

        # Math stays consistent with body text.
        'mathtext.fontset':   'dejavusans',

        # Axes / spines — thin, closed frame (all four sides) for a cleaner
        # W&F / Nature look where every panel looks like a deliberate box.
        'axes.linewidth':     0.6,
        'axes.edgecolor':     '#333333',
        'axes.labelcolor':    '#1a1a1a',
        'axes.titlecolor':    '#1a1a1a',
        'axes.titleweight':   'bold',
        'axes.titlepad':      6.0,
        'axes.labelpad':      3.5,
        'axes.spines.top':    True,
        'axes.spines.right':  True,
        'axes.spines.bottom': True,
        'axes.spines.left':   True,
        'axes.prop_cycle':    mpl.cycler(color=[
            COLORS['pred'], COLORS['peak'], COLORS['ofs'], COLORS['stofs'],
            COLORS['accent'], _TOL_BRIGHT['cyan'], COLORS['obs'],
        ]),

        # Ticks — outward, thin.
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'xtick.major.size':   3.0,
        'ytick.major.size':   3.0,
        'xtick.color':        '#333333',
        'ytick.color':        '#333333',

        # Grid — subtle, behind data.
        'axes.grid':          True,
        'axes.axisbelow':     True,
        'grid.color':         '#dddddd',
        'grid.linestyle':     '-',
        'grid.linewidth':     0.4,
        'grid.alpha':         0.8,

        # Lines / markers.
        'lines.linewidth':    1.2,
        'lines.markersize':   3.5,
        'lines.solid_capstyle': 'round',

        # Legend — no border, light background.
        'legend.frameon':     False,
        'legend.borderaxespad': 0.6,
        'legend.handlelength': 1.6,
        'legend.columnspacing': 1.2,

        # Figure / savefig.
        'figure.dpi':         110,
        'figure.facecolor':   'white',
        'figure.titleweight': 'bold',
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.03,
        'savefig.facecolor':  'white',
        'savefig.format':     'png',

        # Image defaults — viridis for sequential maps.
        'image.cmap':         'viridis',
        'image.interpolation': 'nearest',

        # PDF / PS — embed TrueType (Type42) for clean journal PDFs.
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })


# =============================================================================
# HELPERS
# =============================================================================
def savefig(fig, path, formats=('png',), **kwargs):
    """Save a figure in one or more formats with publication defaults."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base, ext = os.path.splitext(path)
    ext = ext.lstrip('.').lower() if ext else None
    fmts = (ext,) if ext else formats
    for fmt in fmts:
        out = f'{base}.{fmt}' if ext is None else path
        fig.savefig(out, format=fmt, **kwargs)


def annotate_panel(ax, letter, x=-0.14, y=1.05):
    """Add a bold panel letter (a, b, c, ...) at axes-relative coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')


def style_axis_clean(ax, grid_axis='y'):
    """Apply the paper-standard closed-frame axis treatment to a single ax."""
    ax.grid(axis=grid_axis, linestyle='-', linewidth=0.4, color='#dddddd', alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right', 'left', 'bottom'):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color('#333333')


def title(ax, text, subtitle=None):
    """Paper-standard bold title with optional light subtitle underneath.

    Uses a fixed point-offset above the axes top so that the title stays
    the same visual distance from the axes regardless of how tall that
    axes is — axes-relative coords broke on the tall per-station panel.
    """
    from matplotlib.transforms import ScaledTranslation
    fig = ax.figure

    title_offset = ScaledTranslation(0.0, 18 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, text,
            transform=ax.transAxes + title_offset,
            fontsize=9.5, fontweight='bold', color='#1a1a1a',
            va='baseline', ha='left')

    if subtitle:
        sub_offset = ScaledTranslation(0.0, 6 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, subtitle,
                transform=ax.transAxes + sub_offset,
                fontsize=7.5, color='#666666',
                va='baseline', ha='left')
        pad_pts = 26
    else:
        pad_pts = 14
    # Reserve vertical space above the axes so the title does not overlap
    # anything above it in a tight_layout / gridspec.
    ax.set_title(' ', pad=pad_pts)


if __name__ == '__main__':
    # Self-test: make one tiny figure to verify the style compiles.
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    import numpy as np
    setup()
    x = np.linspace(0, 48, 200)
    fig, ax = plt.subplots(figsize=FIG['single'])
    ax.plot(x, np.sin(x / 4), color=COLORS['obs'], label='observed')
    ax.plot(x, np.sin(x / 4) + 0.1, color=COLORS['pred'], label='model')
    ax.fill_between(x, np.sin(x / 4) - 0.15, np.sin(x / 4) + 0.15,
                    color=COLORS['envelope'], alpha=0.18, linewidth=0)
    ax.set_xlabel('lead time (h)')
    ax.set_ylabel('residual (m)')
    title(ax, 'Style self-test', subtitle='single-column preset · Tol bright palette')
    ax.legend(loc='upper right')
    savefig(fig, 'outputs/_style_test.png')
    plt.close(fig)
    print('Wrote outputs/_style_test.png')


# =============================================================================
# Paper palette (decided 2026-07-24, Colorgorical + manual picks).
# One hue = one meaning, everywhere. Scripts currently hardcode these
# values; this block is the single source of truth for the polish sweep.
# =============================================================================
PAPER_PALETTE = {
    'ai':        '#208eb7',   # AI — deterministic AND ERA5-driven (never co-plot)
    'stofs':     '#c46fd8',   # STOFS-2D physics
    'ensemble':  '#ffa500',   # AI 93-member ensemble (curves, bars, ribbons)
    'minor':     '#fab899',   # fig-5 reliability, minor (light ensemble sibling)
    'moderate':  '#781486',   # fig-5 reliability, moderate
    'observed':  '#1a1a1a',   # truth = ink (outside the palette)
    'boundary':  '#666666',   # training-max / beyond-support (always dashed)
    'neutral':   '#555555',   # location dots, fig-2a gauge scatter
    # fig-1 map colorbar stays PRGn (locked figure, independent of lines)
}
