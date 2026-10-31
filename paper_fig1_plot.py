"""
paper_fig1_plot.py — v4.1 Figure 1: one system vs the operational fleet
(writes docs/paper/figures/v4/fig1.png).

Layout (Round 4 revision 2026-07-16, `docs/paper/panel_discussion_on_figures.md`):
  (a) left ~60% width, vertically centered — plain lat/lon CONUS map, white
      land and ocean, every STOFS-matched gauge as a fixed-size dot colored by
      MAE skill (1 − CEFNet/physics; teal = CEFNet better, orange = physics,
      clipped at ±0.6); slim vertical colorbar in the open Atlantic east of
      Florida. No OFS overlays, no annotation text (headline numbers go to
      the manuscript caption — printed to console here).
  (b) top-right — 2-row forest plot (bias-corrected STOFS national; regional
      OFS 7-model suite pooled — per-model rows go to SI): per-gauge points
      colored by skill (same palette as the map), n-weighted pooled skill
      (diamond) with a gauge-bootstrap 95% CI, parity line, per-row
      "pooled MAE (cm): CEFNet / physics" register + n.
  (c) bottom-right (taller) — pooled MAE (cm) vs lead time (1–72 h), CEFNet vs
      STOFS, station-bootstrap CI bands; the approximate STOFS DA window is
      shaded but unlabeled (explained in the caption). (Reserved: persistence
      / tide-only curves join this panel when the simple-baseline runs exist.)

The win is stated once, in one figure. Everything renders from the fig1 CSVs
with zero manual touch-up so the figure regenerates when the final run lands.

Reads:
    {PROCESS_OUT_DIR}/fig1_stofs_skill.csv
    {PROCESS_OUT_DIR}/fig1_ofs_skill.csv
    {PROCESS_OUT_DIR}/fig1_lead_mae.csv      (paper_fig1_lead.py)
Writes:
    {FIG_OUT_DIR}/fig1.png                   (decided 3-panel main figure)
    {FIG_OUT_DIR}/fig1_main_peakamp.png      (SI variant, legacy 2-panel layout)
    {FIG_OUT_DIR}/fig1_main_peakthr.png      (SI variant, legacy 2-panel layout)

Config: paper_fig1_config.json  (v2: X_PAPER_FIG1_CONFIG=paper_fig1_config_v2.json)
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from o_style import setup, savefig, annotate_panel, PAPER_FONT as F
from paper_common import eligible_station_ids


# v2 is the active paper track — default there; set X_PAPER_FIG1_CONFIG to
# override (e.g. back to the v1 paper_fig1_config.json).
CONFIG_PATH = os.environ.get('X_PAPER_FIG1_CONFIG', 'paper_fig1_config_v2.json')
SKILL_CLIP = 0.6          # diverging colour limit on the map (|skill|)
# custom diverging map: teal (positive) = CEFNet better, orange (negative) =
# physics better, white at 0
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    'cefnet_div', ['#ffa500', '#ffffff', '#0fb381'])
N_BOOT = 2000             # gauge-bootstrap resamples for pooled-skill CIs
FOREST_XLIM = (-0.42, 0.92)
AI_COLOR = '#208eb7'      # paper palette 2026-07-24 (AI, det & ERA5)
PHYS_COLOR = '#c46fd8'    # paper palette 2026-07-24 (STOFS-2D)
TEXT = '#1a1a1a'          # all in-panel text black (Round 4 review)

# Legacy SI variants (peak metrics) keep the old stacked map+forest layout.
SI_VARIANTS = [
    {'skill_col': 'peakamp_skill', 'ai_col': 'ai_peakamp_mae_cm',
     'phys_fmt': '{p}_peakamp_mae_cm', 'w_col': 'n_peak_cycles',
     'label': 'peak-amplitude skill (1 − CEFNet / physics)',
     'out': 'fig1_main_peakamp.png'},
    {'skill_col': 'peak_mae_skill', 'ai_col': 'ai_peak_mae_cm',
     'phys_fmt': '{p}_peak_mae_cm', 'w_col': 'n_peak',
     'label': 'peak-MAE skill (1 − CEFNet / physics)',
     'out': 'fig1_main_peakthr.png'},
]
MAIN_VARIANT = {'skill_col': 'mae_skill', 'ai_col': 'ai_mae_cm',
                'phys_fmt': '{p}_mae_cm', 'w_col': 'n_pairs',
                'label': 'MAE skill (1 − CEFNet / physics)', 'out': 'fig1.png'}


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Pooled skill + bootstrap CI (over gauges, n-weighted)
# =============================================================================
def pooled_skill(df, ai_col, phys_col, w_col):
    d = df.dropna(subset=[ai_col, phys_col, w_col])
    d = d[d[w_col] > 0]
    if d.empty:
        return np.nan
    w = d[w_col].to_numpy(float)
    ai = float(np.sum(d[ai_col].to_numpy(float) * w))
    ph = float(np.sum(d[phys_col].to_numpy(float) * w))
    return 1.0 - ai / ph if ph > 0 else np.nan


def pooled_mae_cm(df, col, w_col):
    d = df.dropna(subset=[col, w_col])
    d = d[d[w_col] > 0]
    if d.empty:
        return np.nan
    return float(np.average(d[col].to_numpy(float),
                            weights=d[w_col].to_numpy(float)))


def bootstrap_ci(df, ai_col, phys_col, w_col, n_boot=N_BOOT, seed=0):
    """95% CI of the n-weighted pooled skill, resampling gauges (rows)."""
    d = df.dropna(subset=[ai_col, phys_col, w_col]).reset_index(drop=True)
    d = d[d[w_col] > 0].reset_index(drop=True)
    if len(d) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ai = d[ai_col].to_numpy(float) * d[w_col].to_numpy(float)
    ph = d[phys_col].to_numpy(float) * d[w_col].to_numpy(float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    s = 1.0 - ai[idx].sum(axis=1) / ph[idx].sum(axis=1)
    return tuple(np.percentile(s, [2.5, 97.5]))


# =============================================================================
# Panel (a): money map (Lambert conformal, white land) with OFS hulls
# =============================================================================
def _base_map(ax, ccrs, cfeature):
    ax.set_extent([-125.5, -66, 23.5, 50.5], crs=ccrs.PlateCarree())
    ax.set_facecolor('white')
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#bbbbbb',
                   linewidth=0.18, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black',
                   linewidth=0.3, zorder=2)
    ax.spines['geo'].set_visible(False)


def _map_panel(ax, stofs, ccrs, cfeature, v):
    _base_map(ax, ccrs, cfeature)
    col = v['skill_col']
    d = stofs.dropna(subset=['lat', 'lon', col]).copy()
    # Draw order: highest skill first so near-zero/negative gauges land on top
    # and cannot hide underneath teal neighbours in dense clusters.
    d = d.sort_values(col, ascending=False)
    # Light gray edge (not white): near-zero gauges are almost white in the
    # diverging palette and would vanish on the white land otherwise.
    sc = ax.scatter(d['lon'], d['lat'], c=d[col].clip(-SKILL_CLIP, SKILL_CLIP),
                    s=15, cmap=CMAP, vmin=-SKILL_CLIP, vmax=SKILL_CLIP,
                    edgecolors='black', linewidths=0.1,
                    transform=ccrs.PlateCarree(), zorder=5)
    # Headline numbers go to the manuscript caption, not the figure — print
    # them so caption drafting has the exact values.
    pool = pooled_skill(d, v['ai_col'], v['phys_fmt'].format(p='stofs'),
                        v['w_col'])
    n_better = int((d[col] > 0).sum())
    print(f"  caption numbers [{v['out']}]: {len(d)} gauges vs bias-corrected "
          f"STOFS-2D-Global, pooled skill {pool:+.2f}, AI lower MAE at "
          f"{n_better}/{len(d)} ({n_better / len(d):.0%})")
    # Slim vertical colorbar in the open Atlantic east of Florida/Georgia.
    cax = ax.inset_axes([0.898, 0.008, 0.022, 0.30])
    # extend='max': dot colors are capped at +0.6 but per-gauge skill reaches
    # ~0.87 (visible in panel b) — the arrow tip says so (figure review, Round 4).
    cb = plt.colorbar(sc, cax=cax, orientation='vertical', extend='max')
    cb.set_ticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=F['small'], width=0.5, length=2, color=TEXT,
                      labelcolor=TEXT)
    # Title under the bar (open ocean below). "MAE skill" sits above the skill
    # definition, which renders as a parenthesized stacked fraction one size
    # smaller; the map compares only vs STOFS-2D, so the denominator names that
    # model specifically (panel b pools two physics families, so it keeps the
    # generic "physics").
    cb.ax.text(0.5, -0.13, 'MAE skill', fontsize=F['small'], color=TEXT,
               ha='center', va='top', transform=cb.ax.transAxes)
    cb.ax.text(0.5, -0.34,
               r'$\left(1-\dfrac{\mathrm{CEFNet}}{\mathrm{STOFS\text{-}2D}}\right)$',
               fontsize=5.0, color=TEXT, ha='center', va='top',
               transform=cb.ax.transAxes)
    return sc


# =============================================================================
# Panel (b): forest plot with cm registers (2-row main; 8-row SI)
# =============================================================================
def _forest_rows(stofs, ofs, v, collapse_ofs=True):
    """Row dicts top-to-bottom. Main figure: STOFS + the pooled 7-model OFS
    suite (each gauge matched to exactly one OFS, so pooling counts nothing
    twice; per-model rows go to SI). SI variants keep the per-model rows."""
    # Short display names; the "bias-corrected" / "7-model suite" qualifiers
    # live in the caption (Ehsan, post-Round-4).
    rows = [{'name': 'STOFS-2D', 'df': stofs,
             'phys_col': v['phys_fmt'].format(p='stofs')}]
    ph_o = v['phys_fmt'].format(p='ofs')
    if collapse_ofs:
        rows.append({'name': 'Regional OFS', 'df': ofs, 'phys_col': ph_o})
        return rows
    pooled = []
    for m, d in ofs.groupby('ofs_model'):
        pooled.append((m, pooled_skill(d, v['ai_col'], ph_o, v['w_col']), d))
    pooled.sort(key=lambda t: -(t[1] if np.isfinite(t[1]) else -9))
    for m, _, d in pooled:
        rows.append({'name': m.upper(), 'df': d, 'phys_col': ph_o})
    return rows


def _forest_panel(ax, stofs, ofs, v, registers=True, collapse_ofs=True,
                  rng_seed=0):
    col, ai_col, w_col = v['skill_col'], v['ai_col'], v['w_col']
    rows = _forest_rows(stofs, ofs, v, collapse_ofs=collapse_ofs)
    rng = np.random.default_rng(rng_seed)
    n_rows = len(rows)
    reg_x = FOREST_XLIM[1] + 0.04
    for i, r in enumerate(rows):
        y = n_rows - 1 - i
        d = r['df'].dropna(subset=[col]).copy()
        x = d[col].clip(*FOREST_XLIM).to_numpy(float)
        jit = rng.uniform(-0.17, 0.17, size=len(x))
        # Points colored by their skill on the map's scale — one color meaning
        # across panels (teal = CEFNet better).
        ax.scatter(x, y + jit, s=7, c=d[col].clip(-SKILL_CLIP, SKILL_CLIP),
                   cmap=CMAP, vmin=-SKILL_CLIP, vmax=SKILL_CLIP, alpha=0.85,
                   edgecolors='black', linewidths=0.1, zorder=3)
        pool = pooled_skill(d, ai_col, r['phys_col'], w_col)
        lo, hi = bootstrap_ci(d, ai_col, r['phys_col'], w_col, seed=rng_seed + i)
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=TEXT, lw=1.2, zorder=5,
                    solid_capstyle='butt')
        ax.scatter([pool], [y], marker='D', s=24, facecolor='#d9d9d9',
                   edgecolor=TEXT, linewidth=0.9, zorder=6)
        if registers:
            ai = pooled_mae_cm(d, ai_col, w_col)
            ph = pooled_mae_cm(d, r['phys_col'], w_col)
            ax.text(reg_x, y, f"{ai:.1f} / {ph:.1f} cm",
                    fontsize=F['label_sm'], va='center', ha='left', color=TEXT)
        else:
            ax.text(reg_x, y, f"{pool:+.2f}  (n={len(d)})",
                    fontsize=7, va='center', ha='left', color=TEXT)
    if registers:
        ax.text(reg_x, n_rows - 0.42, 'MAE (cm):\nCEFNet / physics',
                fontsize=F['small'], color=TEXT, ha='left', va='bottom',
                linespacing=1.3)
    # Dashed zero line = parity (equal error); explained in the caption.
    ax.axvline(0.0, color='black', lw=0.5, ls=(0, (4, 2)), zorder=2)
    # Row names sit ABOVE their point clouds (not as inward y-tick labels) so
    # far-left gauges can never collide with them.
    ax.set_yticks([])
    from matplotlib.transforms import ScaledTranslation
    name_tr = (ax.get_yaxis_transform()
               + ScaledTranslation(-8 / 72, 2 / 72, ax.figure.dpi_scale_trans))
    for i, r in enumerate(rows):
        ax.text(0.0, n_rows - 1 - i + 0.26, r['name'], fontsize=7,
                color=TEXT, ha='left', va='bottom', transform=name_tr)
    ax.set_ylim(-0.6, n_rows - 0.4 + 0.35)
    ax.set_xlim(FOREST_XLIM[0], FOREST_XLIM[1] + 0.42)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.5, 0.75])
    ax.set_xlabel(v['label'], fontsize=F['label'], labelpad=1.5, color=TEXT)
    ax.tick_params(axis='x', labelsize=F['tick'], colors=TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=-6)  # labels hug the (hidden) axis
    ax.grid(False)


# =============================================================================
# Panel (c): pooled MAE vs lead time with station-bootstrap CI
# =============================================================================
def _lead_curves(lead_df, n_boot=1000, seed=0):
    """Pooled MAE-vs-lead for AI and STOFS + 95% station-bootstrap bands.
    Returns dict of arrays indexed by sorted lead hours."""
    leads = np.sort(lead_df['lead_hour'].unique())
    piv_n = lead_df.pivot_table(index='station_id', columns='lead_hour',
                                values='n', fill_value=0.0)
    piv_n = piv_n.reindex(columns=leads, fill_value=0.0)
    out = {'lead': leads}
    rng = np.random.default_rng(seed)
    n = piv_n.to_numpy(float)                       # S x L
    idx = rng.integers(0, n.shape[0], size=(n_boot, n.shape[0]))
    for key, col in (('ai', 'ai_mae_cm'), ('ph', 'stofs_mae_cm')):
        piv = lead_df.pivot_table(index='station_id', columns='lead_hour',
                                  values=col, fill_value=0.0)
        piv = piv.reindex(columns=leads, fill_value=0.0)
        s = piv.to_numpy(float) * n                 # abs-error sums, S x L
        out[key] = s.sum(axis=0) / n.sum(axis=0)
        boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)   # n_boot x L
        out[f'{key}_lo'], out[f'{key}_hi'] = np.percentile(boot, [2.5, 97.5],
                                                           axis=0)
    return out


def _lead_panel(ax, lead_df):
    c = _lead_curves(lead_df)
    # Approximate end of the STOFS DA-influenced window — dashed marker line,
    # unlabeled; the caption explains it.
    ax.axvline(12, color='black', lw=0.5, ls=(0, (4, 2)), zorder=1)
    for key, color, label in (('ph', PHYS_COLOR, 'STOFS-2D'),
                              ('ai', AI_COLOR, 'CEFNet')):
        ax.fill_between(c['lead'], c[f'{key}_lo'], c[f'{key}_hi'],
                        color=color, alpha=0.20, lw=0, zorder=2)
        ax.plot(c['lead'], c[key], color=color, lw=1.0, label=label, zorder=3)
    # Reserved: persistence / tide-only baselines join here (dashed gray) once
    # the simple-baseline runs exist.
    top = float(np.nanmax(c['ph_hi']))
    ax.set_xlim(1, 72)
    ax.set_ylim(0, top * 1.10)
    ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax.set_xlabel('lead time (h)', fontsize=F['label'], labelpad=1.5, color=TEXT)
    ax.set_ylabel('MAE (cm)', fontsize=F['label'], labelpad=2, color=TEXT)
    ax.tick_params(labelsize=F['tick'], colors=TEXT)
    ax.legend(fontsize=F['legend'], loc='lower right', frameon=False,
              borderaxespad=0.4, handlelength=1.4, labelcolor=TEXT)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =============================================================================
# Build
# =============================================================================
def _gate(df, lbl, phys, elig):
    n0 = len(df)
    if elig is not None:
        df = df[df['station_id'].astype(str).isin(elig)].copy()
    w = df['n_pairs']
    ai = np.average(df['ai_mae_cm'], weights=w)
    ph = np.average(df[f'{phys}_mae_cm'], weights=w)
    nb = int((df['mae_skill'] > 0).sum())
    print(f"  gate[{lbl}]: kept {len(df)}/{n0} gauges"
          f" | pooled AI={ai:.2f} {phys}={ph:.2f} skill={1 - ai / ph:+.3f}"
          f" | AI-better {nb}/{len(df)}")
    return df


def _render_main(stofs, ofs, lead_df, ccrs, cfeature, fig_dir):
    """Decided 3-panel layout: map left (~60%), forest + lead right."""
    v = MAIN_VARIANT
    fig = plt.figure(figsize=(7.09, 3.15))     # 180 x 80 mm
    # Margins designed into the canvas (tight-crop disabled at save), matching
    # Figure 2: visible side air, minimal top/bottom (Ehsan, Fig-2 Round 4).
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0],
                          height_ratios=[0.95, 1.05],
                          left=0.045, right=0.958, top=0.935, bottom=0.115,
                          wspace=0.238, hspace=0.50)
    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax_for = fig.add_subplot(gs[0, 1])
    ax_lead = fig.add_subplot(gs[1, 1])

    _map_panel(ax_map, stofs, ccrs, cfeature, v)
    _forest_panel(ax_for, stofs, ofs, v, registers=True, collapse_ofs=True)
    if lead_df is not None and not lead_df.empty:
        _lead_panel(ax_lead, lead_df)
    else:
        ax_lead.text(0.5, 0.5, 'lead data pending\n(paper_fig1_lead.py)',
                     ha='center', va='center', fontsize=7, color='#888888',
                     transform=ax_lead.transAxes)
        ax_lead.set_xticks([]); ax_lead.set_yticks([])
    # Thin black frame: 0.5 pt spines and ticks on the data panels.
    for axx in (ax_for, ax_lead):
        for sp in axx.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('black')
        axx.tick_params(width=0.5, color='black')
    # Panel letters in figure coords so a/b/c align regardless of how the
    # projected map shrinks inside its gridspec cell.
    bf, bl = ax_for.get_position(), ax_lead.get_position()
    from matplotlib.transforms import ScaledTranslation

    def _pt_off(dx_pt, dy_pt):
        return (fig.transFigure
                + ScaledTranslation(dx_pt / 72, dy_pt / 72, fig.dpi_scale_trans))
    # nudge in points: a down 12; b right 12 + down 12; c right 12
    fig.text(0.042, bf.y1 - 0.013, 'a', fontsize=F['panel'], fontweight='bold',
             va='bottom', ha='left', transform=_pt_off(0, -12))
    fig.text(bf.x0 - 0.093, bf.y1 - 0.013, 'b', fontsize=F['panel'],
             fontweight='bold', va='bottom', ha='left', transform=_pt_off(12, -12))
    fig.text(bl.x0 - 0.093, bl.y1 + 0.005, 'c', fontsize=F['panel'],
             fontweight='bold', va='bottom', ha='left', transform=_pt_off(12, 0))

    out = os.path.join(fig_dir, v['out'])
    # Keep the full designed canvas — the o_style default tight-crop would
    # trim the side margins right back off. SI variants keep the default.
    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, out, dpi=600)   # main figure at print resolution
    plt.close(fig)
    print(f"  wrote {out}")


def _render_si(stofs, ofs, ccrs, cfeature, v, fig_dir):
    """Legacy stacked layout for the peak-metric SI variants."""
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.15],
                          left=0.13, right=0.97, top=0.99, bottom=0.075,
                          hspace=0.06)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_for = fig.add_subplot(gs[1, 0])
    _map_panel(ax_map, stofs, ccrs, cfeature, v)
    _forest_panel(ax_for, stofs, ofs, v, registers=False, collapse_ofs=False)
    annotate_panel(ax_map, 'a', x=-0.01, y=1.0)
    annotate_panel(ax_for, 'b', x=-0.16, y=1.02)
    out = os.path.join(fig_dir, v['out'])
    savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


def build(cfg):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        print(f"  [skip Fig 1] cartopy unavailable: {e}")
        return

    proc = cfg['PROCESS_OUT_DIR']
    stofs = pd.read_csv(os.path.join(proc, 'fig1_stofs_skill.csv'),
                        dtype={'station_id': str})
    ofs = pd.read_csv(os.path.join(proc, 'fig1_ofs_skill.csv'),
                      dtype={'station_id': str})
    elig = eligible_station_ids(cfg)
    stofs = _gate(stofs, 'STOFS', 'stofs', elig)
    ofs = _gate(ofs, 'OFS', 'ofs', elig)

    lead_path = os.path.join(proc, 'fig1_lead_mae.csv')
    lead_df = None
    if os.path.exists(lead_path):
        lead_df = pd.read_csv(lead_path, dtype={'station_id': str})
        lead_df = lead_df[lead_df['station_id'].isin(set(stofs['station_id']))]
        print(f"  lead curve: {lead_df['station_id'].nunique()} gauges, "
              f"{len(lead_df)} station-lead rows")
    else:
        print(f"  [warn] {lead_path} missing - panel (c) placeholder")

    fig_dir = cfg.get('FIG_OUT_DIR', 'docs/paper/figures')
    os.makedirs(fig_dir, exist_ok=True)
    _render_main(stofs, ofs, lead_df, ccrs, cfeature, fig_dir)
    for v in SI_VARIANTS:
        if v['skill_col'] not in stofs.columns or v['skill_col'] not in ofs.columns:
            print(f"  [skip {v['out']}] column '{v['skill_col']}' missing")
            continue
        _render_si(stofs, ofs, ccrs, cfeature, v, fig_dir)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    setup()
    cfg = load_config()
    os.makedirs(cfg.get('FIG_OUT_DIR', 'docs/paper/figures'), exist_ok=True)
    print('===== Figure 1: one system vs the operational fleet =====')
    build(cfg)
