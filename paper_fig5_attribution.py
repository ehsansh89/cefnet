"""
paper_fig5.py — Figure 5 (story v2): the anatomy of the remaining error.

The weather-swap experiment (referendum item R6): the SAME v2 model (seed_1) is
scored twice over matched cycles — once forced by the operational GFS archive
(what was available at forecast time), once by ERA5 (the weather that actually
happened). Error with ERA5 = the model+observation floor; the GFS−ERA5 gap =
error inherited from the atmospheric forecast, unfixable by any surge model.

Initial build: two panels (the triptych and the variance split join later).
  (a) CONUS map of per-gauge MAE(GFS) − MAE(ERA5) in cm — red = input-limited
      (error sheds under perfect weather), ~0 = model-limited.
  (b) Pooled MAE vs lead hour, both forcings: the flat ERA5 curve is the floor;
      the shaded gap above it is the atmospheric (inherited) band; the early-
      lead dip is the 24 h observation-lookback benefit decaying.

Compute is cached: per-gauge and per-lead tables are written next to the fig1
tables and reused on re-run (delete them or set FORCE below to recompute).

Reads:
    outputs/inference/{run}/region_XX/{era5_span}/forecast.nc        (ERA5, seed_1)
    outputs/paper/inference/{run}/region_XX/gfs_00z/seed_1/forecast.nc (GFS)
    {WATER_LEVELS_H5}  (v2 observed residuals)
    {PROCESS_OUT_DIR}/fig1_stofs_skill.csv   (scored-gauge set + lat/lon)
Writes:
    {PROCESS_OUT_DIR}/fig5_attribution_gauge.csv
    {PROCESS_OUT_DIR}/fig5_attribution_lead.csv
    {PROCESS_OUT_DIR}/fig5_attribution_summary.json
    {FIG_OUT_DIR}/fig5_anatomy.png

Config: paper_fig1_config_v2.json (hardcoded — v2 is the paper namespace).
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from o_style import setup, COLORS, savefig, annotate_panel
from paper_common import eligible_station_ids


CONFIG_PATH = 'paper_fig1_config_v2.json'
ERA5_SPAN = '20240101_20251130'
ERA5_ROOT = 'outputs/inference'          # deterministic ERA5 runs (seed_1)
FORCE = False                            # True -> recompute even if cached

MAP_CMAP = 'RdBu_r'                      # red (positive) = input-limited


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Compute: matched-cycle ERA5-vs-GFS errors
# =============================================================================
def _station_ids(ds):
    ids = ds['station'].values
    return [s.decode() if isinstance(s, bytes) else str(s) for s in ids]


def _load_obs_matrix(h5_path, station_ids, hourly):
    """(T, S) observed residual matrix on the hourly grid, NaN where absent."""
    obs = np.full((len(hourly), len(station_ids)), np.nan, dtype=np.float32)
    with pd.HDFStore(h5_path, 'r') as h:
        for j, sid in enumerate(station_ids):
            key = f'/station_{sid}'
            if key not in h:
                continue
            r = h[key]['residual']
            if getattr(r.index, 'tz', None) is not None:
                r.index = r.index.tz_localize(None)
            obs[:, j] = r.reindex(hourly).values
    return obs


def _region_errors(cfg, region, gauge_set):
    """Per-(cycle, lead, station) absolute errors for both forcings on matched
    samples. Returns None if either archive is missing."""
    import xarray as xr
    run = cfg['RUN_NAME']
    p_era = os.path.join(ERA5_ROOT, run, f'region_{region:02d}', ERA5_SPAN,
                         'forecast.nc')
    p_gfs = os.path.join(cfg['PAPER_INFER_DIR'], run, f'region_{region:02d}',
                         'gfs_00z', 'seed_1', 'forecast.nc')
    if not (os.path.exists(p_era) and os.path.exists(p_gfs)):
        print(f"  [skip region {region}] missing archive "
              f"({'ERA5' if not os.path.exists(p_era) else 'GFS'})")
        return None

    ds_e = xr.open_dataset(p_era)
    ds_g = xr.open_dataset(p_gfs)
    ds_e = ds_e.assign_coords(station=_station_ids(ds_e))
    ds_g = ds_g.assign_coords(station=_station_ids(ds_g))

    cyc = np.intersect1d(ds_e['cycle_time'].values, ds_g['cycle_time'].values)
    st = [s for s in ds_e['station'].values.tolist()
          if s in set(ds_g['station'].values.tolist()) and s in gauge_set]
    if len(cyc) == 0 or len(st) == 0:
        print(f"  [skip region {region}] no matched cycles/stations")
        return None

    e = ds_e['residual_mean'].sel(cycle_time=cyc, station=st).values  # (C,L,S)
    g = ds_g['residual_mean'].sel(cycle_time=cyc, station=st).values
    leads = ds_e['lead_hour'].values.astype(int)
    ds_e.close(); ds_g.close()

    t0 = pd.Timestamp(cyc.min())
    t1 = pd.Timestamp(cyc.max()) + pd.Timedelta(hours=int(leads.max()) + 1)
    hourly = pd.date_range(t0, t1, freq='h')
    obs = _load_obs_matrix(cfg['WATER_LEVELS_H5'], st, hourly)

    cyc_pos = ((cyc - hourly[0].to_datetime64())
               / np.timedelta64(1, 'h')).astype(int)          # (C,)
    valid_pos = cyc_pos[:, None] + leads[None, :]             # (C, L)
    o = obs[valid_pos, :]                                     # (C, L, S)

    mask = np.isfinite(e) & np.isfinite(g) & np.isfinite(o)
    err_e = np.where(mask, np.abs(e - o), np.nan)
    err_g = np.where(mask, np.abs(g - o), np.nan)
    peak = mask & (np.abs(o) >= float(cfg.get('PEAK_THRESHOLD_M', 0.30)))
    print(f"  region {region}: {len(cyc)} matched cycles, {len(st)} gauges, "
          f"{int(mask.sum()):,} samples ({int(peak.sum()):,} peak)")
    return dict(stations=st, leads=leads, err_e=err_e, err_g=err_g,
                obs=o, mask=mask, peak=peak, region=region)


def compute(cfg):
    """Aggregate per-gauge and per-lead tables over all regions."""
    stofs = pd.read_csv(os.path.join(cfg['PROCESS_OUT_DIR'],
                                     'fig1_stofs_skill.csv'),
                        dtype={'station_id': str})
    elig = eligible_station_ids(cfg)
    scored = stofs[stofs['is_gl'] == 0]
    if elig is not None:
        scored = scored[scored['station_id'].isin(elig)]
    gauge_set = set(scored['station_id'])
    meta = scored.set_index('station_id')[['lat', 'lon', 'name']]

    gauge_rows, lead_acc = [], {}
    pool = dict(ae=0.0, ag=0.0, n=0, ae_pk=0.0, ag_pk=0.0, n_pk=0)
    for region in cfg['REGIONS']:
        r = _region_errors(cfg, region, gauge_set)
        if r is None:
            continue
        err_e, err_g, peak, mask = r['err_e'], r['err_g'], r['peak'], r['mask']
        # --- per gauge ---
        for j, sid in enumerate(r['stations']):
            m = mask[:, :, j]
            n = int(m.sum())
            if n == 0:
                continue
            npk = int(peak[:, :, j].sum())
            row = dict(
                station_id=sid, region_id=region, n=n,
                era5_mae_cm=100 * np.nanmean(err_e[:, :, j]),
                gfs_mae_cm=100 * np.nanmean(err_g[:, :, j]),
                n_peak=npk,
                era5_peak_mae_cm=(100 * np.nanmean(err_e[:, :, j][peak[:, :, j]])
                                  if npk else np.nan),
                gfs_peak_mae_cm=(100 * np.nanmean(err_g[:, :, j][peak[:, :, j]])
                                 if npk else np.nan),
            )
            row['delta_cm'] = row['gfs_mae_cm'] - row['era5_mae_cm']
            gauge_rows.append(row)
        # --- per lead (pooled over gauges/cycles) ---
        for li, lead in enumerate(r['leads']):
            a = lead_acc.setdefault(int(lead), dict(ae=0.0, ag=0.0, n=0))
            m = mask[:, li, :]
            a['ae'] += float(np.nansum(err_e[:, li, :]))
            a['ag'] += float(np.nansum(err_g[:, li, :]))
            a['n'] += int(m.sum())
        # --- pooled + peak-pooled ---
        pool['ae'] += float(np.nansum(err_e)); pool['ag'] += float(np.nansum(err_g))
        pool['n'] += int(mask.sum())
        pool['ae_pk'] += float(np.nansum(err_e[peak]))
        pool['ag_pk'] += float(np.nansum(err_g[peak]))
        pool['n_pk'] += int(peak.sum())

    gauges = pd.DataFrame(gauge_rows)
    gauges = gauges.join(meta, on='station_id')
    lead = pd.DataFrame([
        dict(lead_hour=k, n=v['n'],
             era5_mae_cm=100 * v['ae'] / v['n'],
             gfs_mae_cm=100 * v['ag'] / v['n'])
        for k, v in sorted(lead_acc.items()) if v['n'] > 0])

    era5_mae = 100 * pool['ae'] / pool['n']
    gfs_mae = 100 * pool['ag'] / pool['n']
    era5_pk = 100 * pool['ae_pk'] / pool['n_pk'] if pool['n_pk'] else np.nan
    gfs_pk = 100 * pool['ag_pk'] / pool['n_pk'] if pool['n_pk'] else np.nan
    early = lead[lead['lead_hour'] <= 12]['era5_mae_cm'].mean()
    late = lead[lead['lead_hour'] >= 61]['era5_mae_cm'].mean()
    summary = dict(
        n_samples=pool['n'], n_peak=pool['n_pk'], n_gauges=len(gauges),
        pooled_era5_mae_cm=round(era5_mae, 3),
        pooled_gfs_mae_cm=round(gfs_mae, 3),
        atmospheric_share_of_mae=round(1 - era5_mae / gfs_mae, 4),
        peak_era5_mae_cm=round(era5_pk, 3), peak_gfs_mae_cm=round(gfs_pk, 3),
        peak_closure_fraction=round(1 - era5_pk / gfs_pk, 4),
        era5_lead_flatness_cm=dict(lead_01_12=round(early, 3),
                                   lead_61_72=round(late, 3),
                                   rise=round(late - early, 3)),
        note=('same v2 model seed_1, matched 00Z cycles; residual space; '
              'peak = |obs residual| >= PEAK_THRESHOLD_M'),
    )
    return gauges, lead, summary


# =============================================================================
# Plot
# =============================================================================
def _base_map(ax, ccrs, cfeature):
    ax.set_extent([-128, -65, 20, 55], crs=ccrs.PlateCarree())
    ax.set_facecolor('white')
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#888888',
                   linewidth=0.18, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black',
                   linewidth=0.25, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render(cfg, gauges, lead, summary):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        print(f"  [skip Fig 5] cartopy unavailable: {e}")
        return

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.9, 1.25],
                          left=0.10, right=0.97, top=0.99, bottom=0.08,
                          hspace=0.14)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_ld = fig.add_subplot(gs[1, 0])

    # ---- (a) map of GFS − ERA5 MAE ----
    _base_map(ax_map, ccrs, cfeature)
    d = gauges.dropna(subset=['lat', 'lon', 'delta_cm'])
    vlim = max(0.5, float(np.nanpercentile(np.abs(d['delta_cm']), 98)))
    sc = ax_map.scatter(d['lon'], d['lat'], c=d['delta_cm'],
                        s=14 + 46 * np.sqrt(d['n'] / d['n'].max()),
                        cmap=MAP_CMAP, vmin=-vlim, vmax=vlim,
                        edgecolors='black', linewidths=0.12,
                        transform=ccrs.PlateCarree(), zorder=5)
    frac_in = float((d['delta_cm'] > 0).mean())
    ax_map.text(0.985, 0.06,
                f"{len(d)} gauges · same model, weather swapped\n"
                f"error sheds under perfect weather at {frac_in:.0%}",
                transform=ax_map.transAxes, ha='right', va='bottom',
                fontsize=7, color='#1a1a1a',
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
    annotate_panel(ax_map, 'a', x=-0.01, y=1.0)
    pm = ax_map.get_position()
    cax = fig.add_axes([pm.x0 + 0.015, pm.y0 + 0.04, 0.010, pm.height * 0.42])
    cb = fig.colorbar(sc, cax=cax, orientation='vertical')
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.ax.set_title('GFS−ERA5\nMAE (cm)', fontsize=6, pad=3)

    # ---- (b) MAE vs lead, floor + inherited band ----
    x = lead['lead_hour'].to_numpy(float)
    ye = lead['era5_mae_cm'].to_numpy(float)
    yg = lead['gfs_mae_cm'].to_numpy(float)
    ax_ld.fill_between(x, 0, ye, color='#BBBBBB', alpha=0.35, zorder=1,
                       label='model + observation floor (ERA5-forced)')
    ax_ld.fill_between(x, ye, yg, color='#EE6677', alpha=0.30, zorder=2,
                       label='inherited from the weather forecast')
    ax_ld.plot(x, ye, color='#555555', lw=1.3, zorder=3)
    ax_ld.plot(x, yg, color='#B0413E', lw=1.3, zorder=3)
    ax_ld.text(x[-1] - 1, ye[-1] - 0.32, 'ERA5-forced (perfect weather)',
               fontsize=6.5, ha='right', va='top', color='#555555')
    ax_ld.text(x[-1] - 1, yg[-1] + 0.12, 'GFS-forced (operational)',
               fontsize=6.5, ha='right', va='bottom', color='#B0413E')
    fl = summary['era5_lead_flatness_cm']
    ax_ld.text(0.02, 0.96,
               f"floor: {fl['lead_01_12']:.1f} cm (1–12 h) → "
               f"{fl['lead_61_72']:.1f} cm (61–72 h)\n"
               f"peak samples (|obs| ≥ {cfg.get('PEAK_THRESHOLD_M', 0.3):.2f} m): "
               f"GFS {summary['peak_gfs_mae_cm']:.1f} → "
               f"ERA5 {summary['peak_era5_mae_cm']:.1f} cm "
               f"({summary['peak_closure_fraction']:.0%} closes)",
               transform=ax_ld.transAxes, ha='left', va='top', fontsize=7,
               color='#1a1a1a')
    ax_ld.set_xlim(1, 72)
    ax_ld.set_ylim(0, max(yg) * 1.28)
    ax_ld.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax_ld.set_xlabel('lead time (h)')
    ax_ld.set_ylabel('MAE (cm)')
    ax_ld.legend(loc='lower right', fontsize=6.5, frameon=False)
    ax_ld.spines['top'].set_visible(False)
    ax_ld.spines['right'].set_visible(False)
    ax_ld.grid(False)
    annotate_panel(ax_ld, 'b', x=-0.10, y=1.03)

    fig_dir = cfg.get('FIG_OUT_DIR', 'docs/paper/figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, 'fig5_anatomy.png')
    savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    setup()
    cfg = load_config()
    out_dir = cfg['PROCESS_OUT_DIR']
    p_gauge = os.path.join(out_dir, 'fig5_attribution_gauge.csv')
    p_lead = os.path.join(out_dir, 'fig5_attribution_lead.csv')
    p_sum = os.path.join(out_dir, 'fig5_attribution_summary.json')

    print('===== Figure 5 (v2 story): anatomy of the remaining error =====')
    if FORCE or not (os.path.exists(p_gauge) and os.path.exists(p_lead)
                     and os.path.exists(p_sum)):
        gauges, lead, summary = compute(cfg)
        gauges.to_csv(p_gauge, index=False)
        lead.to_csv(p_lead, index=False)
        with open(p_sum, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  cached tables -> {out_dir}")
    else:
        gauges = pd.read_csv(p_gauge, dtype={'station_id': str})
        lead = pd.read_csv(p_lead)
        with open(p_sum) as f:
            summary = json.load(f)
        print("  using cached tables (set FORCE=True to recompute)")

    print(f"  R6 verdict: pooled ERA5 {summary['pooled_era5_mae_cm']:.2f} vs "
          f"GFS {summary['pooled_gfs_mae_cm']:.2f} cm "
          f"(atmospheric share {summary['atmospheric_share_of_mae']:.1%}) | "
          f"peak closure {summary['peak_closure_fraction']:.1%} | "
          f"floor rise {summary['era5_lead_flatness_cm']['rise']:+.2f} cm over 72 h")
    render(cfg, gauges, lead, summary)
