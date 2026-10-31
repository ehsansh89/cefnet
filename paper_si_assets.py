"""Build all figures and LaTeX table fragments for the SI PDF.

Reads the v2 evaluation outputs (outputs/paper/v2), the v2 station inventory,
the region geometry, and the calibration parameter files, and writes every
Supplementary figure and table into docs/paper/si/.
Gives: si/figs/*.png and si/tables/*.tex, ready for si.tex to include.
"""
import os
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from o_style import setup, PAPER_PALETTE, FIG, savefig
    setup()

    SI = os.path.join('docs', 'paper', 'si')
    FIGS = os.path.join(SI, 'figs')
    TABS = os.path.join(SI, 'tables')
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(TABS, exist_ok=True)

    V2 = os.path.join('outputs', 'paper', 'v2')
    C_AI, C_ST = PAPER_PALETTE['ai'], PAPER_PALETTE['stofs']
    C_MIN, C_MOD = PAPER_PALETTE['minor'], PAPER_PALETTE['moderate']

    # ------------------------------------------------------------------
    # S1 — regional inventory table + domain map
    # ------------------------------------------------------------------
    st = pd.read_csv(os.path.join('data', 'stations_processed_v2', 'stations_processed.csv'))
    with open(os.path.join('GIS', 'regions', 'regions_meta.json')) as f:
        meta = json.load(f)

    regimes = {1: 'Northeast', 2: 'Mid-Atlantic', 3: 'Mid-Atlantic', 4: 'Southeast',
               5: 'Southeast', 6: 'Eastern Gulf', 7: 'Western Gulf', 8: 'West Coast',
               9: 'West Coast', 10: 'Pacific Northwest'}

    rows = []
    for i, m in enumerate(meta, start=1):
        npz = np.load(os.path.join('GIS', 'regions', f'region_{i:02d}',
                                   f'region_{i:02d}_lr_latlon.npz'))
        key = [k for k in npz.files if 'lat' in k.lower()][0]
        shp = npz[key].shape
        dy, dx = (shp[0] - 1) * 16, (shp[1] - 1) * 16
        n_g = int((st['region'] == i).sum())
        assert dx == dy == 1008, f'unexpected LR grid size for region {i}: {dx}x{dy}'
        sub = st[st['region'] == i]
        ids = ', '.join(str(s) for s in sorted(sub['station_id'].astype(str)))
        rows.append((i, regimes[i], n_g, ids))

    with open(os.path.join(TABS, 's1_regions.tex'), 'w') as f:
        f.write("\\begin{tabular}{clcp{7.2cm}}\n\\toprule\n"
                "Region & Coast & Gauges & NOAA gauge IDs \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\\n")
        f.write(f"\\midrule\n\\multicolumn{{2}}{{l}}{{Total}} & {st.shape[0]} & \\\\\n"
                "\\bottomrule\n\\end{tabular}\n")

    # Domain map: same base style as the main text's Fig. 1a (white land,
    # 50m states + coastline), one color per region for box + gauges + label
    import geopandas as gpd
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    REGION_COLORS = ['#4477AA', '#EE6677', '#228833', '#A99A28', '#AA3377',
                     '#2E96C8', '#D8621F', '#43389F', '#2C8A7C', '#882255']
    fig = plt.figure(figsize=(7.6, 4.4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-129.5, -61, 19.5, 51.5], crs=ccrs.PlateCarree())
    ax.set_facecolor('white')
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#bbbbbb',
                   linewidth=0.18, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black',
                   linewidth=0.3, zorder=2)
    ax.spines['geo'].set_visible(False)
    for i in range(1, 11):
        col = REGION_COLORS[i - 1]
        b = gpd.read_file(os.path.join('GIS', 'regions', f'region_{i:02d}',
                                       f'region_{i:02d}_lr_boundary.shp'))
        geom = b.geometry.iloc[0]
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=mcolors.to_rgba(col, 0.05),
                          edgecolor=col, linewidth=1.1, zorder=3)
        c = geom.centroid
        ax.text(c.x, c.y, str(i), transform=ccrs.PlateCarree(), fontsize=9,
                fontweight='bold', color=col, ha='center', va='center', zorder=6)
        sub = st[st['region'] == i]
        ax.scatter(sub['lon'], sub['lat'], transform=ccrs.PlateCarree(), s=9,
                   color=col, zorder=5, linewidths=0.35, edgecolors='white')
    savefig(fig, os.path.join(FIGS, 'fig_s1_domains.png'))
    plt.close(fig)

    # ------------------------------------------------------------------
    # S4 — per-system comparison: pooled table + per-gauge skill strips
    # ------------------------------------------------------------------
    with open(os.path.join(V2, 'fig1_process_summary.json')) as f:
        summ = json.load(f)

    ofs_names = ['cbofs', 'dbofs', 'tbofs', 'ngofs2', 'gomofs', 'sfbofs', 'wcofs']
    labels = {'cbofs': 'CBOFS', 'dbofs': 'DBOFS', 'tbofs': 'TBOFS', 'ngofs2': 'NGOFS2',
              'gomofs': 'GoMOFS', 'sfbofs': 'SFBOFS', 'wcofs': 'WCOFS'}

    def block(name):
        return summ.get('ofs', {}).get(name)

    ofs_g = pd.read_csv(os.path.join(V2, 'fig1_ofs_skill.csv'))
    stofs_g = pd.read_csv(os.path.join(V2, 'fig1_stofs_skill.csv'))

    tab_rows, strip = [], []
    stofs_b = summ.get('stofs', {})
    tab_rows.append(('STOFS-2D', stofs_b.get('n_gauges', ''),
                     stofs_b.get('n_pairs', ''), stofs_b.get('ai_mae_cm', ''),
                     stofs_b.get('stofs_mae_cm', ''), stofs_b.get('mae_skill', '')))
    strip.append(('STOFS-2D', stofs_g['mae_skill'].dropna().values,
                  stofs_b.get('mae_skill', np.nan)))
    for name in ofs_names:
        b = block(name)
        sub_rows = ofs_g[ofs_g['ofs_model'] == name]
        sub = sub_rows['mae_skill'].dropna().values
        if b is None and len(sub) == 0:
            continue
        b = b or {}
        phys = b.get('ofs_mae_cm', b.get(f'{name}_mae_cm', b.get('phys_mae_cm', '')))
        # gauge count: the per-gauge CSV folds a dual-domain gauge's pairs into
        # its primary model's row (Chesapeake City sits in both the CBOFS and
        # DBOFS domains), so infer the per-model count from the pair totals
        n_g = len(sub_rows)
        if b.get('n_pairs') and len(sub_rows):
            per_gauge = sub_rows['n_pairs'].median()
            n_g += max(0, int(round(
                (b['n_pairs'] - sub_rows['n_pairs'].sum()) / per_gauge)))
        tab_rows.append((labels[name], n_g, b.get('n_pairs', ''),
                         b.get('ai_mae_cm', ''), phys, b.get('mae_skill', '')))
        strip.append((labels[name], sub, b.get('mae_skill', np.nan)))

    # pooled-OFS row (pairs-weighted across the seven systems)
    blocks = [block(n) for n in ofs_names if block(n)]
    n_tot = sum(b['n_pairs'] for b in blocks)
    ai_p = sum(b['ai_mae_cm'] * b['n_pairs'] for b in blocks) / n_tot
    ph_p = sum(b['ofs_mae_cm'] * b['n_pairs'] for b in blocks) / n_tot
    tab_rows.append(('All regional OFS (pooled)', ofs_g['station_id'].nunique(), n_tot,
                     round(ai_p, 2), round(ph_p, 2), round(1 - ai_p / ph_p, 3)))

    def fmt(v, nd=2):
        if v == '' or v is None:
            return '--'
        if isinstance(v, (int, np.integer)):
            return f'{v:,}'
        if isinstance(v, float):
            return f'{v:,.{nd}f}' if abs(v) < 10 else f'{v:,.1f}'
        return str(v)

    with open(os.path.join(TABS, 's4_ofs.tex'), 'w') as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n"
                "System & Gauges & Matched pairs & CEFNet MAE (cm) & Physics MAE (cm) & Skill \\\\\n\\midrule\n")
        for r in tab_rows:
            f.write(' & '.join([str(r[0]), fmt(r[1]), fmt(r[2]), fmt(r[3]), fmt(r[4]),
                                fmt(r[5], 3)]) + ' \\\\\n')
        f.write("\\bottomrule\n\\end{tabular}\n")

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for j, (lab, vals, pooled) in enumerate(strip):
        y = np.full(len(vals), j) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
        ax.scatter(vals, y, s=9, color='#8ec6de', edgecolors='black',
                   linewidths=0.15)
        if np.isfinite(pooled):
            ax.scatter([pooled], [j], marker='D', s=34, color='#e8871a',
                       zorder=5, edgecolors='black', linewidths=0.4)
    ax.axvline(0, color='#333333', linewidth=0.7)
    ax.set_yticks(range(len(strip)))
    ax.set_yticklabels([s[0] for s in strip])
    ax.set_xlabel('MAE skill of CEFNet against the physics system (per gauge)')
    ax.invert_yaxis()
    savefig(fig, os.path.join(FIGS, 'fig_s4_ofs.png'))
    plt.close(fig)

    # ------------------------------------------------------------------
    # S5 — STOFS-2D anomaly correction: corrected vs raw, by lead hour
    # ------------------------------------------------------------------
    lead_c = pd.read_csv(os.path.join(V2, 'fig1_lead_mae.csv'))
    lead_r = pd.read_csv(os.path.join(V2, 'fig1_lead_mae_noanomaly.csv'))

    def pool(df, col):
        g = df.groupby('lead_hour')
        return g.apply(lambda d: np.average(d[col], weights=d['n']))

    ai = pool(lead_c, 'ai_mae_cm')
    st_c = pool(lead_c, 'stofs_mae_cm')
    st_r = pool(lead_r, 'stofs_mae_cm')

    fig, ax = plt.subplots(figsize=FIG['double'])
    ax.plot(st_r.index, st_r.values, color=C_ST, linestyle='--',
            label='STOFS-2D, raw (no anomaly correction)')
    ax.plot(st_c.index, st_c.values, color=C_ST, label='STOFS-2D, anomaly-corrected')
    ax.plot(ai.index, ai.values, color=C_AI, label='CEFNet')
    ax.axvline(12, color='#666666', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Forecast lead hour')
    ax.set_ylabel('Pooled MAE (cm)')
    ax.set_xlim(1, 72)
    ax.set_ylim(0, None)
    ax.legend(loc='center right')
    savefig(fig, os.path.join(FIGS, 'fig_s5_lead.png'))
    plt.close(fig)

    # ------------------------------------------------------------------
    # S6 — de-biasing sensitivity table
    # ------------------------------------------------------------------
    deb = pd.read_csv(os.path.join(V2, 'fig1_debias_sensitivity.csv'))
    name_map = {'STOFS': 'STOFS-2D'}
    with open(os.path.join(TABS, 's6_debias.tex'), 'w') as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n"
                "System & Gauges & Skill, trend removal & Skill, datum offset & Difference \\\\\n\\midrule\n")
        for _, r in deb.iterrows():
            nm = name_map.get(r['comparand'], labels.get(r['comparand'], str(r['comparand']).upper()))
            f.write(f"{nm} & {int(r['n_gauges'])} & {r['skill_slr']:.3f} & "
                    f"{r['skill_datum']:.3f} & {r['skill_drop']:+.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------------------------
    # S9 — calibration: gamma table, scoreboard table, reliability figure
    # ------------------------------------------------------------------
    cal_root = os.path.join('outputs', 'uq_postprocessing', 'nn_24lb_72fh_v2',
                            'gefs_psfc_v2_raw', '20240101_20251201')
    lead_bins = ['1-12', '13-24', '25-36', '37-48', '49-72']
    with open(os.path.join(TABS, 's9_gamma.tex'), 'w') as f:
        f.write("\\begin{tabular}{l" + "c" * len(lead_bins) + "}\n\\toprule\n"
                "Region & " + " & ".join(f"{b} h" for b in lead_bins) + " \\\\\n\\midrule\n")
        for i in range(1, 11):
            with open(os.path.join(cal_root, f'region_{i:02d}', 'calibration_v2_params.json')) as g:
                p = json.load(g)
            gam = p['directions']['fit2024_eval2025']['gamma']
            f.write(f"{i} & " + " & ".join(f"{gam[b]:.2f}" for b in lead_bins) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(V2, 'fig2_summary.json')) as f:
        f2 = json.load(f)
    sb = f2['reliability_scoreboard']
    dir_lab = {'fit2024_eval2025': 'Fit 2024, evaluate 2025',
               'fit2025_eval2024': 'Fit 2025, evaluate 2024'}
    with open(os.path.join(TABS, 's9_scoreboard.tex'), 'w') as f:
        f.write("\\begin{tabular}{llccccc}\n\\toprule\n"
                "Split & Severity & Product & Brier & BSS & ECE & Event rate \\\\\n\\midrule\n")
        for d in ['fit2024_eval2025', 'fit2025_eval2024']:
            for th in ['minor', 'moderate']:
                for prod, plab in [('raw', 'raw ensemble'), ('isotonic', 'calibrated')]:
                    b = sb[d][th][prod]
                    f.write(f"{dir_lab[d]} & {th} & {plab} & {b['brier']:.4f} & "
                            f"{b['bss_vs_climatology']:.2f} & {b['ece'] * 100:.1f}\\% & "
                            f"{b['event_rate'] * 100:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    rel = pd.read_csv(os.path.join(V2, 'fig2_reliability.csv'))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for r_i, d in enumerate(['fit2024_eval2025', 'fit2025_eval2024']):
        for c_i, th in enumerate(['minor', 'moderate']):
            ax = axes[r_i, c_i]
            col = C_MIN if th == 'minor' else C_MOD
            ax.plot([0, 1], [0, 1], color='#bbbbbb', linewidth=0.7)
            for prod, ls, lw, a in [('raw', '--', 1.0, 0.65), ('isotonic', '-', 1.4, 1.0)]:
                s = rel[(rel['direction'] == d) & (rel['threshold'] == th)
                        & (rel['product'] == prod) & (rel['n'] >= 100)]
                ax.plot(s['pred_mean'], s['obs_freq'], ls, color=col, linewidth=lw,
                        alpha=a, marker='o', markersize=2.5,
                        label='raw ensemble' if prod == 'raw' else 'calibrated')
                if prod == 'isotonic' and s['obs_freq_lo'].notna().any():
                    ax.fill_between(s['pred_mean'], s['obs_freq_lo'], s['obs_freq_hi'],
                                    color=col, alpha=0.15, linewidth=0)
            ax.set_title(f"{th} flooding $\\cdot$ {dir_lab[d]}", fontsize=8)
            if r_i == 1:
                ax.set_xlabel('Forecast probability')
            if c_i == 0:
                ax.set_ylabel('Observed frequency')
            if r_i == 0 and c_i == 0:
                ax.legend(loc='upper left')
    savefig(fig, os.path.join(FIGS, 'fig_s9_reliability.png'))
    plt.close(fig)

    # ------------------------------------------------------------------
    # S10 — warning statistics table (subset of systems)
    # ------------------------------------------------------------------
    warn = pd.read_csv(os.path.join(V2, 'fig4_table_stats.csv'))
    sys_lab = {'ai_det': 'CEFNet deterministic', 'stofs': 'STOFS-2D'}
    for c in range(10, 100, 10):
        sys_lab[f'ens{c}'] = f'Ensemble, {c}\\% cutoff'

    def _bold_rows(rows):
        """rows: list of (label_sev, label_sys, ev, ca, mi, fa, pod, csi)
        numeric; returns formatted lines with per-column best bolded."""
        def fmt_i(v):
            return f"{v:,}"

        def fmt_f(v):
            return f"{v:.2f}"

        best = {
            3: max(fmt_i(r[3]) for r in rows),
            4: min(rows, key=lambda r: r[4])[4],
            5: min(rows, key=lambda r: r[5])[5],
            6: max(fmt_f(r[6]) for r in rows),
            7: max(fmt_f(r[7]) for r in rows),
        }
        best[3] = max((r[3] for r in rows))
        out = []
        for r in rows:
            cells = [fmt_i(r[2])]
            for idx, v in [(3, r[3]), (4, r[4]), (5, r[5])]:
                t = fmt_i(v)
                if v == best[idx]:
                    t = "\\textbf{" + t + "}"
                cells.append(t)
            for idx, v in [(6, r[6]), (7, r[7])]:
                t = fmt_f(v)
                if t == best[idx]:
                    t = "\\textbf{" + t + "}"
                cells.append(t)
            out.append(f"{r[0]} & {r[1]} & " + " & ".join(cells) + " \\\\\n")
        return out

    with open(os.path.join(TABS, 's10_warning.tex'), 'w') as f:
        f.write("\\begin{tabular}{llcccccc}\n\\toprule\n"
                "Severity & System & Events & Caught & Missed & False & POD & CSI \\\\\n\\midrule\n")
        for th in ['minor', 'moderate', 'major']:
            block = []
            for sy in ['ai_det', 'stofs'] + [f'ens{c}' for c in range(10, 100, 10)]:
                r = warn[(warn['threshold'] == th) & (warn['system'] == sy)]
                if r.empty:
                    continue
                r = r.iloc[0]
                block.append((th if sy == 'ai_det' else '', sys_lab[sy],
                              int(r['events']), int(r['caught']),
                              int(r['missed']), int(r['false_warns']),
                              float(r['POD']), float(r['CSI'])))
            for line in _bold_rows(block):
                f.write(line)
            f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------------------------
    # In-support warning statistics (companion to the s10 table): the same
    # counts restricted to windows with >= 3 independent 1980-2021 training
    # events at the observed peak, using the same step lookup as the Fig-3
    # pipeline (paper_fig5_extrap events_at_level).
    # ------------------------------------------------------------------
    import glob as _glob
    SEAM = pd.Timestamp('2024-05-14T12:00')
    UQ = os.path.join('outputs', 'uq_postprocessing', 'nn_24lb_72fh_v2',
                      'gefs_psfc_v2_raw', '20240101_20251201')
    ev = pd.concat([pd.read_parquet(
        os.path.join(rd, 'decision_events.parquet'),
        columns=['cycle_time', 'station_id', 'threshold', 'obs_event',
                 'obs_valid', 'ai_prob_raw', 'stofs_event', 'stofs_valid'])
        for rd in sorted(_glob.glob(os.path.join(UQ, 'region_*')))],
        ignore_index=True)
    ev['station_id'] = ev['station_id'].astype(str)
    det = pd.read_parquet(os.path.join(V2, 'fig2_warn_detgfs.parquet'))
    det['station_id'] = det['station_id'].astype(str)
    ev = ev.merge(det, on=['cycle_time', 'station_id', 'threshold'], how='left')
    ev = ev[(ev['cycle_time'] >= SEAM) & ev['obs_valid'] & ev['stofs_valid']
            & ev['det_gfs_valid'].fillna(False)]
    common = set.intersection(*[
        set(ev.loc[ev['threshold'] == t, 'station_id'])
        for t in ['minor', 'moderate', 'major']])
    ev = ev[ev['station_id'].isin(common)]

    win = pd.read_parquet(os.path.join(V2, 'fig5_extremes_windows.parquet'),
                          columns=['cycle_time', 'station_id', 'obs_res_max'])
    win['station_id'] = win['station_id'].astype(str)
    sup = pd.read_parquet(os.path.join(V2, 'fig5_extrap_support_levels.parquet'))
    sup['sid'] = sup['sid'].astype(str)

    def _support(g):
        s = sup[sup['sid'] == g['station_id'].iloc[0]].sort_values('level')
        if s.empty:
            return pd.Series(np.nan, index=g.index)
        idx = np.clip(np.searchsorted(s['level'].values,
                                      g['obs_res_max'].values, side='right') - 1,
                      0, len(s) - 1)
        return pd.Series(s['events'].values[idx], index=g.index)

    win['support'] = win.groupby('station_id', group_keys=False).apply(_support)
    ev = ev.merge(win, on=['cycle_time', 'station_id'], how='left')
    ins = ev[ev['support'] >= 3]

    def _wscores(d, warn):
        obs = d['obs_event'].astype(bool)
        h = int((warn & obs).sum())
        m = int((~warn & obs).sum())
        fw = int((warn & ~obs).sum())
        return h, m, fw, h / (h + m), h / (h + m + fw)

    with open(os.path.join(TABS, 's10_insupport.tex'), 'w') as f:
        f.write("\\begin{tabular}{llcccccc}\n\\toprule\n"
                "Severity & System & Events & Caught & Missed & False & POD & CSI \\\\\n\\midrule\n")
        for th in ['minor', 'moderate', 'major']:
            d = ins[ins['threshold'] == th]
            pairs = [('ai_det', d['det_gfs_event'].astype(bool).values),
                     ('stofs', d['stofs_event'].astype(bool).values)]
            pairs += [(f'ens{c}', (d['ai_prob_raw'] >= c / 100).values)
                      for c in range(10, 100, 10)]
            block = []
            for sy, warn in pairs:
                h, m, fw, pod, csi = _wscores(d, pd.Series(warn, index=d.index))
                block.append((th if sy == 'ai_det' else '', sys_lab[sy],
                              h + m, h, m, fw, pod, csi))
            for line in _bold_rows(block):
                f.write(line)
            f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        # provenance: starved-window catches, quoted in the SI caption
        for th in ['minor', 'moderate', 'major']:
            d = ev[(ev['threshold'] == th) & ev['obs_event'].astype(bool)
                   & (ev['support'] < 3)]
            f.write(f"% starved {th}: n={len(d)} "
                    f"caught_ai={int((d['det_gfs_event'] == True).sum())} "
                    f"caught_stofs={int((d['stofs_event'] == True).sum())}\n")

    # ------------------------------------------------------------------
    # Attention-evidence figure (v2 model, six hurricane landfalls):
    # 3 rows x 6 storms, upright in each region's rotated grid frame.
    # Row 1 pressure anomaly + wind vectors, row 2 deepest-block attention
    # with the hourly attention-maximum track, row 3 Grad-CAM importance.
    # ------------------------------------------------------------------
    from scipy.ndimage import zoom as nd_zoom
    from cartopy.io import shapereader

    ATT_DIR = os.path.join('outputs', 'attention', 'nn_24lb_72fh_v2')
    STORM_SEASON = {'idalia': 2023, 'debby': 2024, 'helene': 2024,
                    'ian': 2022, 'milton': 2024, 'beryl': 2024}
    STORM_ORDER = ['idalia', 'debby', 'helene', 'ian', 'milton', 'beryl']
    I_LF = 48

    def _hav(lat1, lon1, lat2, lon2):
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(a))

    def _track(name):
        for fn in ['ibtracs_last3years.csv', 'ibtracs_NA_full.csv']:
            tdf = pd.read_csv(os.path.join('data', 'storm_tracks', fn),
                              skiprows=[1], low_memory=False)
            m = (tdf['NAME'].str.upper() == name.upper()) & \
                (tdf['SEASON'].astype(int) == STORM_SEASON[name])
            if m.any():
                t = pd.to_datetime(tdf.loc[m, 'ISO_TIME'])
                la = pd.to_numeric(tdf.loc[m, 'LAT'], errors='coerce')
                lo = pd.to_numeric(tdf.loc[m, 'LON'], errors='coerce')
                ok = la.notna() & lo.notna()
                t, la, lo = t[ok], la[ok].values, lo[ok].values
                hourly = pd.date_range(t.iloc[0], t.iloc[-1], freq='1h')
                hv = hourly.astype('int64')
                tv = t.astype('int64').values
                return hourly, np.interp(hv, tv, la), np.interp(hv, tv, lo)
        raise RuntimeError(f'no track for {name}')

    _coast = list(shapereader.Reader(shapereader.natural_earth(
        resolution='50m', category='physical', name='coastline')).geometries())

    def _make_T(lat_g, lon_g):
        ii, jj = np.mgrid[0:64, 0:64]
        A = np.column_stack([lon_g.ravel(), lat_g.ravel(),
                             np.ones(lon_g.size)])
        cx = np.linalg.lstsq(A, jj.ravel(), rcond=None)[0]
        cy = np.linalg.lstsq(A, ii.ravel(), rcond=None)[0]
        flipy, flipx = cy[1] < 0, cx[0] < 0

        def T(lat, lon):
            lat = np.asarray(lat, dtype=float)
            lon = np.asarray(lon, dtype=float)
            x = cx[0] * lon + cx[1] * lat + cx[2]
            y = cy[0] * lon + cy[1] * lat + cy[2]
            if flipx:
                x = 63.0 - x
            if flipy:
                y = 63.0 - y
            return x, y
        return T, flipy, flipx

    def _orient(fld, flipy, flipx):
        if flipy:
            fld = np.flip(fld, axis=0)
        if flipx:
            fld = np.flip(fld, axis=1)
        return fld

    def _base(ax, T, hlo, hla, s_meta, coast='white'):
        for geom in _coast:
            for line in getattr(geom, 'geoms', [geom]):
                lon, lat = (np.array(line.coords)[:, 0],
                            np.array(line.coords)[:, 1])
                x, y = T(lat, lon)
                keep = (x > -40) & (x < 104) & (y > -40) & (y < 104)
                if keep.any():
                    ax.plot(np.where(keep, x, np.nan),
                            np.where(keep, y, np.nan),
                            color=coast, linewidth=0.6, zorder=4)
        x, y = T(hla, hlo)
        ax.plot(x, y, color='#d62828', linewidth=1.3, zorder=5)
        lx, ly = T(s_meta['storm']['landfall_lat'],
                   s_meta['storm']['landfall_lon'])
        ax.plot([lx], [ly], marker='x', color='#d62828', markersize=6,
                mew=1.6, zorder=6)
        x0, y0 = T(s_meta['storm']['landfall_lat'],
                   s_meta['storm']['landfall_lon'])
        x1, y1 = T(s_meta['storm']['landfall_lat'] + 1.0,
                   s_meta['storm']['landfall_lon'])
        nv = np.array([x1 - x0, y1 - y0])
        nv = nv / np.linalg.norm(nv) * 7.5
        ax.annotate('N', xy=(6 + nv[0], 55 + nv[1]), xytext=(6, 55),
                    arrowprops=dict(arrowstyle='-|>', color='0.25', lw=1.4),
                    color='0.25', fontsize=11, ha='center', va='center',
                    annotation_clip=False)
        ax.set_xlim(-0.5, 63.5)
        ax.set_ylim(-0.5, 63.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    att_ok = all(os.path.exists(os.path.join(ATT_DIR, s, 'cbam_attention.npz'))
                 for s in STORM_ORDER)
    if att_ok:
        figA = plt.figure(figsize=(16.6, 8.6), constrained_layout=True)
        rows = {0: [], 1: [], 2: []}
        imsA = {}
        sc_ref = None
        XX, YY = np.meshgrid(np.arange(64), np.arange(64))
        for k, sname in enumerate(STORM_ORDER):
            sdir = os.path.join(ATT_DIR, sname)
            zz = np.load(os.path.join(sdir, 'cbam_attention.npz'))
            s_meta = json.load(open(os.path.join(sdir, 'storm_meta.json')))
            cyc = pd.Timestamp(s_meta['cycle_time_utc'])
            lf = pd.Timestamp(s_meta['landfall_time_utc'])
            lat_g, lon_g = zz['lat_grid'], zz['lon_grid']
            T, flipy, flipx = _make_T(lat_g, lon_g)
            Uw, Vw = zz['lr_descaled'][I_LF, 0], zz['lr_descaled'][I_LF, 1]
            psfc = zz['lr_descaled'][:, 2]
            panom = psfc[I_LF] - psfc.mean(axis=0)
            tt, hla, hlo = _track(sname)

            def _eyes(layer, mode):
                a = zz[layer][:, 0]
                ahw = a.shape[-1]
                fb = 64 // ahw
                cla = lat_g.reshape(ahw, fb, ahw, fb).mean(axis=(1, 3))
                clo = lon_g.reshape(ahw, fb, ahw, fb).mean(axis=(1, 3))
                pts = []
                for i in range(a.shape[0]):
                    t = cyc + pd.Timedelta(hours=i - 24)
                    if abs((t - lf).total_seconds()) > 24 * 3600:
                        continue
                    j = np.searchsorted(tt, t)
                    if j <= 0 or j >= len(tt):
                        continue
                    sl, so = hla[j], hlo[j]
                    if _hav(cla, clo, sl, so).min() > 25:
                        continue
                    pick = np.argmin if mode == 'min' else np.argmax
                    idx = np.unravel_index(pick(a[i]), a[i].shape)
                    pts.append(((t - lf).total_seconds() / 3600.0,
                                cla[idx], clo[idx], sl, so))
                pts = np.array(pts)
                m = np.median(_hav(pts[:, 1], pts[:, 2],
                                   pts[:, 3], pts[:, 4]))
                return pts, m

            eyes1, med1 = _eyes('sp_layer1', 'min')
            eyes3, med3 = _eyes('sp_layer3', 'max')

            ax = figA.add_subplot(3, 6, k + 1)
            imsA['p'] = ax.pcolormesh(XX, YY, _orient(panom, flipy, flipx),
                                      cmap='RdBu_r', vmin=-22, vmax=22,
                                      shading='gouraud')
            stq = 6
            bx, by = T(lat_g[::stq, ::stq], lon_g[::stq, ::stq])
            sv = 0.02
            tipx, tipy = T(lat_g[::stq, ::stq] + Vw[::stq, ::stq] * sv,
                           lon_g[::stq, ::stq] + Uw[::stq, ::stq] * sv /
                           np.cos(np.radians(lat_g[::stq, ::stq])))
            ax.quiver(bx, by, tipx - bx, tipy - by, color='0.35', scale=28,
                      width=0.004)
            _base(ax, T, hlo, hla, s_meta, coast='black')
            ax.set_title(f"{sname.capitalize()} ({STORM_SEASON[sname]})",
                         fontsize=16)
            if k == 0:
                ax.set_ylabel('pressure anomaly\n+ wind vectors',
                              fontsize=13)
            rows[0].append(ax)

            ax = figA.add_subplot(3, 6, k + 7)
            amap = zz['sp_layer1'][I_LF, 0]
            amap = (amap - amap.min()) / (amap.max() - amap.min())
            big = np.clip(nd_zoom(amap, 2, order=3), 0, 1)
            imsA['a'] = ax.pcolormesh(XX, YY, _orient(big, flipy, flipx),
                                      cmap='cividis', vmin=0, vmax=1,
                                      shading='gouraud')
            _base(ax, T, hlo, hla, s_meta)
            ex, ey = T(eyes1[:, 1], eyes1[:, 2])
            ax.scatter(ex, ey, color='white', s=26, edgecolors='black',
                       linewidths=0.7, zorder=6)
            ax.set_title(f"{med1:.0f} km, {len(eyes1)} h", fontsize=13)
            if k == 0:
                ax.set_ylabel('attention\n(block 1)', fontsize=13)
            rows[1].append(ax)

            ax = figA.add_subplot(3, 6, k + 13)
            amap3 = zz['sp_layer3'][I_LF, 0]
            amap3 = (amap3 - amap3.min()) / (amap3.max() - amap3.min())
            big3 = np.clip(nd_zoom(amap3, 4, order=3), 0, 1)
            imsA['a'] = ax.pcolormesh(XX, YY, _orient(big3, flipy, flipx),
                                      cmap='cividis', vmin=0, vmax=1,
                                      shading='gouraud')
            _base(ax, T, hlo, hla, s_meta)
            ex, ey = T(eyes3[:, 1], eyes3[:, 2])
            ax.scatter(ex, ey, color='white', s=26, edgecolors='black',
                       linewidths=0.7, zorder=6)
            ax.set_title(f"{med3:.0f} km, {len(eyes3)} h", fontsize=13)
            if k == 0:
                ax.set_ylabel('attention\n(block 3)', fontsize=13)
            rows[2].append(ax)

        cbp = figA.colorbar(imsA['p'], ax=rows[0], shrink=0.85, pad=0.01)
        cbp.set_label('pressure anomaly (hPa)', fontsize=13)
        cbp.ax.tick_params(labelsize=11)
        cba = figA.colorbar(imsA['a'], ax=rows[1] + rows[2], shrink=0.5,
                            pad=0.01)
        cba.set_label('attention (normalized)', fontsize=13)
        cba.ax.tick_params(labelsize=11)
        figA.savefig(os.path.join(FIGS, 'fig_s2_attention.png'), dpi=200)
        plt.close(figA)
    else:
        print('MISSING: attention captures for fig_s2_attention')

    print('SI assets written to', SI)
