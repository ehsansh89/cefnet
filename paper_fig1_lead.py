"""
paper_fig1_lead.py — Figure 1 panel (c) backing data: MAE vs lead time.

Re-runs the STOFS pairing exactly as paper_fig1_process.py does (3-seed-mean AI
forecast, 00/06/12/18Z grid, all regions) but aggregates the absolute errors by
(station, lead_hour) instead of collapsing the lead dimension, so the plot
script can draw a pooled MAE-vs-lead curve with a station-bootstrap CI.

Runs the pairing twice: against the bias-corrected product ('cwl', the panel-c
comparison) and against the raw model output ('cwl_noanomaly') — the raw curve
diagnoses whether the early-lead MAE bump comes from the operational bias
correction tapering off or from the model itself (SI sensitivity).

Gauges are restricted to the exact station set already shipped in
fig1_stofs_skill.csv (same inclusion gate, same Great-Lakes drop) so panel (c)
describes the same population as panels (a)/(b).

Reads:
    same inputs as paper_fig1_process.py (inference archives, STOFS per-station
    files, processed gauges) plus {PROCESS_OUT_DIR}/fig1_stofs_skill.csv
Writes:
    {PROCESS_OUT_DIR}/fig1_lead_mae.csv               (vs 'cwl')
    {PROCESS_OUT_DIR}/fig1_lead_mae_noanomaly.csv     (vs 'cwl_noanomaly')
        station_id, lead_hour, n, ai_mae_cm, stofs_mae_cm

Config: paper_fig1_config.json  (v2: X_PAPER_FIG1_CONFIG=paper_fig1_config_v2.json)
"""

import os
import warnings

import pandas as pd

import paper_fig1_process as p1


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    cfg = p1.load_config()
    p1.WL_H5 = cfg.get('WATER_LEVELS_H5', p1.WL_H5)
    out_dir = cfg.get('PROCESS_OUT_DIR', 'outputs/paper')
    hours = cfg.get('STOFS_CYCLE_HOURS', [0, 6, 12, 18])
    seeds = cfg['SEEDS']

    skill_csv = os.path.join(out_dir, 'fig1_stofs_skill.csv')
    keep = set(pd.read_csv(skill_csv, dtype={'station_id': str})['station_id'])
    print(f"===== Fig 1 (c): MAE vs lead - {len(keep)} gauges from {skill_csv} =====")
    print(f"Observed gauges: {p1.WL_H5}")

    sources = {'cwl': 'fig1_lead_mae.csv',
               'cwl_noanomaly': 'fig1_lead_mae_noanomaly.csv'}
    parts = {s: [] for s in sources}
    for region_id in cfg['REGIONS']:
        fc = p1.seed_mean_fc(cfg, region_id, hours, seeds)
        if fc is None:
            print(f"  region_{region_id:02d}: no archives, skip")
            continue
        for source in sources:
            scfg = dict(cfg, STOFS_SOURCE=source)
            pr = p1.pair_region_stofs(scfg, region_id, fc)
            if pr is None or pr.empty:
                print(f"  region_{region_id:02d} [{source}]: no STOFS pairs")
                continue
            pr = pr[pr['station_id'].astype(str).isin(keep)]
            d = pr.dropna(subset=['ai_residual', 'stofs_residual',
                                  'obs_residual']).copy()
            if d.empty:
                print(f"  region_{region_id:02d} [{source}]: no gated pairs")
                continue
            d['ai_ae'] = (d['ai_residual'] - d['obs_residual']).abs()
            d['ph_ae'] = (d['stofs_residual'] - d['obs_residual']).abs()
            g = (d.groupby(['station_id', 'lead_hour'])
                  .agg(n=('ai_ae', 'size'), ai_mae=('ai_ae', 'mean'),
                       ph_mae=('ph_ae', 'mean'))
                  .reset_index())
            parts[source].append(g)
            print(f"  region_{region_id:02d} [{source}]: "
                  f"gauges={g['station_id'].nunique():3d} "
                  f"pairs={int(g['n'].sum()):,}")

    for source, fname in sources.items():
        if not parts[source]:
            print(f"  [{source}] no pairs produced - nothing to write")
            continue
        allg = pd.concat(parts[source], ignore_index=True)
        allg['ai_mae_cm'] = (100 * allg.pop('ai_mae')).round(3)
        allg['stofs_mae_cm'] = (100 * allg.pop('ph_mae')).round(3)
        out = os.path.join(out_dir, fname)
        allg.to_csv(out, index=False)
        pooled = allg.groupby('lead_hour').apply(
            lambda x: pd.Series({
                'ai': (x['ai_mae_cm'] * x['n']).sum() / x['n'].sum(),
                'ph': (x['stofs_mae_cm'] * x['n']).sum() / x['n'].sum()}))
        print(f"  [{source}] pooled AI MAE: {pooled['ai'].iloc[0]:.2f} cm @1h"
              f" -> {pooled['ai'].iloc[-1]:.2f} cm @72h; "
              f"STOFS {pooled['ph'].iloc[0]:.2f} -> {pooled['ph'].iloc[-1]:.2f} cm"
              f" (peak {pooled['ph'].max():.2f} @ {int(pooled['ph'].idxmax())}h)")
        print(f"  wrote {out} ({len(allg)} station-lead rows)")
