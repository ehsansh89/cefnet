"""
paper_fig1_process.py — Figure 1 (AI beats physics): pairing + per-gauge skill.

Turns the Phase-A/B inference archives into figure-ready per-gauge skill tables:

  1. Forms the deterministic AI forecast as the **3-seed mean** of residual_mean
     (single-seed is computed too, for a robustness line).
  2. Pairs AI vs **STOFS-2D-Global** on the 00/06/12/18Z grid (cwl + slr_detrend),
     all 10 regions.
  3. Pairs AI vs each **regional OFS** using the catalog
     (cbofs/tbofs @ 00/06/12/18, ngofs2 @ 03/09/15/21; datum_offset), reusing the
     SAME archives where hours coincide.
  4. Drops Great Lakes gauges (is_gl), keeps coastal/estuary gauges (flags riverine
     placeholder = none available in metadata), and writes per-gauge tables.

Reuses the tested pairing logic from x_stofs_pair.py / x_ofs_pair.py — one pairing
code path, fed the seed-mean forecast.

Reads:
    outputs/paper/inference/{run}/region_XX/gfs_{HH}z/seed_Y/forecast.nc
    outputs/paper/inference/{run}/needed_cycles.json   (catalog)
    data/stofs/per_station/{sid}__stofs2d.nc
    data/ofs/per_station/{sid}__{model}.nc
    data/water_levels_processed/processed_water_levels.h5
    data/ofs/datum_offsets.csv
    data/stations_processed/stations_processed.csv

Writes:
    outputs/paper/fig1_stofs_skill.csv     (per gauge, all regions)
    outputs/paper/fig1_ofs_skill.csv       (per gauge x ofs model)
    outputs/paper/fig1_process_summary.json

Config: paper_fig1_config.json
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from x_stofs_pair import build_pairs_for_station as stofs_pairs
from x_ofs_pair import build_pairs_for_station as ofs_pairs, load_obs_for_station


CONFIG_PATH = os.environ.get('X_PAPER_FIG1_CONFIG', 'paper_fig1_config_v2.json')
# Default observed-gauge store (v1). Rebound in __main__ from cfg['WATER_LEVELS_H5']
# so the v2 config routes pairing onto the v2-detrended gauges. The pairing
# functions read this as a module global, so rebinding it at module scope (inside
# __main__) before they're called is sufficient.
WL_H5 = 'data/water_levels_processed/processed_water_levels.h5'


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Seed-mean forecast assembly
# =============================================================================
def seed_mean_fc(cfg, region_id, hours, seeds):
    """Build one in-memory forecast Dataset whose residual_mean is the mean over
    `seeds`, concatenated across the given cycle `hours`. Returns None if nothing
    on disk."""
    run = cfg['RUN_NAME']
    base = cfg['PAPER_INFER_DIR']
    parts = []
    for h in hours:
        opened = []
        for s in seeds:
            p = os.path.join(base, run, f'region_{region_id:02d}',
                             f'gfs_{h:02d}z', f'seed_{s}', 'forecast.nc')
            if os.path.exists(p):
                opened.append(xr.open_dataset(p))
        if not opened:
            continue
        rm = (sum(d.residual_mean for d in opened) / len(opened)).load()
        src = opened[0]
        fc_h = xr.Dataset(
            {'residual_mean': rm},
            coords={'cycle_time': src.cycle_time.values,
                    'lead_hour': src.lead_hour.values,
                    'station': src.station.values},
        )
        for d in opened:
            d.close()
        parts.append(fc_h)
    if not parts:
        return None
    return xr.concat(parts, dim='cycle_time')


# =============================================================================
# Pairing
# =============================================================================
def pair_region_stofs(cfg, region_id, fc):
    """All (AI, STOFS, obs) pairs for a region's gauges."""
    source = cfg.get('STOFS_SOURCE', 'cwl')
    method = cfg.get('STOFS_RESIDUAL_METHOD', 'slr_detrend')
    rows = []
    sids = [str(s) for s in fc.station.values]
    with pd.HDFStore(WL_H5, 'r') as wl:
        for sid in sids:
            pth = f'data/stofs/per_station/{sid}__stofs2d.nc'
            if not os.path.exists(pth):
                continue
            obs = load_obs_for_station(wl, sid)
            if obs is None:
                continue
            df = stofs_pairs(fc, pth, obs, sid, source=source, method=method)
            if df is not None:
                rows.append(df)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    # STOFS product seam: NOAA's v2.1.7 anomaly correction starts 2024-05-14
    # 12Z; before it the archived product is a different (uncorrected) model.
    # STOFS_MIN_CYCLE in the config drops the pre-correction era so the
    # comparison is against the current operational product only.
    min_cyc = cfg.get('STOFS_MIN_CYCLE')
    if min_cyc:
        out = out[out['cycle_time'] >= pd.Timestamp(min_cyc)]
    out['region_id'] = region_id
    return out


def pair_region_ofs(cfg, region_id, ofs_model, fc, datum, method):
    """All (AI, OFS, obs) pairs for a region's gauges against one OFS model.
    `method` is chosen per-model by the caller (datum_offset where a datum
    bridge exists, else slr_detrend)."""
    rows = []
    sids = [str(s) for s in fc.station.values]
    with pd.HDFStore(WL_H5, 'r') as wl:
        for sid in sids:
            pth = f'data/ofs/per_station/{sid}__{ofs_model}.nc'
            if not os.path.exists(pth):
                continue
            off = None
            if method == 'datum_offset':
                r = datum[datum['station_id'] == sid]
                if r.empty:
                    continue
                off = float(r['offset'].iloc[0])
            obs = load_obs_for_station(wl, sid)
            if obs is None:
                continue
            df = ofs_pairs(fc, pth, obs, sid, method=method, datum_offset=off)
            if df is not None:
                rows.append(df)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out['region_id'] = region_id
    out['ofs_model'] = ofs_model
    return out


# =============================================================================
# Per-gauge skill
# =============================================================================
def per_gauge_skill(pairs, phys_resid_col, phys_name, peak_thr):
    """Per-station skill from a pairs frame. Three flavours:
      - overall MAE skill (all matched (cycle, lead) samples);
      - peak MAE skill (threshold-conditioned: timesteps with |obs| >= peak_thr);
      - per-cycle peak-amplitude skill (see per_gauge_peak_amp), merged in.
    """
    d = pairs.dropna(subset=['ai_residual', phys_resid_col, 'obs_residual']).copy()
    d['ai_ae'] = (d['ai_residual'] - d['obs_residual']).abs()
    d['ph_ae'] = (d[phys_resid_col] - d['obs_residual']).abs()
    d['is_peak'] = d['obs_residual'].abs() >= peak_thr
    rows = []
    for sid, g in d.groupby('station_id'):
        pk = g[g['is_peak']]
        ai = 100 * g['ai_ae'].mean()
        ph = 100 * g['ph_ae'].mean()
        ai_pk = 100 * pk['ai_ae'].mean() if len(pk) else np.nan
        ph_pk = 100 * pk['ph_ae'].mean() if len(pk) else np.nan
        rows.append({
            'station_id': sid,
            'region_id': int(g['region_id'].iloc[0]),
            'ofs_model': g['ofs_model'].iloc[0] if 'ofs_model' in g else '',
            'n_pairs': int(len(g)),
            'ai_mae_cm': round(ai, 3),
            f'{phys_name}_mae_cm': round(ph, 3),
            'mae_skill': round(1 - ai / ph, 4) if ph > 0 else np.nan,
            'n_peak': int(len(pk)),
            'ai_peak_mae_cm': round(ai_pk, 3) if np.isfinite(ai_pk) else np.nan,
            f'{phys_name}_peak_mae_cm': round(ph_pk, 3) if np.isfinite(ph_pk) else np.nan,
            'peak_mae_skill': round(1 - ai_pk / ph_pk, 4)
            if (np.isfinite(ai_pk) and np.isfinite(ph_pk) and ph_pk > 0) else np.nan,
        })
    skill = pd.DataFrame(rows)
    amp = per_gauge_peak_amp(d, phys_resid_col, phys_name, peak_thr)
    return skill.merge(amp, on='station_id', how='left')


def per_gauge_peak_amp(d, phys_resid_col, phys_name, peak_thr):
    """Per-cycle peak-amplitude skill. Within each forecast cycle, the 'peak' is
    the window-max residual (highest water the forecast covered); we compare each
    model's predicted peak height to the observed peak height (timing-independent
    amplitude error), averaged over cycles whose observed peak clears peak_thr
    (a real high-water event). This is the flood-relevant 'did you get the surge
    height' metric, complementing the threshold-conditioned peak MAE above.

    `d` is the already-NaN-dropped pairs frame (needs cycle_time + *_residual).
    Returns per-station: n_peak_cycles, ai_peakamp_mae_cm,
    {phys}_peakamp_mae_cm, peakamp_skill.
    """
    cols = ['station_id', 'n_peak_cycles', 'ai_peakamp_mae_cm',
            f'{phys_name}_peakamp_mae_cm', 'peakamp_skill']
    gc = d.groupby(['station_id', 'cycle_time'], observed=True).agg(
        obs_peak=('obs_residual', 'max'),
        ai_peak=('ai_residual', 'max'),
        ph_peak=(phys_resid_col, 'max'),
    ).reset_index()
    gc = gc[gc['obs_peak'] >= peak_thr]
    if gc.empty:
        return pd.DataFrame(columns=cols)
    gc['ai_amp_ae'] = (gc['ai_peak'] - gc['obs_peak']).abs()
    gc['ph_amp_ae'] = (gc['ph_peak'] - gc['obs_peak']).abs()
    agg = gc.groupby('station_id').agg(
        n_peak_cycles=('obs_peak', 'size'),
        ai_peakamp_mae_cm=('ai_amp_ae', 'mean'),
        ph_peakamp_mae_cm=('ph_amp_ae', 'mean'),
    ).reset_index()
    agg['ai_peakamp_mae_cm'] = (100 * agg['ai_peakamp_mae_cm']).round(3)
    agg['ph_peakamp_mae_cm'] = (100 * agg['ph_peakamp_mae_cm']).round(3)
    agg['peakamp_skill'] = np.where(
        agg['ph_peakamp_mae_cm'] > 0,
        (1 - agg['ai_peakamp_mae_cm'] / agg['ph_peakamp_mae_cm']).round(4),
        np.nan)
    agg['n_peak_cycles'] = agg['n_peak_cycles'].astype(int)
    return agg.rename(columns={'ph_peakamp_mae_cm': f'{phys_name}_peakamp_mae_cm'})


def pooled_mae(pairs, phys_resid_col):
    d = pairs.dropna(subset=['ai_residual', phys_resid_col, 'obs_residual'])
    ai = 100 * (d['ai_residual'] - d['obs_residual']).abs().mean()
    ph = 100 * (d[phys_resid_col] - d['obs_residual']).abs().mean()
    return float(ai), float(ph), float(1 - ai / ph), int(len(d))


def attach_station_meta(skill_df, stations_csv):
    st = pd.read_csv(stations_csv, dtype={'station_id': str})
    keep = ['station_id', 'name', 'lat', 'lon', 'is_gl']
    keep = [c for c in keep if c in st.columns]
    m = skill_df.merge(st[keep], on='station_id', how='left')
    n0 = len(m)
    if 'is_gl' in m.columns:
        m = m[m['is_gl'].fillna(0).astype(int) != 1]
    return m, n0 - len(m)


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    cfg = load_config()
    WL_H5 = cfg.get('WATER_LEVELS_H5', WL_H5)   # v2: data/water_levels_processed_v2/...
    print(f"Observed gauges: {WL_H5}")
    run = cfg['RUN_NAME']
    seeds = cfg['SEEDS']
    out_dir = cfg.get('PROCESS_OUT_DIR', 'outputs/paper')
    os.makedirs(out_dir, exist_ok=True)
    peak_thr = float(cfg.get('PEAK_THRESHOLD_M', 0.30))
    stofs_hours = cfg.get('STOFS_CYCLE_HOURS', [0, 6, 12, 18])

    catalog = json.load(open(os.path.join(cfg['PAPER_INFER_DIR'], run,
                                          'needed_cycles.json')))
    summary = {}

    # ---- STOFS: all regions, 3-seed mean, 00/06/12/18Z ----
    print('===== STOFS pairing (3-seed mean) =====')
    stofs_all = []
    for region_id in cfg['REGIONS']:
        fc = seed_mean_fc(cfg, region_id, stofs_hours, seeds)
        if fc is None:
            print(f"  region_{region_id:02d}: no archives, skip")
            continue
        pr = pair_region_stofs(cfg, region_id, fc)
        if pr is None or pr.empty:
            print(f"  region_{region_id:02d}: no STOFS pairs")
            continue
        stofs_all.append(pr)
        ai, ph, sk, n = pooled_mae(pr, 'stofs_residual')
        print(f"  region_{region_id:02d}: gauges~{pr['station_id'].nunique():3d} "
              f"AI={ai:.2f} STOFS={ph:.2f} skill={sk:+.3f} n={n:,}")

    stofs_skill = pd.DataFrame()
    if stofs_all:
        stofs_pairs_all = pd.concat(stofs_all, ignore_index=True)
        stofs_skill = per_gauge_skill(stofs_pairs_all, 'stofs_residual', 'stofs', peak_thr)
        stofs_skill, n_gl = attach_station_meta(stofs_skill, cfg['STATIONS_CSV'])
        p = os.path.join(out_dir, 'fig1_stofs_skill.csv')
        stofs_skill.to_csv(p, index=False)
        ai, ph, sk, n = pooled_mae(stofs_pairs_all, 'stofs_residual')
        print(f"  POOLED STOFS: AI={ai:.2f} STOFS={ph:.2f} skill={sk:+.3f} "
              f"gauges={len(stofs_skill)} (dropped {n_gl} Great-Lakes) -> {p}")
        summary['stofs'] = {'ai_mae_cm': round(ai, 2), 'stofs_mae_cm': round(ph, 2),
                            'mae_skill': round(sk, 3), 'n_pairs': n,
                            'n_gauges': int(len(stofs_skill)),
                            'ai_better_frac': round(float((stofs_skill['mae_skill'] > 0).mean()), 3)}

        # robustness: single-seed (seed_1) pooled, regions present
        s1 = []
        for region_id in cfg['REGIONS']:
            fc1 = seed_mean_fc(cfg, region_id, stofs_hours, [seeds[0]])
            if fc1 is None:
                continue
            pr1 = pair_region_stofs(cfg, region_id, fc1)
            if pr1 is not None:
                s1.append(pr1)
        if s1:
            ai1, ph1, sk1, _ = pooled_mae(pd.concat(s1, ignore_index=True), 'stofs_residual')
            print(f"  robustness: single-seed AI={ai1:.2f} (3-seed {ai:.2f})")
            summary['stofs']['single_seed_ai_mae_cm'] = round(ai1, 2)

    # ---- OFS: per catalog ----
    print('\n===== OFS pairing (3-seed mean, datum_offset) =====')
    datum = pd.DataFrame()
    if os.path.exists(cfg['OFS_DATUM_CSV']):
        datum = pd.read_csv(cfg['OFS_DATUM_CSV'], dtype={'station_id': str})
    ofs_all = []
    ofs_methods = {}
    for model, spec in catalog.get('ofs', {}).items():
        hours = spec['cycle_hours']
        dmodel = datum[datum['ofs_model'] == model] if not datum.empty else pd.DataFrame(columns=['station_id', 'offset'])
        # Uniform residual method for all physics models (see config _process note);
        # default slr_detrend, identical treatment to the detrended gauge.
        method = cfg.get('OFS_RESIDUAL_METHOD', 'slr_detrend')
        ofs_methods[model] = method
        print(f"  -- {model}: method={method} --")
        for region_id in spec['regions']:
            fc = seed_mean_fc(cfg, region_id, hours, seeds)
            if fc is None:
                print(f"  {model} region_{region_id:02d}: no archives, skip")
                continue
            pr = pair_region_ofs(cfg, region_id, model, fc, dmodel, method)
            if pr is None or pr.empty:
                print(f"  {model} region_{region_id:02d}: no OFS pairs")
                continue
            ofs_all.append(pr)
            ai, ph, sk, n = pooled_mae(pr, 'ofs_residual')
            print(f"  {model} region_{region_id:02d}: gauges~{pr['station_id'].nunique():3d} "
                  f"AI={ai:.2f} OFS={ph:.2f} skill={sk:+.3f} n={n:,}")

    if ofs_all:
        ofs_pairs_all = pd.concat(ofs_all, ignore_index=True)
        ofs_skill = per_gauge_skill(ofs_pairs_all, 'ofs_residual', 'ofs', peak_thr)
        # carry ofs_model through (per_gauge_skill kept first model per station; here
        # a station maps to one model, so it's fine)
        ofs_skill, n_gl = attach_station_meta(ofs_skill, cfg['STATIONS_CSV'])
        p = os.path.join(out_dir, 'fig1_ofs_skill.csv')
        ofs_skill.to_csv(p, index=False)
        print(f"  wrote {p} ({len(ofs_skill)} gauges, dropped {n_gl} Great-Lakes)")
        summary['ofs'] = {}
        for model in ofs_skill['ofs_model'].unique():
            sub = ofs_pairs_all[ofs_pairs_all['ofs_model'] == model]
            ai, ph, sk, n = pooled_mae(sub, 'ofs_residual')
            summary['ofs'][model] = {'ai_mae_cm': round(ai, 2), 'ofs_mae_cm': round(ph, 2),
                                     'mae_skill': round(sk, 3), 'n_pairs': n,
                                     'method': ofs_methods.get(model)}
            print(f"  POOLED {model}: AI={ai:.2f} OFS={ph:.2f} skill={sk:+.3f}")

    with open(os.path.join(out_dir, 'fig1_process_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {os.path.join(out_dir, 'fig1_process_summary.json')}")
