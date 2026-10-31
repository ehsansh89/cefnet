"""
paper_common.py — shared helpers for the Nature Communications paper figures.

Used by paper_fig*.py and paper_table*.py. These scripts live in the repo root
(like the o_*/d_* scripts) and write their deliverables into
``docs/paper/figures`` and ``docs/paper/tables``.

Design: re-render from the verified backing CSVs under ``outputs/`` rather than
screenshot old PNGs. Every loader is defensive — a missing file returns an empty
frame and the calling panel annotates "data pending" instead of crashing, mirroring
the posture of o_paper_figures.py.

Backing-data schemas this module relies on (verified 2026-06-26):
  stations_processed.csv : station_id,name,state,lat,lon,...,region,...,slr_2025_m
  eval summary_table.csv : region_id,seed,overall_mae_cm,overall_bias_cm,...
  stofs station_skill_table.csv :
      region_id,atm_source,station_id,station_name,lat,lon,n_pairs,
      ai_mae_cm,stofs_mae_cm,mae_skill,ai_peak_mae_cm,stofs_peak_mae_cm,peak_mae_skill
  ofs station_skill_table.csv : same but ofs_model + ofs_* columns
  *_comparison*/.../skill_by_lead.csv :
      n,ai_mae_cm,...,(ofs|stofs)_mae_cm,...,mae_skill_vs_*,lead_hour
  uq .../uq_crpss_stofs.csv : lead_hour,n,ai_crps_cm,baseline_crps_cm,crpss,baseline,region
  uq .../uq_coverage_region_lead.csv : lead_bin,nominal,observed_coverage,mean_interval_width_cm,n,region
  uq .../brier_by_threshold.csv :
      threshold_name,event_definition,n,event_count,event_rate,ai_brier,
      climatology_brier,stofs_brier,ofs_brier,bss_vs_climatology,bss_vs_stofs,bss_vs_ofs
  uq .../uq_variance_decomp.csv :
      lead_hour,n,total_var_cm2,atm_var_cm2,seed_var_cm2,interaction_var_cm2,
      atm_share,seed_share,interaction_share,region
"""

import json
import os
import shutil
from glob import glob

import numpy as np
import pandas as pd


CONFIG_PATH = 'paper_config.json'


# =============================================================================
# Config + paths
# =============================================================================
def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def ensure_dirs(cfg):
    os.makedirs(cfg['FIG_DIR'], exist_ok=True)
    os.makedirs(cfg['TABLE_DIR'], exist_ok=True)


def fig_path(cfg, name):
    return os.path.join(cfg['FIG_DIR'], name)


def table_path(cfg, name):
    return os.path.join(cfg['TABLE_DIR'], name)


def read_csv(path):
    """Read a CSV, returning an empty DataFrame if absent/unreadable."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}")
        return pd.DataFrame()


def copy_if_exists(src, dst):
    if not src or not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def find_first(patterns):
    for p in patterns:
        hits = sorted(glob(p))
        if hits:
            return hits[0]
    return None


# =============================================================================
# Station metadata
# =============================================================================
def station_train_years(cfg, h5_path=None, train_start=None, train_end=None):
    """Per-station count of observed-residual hours falling in the training split,
    expressed in years. Cached to <PROCESS_OUT_DIR>/station_train_years.csv. Used
    by the inclusion gate (a station the model never saw in training cannot be
    reliably scored — e.g. post-cutoff records with borrowed datums).

    The H5 store and the training window are taken from cfg when not passed
    explicitly: cfg['WATER_LEVELS_H5'] and cfg['GATE_TRAIN_{START,END}']. v2
    routes onto the v2-detrended gauges with a 2021 cutoff; v1 (no GATE_* keys)
    falls back to the v1 store and the original 1980–2019 window."""
    h5_path = h5_path or cfg.get('WATER_LEVELS_H5',
                                 'data/water_levels_processed/processed_water_levels.h5')
    train_start = train_start or cfg.get('GATE_TRAIN_START', '1980-01-01')
    train_end = train_end or cfg.get('GATE_TRAIN_END', '2019-12-31')
    out_dir = cfg.get('PROCESS_OUT_DIR', 'outputs/paper')
    cache = os.path.join(out_dir, 'station_train_years.csv')
    if os.path.exists(cache):
        return read_csv(cache).astype({'station_id': str})
    os.makedirs(out_dir, exist_ok=True)
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    rows = []
    with pd.HDFStore(h5_path, 'r') as h:
        for k in h.keys():
            if not k.startswith('/station_'):
                continue
            sid = k.replace('/station_', '')
            try:
                d = h[k]
            except Exception:
                continue
            if 'residual' not in getattr(d, 'columns', []):
                continue
            r = d['residual'].dropna()
            if getattr(r.index, 'tz', None) is not None:
                r.index = r.index.tz_localize(None)
            tr = r[(r.index >= ts) & (r.index <= te)]
            rows.append({'station_id': sid, 'train_yr': round(len(tr) / 8766.0, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def eligible_station_ids(cfg):
    """Set of station_ids meeting the inclusion gate MIN_TRAIN_YEARS. Empty
    MIN_TRAIN_YEARS (or 0) → all stations eligible."""
    min_yr = float(cfg.get('MIN_TRAIN_YEARS', 0) or 0)
    if min_yr <= 0:
        return None
    ty = station_train_years(cfg)
    return set(ty[ty['train_yr'] >= min_yr]['station_id'].astype(str))


def load_stations(cfg):
    sd = read_csv(cfg['STATIONS_CSV'])
    if sd.empty:
        return sd
    if 'region' in sd.columns:
        sd['region'] = pd.to_numeric(sd['region'], errors='coerce').astype('Int64')
    regime_map = {int(k): v for k, v in cfg.get('REGIME_MAP', {}).items()}
    if regime_map and 'region' in sd.columns:
        sd['regime'] = sd['region'].map(regime_map).fillna('other')
    return sd


# =============================================================================
# Skill loaders
# =============================================================================
def load_stofs_station_skill(cfg, atm_source='gfs_archive'):
    """Per-station AI-vs-STOFS skill. Falls back to era5 if gfs_archive absent."""
    p = os.path.join(cfg['STOFS_ALL_DIR'], cfg['RUN_NAME'], 'station_skill_table.csv')
    df = read_csv(p)
    if df.empty:
        return df
    if 'atm_source' in df.columns:
        sub = df[df['atm_source'] == atm_source]
        if sub.empty:
            sub = df[df['atm_source'] == 'era5']
        df = sub
    return df.copy()


def load_ofs_station_skill(cfg, atm_source='gfs_archive'):
    """Per-station AI-vs-OFS skill across regional models."""
    p = os.path.join(cfg['OFS_ALL_DIR'], cfg['RUN_NAME'], 'station_skill_table.csv')
    df = read_csv(p)
    if df.empty:
        return df
    if 'atm_source' in df.columns:
        df = df[df['atm_source'] == atm_source]
    return df.copy()


def load_stofs_pooled(cfg):
    return read_csv(os.path.join(cfg['STOFS_ALL_DIR'], cfg['RUN_NAME'],
                                 'skill_table_pooled.csv'))


def load_ofs_pooled(cfg):
    return read_csv(os.path.join(cfg['OFS_ALL_DIR'], cfg['RUN_NAME'],
                                 'skill_table_pooled.csv'))


def load_eval_pooled(cfg):
    return read_csv(os.path.join(cfg['EVAL_ALL_DIR'], cfg['RUN_NAME'],
                                 cfg['SPLIT'], 'summary_table_pooled.csv'))


def load_eval_summary(cfg):
    return read_csv(os.path.join(cfg['EVAL_ALL_DIR'], cfg['RUN_NAME'],
                                 cfg['SPLIT'], 'summary_table.csv'))


def pooled_lead_curve(cmp_root, run_name, mae_col, skill_col):
    """Aggregate per-region skill_by_lead.csv across regions into one
    n-weighted MAE-vs-lead curve. Returns DataFrame[lead_hour, ai_mae_cm,
    <mae_col>, <skill_col>, n]. Works for both OFS and STOFS comparison trees
    by passing the physics MAE column name (ofs_mae_cm / stofs_mae_cm)."""
    paths = glob(os.path.join(cmp_root, run_name, '**', 'skill_by_lead.csv'),
                 recursive=True)
    if not paths:
        paths = glob(os.path.join(cmp_root, '**', 'skill_by_lead.csv'),
                     recursive=True)
    frames = []
    for p in paths:
        d = read_csv(p)
        if d.empty or 'lead_hour' not in d.columns or mae_col not in d.columns:
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    alld = pd.concat(frames, ignore_index=True)
    alld = alld.dropna(subset=['lead_hour', 'ai_mae_cm', mae_col])
    g = alld.groupby('lead_hour')

    def wmean(col):
        return g.apply(lambda x: np.average(x[col], weights=x['n'])
                       if x['n'].sum() > 0 else np.nan)

    out = pd.DataFrame({
        'lead_hour': sorted(alld['lead_hour'].unique()),
    }).set_index('lead_hour')
    out['ai_mae_cm'] = wmean('ai_mae_cm')
    out[mae_col] = wmean(mae_col)
    out['n'] = g['n'].sum()
    out['mae_skill'] = 1.0 - out['ai_mae_cm'] / out[mae_col]
    return out.reset_index()


# =============================================================================
# UQ file resolution (deep, variant/daterange-nested layout)
# =============================================================================
def _uq_pick(paths, prefer_daterange):
    """From candidate paths, prefer the one whose path contains the target
    date-range (the 2-year archive); else the longest path string."""
    if not paths:
        return None
    pref = [p for p in paths if prefer_daterange and prefer_daterange in p]
    pool = pref if pref else paths
    return sorted(pool, key=len, reverse=True)[0]


def find_uq_allregions(cfg, fname, calibrated=False):
    """Locate a rolled-up (region-columned) UQ csv under uq_all_regions."""
    root = os.path.join(cfg['UQ_ALL_DIR'], cfg['RUN_NAME'])
    paths = glob(os.path.join(root, '**', fname), recursive=True)
    if calibrated:
        cal = [p for p in paths if 'calibrat' in p.lower()]
        paths = cal if cal else []
    else:
        paths = [p for p in paths if 'calibrat' not in p.lower()]
    return _uq_pick(paths, cfg.get('UQ_DATERANGE'))


def find_uq_postproc(cfg, fname, calibrated=False):
    """Glob all per-region copies of a uq_postprocessing csv (2-yr range)."""
    root = os.path.join(cfg['UQ_POST_DIR'], cfg['RUN_NAME'])
    paths = glob(os.path.join(root, '**', fname), recursive=True)
    if cfg.get('UQ_DATERANGE'):
        dr = [p for p in paths if cfg['UQ_DATERANGE'] in p]
        if dr:
            paths = dr
    if calibrated:
        paths = [p for p in paths if 'calibrat' in p.lower()]
    else:
        paths = [p for p in paths if 'calibrat' not in p.lower()]
    return paths


def aggregate_brier(cfg, event_definition='window_peak', calibrated=False):
    """n-weighted aggregate of brier_by_threshold across regions.
    Returns DataFrame[threshold_name, event_rate, bss_vs_climatology,
    bss_vs_stofs, bss_vs_ofs]."""
    paths = find_uq_postproc(cfg, 'brier_by_threshold.csv', calibrated=calibrated)
    frames = [read_csv(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    alld = pd.concat(frames, ignore_index=True)
    if 'event_definition' in alld.columns:
        alld = alld[alld['event_definition'] == event_definition]
    if alld.empty:
        return pd.DataFrame()
    wcols = ['event_rate', 'bss_vs_climatology', 'bss_vs_stofs', 'bss_vs_ofs',
             'ai_brier', 'climatology_brier', 'stofs_brier', 'ofs_brier']
    rows = []
    for name, g in alld.groupby('threshold_name'):
        w = g['n'].to_numpy(dtype=float)
        row = {'threshold_name': name, 'n': float(np.nansum(w))}
        for c in wcols:
            if c in g.columns:
                v = g[c].to_numpy(dtype=float)
                m = np.isfinite(v) & np.isfinite(w)
                row[c] = float(np.average(v[m], weights=w[m])) if m.any() else np.nan
        rows.append(row)
    order = {'minor': 0, 'moderate': 1, 'major': 2}
    out = pd.DataFrame(rows)
    out['_o'] = out['threshold_name'].map(order).fillna(9)
    return out.sort_values('_o').drop(columns='_o').reset_index(drop=True)


def region_mean_by_lead(cfg, fname, value_col, calibrated=False):
    """Average a region-columned UQ csv across regions per lead_hour
    (n-weighted). Used for CRPSS-vs-lead. Returns DataFrame[lead_hour,
    <value_col>, n]."""
    p = find_uq_allregions(cfg, fname, calibrated=calibrated)
    df = read_csv(p)
    if df.empty or 'lead_hour' not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['lead_hour', value_col])
    if 'n' not in df.columns:
        df['n'] = 1.0
    g = df.groupby('lead_hour')
    out = pd.DataFrame({'lead_hour': sorted(df['lead_hour'].unique())}).set_index('lead_hour')
    out[value_col] = g.apply(lambda x: np.average(x[value_col], weights=x['n'])
                             if x['n'].sum() > 0 else np.nan)
    out['n'] = g['n'].sum()
    return out.reset_index()
