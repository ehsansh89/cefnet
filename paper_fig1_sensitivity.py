"""
paper_fig1_sensitivity.py — how much does Fig 1 skill move with the de-bias method?

The worry: converting a physics model (STOFS cwl / OFS zeta) into the gauge's
residual frame is a choice, and the two available choices are not equivalent:

  * slr_detrend  : phys_residual = zeta - harmonic_tide - slr_component
                   (leaves the model's constant datum/mean offset IN -> favors AI)
  * datum_offset : phys_residual = zeta - datum_offset - harmonic_tide
                   (removes the constant offset -> fair dynamic comparison)

This script holds EVERYTHING else fixed (v2 model, 3-seed mean, the SAME gauge set,
the SAME (cycle, lead) samples) and flips ONLY the method, so the change in pooled
MAE-skill is attributable to the de-bias choice alone. It reuses the tested pairing
engines (x_stofs_pair / x_ofs_pair) — one pairing code path — and the same seed-mean
assembly as paper_fig1_process.

To keep STOFS and OFS on identical footing, BOTH datum offsets are per-(gauge,model)
constants estimated from the LEAD-0 nowcast over one window (2022-01-01..2027-01-01),
matched to the v2-detrended gauges: OFS from the existing datum_offsets_v2.csv, STOFS
computed inline here with the same x_stofs_datum_bridge routine.

For each comparand both methods are run on the INTERSECTION of gauges that (a) have a
datum offset, (b) clear the v2 inclusion gate, (c) are not Great Lakes — so the only
thing differing between the two rows is the transform. ai_mae_cm should therefore be
(near) identical across methods; it's printed as a self-check.

Writes: outputs/paper/v2/fig1_debias_sensitivity.csv
Config: paper_fig1_config_v2.json
"""

import json
import os

import numpy as np
import pandas as pd

from paper_fig1_process import seed_mean_fc
from paper_common import eligible_station_ids
from x_stofs_pair import build_pairs_for_station as stofs_pairs
from x_ofs_pair import build_pairs_for_station as ofs_pairs, load_obs_for_station
from x_stofs_datum_bridge import cycle_offset_for_pair as stofs_offset_for_pair


CONFIG_PATH = os.environ.get('X_PAPER_FIG1_CONFIG', 'paper_fig1_config_v2.json')
DATUM_WINDOW = ('2022-01-01', '2027-01-01')   # match datum_offsets_v2.csv methodology
DATUM_MIN_SAMPLES = 200


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def new_acc():
    return {'sae_ai': 0.0, 'sae_ph': 0.0, 'n': 0, 'gauges': set()}


def add_pairs(acc, df, phys_col, sid):
    """Reduce one station's pairs frame to running sums (memory-safe)."""
    if df is None:
        return
    d = df.dropna(subset=['ai_residual', phys_col, 'obs_residual'])
    if d.empty:
        return
    acc['sae_ai'] += (d['ai_residual'] - d['obs_residual']).abs().sum()
    acc['sae_ph'] += (d[phys_col] - d['obs_residual']).abs().sum()
    acc['n'] += len(d)
    acc['gauges'].add(sid)


def summarize(acc):
    if acc['n'] == 0:
        return dict(n_pairs=0, n_gauges=0, ai_mae_cm=np.nan,
                    phys_mae_cm=np.nan, skill=np.nan)
    ai = 100 * acc['sae_ai'] / acc['n']
    ph = 100 * acc['sae_ph'] / acc['n']
    return dict(n_pairs=acc['n'], n_gauges=len(acc['gauges']),
                ai_mae_cm=round(ai, 2), phys_mae_cm=round(ph, 2),
                skill=round(1 - ai / ph, 3) if ph > 0 else np.nan)


def compute_stofs_offsets(wl, sids, source):
    """Per-station STOFS lead-0 datum offset vs the v2 gauge, same window/rule as
    the v2 OFS bridge. Returns {sid: offset_m} for finite offsets only."""
    out = {}
    for sid in sids:
        pth = f'data/stofs/per_station/{sid}__stofs2d.nc'
        if not os.path.exists(pth):
            continue
        rec = stofs_offset_for_pair(pth, wl, sid, source,
                                    DATUM_WINDOW[0], DATUM_WINDOW[1],
                                    DATUM_MIN_SAMPLES)
        if rec and np.isfinite(rec.get('offset', np.nan)):
            out[sid] = float(rec['offset'])
    return out


def run_comparand(label, regions, hours, phys_col, pair_fn, offsets, keep, cfg, seeds):
    """Run both methods for one comparand on the identical eligible gauge set.
    `pair_fn(fc, path, obs, sid, method, off)` returns a pairs frame or None."""
    accs = {'slr_detrend': new_acc(), 'datum_offset': new_acc()}
    for region_id in regions:
        fc = seed_mean_fc(cfg, region_id, hours, seeds)
        if fc is None:
            continue
        sids = [str(s) for s in fc.station.values]
        with pd.HDFStore(cfg['WATER_LEVELS_H5'], 'r') as wl:
            for sid in sids:
                off = offsets.get(sid)
                # identical gauge set for both methods: needs an offset + gate + non-GL
                if off is None or sid not in keep:
                    continue
                obs = load_obs_for_station(wl, sid)
                if obs is None:
                    continue
                for method in ('slr_detrend', 'datum_offset'):
                    df = pair_fn(fc, sid, obs, method, off)
                    add_pairs(accs[method], df, phys_col, sid)
    rows = []
    for method in ('slr_detrend', 'datum_offset'):
        s = summarize(accs[method])
        s.update(comparand=label, method=method)
        rows.append(s)
    return rows


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    seeds = cfg['SEEDS']
    source = cfg.get('STOFS_SOURCE', 'cwl')
    out_dir = cfg.get('PROCESS_OUT_DIR', 'outputs/paper/v2')
    os.makedirs(out_dir, exist_ok=True)

    catalog = json.load(open(os.path.join(cfg['PAPER_INFER_DIR'], cfg['RUN_NAME'],
                                           'needed_cycles.json')))

    # --- gauge-eligibility filters (applied identically to both methods) ---
    gate = eligible_station_ids(cfg)                 # v2 train-years gate (or None)
    st = pd.read_csv(cfg['STATIONS_CSV'], dtype={'station_id': str})
    gl = set(st.loc[st.get('is_gl', 0).fillna(0).astype(int) == 1, 'station_id']) \
        if 'is_gl' in st.columns else set()
    keep_all = (set(st['station_id']) if gate is None else set(gate)) - gl
    print(f"Config: {CONFIG_PATH}   run: {cfg['RUN_NAME']}")
    print(f"Gate: MIN_TRAIN_YEARS={cfg.get('MIN_TRAIN_YEARS')} "
          f"(end {cfg.get('GATE_TRAIN_END')}) -> {len(keep_all)} gauges pre-offset; "
          f"dropped {len(gl)} Great-Lakes")

    # --- OFS datum offsets: reuse the v2 file (same window/rule we use for STOFS) ---
    ofs_datum = pd.read_csv(cfg['OFS_DATUM_CSV'], dtype={'station_id': str})

    results = []

    # ---- STOFS (all regions) ----
    print("\n== STOFS ==")
    with pd.HDFStore(cfg['WATER_LEVELS_H5'], 'r') as wl:
        all_sids = [k.replace('/station_', '') for k in wl.keys()
                    if k.startswith('/station_')]
        stofs_off = compute_stofs_offsets(wl, all_sids, source)
    print(f"  STOFS lead-0 offsets: {len(stofs_off)} gauges "
          f"(median {np.median(list(stofs_off.values()))*100:.1f} cm)")

    def _stofs_pair(fc, sid, obs, method, off):
        pth = f'data/stofs/per_station/{sid}__stofs2d.nc'
        if not os.path.exists(pth):
            return None
        return stofs_pairs(fc, pth, obs, sid, source=source, method=method,
                           datum_offset=off)

    results += run_comparand('STOFS', catalog['stofs']['regions'],
                             catalog['stofs']['cycle_hours'], 'stofs_residual',
                             _stofs_pair, stofs_off, keep_all, cfg, seeds)
    for r in results[-2:]:
        print(f"  {r['method']:<13s} n_gauges={r['n_gauges']:>3d} "
              f"AI={r['ai_mae_cm']:.2f} STOFS={r['phys_mae_cm']:.2f} "
              f"skill={r['skill']:+.3f}")

    # ---- OFS (per catalog model) ----
    for model, spec in catalog.get('ofs', {}).items():
        print(f"\n== OFS: {model} ==")
        moff = ofs_datum[ofs_datum['ofs_model'] == model]
        off_map = dict(zip(moff['station_id'], moff['offset']))
        if not off_map:
            print("  no datum offsets in file, skip")
            continue

        def _ofs_pair(fc, sid, obs, method, off, _m=model):
            pth = f'data/ofs/per_station/{sid}__{_m}.nc'
            if not os.path.exists(pth):
                return None
            return ofs_pairs(fc, pth, obs, sid, method=method, datum_offset=off)

        rows = run_comparand(model, spec['regions'], spec['cycle_hours'],
                             'ofs_residual', _ofs_pair, off_map, keep_all, cfg, seeds)
        results += rows
        for r in rows:
            print(f"  {r['method']:<13s} n_gauges={r['n_gauges']:>3d} "
                  f"AI={r['ai_mae_cm']:.2f} {model.upper()}={r['phys_mae_cm']:.2f} "
                  f"skill={r['skill']:+.3f}")

    # --- tidy wide table: one row per comparand, both methods side by side ---
    df = pd.DataFrame(results)
    wide = []
    for label in df['comparand'].unique():
        g = df[df['comparand'] == label].set_index('method')
        slr, dat = g.loc['slr_detrend'], g.loc['datum_offset']
        wide.append({
            'comparand': label, 'n_gauges': int(dat['n_gauges']),
            'ai_mae_cm': dat['ai_mae_cm'],
            'ai_mae_cm_slr_check': slr['ai_mae_cm'],
            'phys_mae_slr': slr['phys_mae_cm'], 'skill_slr': slr['skill'],
            'phys_mae_datum': dat['phys_mae_cm'], 'skill_datum': dat['skill'],
            'skill_drop': round(float(slr['skill']) - float(dat['skill']), 3),
        })
    wide = pd.DataFrame(wide)
    out_path = os.path.join(out_dir, 'fig1_debias_sensitivity.csv')
    wide.to_csv(out_path, index=False)

    pd.set_option('display.width', 160, 'display.max_columns', 20)
    print("\n================ DE-BIAS SENSITIVITY (v2, same gauges both rows) ============")
    print(wide.to_string(index=False))
    print(f"\nWrote {out_path}")
