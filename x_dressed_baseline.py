"""
x_dressed_baseline.py — The dressed-physics baseline (referendum item R4).

The strongest cheap objection to our ensemble: "take deterministic STOFS and
statistically spread it with its own error history — physics has probabilities
too." We build exactly that, with the SAME year-split + swap protocol as our
own calibration (x_ensemble_calibrate_v2.py), and score it identically:

  For each (station, lead-bin): collect STOFS residual errors
  e = stofs_residual − obs_residual on the FIT year; take 19 quantiles
  (5%..95%). Dressed pseudo-ensemble for the EVAL year:
      member_k(c, l, s) = stofs(c, l, s) − e_q[K−1−k]
  (subtracting the error quantiles in reverse order gives ascending members —
  quantile trajectories, i.e. rank-correlated across lead, the standard simple
  dressing). Every cycle is scored only with parameters from the other year.

Outputs per region (next to verify_arrays.nc):
    dressed_crps_by_lead.csv      lead_hour, n, stofs_det_crps_cm (=MAE),
                                  dressed_crps_cm
    dressed_coverage.csv          nominal, n, coverage, width_cm
    dressed_probs.nc              window-peak exceedance probs per
                                  (cycle, station, threshold) + STOFS/obs
                                  flags — consumed by the Fig-3 decision
                                  analysis and Fig-2 Brier panel.
    dressed_summary.json

Config: reuses x_ensemble_calibrate_v2_config.json (BASE_DIR, SPLIT_YEAR).
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import xarray as xr


CONFIG_PATH = os.environ.get('X_ENSEMBLE_CALIBRATE_V2_CONFIG',
                             'x_ensemble_calibrate_v2_config.json')

LEAD_BINS = [(1, 12), (13, 24), (25, 36), (37, 48), (49, 72)]
Q_LEVELS = np.arange(0.05, 0.951, 0.05)          # 19 pseudo-members
NOMINAL_LEVELS = [0.50, 0.80, 0.90]              # coverage checks
MIN_FIT_SAMPLES = 200
THRESHOLDS = ['minor', 'moderate', 'major']


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def crps_ensemble(forecasts, obs):
    f = np.sort(forecasts, axis=-1)
    y = obs[..., None]
    n = f.shape[-1]
    term1 = np.nanmean(np.abs(f - y), axis=-1)
    i = np.arange(1, n + 1)
    coeffs = (2 * i - n - 1) / (n * n)
    term2 = np.nansum(coeffs * f, axis=-1)
    return term1 - term2


def _lead_bin_index(n_lead):
    idx = np.zeros(n_lead, dtype=int)
    for b, (lo, hi) in enumerate(LEAD_BINS):
        idx[lo - 1:hi] = b
    return idx


def fit_error_quantiles(stofs, obs, fit_mask, lead_bin_idx):
    """(station, bin, K) STOFS-error quantiles on the fit year; NaN when the
    station×bin has < MIN_FIT_SAMPLES valid errors."""
    n_s = stofs.shape[2]
    out = np.full((n_s, len(LEAD_BINS), len(Q_LEVELS)), np.nan, np.float32)
    err = stofs[fit_mask] - obs[fit_mask]                  # (Cf, L, S)
    for b in range(len(LEAD_BINS)):
        e_b = err[:, lead_bin_idx == b, :]                 # (Cf, Lb, S)
        for s in range(n_s):
            e = e_b[:, :, s]
            e = e[np.isfinite(e)]
            if e.size >= MIN_FIT_SAMPLES:
                out[s, b] = np.quantile(e, Q_LEVELS)
    return out


def dressed_members(stofs, eq, lead_bin_idx):
    """(C, L, S) STOFS − reversed error quantiles -> (C, K, L, S) ascending
    pseudo-members (NaN where the station×bin was unfit)."""
    # eq: (S, B, K) -> per-lead (S, L, K)
    eq_l = eq[:, lead_bin_idx, :]                          # (S, L, K)
    rev = eq_l[:, :, ::-1]                                 # reversed quantiles
    # members: stofs (C,L,S) - rev (S,L,K) -> (C,K,L,S)
    return (stofs[:, None, :, :]
            - np.moveaxis(rev, [0, 1, 2], [2, 1, 0])[None, :, :, :])


def score_region(verify_dir, cfg):
    out_sum = os.path.join(verify_dir, 'dressed_summary.json')
    if os.path.exists(out_sum):
        print(f"  [exists, skip] {verify_dir}")
        return
    fc = os.path.join(verify_dir, 'verify_arrays.nc')
    ds = xr.open_dataset(fc)
    stofs = ds.stofs_residual.values.astype(np.float32)    # (C, L, S)
    obs = ds.obs_residual.values.astype(np.float32)
    stofs_wl = ds.stofs_water_level.values.astype(np.float32)
    obs_wl = ds.obs_water_level.values.astype(np.float32)
    tide = ds.tide.values.astype(np.float32)
    slr = ds.slr_component.values.astype(np.float32)
    thr = {t: ds[f'{t}_threshold'].values.astype(np.float32)
           for t in THRESHOLDS}
    cycles = pd.to_datetime(ds.cycle_time.values)
    station_ids = [str(s) for s in ds.station.values]
    ds.close()

    n_c, n_l, n_s = stofs.shape
    lead_bin_idx = _lead_bin_index(n_l)
    split = int(cfg.get('SPLIT_YEAR', 2025))
    mask_a = cycles.year < split
    directions = [('fit2024_eval2025', mask_a, ~mask_a),
                  ('fit2025_eval2024', ~mask_a, mask_a)]

    # Held-out dressed members for EVERY cycle (each from other-year fit).
    dm = np.full((n_c, len(Q_LEVELS), n_l, n_s), np.nan, np.float32)
    for name, fmask, emask in directions:
        if not fmask.any() or not emask.any():
            continue
        eq = fit_error_quantiles(stofs, obs, np.asarray(fmask), lead_bin_idx)
        dm[np.asarray(emask)] = dressed_members(
            stofs[np.asarray(emask)], eq, lead_bin_idx)

    # ---- CRPS by lead (dressed vs deterministic-as-degenerate) ----
    rows = []
    for li in range(n_l):
        f = np.moveaxis(dm[:, :, li, :], 1, -1)            # (C, S, K)
        y = obs[:, li, :]
        det = stofs[:, li, :]
        v = np.isfinite(y) & np.isfinite(det) & np.all(np.isfinite(f), axis=-1)
        if not v.any():
            continue
        rows.append(dict(
            lead_hour=li + 1, n=int(v.sum()),
            stofs_det_crps_cm=100 * float(np.mean(np.abs(det[v] - y[v]))),
            dressed_crps_cm=100 * float(np.mean(crps_ensemble(f[v], y[v]))),
        ))
    crps_df = pd.DataFrame(rows)
    if crps_df.empty:
        # Span lacks one of the two years (e.g. the Oct-2024 pilot) — no
        # held-out dressing is possible; skip gracefully.
        print("  [skip] no held-out dressed samples (single-year span)")
        return
    crps_df.to_csv(os.path.join(verify_dir, 'dressed_crps_by_lead.csv'),
                   index=False)

    # ---- coverage ----
    cov_rows = []
    for a in NOMINAL_LEVELS:
        lo_k = int(round((1 - a) / 2 / 0.05)) - 1
        hi_k = len(Q_LEVELS) - 1 - lo_k
        lo, hi = dm[:, lo_k], dm[:, hi_k]                  # (C, L, S)
        v = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
        if not v.any():
            continue
        cov_rows.append(dict(
            nominal=a, n=int(v.sum()),
            coverage=float(np.mean((lo[v] <= obs[v]) & (obs[v] <= hi[v]))),
            width_cm=100 * float(np.mean(hi[v] - lo[v]))))
    pd.DataFrame(cov_rows).to_csv(
        os.path.join(verify_dir, 'dressed_coverage.csv'), index=False)

    # ---- window-peak exceedance probs (water-level space) ----
    # dressed wl member = dressed residual + tide + slr
    probs = {}
    obs_ev, det_ev = {}, {}
    dm_wl_max = np.nanmax(dm + (tide + slr)[:, None, :, :], axis=2)  # (C,K,S)
    obs_max = np.nanmax(obs_wl, axis=1)                              # (C,S)
    det_max = np.nanmax(stofs_wl, axis=1)
    valid = (np.all(np.isfinite(dm_wl_max), axis=1)
             & np.isfinite(obs_max) & np.isfinite(det_max))          # (C,S)
    for t in THRESHOLDS:
        th = thr[t][None, None, :]
        probs[t] = np.mean(dm_wl_max >= th, axis=1).astype(np.float32)
        obs_ev[t] = (obs_max >= thr[t][None, :])
        det_ev[t] = (det_max >= thr[t][None, :])
    ds_out = xr.Dataset(
        data_vars={
            **{f'dressed_prob_{t}': (('cycle_time', 'station'), probs[t])
               for t in THRESHOLDS},
            **{f'obs_event_{t}': (('cycle_time', 'station'),
                                  obs_ev[t].astype(np.uint8))
               for t in THRESHOLDS},
            **{f'stofs_event_{t}': (('cycle_time', 'station'),
                                    det_ev[t].astype(np.uint8))
               for t in THRESHOLDS},
            'valid': (('cycle_time', 'station'), valid.astype(np.uint8)),
        },
        coords=dict(cycle_time=cycles.values,
                    station=np.asarray(station_ids, dtype=object)),
        attrs=dict(event_definition='window_peak_1_72h',
                   note='held-out year-swap dressing; 19 quantile members'),
    )
    ds_out.to_netcdf(os.path.join(verify_dir, 'dressed_probs.nc'))

    n_ok = int(np.isfinite(dm[:, 0]).sum())
    summary = dict(
        n_dressed_samples=n_ok,
        pooled_dressed_crps_cm=float(np.average(
            crps_df['dressed_crps_cm'], weights=crps_df['n'])),
        pooled_stofs_det_crps_cm=float(np.average(
            crps_df['stofs_det_crps_cm'], weights=crps_df['n'])),
        coverage=cov_rows,
    )
    with open(out_sum, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  dressed CRPS {summary['pooled_dressed_crps_cm']:.2f} cm "
          f"(det {summary['pooled_stofs_det_crps_cm']:.2f})")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    spans = [d for d in glob.glob(os.path.join(cfg['BASE_DIR'], '*'))
             if os.path.isdir(d)]
    for span_dir in spans:
        regs = sorted(d for d in glob.glob(os.path.join(span_dir, 'region_*'))
                      if os.path.exists(os.path.join(d, 'verify_arrays.nc')))
        for rd in regs:
            print(f"Dressed baseline -> {rd}")
            score_region(rd, cfg)
