"""
x_ensemble_calibrate_v2.py — Full-span calibration with year-split + swap CV
(referendum item R3). Successor to the pilot x_ensemble_calibrate.py for the
2-year v2 archive.

Protocol: fit on 2024 -> evaluate held-out 2025, AND fit on 2025 -> evaluate
held-out 2024. Every cycle therefore has calibrated output from parameters it
never influenced; the 'combined' rows pool both held-out halves and are the
paper numbers ("calibrated" = verified out-of-sample over the full span).

Method (kept deliberately simple + auditable):
    member' = median − bias + gamma · (member − median)
per lead-bin: `bias` = mean(ensemble-median − obs) on the fit year;
`gamma` = scalar spread inflation grid-fit so the TARGET_ALPHA central
interval covers TARGET_ALPHA of fit-year observations. Applied identically in
residual and water-level space (tide + SLR cancel in the deviation term).

Memory-safe: streams the 5-D member array in cycle chunks; keeps only
per-sample quantiles (for coverage) and chunked CRPS accumulators. Writes NO
calibrated member arrays — downstream consumers apply the JSON parameters on
the fly.

Reads:
    {BASE_DIR}/{span}/region_XX/verify_arrays.nc
Writes (per region):
    calibration_v2_params.json      {direction: {bias_cm, gamma per lead-bin}}
    calibration_v2_coverage.csv     direction,nominal,n,cov_raw,cov_cal,width_raw_cm,width_cal_cm
    calibration_v2_crps.csv         direction,n,crps_raw_cm,crps_cal_cm

Config: x_ensemble_calibrate_v2_config.json
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
NOMINAL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
GAMMA_GRID = np.arange(1.0, 4.51, 0.02)
CHUNK = 128                     # cycles per streamed chunk


def load_config(path=CONFIG_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(
        BASE_DIR='outputs/uq_postprocessing/nn_24lb_72fh_v2/gefs_psfc_v2_raw',
        TARGET_ALPHA=0.90,
        SPLIT_YEAR=2025,        # cycles before Jan 1 of this year = year A
    )


def _quantile_pairs():
    """(lo, hi) member-quantile levels for each nominal central interval."""
    return [((1 - a) / 2, 1 - (1 - a) / 2) for a in NOMINAL_LEVELS]


def _lead_bin_index(n_lead):
    """lead index (0-based) -> bin index array."""
    idx = np.zeros(n_lead, dtype=int)
    for b, (lo, hi) in enumerate(LEAD_BINS):
        idx[lo - 1:hi] = b
    return idx


def crps_ensemble(forecasts, obs):
    """Empirical CRPS, forecasts (..., M) sorted or not, obs (...)."""
    f = np.sort(forecasts, axis=-1)
    y = obs[..., None]
    n = f.shape[-1]
    term1 = np.nanmean(np.abs(f - y), axis=-1)
    i = np.arange(1, n + 1)
    coeffs = (2 * i - n - 1) / (n * n)
    term2 = np.nansum(coeffs * f, axis=-1)
    return term1 - term2


# =============================================================================
# Pass 1 — per-sample ensemble quantiles (streamed)
# =============================================================================
def collect_quantiles(ds):
    """Median + interval-edge quantiles per (cycle, lead, station).

    Returns dict with 'med' (C,L,S) and 'qlo'/'qhi' lists indexed like
    NOMINAL_LEVELS, plus obs (C,L,S)."""
    n_c = ds.sizes['cycle_time']
    n_l = ds.sizes['lead_hour']
    n_s = ds.sizes['station']
    pairs = _quantile_pairs()
    med = np.full((n_c, n_l, n_s), np.nan, np.float32)
    qlo = [np.full((n_c, n_l, n_s), np.nan, np.float32) for _ in pairs]
    qhi = [np.full((n_c, n_l, n_s), np.nan, np.float32) for _ in pairs]
    for c0 in range(0, n_c, CHUNK):
        c1 = min(c0 + CHUNK, n_c)
        a = ds.ai_residual_member.isel(cycle_time=slice(c0, c1)).values
        a = a.reshape(a.shape[0], -1, n_l, n_s)          # (c, 93, L, S)
        med[c0:c1] = np.nanmedian(a, axis=1)
        for k, (lo, hi) in enumerate(pairs):
            q = np.nanquantile(a, [lo, hi], axis=1)
            qlo[k][c0:c1] = q[0]
            qhi[k][c0:c1] = q[1]
    obs = ds.obs_residual.values.astype(np.float32)
    return dict(med=med, qlo=qlo, qhi=qhi, obs=obs)


# =============================================================================
# Fit on one year
# =============================================================================
def fit_direction(q, fit_mask, lead_bin_idx, target_alpha):
    """Per lead-bin bias + gamma on fit-year samples."""
    med, obs = q['med'], q['obs']
    k90 = NOMINAL_LEVELS.index(target_alpha)
    lo90, hi90 = q['qlo'][k90], q['qhi'][k90]
    bias, gamma = {}, {}
    for b, (blo, bhi) in enumerate(LEAD_BINS):
        sel_l = lead_bin_idx == b
        m = med[fit_mask][:, sel_l, :]
        y = obs[fit_mask][:, sel_l, :]
        lo = lo90[fit_mask][:, sel_l, :]
        hi = hi90[fit_mask][:, sel_l, :]
        v = np.isfinite(m) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if v.sum() < 100:
            bias[b] = 0.0
            gamma[b] = 1.0
            continue
        mv, yv, lov, hiv = m[v], y[v], lo[v], hi[v]
        bb = float(np.mean(mv - yv))
        ctr = mv - bb
        dlo, dhi = lov - mv, hiv - mv           # deviations from the median
        # coverage(gamma) over the grid, vectorized
        cov = np.array([np.mean((ctr + g * dlo <= yv) & (yv <= ctr + g * dhi))
                        for g in GAMMA_GRID])
        gg = float(GAMMA_GRID[int(np.argmin(np.abs(cov - target_alpha)))])
        bias[b] = bb
        gamma[b] = gg
    return bias, gamma


# =============================================================================
# Evaluate on the held-out year
# =============================================================================
def eval_direction(q, eval_mask, lead_bin_idx, bias, gamma):
    """Held-out coverage/width per nominal level, raw vs calibrated."""
    med, obs = q['med'], q['obs']
    b_arr = np.array([bias[b] for b in range(len(LEAD_BINS))],
                     np.float32)[lead_bin_idx]        # (L,)
    g_arr = np.array([gamma[b] for b in range(len(LEAD_BINS))],
                     np.float32)[lead_bin_idx]
    rows = []
    for k, a in enumerate(NOMINAL_LEVELS):
        lo = q['qlo'][k][eval_mask]
        hi = q['qhi'][k][eval_mask]
        m = med[eval_mask]
        y = obs[eval_mask]
        v = np.isfinite(m) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if not v.any():
            continue
        ctr = (m - b_arr[None, :, None])[v]
        glo = (m + g_arr[None, :, None] * (lo - m) - b_arr[None, :, None])[v]
        ghi = (m + g_arr[None, :, None] * (hi - m) - b_arr[None, :, None])[v]
        yv, lov, hiv = y[v], lo[v], hi[v]
        rows.append(dict(
            nominal=a, n=int(v.sum()),
            cov_raw=float(np.mean((lov <= yv) & (yv <= hiv))),
            cov_cal=float(np.mean((glo <= yv) & (yv <= ghi))),
            width_raw_cm=float(np.mean(hiv - lov)) * 100,
            width_cal_cm=float(np.mean(ghi - glo)) * 100,
        ))
    return rows


def eval_crps(ds, eval_idx, lead_bin_idx, bias, gamma):
    """Held-out CRPS raw vs calibrated, streamed over cycle chunks."""
    n_l = ds.sizes['lead_hour']
    n_s = ds.sizes['station']
    b_arr = np.array([bias[b] for b in range(len(LEAD_BINS))],
                     np.float32)[lead_bin_idx][None, :, None]
    g_arr = np.array([gamma[b] for b in range(len(LEAD_BINS))],
                     np.float32)[lead_bin_idx][None, :, None]
    obs = ds.obs_residual.values.astype(np.float32)
    s_raw = s_cal = 0.0
    n_tot = 0
    for c0 in range(0, len(eval_idx), CHUNK):
        sel = eval_idx[c0:c0 + CHUNK]
        a = ds.ai_residual_member.isel(cycle_time=sel).values
        a = a.reshape(a.shape[0], -1, n_l, n_s)          # (c, 93, L, S)
        y = obs[sel]
        med = np.nanmedian(a, axis=1)                    # (c, L, S)
        v = np.isfinite(y) & np.isfinite(med) & np.all(np.isfinite(a), axis=1)
        if not v.any():
            continue
        f = np.moveaxis(a, 1, -1)[v]                     # (N, 93)
        yv = y[v]
        s_raw += float(np.nansum(crps_ensemble(f, yv)))
        medv = med[v][:, None]
        cal = medv + np.broadcast_to(g_arr, med.shape)[v][:, None] * (f - medv) \
            - np.broadcast_to(b_arr, med.shape)[v][:, None]
        s_cal += float(np.nansum(crps_ensemble(cal, yv)))
        n_tot += int(v.sum())
    if n_tot == 0:
        return np.nan, np.nan, 0
    return 100 * s_raw / n_tot, 100 * s_cal / n_tot, n_tot


# =============================================================================
# Region driver
# =============================================================================
def calibrate_region(verify_dir, cfg):
    fc_path = os.path.join(verify_dir, 'verify_arrays.nc')
    out_json = os.path.join(verify_dir, 'calibration_v2_params.json')
    out_cov = os.path.join(verify_dir, 'calibration_v2_coverage.csv')
    out_crps = os.path.join(verify_dir, 'calibration_v2_crps.csv')
    if os.path.exists(out_cov) and os.path.exists(out_json):
        print(f"  [exists, skip] {verify_dir}")
        return
    ds = xr.open_dataset(fc_path, chunks={'cycle_time': CHUNK})
    years = pd.to_datetime(ds.cycle_time.values).year
    split = int(cfg.get('SPLIT_YEAR', 2025))
    mask_a = years < split                 # 2024
    mask_b = ~mask_a                       # 2025+
    lead_bin_idx = _lead_bin_index(ds.sizes['lead_hour'])
    target = float(cfg.get('TARGET_ALPHA', 0.90))

    q = collect_quantiles(ds)
    directions = [('fit2024_eval2025', mask_a, mask_b),
                  ('fit2025_eval2024', mask_b, mask_a)]
    params, cov_rows, crps_rows = {}, [], []
    for name, fmask, emask in directions:
        bias, gamma = fit_direction(q, fmask, lead_bin_idx, target)
        params[name] = dict(
            bias_cm={f'{lo}-{hi}': round(100 * bias[b], 3)
                     for b, (lo, hi) in enumerate(LEAD_BINS)},
            gamma={f'{lo}-{hi}': round(gamma[b], 3)
                   for b, (lo, hi) in enumerate(LEAD_BINS)},
        )
        for row in eval_direction(q, emask, lead_bin_idx, bias, gamma):
            cov_rows.append(dict(direction=name, **row))
        crps_raw, crps_cal, n = eval_crps(
            ds, np.where(emask)[0], lead_bin_idx, bias, gamma)
        crps_rows.append(dict(direction=name, n=n,
                              crps_raw_cm=round(crps_raw, 3),
                              crps_cal_cm=round(crps_cal, 3)))
        print(f"  {name}: gamma={list(params[name]['gamma'].values())} "
              f"CRPS {crps_raw:.2f}->{crps_cal:.2f} cm (n={n:,})")
    ds.close()

    # Combined held-out rows (n-weighted union of the two directions).
    cov = pd.DataFrame(cov_rows)
    comb = []
    for a, g in cov.groupby('nominal'):
        w = g['n'].to_numpy(float)
        comb.append(dict(direction='combined', nominal=a, n=int(w.sum()),
                         **{c: float(np.average(g[c], weights=w))
                            for c in ['cov_raw', 'cov_cal',
                                      'width_raw_cm', 'width_cal_cm']}))
    cov = pd.concat([cov, pd.DataFrame(comb)], ignore_index=True)
    crps = pd.DataFrame(crps_rows)
    w = crps['n'].to_numpy(float)
    crps = pd.concat([crps, pd.DataFrame([dict(
        direction='combined', n=int(w.sum()),
        crps_raw_cm=float(np.average(crps['crps_raw_cm'], weights=w)),
        crps_cal_cm=float(np.average(crps['crps_cal_cm'], weights=w)),
    )])], ignore_index=True)

    with open(out_json, 'w') as f:
        json.dump(dict(method='bias+spread_inflation', target_alpha=target,
                       lead_bins=[f'{lo}-{hi}' for lo, hi in LEAD_BINS],
                       split_year=split, directions=params), f, indent=2)
    cov.to_csv(out_cov, index=False)
    crps.to_csv(out_crps, index=False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    spans = [d for d in glob.glob(os.path.join(cfg['BASE_DIR'], '*'))
             if os.path.isdir(d)]
    for span_dir in spans:
        regs = sorted(d for d in glob.glob(os.path.join(span_dir, 'region_*'))
                      if os.path.exists(os.path.join(d, 'verify_arrays.nc')))
        for rd in regs:
            print(f"Calibrate v2 -> {rd}")
            calibrate_region(rd, cfg)
