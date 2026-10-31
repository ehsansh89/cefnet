"""
o_ensemble_uq.py — Score one region/source/span for raw (or calibrated)
ensemble forecasts. Produces diagnostic UQ figures and tables.

Reads:
    outputs/uq_postprocessing/{run}/{exp_tag}/{span}/region_XX/verify_arrays.nc
    optional: .../calibrated_{method}/verify_arrays_calibrated.nc

Writes (per region, per raw|calibrated_{method}):
    metrics_overall.csv
    metrics_by_lead.csv
    metrics_by_station.csv
    metrics_by_lead_bin.csv
    coverage_by_nominal.csv
    rank_histogram.csv
    crps_by_lead.csv
    crpss_by_lead_vs_stofs.csv
    crpss_by_lead_vs_ofs.csv
    spread_skill_bins.csv
    sharpness_by_lead.csv
    brier_by_threshold.csv
    reliability_bins.csv
    roc_pr_by_threshold.csv
    extreme_tail_metrics.csv
    figures/fig_uq_r{1..10}*.png
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from o_style import setup, COLORS, FIG, savefig


CONFIG_PATH = os.environ.get('X_ENSEMBLE_UQ_CONFIG', 'o_ensemble_uq_config_v2.json')

LEAD_BINS = [(1, 12), (13, 24), (25, 36), (37, 48), (49, 72)]
NOMINAL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
PROB_BINS = np.linspace(0.0, 1.0, 11)        # 10 reliability bins


# =============================================================================
# CORE METRIC HELPERS
# =============================================================================
def flatten_members(arr):
    """(cyc, atm, seed, lead, sta) -> (cyc, member_flat, lead, sta)."""
    n_c, n_a, n_s, n_l, n_st = arr.shape
    return arr.reshape(n_c, n_a * n_s, n_l, n_st)


def crps_ensemble(forecasts, obs):
    """Empirical CRPS for an ensemble.

    forecasts: (..., M) along the last axis.
    obs:       (...,)
    Returns CRPS with the same broadcast shape as obs (last axis dropped).
    Definition: CRPS = mean|f_i - y| - 0.5 * mean|f_i - f_j|.
    """
    f = np.sort(forecasts, axis=-1)
    y = obs[..., None]
    n = f.shape[-1]
    if n == 0:
        return np.full(obs.shape, np.nan)
    # mean |f - y|
    term1 = np.nanmean(np.abs(f - y), axis=-1)
    # 0.5 * mean |f_i - f_j| via sorted formula:
    # mean|f_i - f_j| = (2/(n^2)) sum_i (2i - n - 1) f_(i)
    i = np.arange(1, n + 1)
    coeffs = (2 * i - n - 1) / (n * n)
    term2 = 0.5 * 2.0 * np.nansum(coeffs * f, axis=-1)
    return term1 - term2


def coverage_for_level(forecasts, obs, alpha):
    """Empirical coverage of the central alpha interval (no monotonicity tests).

    forecasts: (N, M) - one forecast per row of an ensemble of M.
    obs:       (N,)
    Returns (coverage, mean_interval_width).

    Masks invalid rows first, then uses plain `np.percentile` on the clean
    subset (both bounds in one call) — far faster than `nanpercentile` over the
    full array, which dominated runtime on the 2-year archive.
    """
    valid = np.isfinite(obs) & np.all(np.isfinite(forecasts), axis=1)
    if not valid.any():
        return np.nan, np.nan
    f = forecasts[valid]
    y = obs[valid]
    lo = (1.0 - alpha) / 2.0 * 100.0
    hi = (1.0 - (1.0 - alpha) / 2.0) * 100.0
    f_lo, f_hi = np.percentile(f, [lo, hi], axis=1)
    inside = (y >= f_lo) & (y <= f_hi)
    return float(inside.mean()), float(np.mean(f_hi - f_lo))


def rank_of_obs(forecasts, obs):
    """Empirical rank of obs among the M ensemble members + 1 ties broken
    uniformly at random (small ties are unlikely with float32). Returns int
    rank in [1, M+1]. NaNs in obs/forecasts -> NaN rank.
    """
    M = forecasts.shape[-1]
    rng = np.random.default_rng(0)
    flat_f = forecasts.reshape(-1, M)
    flat_y = obs.reshape(-1)
    ranks = np.full(flat_y.shape, np.nan, dtype=np.float32)
    # Vectorized: rank = (#members below obs) + 1, with a uniform tie-break
    # offset in [0, #equal] — matches the original element-wise semantics but
    # without the per-point Python loop (which hung on the 2-year archive).
    valid = np.isfinite(flat_y) & np.all(np.isfinite(flat_f), axis=1)
    if valid.any():
        fv = flat_f[valid]
        yv = flat_y[valid][:, None]
        below = (fv < yv).sum(axis=1)
        equal = (fv == yv).sum(axis=1)
        offset = rng.integers(0, equal + 1)        # 0..equal inclusive
        ranks[valid] = (below + 1 + offset).astype(np.float32)
    return ranks.reshape(obs.shape)


# =============================================================================
# METRIC COMPUTATIONS
# =============================================================================
def score_overall(ai_ens, obs, stofs, ofs):
    """`ai_ens` (cyc, M, lead, sta), `obs/stofs/ofs` (cyc, lead, sta).
    Returns dict of headline numbers (cm units for residuals)."""
    ai_med = np.nanmedian(ai_ens, axis=1)
    ai_mean = np.nanmean(ai_ens, axis=1)
    mask = np.isfinite(obs)

    def _stats(pred, obs_, mask_):
        m = mask_ & np.isfinite(pred)
        if not m.any():
            return dict(n=0, mae_cm=np.nan, rmse_cm=np.nan, bias_cm=np.nan,
                         p95_abs_error_cm=np.nan)
        d = (pred - obs_)[m]
        return dict(
            n=int(m.sum()),
            mae_cm=float(np.mean(np.abs(d)) * 100),
            rmse_cm=float(np.sqrt(np.mean(d ** 2)) * 100),
            bias_cm=float(np.mean(d) * 100),
            p95_abs_error_cm=float(np.percentile(np.abs(d), 95) * 100),
        )
    out = {}
    for name, pred in [('ai_median', ai_med), ('ai_mean', ai_mean),
                        ('stofs', stofs), ('regional_ofs', ofs)]:
        out[name] = _stats(pred, obs, mask)
    return out


def score_by_lead(ai_ens, obs, stofs, ofs):
    n_l = ai_ens.shape[2]
    rows = []
    ai_med = np.nanmedian(ai_ens, axis=1)
    ai_mean = np.nanmean(ai_ens, axis=1)
    for l in range(n_l):
        for system, pred in [('ai_median', ai_med[:, l]),
                               ('ai_mean', ai_mean[:, l]),
                               ('stofs', stofs[:, l]),
                               ('regional_ofs', ofs[:, l])]:
            y = obs[:, l]
            m = np.isfinite(pred) & np.isfinite(y)
            if not m.any():
                rows.append(dict(forecast_system=system, lead_hour=l + 1,
                                  n=0, mae_cm=np.nan, rmse_cm=np.nan,
                                  bias_cm=np.nan, p95_abs_error_cm=np.nan))
                continue
            d = pred[m] - y[m]
            rows.append(dict(
                forecast_system=system, lead_hour=l + 1, n=int(m.sum()),
                mae_cm=float(np.mean(np.abs(d)) * 100),
                rmse_cm=float(np.sqrt(np.mean(d ** 2)) * 100),
                bias_cm=float(np.mean(d) * 100),
                p95_abs_error_cm=float(np.percentile(np.abs(d), 95) * 100),
            ))
    return pd.DataFrame(rows)


def score_by_station(ai_ens, obs, station_ids):
    n_st = ai_ens.shape[-1]
    ai_med = np.nanmedian(ai_ens, axis=1)
    ai_mean = np.nanmean(ai_ens, axis=1)
    rows = []
    for s in range(n_st):
        for system, pred in [('ai_median', ai_med[..., s]),
                               ('ai_mean', ai_mean[..., s])]:
            y = obs[..., s]
            m = np.isfinite(pred) & np.isfinite(y)
            if not m.any():
                continue
            d = pred[m] - y[m]
            rows.append(dict(
                station_id=str(station_ids[s]), forecast_system=system,
                n=int(m.sum()),
                mae_cm=float(np.mean(np.abs(d)) * 100),
                rmse_cm=float(np.sqrt(np.mean(d ** 2)) * 100),
                bias_cm=float(np.mean(d) * 100),
            ))
    return pd.DataFrame(rows)


def coverage_table(ai_ens, obs):
    """Per nominal level + per lead bin: empirical coverage + mean width."""
    rows = []
    n_l = ai_ens.shape[2]
    for (lo, hi) in LEAD_BINS:
        l0 = lo - 1
        l1 = min(hi, n_l)
        flat_f = ai_ens[:, :, l0:l1, :].transpose(0, 2, 3, 1)
        flat_f = flat_f.reshape(-1, flat_f.shape[-1])
        flat_y = obs[:, l0:l1, :].reshape(-1)
        for alpha in NOMINAL_LEVELS:
            cov, width = coverage_for_level(flat_f, flat_y, alpha)
            rows.append(dict(
                lead_bin=f'{lo}-{hi}', nominal=alpha,
                observed_coverage=cov, mean_interval_width_cm=width * 100
                  if np.isfinite(width) else np.nan,
                n=int(np.isfinite(flat_y).sum()),
            ))
    # Overall (all leads)
    flat_f = ai_ens.transpose(0, 2, 3, 1).reshape(-1, ai_ens.shape[1])
    flat_y = obs.reshape(-1)
    for alpha in NOMINAL_LEVELS:
        cov, width = coverage_for_level(flat_f, flat_y, alpha)
        rows.append(dict(
            lead_bin='all', nominal=alpha,
            observed_coverage=cov, mean_interval_width_cm=width * 100
              if np.isfinite(width) else np.nan,
            n=int(np.isfinite(flat_y).sum()),
        ))
    return pd.DataFrame(rows)


def rank_histogram_table(ai_ens, obs, max_rows=200000):
    """Compute rank-histogram counts per lead bin + overall."""
    M = ai_ens.shape[1]
    rows = []
    for (lo, hi) in LEAD_BINS:
        l0 = lo - 1
        l1 = min(hi, ai_ens.shape[2])
        flat_f = ai_ens[:, :, l0:l1, :].transpose(0, 2, 3, 1)
        flat_f = flat_f.reshape(-1, M)
        flat_y = obs[:, l0:l1, :].reshape(-1)
        # Mask + optional subsample for compute.
        valid = np.isfinite(flat_y) & np.all(np.isfinite(flat_f), axis=1)
        flat_f = flat_f[valid]
        flat_y = flat_y[valid]
        if flat_y.size > max_rows:
            idx = np.random.default_rng(42).choice(flat_y.size, max_rows,
                                                       replace=False)
            flat_f = flat_f[idx]; flat_y = flat_y[idx]
        r = rank_of_obs(flat_f, flat_y)
        for k in range(1, M + 2):
            rows.append(dict(
                lead_bin=f'{lo}-{hi}',
                rank=k,
                count=int(np.sum(r == k)),
                expected_count=float(r.size / (M + 1)) if r.size else np.nan,
                n=int(r.size),
            ))
    return pd.DataFrame(rows)


def crps_by_lead(ai_ens, obs):
    """Empirical CRPS by lead, in cm. Forecasts are residuals (m)."""
    n_l = ai_ens.shape[2]
    rows = []
    for l in range(n_l):
        f = ai_ens[:, :, l, :].transpose(0, 2, 1).reshape(-1, ai_ens.shape[1])
        y = obs[:, l, :].reshape(-1)
        valid = np.isfinite(y) & np.all(np.isfinite(f), axis=1)
        if not valid.any():
            rows.append(dict(lead_hour=l + 1, n=0,
                              ai_crps_cm=np.nan,
                              ai_median_mae_cm=np.nan,
                              ai_mean_mae_cm=np.nan,
                              ensemble_spread_p90_cm=np.nan,
                              ensemble_iqr_cm=np.nan))
            continue
        f_v = f[valid]; y_v = y[valid]
        crps = crps_ensemble(f_v, y_v) * 100.0
        spread_p90 = (np.percentile(f_v, 95, axis=1)
                       - np.percentile(f_v, 5, axis=1)) * 100.0
        iqr = (np.percentile(f_v, 75, axis=1)
                - np.percentile(f_v, 25, axis=1)) * 100.0
        med = np.median(f_v, axis=1)
        rows.append(dict(
            lead_hour=l + 1, n=int(valid.sum()),
            ai_crps_cm=float(np.mean(crps)),
            ai_median_mae_cm=float(np.mean(np.abs(med - y_v)) * 100),
            ai_mean_mae_cm=float(np.mean(np.abs(np.mean(f_v, axis=1) - y_v))
                                    * 100),
            ensemble_spread_p90_cm=float(np.mean(spread_p90)),
            ensemble_iqr_cm=float(np.mean(iqr)),
        ))
    return pd.DataFrame(rows)


def crpss_vs_deterministic(ai_ens, obs, baseline, name):
    """CRPSS vs a deterministic baseline. baseline (cyc, lead, sta)."""
    n_l = ai_ens.shape[2]
    rows = []
    for l in range(n_l):
        f = ai_ens[:, :, l, :].transpose(0, 2, 1).reshape(-1, ai_ens.shape[1])
        y = obs[:, l, :].reshape(-1)
        b = baseline[:, l, :].reshape(-1)
        valid = (np.isfinite(y) & np.all(np.isfinite(f), axis=1)
                  & np.isfinite(b))
        if not valid.any():
            rows.append(dict(lead_hour=l + 1, n=0, ai_crps_cm=np.nan,
                              baseline_crps_cm=np.nan, crpss=np.nan,
                              baseline=name))
            continue
        ai_crps = float(np.mean(crps_ensemble(f[valid], y[valid])) * 100)
        base_crps = float(np.mean(np.abs(b[valid] - y[valid])) * 100)
        crpss = (1.0 - ai_crps / base_crps) if base_crps > 0 else np.nan
        rows.append(dict(
            lead_hour=l + 1, n=int(valid.sum()),
            ai_crps_cm=ai_crps, baseline_crps_cm=base_crps,
            crpss=crpss, baseline=name,
        ))
    return pd.DataFrame(rows)


def spread_skill_bins(ai_ens, obs, n_bins=10):
    """For each lead bin, bin |error_of_mean| by ensemble spread."""
    rows = []
    for (lo, hi) in LEAD_BINS:
        l0 = lo - 1
        l1 = min(hi, ai_ens.shape[2])
        f = ai_ens[:, :, l0:l1, :].transpose(0, 2, 3, 1)
        f = f.reshape(-1, f.shape[-1])
        y = obs[:, l0:l1, :].reshape(-1)
        valid = np.isfinite(y) & np.all(np.isfinite(f), axis=1)
        if not valid.any():
            continue
        f = f[valid]; y = y[valid]
        spread_p90 = (np.percentile(f, 95, axis=1)
                       - np.percentile(f, 5, axis=1))
        err = np.abs(np.median(f, axis=1) - y)
        if spread_p90.size < 50:
            continue
        edges = np.quantile(spread_p90, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        bin_idx = np.digitize(spread_p90, edges[1:-1])
        for b in range(len(edges) - 1):
            m = (bin_idx == b)
            if not m.any():
                continue
            rows.append(dict(
                lead_bin=f'{lo}-{hi}',
                spread_bin=b,
                spread_bin_left_cm=float(edges[b]) * 100,
                spread_bin_right_cm=float(edges[b + 1]) * 100,
                n=int(m.sum()),
                mean_spread_cm=float(np.mean(spread_p90[m])) * 100,
                median_abs_error_cm=float(np.median(err[m])) * 100,
                rmse_cm=float(np.sqrt(np.mean(err[m] ** 2))) * 100,
                mae_cm=float(np.mean(err[m])) * 100,
            ))
    return pd.DataFrame(rows)


def sharpness_by_lead(ai_ens):
    """Mean and median p05-p95 + p10-p90 widths per lead, in cm."""
    n_l = ai_ens.shape[2]
    rows = []
    for l in range(n_l):
        f = ai_ens[:, :, l, :].transpose(0, 2, 1).reshape(-1, ai_ens.shape[1])
        v = np.all(np.isfinite(f), axis=1)
        if not v.any():
            rows.append(dict(lead_hour=l + 1, n=0,
                              mean_p05_p95_width_cm=np.nan,
                              median_p05_p95_width_cm=np.nan,
                              mean_p10_p90_width_cm=np.nan,
                              median_p10_p90_width_cm=np.nan))
            continue
        f = f[v]
        w90 = (np.percentile(f, 95, axis=1)
                - np.percentile(f, 5, axis=1)) * 100
        w80 = (np.percentile(f, 90, axis=1)
                - np.percentile(f, 10, axis=1)) * 100
        rows.append(dict(
            lead_hour=l + 1, n=int(v.sum()),
            mean_p05_p95_width_cm=float(np.mean(w90)),
            median_p05_p95_width_cm=float(np.median(w90)),
            mean_p10_p90_width_cm=float(np.mean(w80)),
            median_p10_p90_width_cm=float(np.median(w80)),
        ))
    return pd.DataFrame(rows)


def brier_and_reliability(ai_wl, obs_wl, stofs_wl, ofs_wl, thr_arr,
                           threshold_name, event_kind='pointwise',
                           climatology_rate=None):
    """ai_wl (cyc, M, lead, sta); obs_wl/stofs_wl/ofs_wl (cyc, lead, sta);
    thr_arr (sta,). Returns (brier_df, reliability_df).

    event_kind='pointwise': event per (cyc, lead, sta).
    event_kind='window_peak': event per (cyc, sta), where peak = max over lead 1..L.
    """
    if event_kind == 'window_peak':
        # collapse lead axis
        ai_peak = np.nanmax(ai_wl, axis=2)        # (cyc, M, sta)
        obs_peak = np.nanmax(obs_wl, axis=1)      # (cyc, sta)
        stofs_peak = np.nanmax(stofs_wl, axis=1)
        ofs_peak = np.nanmax(ofs_wl, axis=1)

        thr_b = thr_arr[None, :]                  # (1, sta)
        obs_evt = (obs_peak >= thr_b).astype(np.float32)
        ai_prob = np.nanmean((ai_peak >= thr_b[:, None, :]).astype(np.float32),
                              axis=1)            # (cyc, sta)
        stofs_evt = (stofs_peak >= thr_b).astype(np.float32)
        ofs_evt = (ofs_peak >= thr_b).astype(np.float32)
        mask = np.isfinite(obs_peak) & np.isfinite(thr_b)
    else:
        thr_b = thr_arr[None, None, :]            # (1, 1, sta)
        obs_evt = (obs_wl >= thr_b).astype(np.float32)
        ai_prob = np.nanmean(
            (ai_wl >= thr_b[:, None, :, :]).astype(np.float32), axis=1,
        )
        stofs_evt = (stofs_wl >= thr_b).astype(np.float32)
        ofs_evt = (ofs_wl >= thr_b).astype(np.float32)
        mask = np.isfinite(obs_wl) & np.isfinite(thr_b)

    valid = mask & np.isfinite(ai_prob) & np.isfinite(obs_evt)
    if not valid.any():
        return pd.DataFrame(), pd.DataFrame()

    ai_p_v = ai_prob[valid]
    obs_v = obs_evt[valid]
    n = int(valid.sum())
    event_count = int(obs_v.sum())
    event_rate = event_count / max(n, 1)
    ai_brier = float(np.mean((ai_p_v - obs_v) ** 2))
    clim_rate = climatology_rate if climatology_rate is not None else event_rate
    clim_brier = float(np.mean((clim_rate - obs_v) ** 2))

    stofs_v = stofs_evt[valid]
    ofs_v = ofs_evt[valid]
    stofs_brier = (float(np.mean((stofs_v - obs_v) ** 2))
                    if np.isfinite(stofs_v).any() else np.nan)
    ofs_brier = (float(np.mean((ofs_v - obs_v) ** 2))
                  if np.isfinite(ofs_v).any() else np.nan)

    brier_row = dict(
        threshold_name=threshold_name, event_definition=event_kind,
        n=n, event_count=event_count, event_rate=event_rate,
        ai_brier=ai_brier, climatology_brier=clim_brier,
        stofs_brier=stofs_brier, ofs_brier=ofs_brier,
        bss_vs_climatology=(1.0 - ai_brier / clim_brier) if clim_brier > 0 else np.nan,
        bss_vs_stofs=(1.0 - ai_brier / stofs_brier)
                       if stofs_brier and stofs_brier > 0 else np.nan,
        bss_vs_ofs=(1.0 - ai_brier / ofs_brier)
                     if ofs_brier and ofs_brier > 0 else np.nan,
    )

    # Reliability bins.
    bins = PROB_BINS
    bin_idx = np.digitize(ai_p_v, bins[1:-1])
    rel_rows = []
    for b in range(len(bins) - 1):
        m = (bin_idx == b)
        if not m.any():
            continue
        rel_rows.append(dict(
            threshold_name=threshold_name, event_definition=event_kind,
            prob_bin_left=float(bins[b]), prob_bin_right=float(bins[b + 1]),
            prob_bin_center=float(0.5 * (bins[b] + bins[b + 1])),
            n=int(m.sum()),
            predicted_probability_mean=float(np.mean(ai_p_v[m])),
            observed_frequency=float(np.mean(obs_v[m])),
        ))
    return pd.DataFrame([brier_row]), pd.DataFrame(rel_rows)


def extreme_tail_metrics(ai_ens, obs):
    """Coverage on extreme observed-residual quantiles."""
    flat_f = ai_ens.transpose(0, 2, 3, 1).reshape(-1, ai_ens.shape[1])
    flat_y = obs.reshape(-1)
    valid = np.isfinite(flat_y) & np.all(np.isfinite(flat_f), axis=1)
    flat_f = flat_f[valid]; flat_y = flat_y[valid]
    rows = []
    if flat_y.size == 0:
        return pd.DataFrame()
    p05 = np.percentile(flat_f, 5, axis=1)
    p95 = np.percentile(flat_f, 95, axis=1)
    p10 = np.percentile(flat_f, 10, axis=1)
    p90 = np.percentile(flat_f, 90, axis=1)
    median = np.median(flat_f, axis=1)
    above_p95 = (flat_y > p95).astype(np.float32).mean()
    below_p05 = (flat_y < p05).astype(np.float32).mean()
    cov_p90 = ((flat_y >= p05) & (flat_y <= p95)).astype(np.float32).mean()
    cov_p80 = ((flat_y >= p10) & (flat_y <= p90)).astype(np.float32).mean()

    abs_y = np.abs(flat_y)
    for tail_name, q in [('all', 0.0), ('top10pct', 0.9),
                           ('top5pct', 0.95), ('top1pct', 0.99)]:
        if q == 0.0:
            sel = np.ones_like(abs_y, dtype=bool)
        else:
            thr_val = np.percentile(abs_y, q * 100)
            sel = abs_y >= thr_val
        if not sel.any():
            continue
        rows.append(dict(
            tail_definition=tail_name,
            n=int(sel.sum()),
            median_abs_error_cm=float(np.median(
                np.abs(median[sel] - flat_y[sel])) * 100),
            p05_p95_coverage=float(((flat_y[sel] >= p05[sel])
                                      & (flat_y[sel] <= p95[sel])).mean()),
            p10_p90_coverage=float(((flat_y[sel] >= p10[sel])
                                      & (flat_y[sel] <= p90[sel])).mean()),
            mean_interval_width_cm=float(np.mean(p95[sel] - p05[sel]) * 100),
            obs_above_p95_rate=float((flat_y[sel] > p95[sel]).mean()),
            obs_below_p05_rate=float((flat_y[sel] < p05[sel]).mean()),
        ))
    return pd.DataFrame(rows)


# =============================================================================
# FIGURE HELPERS
# =============================================================================
def fig_coverage_curve(cov_df, out_path):
    setup()
    fig, ax = plt.subplots(figsize=FIG['single'])
    overall = cov_df[cov_df.lead_bin == 'all']
    ax.plot([0, 1], [0, 1], color='#999', linewidth=0.8, linestyle='--')
    ax.plot(overall.nominal, overall.observed_coverage,
             color=COLORS['pred'], marker='o', linewidth=1.4, label='all leads')
    for (lo, hi) in LEAD_BINS:
        sub = cov_df[cov_df.lead_bin == f'{lo}-{hi}']
        if sub.empty:
            continue
        ax.plot(sub.nominal, sub.observed_coverage, linewidth=0.8, alpha=0.7,
                 label=f'{lo}-{hi}h')
    ax.set_xlabel('nominal interval'); ax.set_ylabel('empirical coverage')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Coverage by nominal level')
    savefig(fig, out_path)
    plt.close(fig)


def fig_rank_hist(rank_df, n_members, out_path):
    setup()
    bins = sorted(rank_df.lead_bin.unique())
    n_panels = len(bins)
    fig, axes = plt.subplots(1, n_panels, figsize=(FIG['hero'][0], 2.6),
                              sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, lb in zip(axes, bins):
        sub = rank_df[rank_df.lead_bin == lb]
        total = sub['count'].sum()
        if total == 0:
            ax.set_title(f'lead {lb}h (no data)', fontsize=8)
            continue
        ax.bar(sub['rank'], sub['count'] / total, color=COLORS['pred'],
                width=1.0, edgecolor='none')
        ax.axhline(1 / (n_members + 1), color='#999', linewidth=0.7,
                    linestyle='--')
        ax.set_title(f'lead {lb}h', fontsize=8)
        ax.set_xlabel('rank')
    axes[0].set_ylabel('frequency')
    fig.suptitle(f'Rank histogram (M+1={n_members + 1} bins)', fontsize=10)
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def fig_crps_by_lead(crps_df, vs_stofs_df, vs_ofs_df, out_path):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(FIG['hero'][0], 2.8), sharex=True)
    ax = axes[0]
    ax.plot(crps_df.lead_hour, crps_df.ai_crps_cm, color=COLORS['pred'],
             linewidth=1.4, label='AI CRPS')
    if vs_stofs_df is not None and not vs_stofs_df.empty:
        ax.plot(vs_stofs_df.lead_hour, vs_stofs_df.baseline_crps_cm,
                 color=COLORS['stofs'], linewidth=1.2, label='STOFS |err|')
    if vs_ofs_df is not None and not vs_ofs_df.empty:
        ax.plot(vs_ofs_df.lead_hour, vs_ofs_df.baseline_crps_cm,
                 color=COLORS['ofs'], linewidth=1.2, label='OFS |err|')
    ax.set_xlabel('lead hour'); ax.set_ylabel('CRPS / |err| (cm)')
    ax.legend(fontsize=7); ax.set_title('Continuous skill')

    ax = axes[1]
    if vs_stofs_df is not None and not vs_stofs_df.empty:
        ax.plot(vs_stofs_df.lead_hour, vs_stofs_df.crpss,
                 color=COLORS['stofs'], linewidth=1.4, label='vs STOFS')
    if vs_ofs_df is not None and not vs_ofs_df.empty:
        ax.plot(vs_ofs_df.lead_hour, vs_ofs_df.crpss,
                 color=COLORS['ofs'], linewidth=1.4, label='vs OFS')
    ax.axhline(0, color='#999', linewidth=0.7, linestyle='--')
    ax.set_xlabel('lead hour'); ax.set_ylabel('CRPSS')
    ax.legend(fontsize=7); ax.set_title('Probabilistic skill score')

    ax = axes[2]
    ax.plot(crps_df.lead_hour, crps_df.ai_median_mae_cm, color=COLORS['pred'],
             linewidth=1.4, label='AI median MAE')
    ax.set_xlabel('lead hour'); ax.set_ylabel('MAE (cm)')
    ax.legend(fontsize=7); ax.set_title('Deterministic center MAE')
    fig.tight_layout(); savefig(fig, out_path); plt.close(fig)


def fig_spread_skill(ss_df, sh_df, out_path):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(FIG['double'][0], 2.8))
    ax = axes[0]
    if not ss_df.empty:
        for (lo, hi) in LEAD_BINS:
            sub = ss_df[ss_df.lead_bin == f'{lo}-{hi}']
            if sub.empty:
                continue
            ax.plot(sub.mean_spread_cm, sub.mae_cm, marker='o',
                     linewidth=1.0, alpha=0.85, label=f'{lo}-{hi}h')
    ax.set_xlabel('ensemble p05-p95 spread (cm)')
    ax.set_ylabel('|error of median| (cm)')
    ax.set_title('Spread vs skill')
    ax.legend(fontsize=7)

    ax = axes[1]
    if not sh_df.empty:
        ax.plot(sh_df.lead_hour, sh_df.mean_p05_p95_width_cm,
                 color=COLORS['pred'], linewidth=1.4, label='p05-p95')
        ax.plot(sh_df.lead_hour, sh_df.mean_p10_p90_width_cm,
                 color=COLORS['cyan' if 'cyan' in COLORS else 'envelope'],
                 linewidth=1.0, label='p10-p90')
    ax.set_xlabel('lead hour'); ax.set_ylabel('interval width (cm)')
    ax.set_title('Sharpness'); ax.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, out_path); plt.close(fig)


def fig_reliability(rel_df, out_path):
    if rel_df.empty:
        return
    setup()
    thresholds = list(rel_df.threshold_name.unique())
    fig, axes = plt.subplots(2, len(thresholds),
                              figsize=(FIG['hero'][0],
                                          3.6 if len(thresholds) > 1 else 3.6),
                              sharex=True, sharey=True)
    if len(thresholds) == 1:
        axes = axes[:, None]
    for j, t in enumerate(thresholds):
        for i, kind in enumerate(['pointwise', 'window_peak']):
            ax = axes[i, j]
            sub = rel_df[(rel_df.threshold_name == t)
                          & (rel_df.event_definition == kind)]
            ax.plot([0, 1], [0, 1], color='#999', linewidth=0.7, linestyle='--')
            if not sub.empty:
                sz = 8 + 60 * (sub.n / max(sub.n.max(), 1))
                ax.scatter(sub.predicted_probability_mean,
                            sub.observed_frequency,
                            s=sz, color=COLORS['pred'], alpha=0.85)
                ax.plot(sub.predicted_probability_mean,
                         sub.observed_frequency,
                         color=COLORS['pred'], linewidth=1.0)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f'{t} - {kind}', fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel('predicted prob.')
    for ax in axes[:, 0]:
        ax.set_ylabel('observed freq.')
    fig.tight_layout(); savefig(fig, out_path); plt.close(fig)


def fig_fan_plot(ai_ens, obs_wl, tide, station_id, station_name, cycle_time,
                  thresholds, out_path):
    """Fan plot for one station, one cycle.

    ai_ens (M, lead) for one (cycle, sta). obs_wl/tide same shape minus M.
    thresholds dict with minor/moderate/major or NaN.
    """
    setup()
    fig, ax = plt.subplots(figsize=FIG['double'])
    lead = np.arange(1, ai_ens.shape[1] + 1)
    p05 = np.percentile(ai_ens, 5, axis=0)
    p25 = np.percentile(ai_ens, 25, axis=0)
    p50 = np.percentile(ai_ens, 50, axis=0)
    p75 = np.percentile(ai_ens, 75, axis=0)
    p95 = np.percentile(ai_ens, 95, axis=0)
    ax.fill_between(lead, p05, p95, color=COLORS['pred'], alpha=0.18,
                     linewidth=0, label='p05-p95')
    ax.fill_between(lead, p25, p75, color=COLORS['pred'], alpha=0.30,
                     linewidth=0, label='p25-p75')
    ax.plot(lead, p50, color=COLORS['pred'], linewidth=1.4, label='AI median')
    if obs_wl is not None:
        ax.plot(lead, obs_wl, color=COLORS['obs'], linewidth=1.1,
                 label='observed')
    for k, c in [('minor', '#FFC700'),
                  ('moderate', '#FF8C00'),
                  ('major', COLORS['peak'])]:
        v = thresholds.get(k)
        if v is not None and np.isfinite(v):
            ax.axhline(v, color=c, linewidth=0.6, linestyle='--',
                        alpha=0.7, label=f'{k} ({v:.2f} m)')
    ax.set_xlabel('lead hour')
    ax.set_ylabel('water level (m)')
    ax.set_title(f'Fan plot — {station_name} ({station_id}) — cycle '
                  f'{pd.Timestamp(cycle_time):%Y-%m-%d %H}Z')
    ax.legend(fontsize=6.5, ncol=3, loc='upper right')
    fig.tight_layout(); savefig(fig, out_path); plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def run_region(verify_dir, cfg, kind='raw'):
    """Score the verify archive of `kind` ('raw' or 'calibrated_<method>').

    For 'raw': reads `{verify_dir}/verify_arrays.nc` and writes under `raw/`.
    For 'calibrated_<method>': reads `{verify_dir}/calibrated_<method>/
    verify_arrays_calibrated.nc` and writes to the same `calibrated_<method>/`
    folder (no `raw/` subdir).
    """
    if kind == 'raw':
        out_dir_raw = os.path.join(verify_dir, 'raw')
        fc_path = os.path.join(verify_dir, 'verify_arrays.nc')
    else:
        out_dir_raw = os.path.join(verify_dir, kind)
        fc_path = os.path.join(verify_dir, kind, 'verify_arrays_calibrated.nc')
    os.makedirs(out_dir_raw, exist_ok=True)
    fig_dir = os.path.join(out_dir_raw, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(fc_path):
        return None
    ds = xr.open_dataset(fc_path)
    station_ids = [str(s) for s in ds.station.values]
    station_names = [str(s) for s in ds.station_name.values]
    minor = ds.minor_threshold.values
    moderate = ds.moderate_threshold.values
    major = ds.major_threshold.values
    cycles = pd.to_datetime(ds.cycle_time.values)
    n_members = int(ds.sizes['atm_member'] * ds.sizes['model_seed'])

    ai_res5 = ds.ai_residual_member.values.astype(np.float32)
    ai_wl5 = ds.ai_water_level_member.values.astype(np.float32)
    obs_res = ds.obs_residual.values.astype(np.float32)
    obs_wl = ds.obs_water_level.values.astype(np.float32)
    stofs_res = ds.stofs_residual.values.astype(np.float32)
    stofs_wl = ds.stofs_water_level.values.astype(np.float32)
    ofs_res = ds.regional_ofs_residual.values.astype(np.float32)
    ofs_wl = ds.regional_ofs_water_level.values.astype(np.float32)
    ds.close()

    ai_ens = flatten_members(ai_res5)
    ai_wl_ens = flatten_members(ai_wl5)

    # ----- deterministic tables -----
    overall = score_overall(ai_ens, obs_res, stofs_res, ofs_res)
    with open(os.path.join(out_dir_raw, 'metrics_overall.json'), 'w') as f:
        json.dump(overall, f, indent=2, default=str)
    score_by_lead(ai_ens, obs_res, stofs_res, ofs_res).to_csv(
        os.path.join(out_dir_raw, 'metrics_by_lead.csv'), index=False)
    score_by_station(ai_ens, obs_res, station_ids).to_csv(
        os.path.join(out_dir_raw, 'metrics_by_station.csv'), index=False)

    # ----- probabilistic tables -----
    cov_df = coverage_table(ai_ens, obs_res)
    cov_df.to_csv(os.path.join(out_dir_raw, 'coverage_by_nominal.csv'),
                  index=False)
    rank_df = rank_histogram_table(ai_ens, obs_res)
    rank_df.to_csv(os.path.join(out_dir_raw, 'rank_histogram.csv'), index=False)

    crps_df = crps_by_lead(ai_ens, obs_res)
    crps_df.to_csv(os.path.join(out_dir_raw, 'crps_by_lead.csv'), index=False)
    vs_stofs_df = crpss_vs_deterministic(ai_ens, obs_res, stofs_res, 'stofs')
    vs_stofs_df.to_csv(os.path.join(out_dir_raw, 'crpss_by_lead_vs_stofs.csv'),
                       index=False)
    vs_ofs_df = crpss_vs_deterministic(ai_ens, obs_res, ofs_res, 'regional_ofs')
    vs_ofs_df.to_csv(os.path.join(out_dir_raw, 'crpss_by_lead_vs_ofs.csv'),
                     index=False)

    ss_df = spread_skill_bins(ai_ens, obs_res)
    ss_df.to_csv(os.path.join(out_dir_raw, 'spread_skill_bins.csv'), index=False)
    sh_df = sharpness_by_lead(ai_ens)
    sh_df.to_csv(os.path.join(out_dir_raw, 'sharpness_by_lead.csv'), index=False)

    brier_rows = []
    rel_rows = []
    for name, arr in [('minor', minor), ('moderate', moderate),
                        ('major', major)]:
        for kind in ['pointwise', 'window_peak']:
            br, rl = brier_and_reliability(
                ai_wl_ens, obs_wl, stofs_wl, ofs_wl,
                arr, name, event_kind=kind,
            )
            if not br.empty:
                brier_rows.append(br)
            if not rl.empty:
                rel_rows.append(rl)
    if brier_rows:
        pd.concat(brier_rows, ignore_index=True).to_csv(
            os.path.join(out_dir_raw, 'brier_by_threshold.csv'), index=False)
    rel_df = (pd.concat(rel_rows, ignore_index=True)
              if rel_rows else pd.DataFrame())
    if not rel_df.empty:
        rel_df.to_csv(os.path.join(out_dir_raw, 'reliability_bins.csv'),
                       index=False)

    extreme_tail_metrics(ai_ens, obs_res).to_csv(
        os.path.join(out_dir_raw, 'extreme_tail_metrics.csv'), index=False)

    # ----- figures -----
    fig_coverage_curve(cov_df, os.path.join(fig_dir, 'fig_uq_r2_coverage.png'))
    fig_rank_hist(rank_df, n_members,
                   os.path.join(fig_dir, 'fig_uq_r3_rank_hist.png'))
    fig_crps_by_lead(crps_df, vs_stofs_df, vs_ofs_df,
                      os.path.join(fig_dir, 'fig_uq_r4_crps.png'))
    fig_spread_skill(ss_df, sh_df,
                      os.path.join(fig_dir, 'fig_uq_r5_spread_skill.png'))
    fig_reliability(rel_df,
                     os.path.join(fig_dir, 'fig_uq_r6_reliability.png'))

    # Fan plot: pick the highest observed-peak (cycle, station) and plot.
    obs_peak = np.nanmax(obs_wl, axis=1)  # (cyc, sta)
    if np.isfinite(obs_peak).any():
        c_idx, s_idx = np.unravel_index(np.nanargmax(obs_peak), obs_peak.shape)
        thresholds = dict(minor=float(minor[s_idx]) if np.isfinite(minor[s_idx]) else np.nan,
                           moderate=float(moderate[s_idx]) if np.isfinite(moderate[s_idx]) else np.nan,
                           major=float(major[s_idx]) if np.isfinite(major[s_idx]) else np.nan)
        fig_fan_plot(
            ai_wl_ens[c_idx, :, :, s_idx], obs_wl[c_idx, :, s_idx],
            None,
            station_ids[s_idx], station_names[s_idx], cycles[c_idx],
            thresholds,
            os.path.join(fig_dir, 'fig_uq_r1_fan_plot_top_event.png'),
        )
    return out_dir_raw


def main(cfg):
    bases = [d for d in glob.glob(os.path.join(cfg['BASE_DIR'], '*'))
              if os.path.isdir(d)]
    for span_dir in bases:
        regs = sorted(d for d in glob.glob(os.path.join(span_dir, 'region_*'))
                       if os.path.isdir(d)
                       and os.path.exists(os.path.join(d, 'verify_arrays.nc')))
        for rd in regs:
            if os.path.exists(os.path.join(rd, 'raw', 'metrics_overall.json')):
                print(f"UQ -> {rd} [raw exists, skip]")
            else:
                print(f"UQ -> {rd} [raw]")
                run_region(rd, cfg, kind='raw')
            for cal_dir in sorted(glob.glob(os.path.join(rd,
                                                           'calibrated_*'))):
                method = os.path.basename(cal_dir)
                print(f"UQ -> {rd} [{method}]")
                run_region(rd, cfg, kind=method)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    else:
        cfg = dict(BASE_DIR='outputs/uq_postprocessing/nn_24lb_72fh/gefs_psfc_v1_raw')
    main(cfg)
