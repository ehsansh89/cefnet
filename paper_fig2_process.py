"""
paper_fig2_process.py — Figure 2 backing data (Round 3 decision in
docs/paper/panel_discussion_on_figures.md).

Plain summary: this script answers the two Figure-2 questions — "when we say
40%, does flooding happen 40% of the time?" and "is the ensemble better than
statistically post-processed physics?" — and writes three small files that
paper_fig2_plot.py renders. It computes everything from scratch so nothing is
inherited from the retired paper_fig2.py aggregation.

Computation 1 — RELIABILITY (panel a).
    Flood events use the window-peak definition: "does water level cross the
    NOAA threshold at any lead 1-72 h of this forecast cycle?" The spread
    calibration was tuned for interval coverage and worsens event
    probabilities (figure review, Round 2), so flood probabilities get their own
    layer: isotonic regression on the raw member fraction — chosen a priori,
    fit on 2024 pooled CONUS per threshold, evaluated once on 2025 (headline
    direction). The reverse direction is computed only as a retrospective
    sensitivity. Observed-frequency CIs use a 10-day block bootstrap.

Computation 2 — CRPSS vs lead (panel b).
    Evaluated on 2025 with fit-2024 parameters, strictly matched samples
    (the same cycle-station-lead rows for every system), and every
    probabilistic forecast represented as the SAME 19 quantiles (5-95%) so
    member count cannot fake skill. Systems:
      ai_cal         our ensemble, spread-calibrated (fit-2024 bias+gamma)
      ai_raw         our raw ensemble (context: what calibration adds)
      stofs_det      deterministic STOFS (CRPS of a single line = |error|)
      dressed_stofs  STOFS + its own 2024 error quantiles per gauge/lead-bin
      dressed_ai     the raw AI median + its own 2024 error quantiles — the
                     same-center control that isolates whether flow-dependent
                     ensemble spread adds information
    Pooled CRPSS is one ratio of pooled CRPS sums (never an average of
    regional ratios); bands come from one PAIRED 10-day-block bootstrap
    (same resampled blocks for every system).

Reads (per region, uq base = x_ensemble_calibrate_v2_config.json BASE_DIR):
    decision_events.parquet, verify_arrays.nc, calibration_v2_params.json,
    warning_lead.csv
Writes (PROCESS_OUT_DIR from paper_fig1_config_v2.json = outputs/paper/v2):
    fig2_reliability.csv   direction, threshold, product, bin rows
    fig2_crpss_lead.csv    per-lead CRPS of all systems + CRPSS + CI bands
    fig2_summary.json      Brier/BSS scoreboard, ECE, episode counts, protocol
"""

import os
import json
import glob
import warnings

import numpy as np
import pandas as pd

from paper_common import eligible_station_ids


SPAN = '20240101_20251201'
LEAD_BINS = [(1, 12), (13, 24), (25, 36), (37, 48), (49, 72)]
Q_LEVELS = np.arange(0.05, 0.951, 0.05)      # 19 quantiles for ALL systems
PROB_BIN_EDGES = np.linspace(0.0, 1.0, 11)   # 10 reliability bins
THRESHOLDS = ['minor', 'moderate']           # major -> Fig 3 / Table 3 orbit
BLOCK_DAYS = 10                              # bootstrap block length
N_BOOT = 1000
MIN_FIT_SAMPLES = 200                        # per (station, lead-bin) dressing
CHUNK = 128                                  # cycles per streamed nc chunk
DIRECTIONS = [                               # (name, fit_year, eval_year)
    ('fit2024_eval2025', 2024, 2025),        # headline (operational direction)
    ('fit2025_eval2024', 2025, 2024),        # retrospective sensitivity (SI)
]


def load_cfgs():
    """Gets the two config files; gives (figure cfg, uq cfg)."""
    with open('paper_fig1_config_v2.json') as f:
        fig_cfg = json.load(f)
    with open('x_ensemble_calibrate_v2_config.json') as f:
        uq_cfg = json.load(f)
    return fig_cfg, uq_cfg


def region_dirs(uq_base):
    return sorted(glob.glob(os.path.join(uq_base, SPAN, 'region_*')))


def block_of(times, year):
    """Gets datetimes and the eval year; gives the 10-day block index of each."""
    t0 = pd.Timestamp(f'{year}-01-01')
    return ((pd.DatetimeIndex(times) - t0).days // BLOCK_DAYS).to_numpy()


# =============================================================================
# Computation 1 — reliability of window-peak flood probabilities
# =============================================================================
def load_events(rdirs, elig):
    """Reads decision_events from every region; gives one pooled frame of
    valid rows at eligible gauges with a 'year' column."""
    frames = []
    for rd in rdirs:
        p = os.path.join(rd, 'decision_events.parquet')
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[ev['obs_valid'] & np.isfinite(ev['ai_prob_raw'])].copy()
    if elig is not None:
        ev = ev[ev['station_id'].astype(str).isin(elig)]
    ev['year'] = pd.DatetimeIndex(ev['cycle_time']).year
    return ev


def fit_isotonic(ev_fit, threshold):
    """Fits the a-priori-chosen probability calibration on the fit year.
    Gets fit-year events for one threshold; gives a fitted isotonic model."""
    from sklearn.isotonic import IsotonicRegression
    d = ev_fit[ev_fit['threshold'] == threshold]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                             out_of_bounds='clip')
    iso.fit(d['ai_prob_raw'].to_numpy(float),
            d['obs_event'].to_numpy(float))
    return iso


def bin_reliability(p, y, blocks=None):
    """Gets probabilities p and outcomes y (and optional block ids);
    gives per-bin rows: n, mean predicted p, observed frequency, and — when
    blocks are given — a 10-day block-bootstrap 95% CI on the frequency."""
    idx = np.clip(np.digitize(p, PROB_BIN_EDGES[1:-1]), 0, 9)
    rows = []
    ci = {}
    if blocks is not None:
        nb = int(blocks.max()) + 1
        cnt = np.zeros((nb, 10)); hit = np.zeros((nb, 10))
        np.add.at(cnt, (blocks, idx), 1.0)
        np.add.at(hit, (blocks, idx), y)
        rng = np.random.default_rng(0)
        pick = rng.integers(0, nb, size=(N_BOOT, nb))
        with np.errstate(invalid='ignore', divide='ignore'):
            freq = hit[pick].sum(axis=1) / cnt[pick].sum(axis=1)
        lo, hi = np.nanpercentile(freq, [2.5, 97.5], axis=0)
        ci = {b: (lo[b], hi[b]) for b in range(10)}
    for b in range(10):
        m = idx == b
        n = int(m.sum())
        row = dict(bin_left=PROB_BIN_EDGES[b], bin_right=PROB_BIN_EDGES[b + 1],
                   n=n,
                   pred_mean=float(p[m].mean()) if n else np.nan,
                   obs_freq=float(y[m].mean()) if n else np.nan,
                   obs_freq_lo=np.nan, obs_freq_hi=np.nan)
        if blocks is not None and n:
            row['obs_freq_lo'], row['obs_freq_hi'] = ci[b]
        rows.append(row)
    return rows


def brier_stats(p, y):
    """Gets probabilities and outcomes; gives Brier score, BSS vs the eval
    sample's own base rate, and the expected calibration error (ECE)."""
    b = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    b_clim = float(np.mean((base - y) ** 2))
    idx = np.clip(np.digitize(p, PROB_BIN_EDGES[1:-1]), 0, 9)
    ece = 0.0
    for k in range(10):
        m = idx == k
        if m.any():
            ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return dict(brier=b, bss_vs_climatology=1 - b / b_clim if b_clim > 0 else np.nan,
                ece=float(ece), n=int(len(y)), event_rate=base)


def run_reliability(ev):
    """Runs both directions x thresholds x products; gives (rows for
    fig2_reliability.csv, scoreboard dict for the summary json)."""
    rows, board = [], {}
    for dname, fit_year, eval_year in DIRECTIONS:
        board[dname] = {}
        ev_fit = ev[ev['year'] == fit_year]
        ev_ev = ev[ev['year'] == eval_year]
        for thr in THRESHOLDS:
            iso = fit_isotonic(ev_fit, thr)
            d = ev_ev[ev_ev['threshold'] == thr]
            blocks = block_of(d['cycle_time'], eval_year)
            y = d['obs_event'].to_numpy(float)
            products = {
                'raw': d['ai_prob_raw'].to_numpy(float),
                'spread_cal': d['ai_prob_cal'].to_numpy(float),
                'isotonic': iso.predict(d['ai_prob_raw'].to_numpy(float)),
            }
            ds_ = d[d['stofs_valid'] & np.isfinite(d['dressed_prob'])]
            board[dname][thr] = {}
            for prod, p in products.items():
                ok = np.isfinite(p)
                bl = blocks[ok] if prod == 'isotonic' else None
                for r in bin_reliability(p[ok], y[ok], blocks=bl):
                    rows.append(dict(direction=dname, threshold=thr,
                                     product=prod, **r))
                board[dname][thr][prod] = brier_stats(p[ok], y[ok])
            # dressed STOFS probabilities (SI context: post-processed physics
            # is not automatically honest either) — STOFS-matched rows only.
            pd_ = ds_['dressed_prob'].to_numpy(float)
            yd = ds_['obs_event'].to_numpy(float)
            for r in bin_reliability(pd_, yd):
                rows.append(dict(direction=dname, threshold=thr,
                                 product='dressed_stofs', **r))
            board[dname][thr]['dressed_stofs'] = brier_stats(pd_, yd)
    return rows, board


def episode_counts(rdirs, elig):
    """Counts distinct observed exceedance episodes (from warning_lead.csv)
    per threshold, total and in 2025 — the caption's honest n."""
    frames = []
    for rd in rdirs:
        p = os.path.join(rd, 'warning_lead.csv')
        if os.path.exists(p):
            frames.append(pd.read_csv(p, dtype={'station_id': str}))
    if not frames:
        return {}
    w = pd.concat(frames, ignore_index=True)
    if elig is not None:
        w = w[w['station_id'].isin(elig)]
    w['year'] = pd.DatetimeIndex(w['episode_start']).year
    out = {}
    for thr in ['minor', 'moderate', 'major']:
        d = w[w['threshold'] == thr]
        out[thr] = dict(total=int(len(d)), y2025=int((d['year'] == 2025).sum()))
    return out


# =============================================================================
# Computation 2 — matched-sample CRPS by lead (fit 2024 -> eval 2025)
# =============================================================================
def crps_from_quantiles(q, y):
    """CRPS of an m-member ensemble given as SORTED values along the last
    axis, against observation y. Gets q (..., m) and y (...); gives CRPS (...).
    Uses the standard empirical-ensemble form; for m=1 it reduces to |q-y|."""
    m = q.shape[-1]
    term1 = np.mean(np.abs(q - y[..., None]), axis=-1)
    w = (2.0 * np.arange(m) + 1.0 - m)
    term2 = np.tensordot(q, w, axes=([-1], [0])) / (m * m)
    return term1 - term2


def quantiles_members(mm):
    """Gets a member block (c, 93, L, S); gives its 19 quantiles (c, L, S, 19),
    NaN-aware only when needed (nanquantile is slow)."""
    fn = np.nanquantile if np.isnan(mm).any() else np.quantile
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = fn(mm, Q_LEVELS, axis=1)         # (19, c, L, S)
    return np.moveaxis(q, 0, -1)             # (c, L, S, 19)


def fit_error_quantiles(err):
    """Gets fit-year error samples (n_cyc, L, S); gives per (lead, station)
    19 error quantiles (L, S, 19) expanded from lead-bins, NaN where a
    (station, bin) has fewer than MIN_FIT_SAMPLES finite errors."""
    L, S = err.shape[1], err.shape[2]
    out = np.full((L, S, 19), np.nan, np.float64)
    for lo, hi in LEAD_BINS:
        blk = err[:, lo - 1:hi, :].reshape(-1, S)          # samples x S
        for s in range(S):
            v = blk[:, s]
            v = v[np.isfinite(v)]
            if len(v) >= MIN_FIT_SAMPLES:
                out[lo - 1:hi, s, :] = np.quantile(v, Q_LEVELS)
    return out


def leadbin_params(par_dir):
    """Gets one direction's calibration params; gives per-lead arrays
    (bias in m, gamma) expanded from the lead-bins."""
    bias = np.zeros(72); gamma = np.ones(72)
    for lo, hi in LEAD_BINS:
        key = f'{lo}-{hi}'
        bias[lo - 1:hi] = par_dir['bias_cm'][key] / 100.0
        gamma[lo - 1:hi] = par_dir['gamma'][key]
    return bias, gamma


def region_crps(rd, elig, fit_year, eval_year, acc):
    """Streams one region's verify_arrays: fits 2024 error dressings, then
    accumulates matched per (lead, block) CRPS sums for all systems into acc.
    Gives the set of contributing station ids."""
    import xarray as xr
    ds = xr.open_dataset(os.path.join(rd, 'verify_arrays.nc'))
    sids = np.array([str(s) for s in ds.station.values])
    keep = np.ones(len(sids), bool) if elig is None else np.isin(sids, list(elig))
    if not keep.any():
        ds.close()
        return set()
    cyc = pd.DatetimeIndex(ds.cycle_time.values)
    fit_i = np.where(cyc.year == fit_year)[0]
    ev_i = np.where(cyc.year == eval_year)[0]

    with open(os.path.join(rd, 'calibration_v2_params.json')) as f:
        par = json.load(f)
    bias, gamma = leadbin_params(
        par['directions'][f'fit{fit_year}_eval{eval_year}'])

    # ---- pass 1 (fit year): error samples for the two dressings ----------
    obs_f = ds.obs_water_level.isel(cycle_time=fit_i, station=keep).values
    stofs_f = ds.stofs_water_level.isel(cycle_time=fit_i, station=keep).values
    med_f = np.full_like(obs_f, np.nan)
    for a in range(0, len(fit_i), CHUNK):
        ii = fit_i[a:a + CHUNK]
        mm = ds.ai_water_level_member.isel(cycle_time=ii, station=keep).values
        mm = mm.reshape(mm.shape[0], -1, mm.shape[3], mm.shape[4])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med_f[a:a + len(ii)] = np.nanmedian(mm, axis=1)
    eq_stofs = fit_error_quantiles(stofs_f - obs_f)     # (L, S, 19) ascending
    eq_ai = fit_error_quantiles(med_f - obs_f)
    # center - ascending errors = descending members; flip to ascending
    eq_stofs = eq_stofs[..., ::-1]
    eq_ai = eq_ai[..., ::-1]

    # ---- pass 2 (eval year): matched CRPS accumulation -------------------
    stations_used = set()
    for a in range(0, len(ev_i), CHUNK):
        ii = ev_i[a:a + CHUNK]
        blocks = block_of(cyc[ii], eval_year)
        mm = ds.ai_water_level_member.isel(cycle_time=ii, station=keep).values
        mm = mm.reshape(mm.shape[0], -1, mm.shape[3], mm.shape[4])
        q_raw = quantiles_members(mm)                    # (c, L, S, 19)
        q50 = q_raw[..., 9:10]
        q_cal = q50 - bias[None, :, None, None] + \
            gamma[None, :, None, None] * (q_raw - q50)
        obs = ds.obs_water_level.isel(cycle_time=ii, station=keep).values
        stofs = ds.stofs_water_level.isel(cycle_time=ii, station=keep).values
        d_stofs = stofs[..., None] - eq_stofs[None]      # (c, L, S, 19)
        d_ai = q50[..., 0][..., None] - eq_ai[None]

        crps = dict(
            ai_cal=crps_from_quantiles(q_cal, obs),
            ai_raw=crps_from_quantiles(q_raw, obs),
            stofs_det=np.abs(stofs - obs),
            dressed_stofs=crps_from_quantiles(d_stofs, obs),
            dressed_ai=crps_from_quantiles(d_ai, obs),
        )
        ok = np.isfinite(obs)
        for v in crps.values():
            ok &= np.isfinite(v)
        stations_used |= set(sids[keep][ok.any(axis=(0, 1))])
        c_idx, l_idx, s_idx = np.nonzero(ok)
        b_idx = blocks[c_idx]
        np.add.at(acc['n'], (l_idx, b_idx), 1.0)
        for k, v in crps.items():
            np.add.at(acc[k], (l_idx, b_idx), v[ok])
    ds.close()
    return stations_used


def assemble_crpss(acc):
    """Gets the pooled (lead, block) CRPS sums; gives the per-lead CSV frame
    (pooled-ratio CRPSS + paired block-bootstrap bands) and pooled summary."""
    n = acc['n']
    leads = np.arange(1, 73)
    out = pd.DataFrame({'lead_hour': leads, 'n': n.sum(axis=1).astype(int)})
    for k in ['ai_cal', 'ai_raw', 'stofs_det', 'dressed_stofs', 'dressed_ai']:
        out[f'crps_{k}_cm'] = acc[k].sum(axis=1) / n.sum(axis=1) * 100.0
    pairs = [('stofs', 'stofs_det'), ('dressed_stofs', 'dressed_stofs'),
             ('dressed_ai', 'dressed_ai')]
    rng = np.random.default_rng(0)
    nb = n.shape[1]
    pick = rng.integers(0, nb, size=(N_BOOT, nb))        # PAIRED blocks
    boot = {k: acc[k][:, pick].sum(axis=2) for k in acc}  # (72, N_BOOT)
    for name, k in pairs:
        out[f'crpss_vs_{name}'] = 1.0 - out['crps_ai_cal_cm'] / out[f'crps_{k}_cm']
        with np.errstate(invalid='ignore', divide='ignore'):
            s = 1.0 - boot['ai_cal'] / boot[k]
        lo, hi = np.nanpercentile(s, [2.5, 97.5], axis=1)
        out[f'crpss_vs_{name}_lo'] = lo
        out[f'crpss_vs_{name}_hi'] = hi
    # pooled over all leads (one ratio of grand sums + paired-block CI)
    pooled = {}
    tot = {k: acc[k].sum() for k in acc}
    tot_b = {k: acc[k].sum(axis=0)[pick].sum(axis=1) for k in acc}
    for name, k in pairs:
        with np.errstate(invalid='ignore', divide='ignore'):
            s = 1.0 - tot_b['ai_cal'] / tot_b[k]
        pooled[f'crpss_vs_{name}'] = dict(
            value=1.0 - tot['ai_cal'] / tot[k],
            lo=float(np.nanpercentile(s, 2.5)),
            hi=float(np.nanpercentile(s, 97.5)))
    pooled['crps_cm'] = {k: tot[k] / acc['n'].sum() * 100.0
                         for k in ['ai_cal', 'ai_raw', 'stofs_det',
                                   'dressed_stofs', 'dressed_ai']}
    return out, pooled


# =============================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    print('===== Figure 2 processing (Round 3 protocol) =====')
    fig_cfg, uq_cfg = load_cfgs()
    out_dir = fig_cfg['PROCESS_OUT_DIR']
    os.makedirs(out_dir, exist_ok=True)
    rdirs = region_dirs(uq_cfg['BASE_DIR'])
    print(f'  regions: {len(rdirs)}')
    elig = eligible_station_ids(fig_cfg)
    print(f'  eligible gauges: {len(elig) if elig is not None else "all"}')

    # ---- computation 1: reliability --------------------------------------
    ev = load_events(rdirs, elig)
    print(f'  decision events: {len(ev):,} rows, '
          f"{ev['station_id'].nunique()} gauges")
    rel_rows, board = run_reliability(ev)
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(os.path.join(out_dir, 'fig2_reliability.csv'), index=False)
    print(f"  wrote fig2_reliability.csv ({len(rel)} rows)")
    for thr in THRESHOLDS:
        b = board['fit2024_eval2025'][thr]
        print(f"  [2025 held-out, {thr}]  Brier raw {b['raw']['brier']:.5f}"
              f" | spread-cal {b['spread_cal']['brier']:.5f}"
              f" | isotonic {b['isotonic']['brier']:.5f}"
              f" | ECE raw {b['raw']['ece']:.4f} -> iso {b['isotonic']['ece']:.4f}")

    eps = episode_counts(rdirs, elig)
    print(f'  episodes: {eps}')

    # ---- computation 2: matched CRPS by lead ------------------------------
    fit_year, eval_year = 2024, 2025
    nb = (pd.Timestamp(f'{eval_year}-12-01') -
          pd.Timestamp(f'{eval_year}-01-01')).days // BLOCK_DAYS + 1
    acc = {k: np.zeros((72, nb)) for k in
           ['n', 'ai_cal', 'ai_raw', 'stofs_det', 'dressed_stofs', 'dressed_ai']}
    used = set()
    for rd in rdirs:
        used |= region_crps(rd, elig, fit_year, eval_year, acc)
        print(f'  crps pass done: {os.path.basename(rd)} '
              f'(cum n={int(acc["n"].sum()):,})')
    crpss, pooled = assemble_crpss(acc)
    crpss.to_csv(os.path.join(out_dir, 'fig2_crpss_lead.csv'), index=False)
    print(f"  wrote fig2_crpss_lead.csv | pooled CRPSS "
          f"vs STOFS {pooled['crpss_vs_stofs']['value']:+.3f}, "
          f"vs dressed STOFS {pooled['crpss_vs_dressed_stofs']['value']:+.3f}, "
          f"vs dressed AI {pooled['crpss_vs_dressed_ai']['value']:+.3f}")

    summary = dict(
        protocol=dict(
            event_definition='window_peak_1_72h',
            probability_calibration='isotonic on raw member fraction, '
                                    'chosen a priori, fit 2024 pooled CONUS '
                                    'per threshold, evaluated on 2025',
            spread_calibration='bias+gamma per lead-bin (fit 2024), '
                               'continuous product only',
            quantile_representation=f'{len(Q_LEVELS)} quantiles (5-95%) for '
                                    'all probabilistic systems',
            crpss_estimand='one ratio of pooled CRPS sums, matched samples',
            bootstrap=f'{BLOCK_DAYS}-day paired block bootstrap, '
                      f'{N_BOOT} resamples',
            reliability_bins=10, min_bin_n_for_plot=100,
            min_fit_samples_dressing=MIN_FIT_SAMPLES,
            thresholds_main=THRESHOLDS,
            major_disposition='excluded from panel a (too few episodes, '
                              'negative BSS) - reported in Fig 3/Table 3 orbit',
        ),
        reliability_scoreboard=board,
        episodes=eps,
        n_gauges_reliability=int(ev['station_id'].nunique()),
        n_gauges_crpss=len(used),
        matched_pairs_crpss=int(acc['n'].sum()),
        crpss_pooled=pooled,
    )
    with open(os.path.join(out_dir, 'fig2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print('  wrote fig2_summary.json')
