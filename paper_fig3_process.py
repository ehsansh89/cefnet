"""
paper_fig3_process.py — computes everything behind Figure 3 (the flood-warning
decision figure), per the panel's Round 3 final decision + the review's five
amendments, with one documented correction found at implementation.

In plain English: we turn every forecast system into a flood-warning machine
("warn when the chance of crossing the NOAA threshold is high enough"), count
distinct real flood episodes each machine catches, measure how much needless
alert time it costs (false-alert days), and compare machines at the SAME
false-alert workload. The warning dial of every probabilistic system is
chosen on 2024 only, frozen, then judged on 2025.

THE CORRECTION (documented for the panel): the ratified rule scored false
alarms as unmatched warning EPISODES with one-to-one matching. Half of all
minor flood episodes have another episode within 72 h at the same gauge, so a
continuous (honest) warning covering a week of clustered floods earned ONE
credit and ONE false alarm, while a choppy deterministic line earned one per
burst — the sweep curve came out flat (the dial did nothing) and burden
matching picked a cutoff of 0.07. That punishes every probabilistic system
(ours and physics'). PRIMARY RULE here: an episode is caught if ANY pre-onset
cycle's 72 h forecast window covers its onset (no exclusivity); false-alert
burden = days on false alert per station-year (a flagged cycle whose window
holds no exceedance = 6 false hours). The ratified episode-matching rule is
still computed in full as the SI sensitivity variant ("epimatch").

Stage 1 (slow, cached): dressed-AI and dressed-STOFS window-peak flood
probabilities (19 error-quantile pseudo-members, fitted on 2024 only —
mirrors x_dressed_baseline.py). Cache: outputs/paper/v2/fig3_probs.parquet.

Stage 2 (fast): detection sweeps under both rules, cutoff selection on 2024,
forward 2025 evaluation, the dressed-AI escrow referendum, paired block
bootstrap.

Writes (all under outputs/paper/v2/):
    fig3_tradeoff.csv           primary-rule curves + deterministic points
    fig3_decomp.csv             matched-point episode decomposition (the N)
    fig3_lead.csv               fraction warned >= 6/12/24/36/48 h ahead
    fig3_reliability.csv        referendum supporting test (ens vs dressed AI)
    fig3_tradeoff_epimatch.csv  ratified-rule variant (SI sensitivity)
    fig3_summary.json           frozen cutoffs, burdens, referendum, variant
"""

import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from paper_common import eligible_station_ids


SPAN = '20240101_20251201'
LEAD_BINS = [(1, 12), (13, 24), (25, 36), (37, 48), (49, 72)]
Q_LEVELS = np.arange(0.05, 0.951, 0.05)          # 19 pseudo-members
MIN_FIT_SAMPLES = 200                            # per (station, lead-bin)
THRESHOLDS = ['minor', 'moderate', 'major']
REFERENDUM_THRESHOLDS = ['minor', 'moderate']    # review amendment 2
CHUNK = 96                                       # cycles per streamed chunk
EPISODE_GAP_H = 12                               # frozen episode clustering
WINDOW_H = 72                                    # forecast window (h)
CYCLE_H = 6                                      # forecast cadence (h)
FIT_YEAR, EVAL_YEAR = 2024, 2025
LEAD_MARKS = [6, 12, 24, 36, 48]                 # panel (c) thresholds
BLOCK_DAYS = 10
N_BOOT = 1000
PROB_BIN_EDGES = np.linspace(0.0, 1.0, 11)
GRID_ENS = (np.arange(1, 94) - 0.5) / 93         # >= k of 93 members
GRID_DRESSED = (np.arange(1, 20) - 0.5) / 19     # >= k of 19 pseudo-members

OUT_DIR = 'outputs/paper/v2'
PROBS_CACHE = os.path.join(OUT_DIR, 'fig3_probs.parquet')
REBUILD_PROBS = False

PROB_SYSTEMS = ['ens', 'dai', 'dstofs']
DET_SYSTEMS = ['stofs_det', 'ai_det']
ALL_SYSTEMS = PROB_SYSTEMS + DET_SYSTEMS


# =============================================================================
# Stage 1 — dressed probabilities (fit 2024 only)
# =============================================================================
def _lead_bin_index(n_lead):
    idx = np.zeros(n_lead, dtype=int)
    for b, (lo, hi) in enumerate(LEAD_BINS):
        idx[lo - 1:hi] = b
    return idx


def fit_error_quantiles(center, obs, fit_mask, lead_bin_idx):
    """Gets center forecasts + obs (C,L,S) and the fit-year mask; gives
    (S, bins, 19) error quantiles, NaN where < MIN_FIT_SAMPLES valid errors."""
    n_s = center.shape[2]
    out = np.full((n_s, len(LEAD_BINS), len(Q_LEVELS)), np.nan, np.float32)
    err = center[fit_mask] - obs[fit_mask]
    for b in range(len(LEAD_BINS)):
        e_b = err[:, lead_bin_idx == b, :]
        for s in range(n_s):
            e = e_b[:, :, s]
            e = e[np.isfinite(e)]
            if e.size >= MIN_FIT_SAMPLES:
                out[s, b] = np.quantile(e, Q_LEVELS)
    return out


def dressed_members(center, eq, lead_bin_idx):
    """Gets center (C,L,S) + error quantiles (S,B,19); gives (C,19,L,S)
    ascending pseudo-member trajectories (center minus reversed quantiles)."""
    eq_l = eq[:, lead_bin_idx, :]
    rev = eq_l[:, :, ::-1]
    return (center[:, None, :, :]
            - np.moveaxis(rev, [0, 1, 2], [2, 1, 0])[None, :, :, :])


def build_region_probs(rd):
    """Gets one region dir; gives a frame of dressed-AI and dressed-STOFS
    window-peak probabilities (fit on 2024 only) per cycle x gauge x
    threshold. NaN where the dressing could not be fitted or the center is
    missing for the whole window."""
    ds = xr.open_dataset(os.path.join(rd, 'verify_arrays.nc'))
    cycles = pd.to_datetime(ds.cycle_time.values)
    sids = [str(s) for s in ds.station.values]
    n_c, n_l = ds.sizes['cycle_time'], ds.sizes['lead_hour']
    n_s = ds.sizes['station']
    obs_wl = ds.obs_water_level.values.astype(np.float32)
    stofs_wl = ds.stofs_water_level.values.astype(np.float32)
    thr = {t: ds[f'{t}_threshold'].values.astype(np.float32)
           for t in THRESHOLDS}

    ai_med = np.full((n_c, n_l, n_s), np.nan, np.float32)
    for c0 in range(0, n_c, CHUNK):
        c1 = min(c0 + CHUNK, n_c)
        a = ds.ai_water_level_member.isel(cycle_time=slice(c0, c1)).values
        a = a.reshape(a.shape[0], -1, n_l, n_s)
        ai_med[c0:c1] = np.nanmedian(a, axis=1)
    ds.close()

    fit_mask = np.asarray(cycles.year == FIT_YEAR)
    lead_bin_idx = _lead_bin_index(n_l)
    probs = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN')
        for name, center in [('dai', ai_med), ('dstofs', stofs_wl)]:
            eq = fit_error_quantiles(center, obs_wl, fit_mask, lead_bin_idx)
            dm = dressed_members(center, eq, lead_bin_idx)
            dmax = np.nanmax(dm, axis=2)
            valid = np.all(np.isfinite(dmax), axis=1)
            for t in THRESHOLDS:
                p = np.mean(dmax >= thr[t][None, None, :], axis=1)
                probs[(name, t)] = np.where(valid, p, np.nan).astype(np.float32)

    rows = []
    for t in THRESHOLDS:
        okt = np.isfinite(thr[t])
        for s in range(n_s):
            if not okt[s]:
                continue
            rows.append(pd.DataFrame(dict(
                cycle_time=cycles, station_id=sids[s], threshold=t,
                dai_prob=probs[('dai', t)][:, s],
                dstofs_prob=probs[('dstofs', t)][:, s])))
    return pd.concat(rows, ignore_index=True)


def build_probs(rdirs):
    """Builds (or loads) the pooled dressed-probability cache."""
    if os.path.exists(PROBS_CACHE) and not REBUILD_PROBS:
        print(f'[cache] {PROBS_CACHE}')
        return pd.read_parquet(PROBS_CACHE)
    frames = []
    for rd in rdirs:
        print(f'  dressing (fit {FIT_YEAR}) -> {os.path.basename(rd)}')
        frames.append(build_region_probs(rd))
    probs = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    probs.to_parquet(PROBS_CACHE, index=False)
    return probs


def validate_dstofs(rdirs, probs):
    """Sanity check: our 2025 dressed-STOFS probs must match the existing
    dressed_probs.nc forward direction (both are fit-2024). Gives max |diff|."""
    rd = rdirs[0]
    dp = xr.open_dataset(os.path.join(rd, 'dressed_probs.nc'))
    ref = dp['dressed_prob_minor'].to_dataframe().reset_index()
    dp.close()
    ref['station_id'] = ref['station'].astype(str)
    ref = ref[pd.DatetimeIndex(ref['cycle_time']).year == EVAL_YEAR]
    ours = probs[(probs['threshold'] == 'minor')
                 & (pd.DatetimeIndex(probs['cycle_time']).year == EVAL_YEAR)]
    m = ref.merge(ours, on=['cycle_time', 'station_id'], how='inner')
    d = (m['dressed_prob_minor'] - m['dstofs_prob']).abs()
    return float(np.nanmax(d.to_numpy())) if len(m) else np.nan


# =============================================================================
# Stage 2 — loading
# =============================================================================
def load_events(rdirs, elig):
    """Reads decision_events from every region; gives one pooled frame at
    eligible gauges (all validity flags kept — filtering happens per use)."""
    cols = ['cycle_time', 'station_id', 'threshold', 'obs_event', 'obs_valid',
            'ai_prob_raw', 'ai_det_event', 'stofs_event', 'stofs_valid']
    frames = [pd.read_parquet(os.path.join(rd, 'decision_events.parquet'),
                              columns=cols) for rd in rdirs]
    ev = pd.concat(frames, ignore_index=True)
    ev['station_id'] = ev['station_id'].astype(str)
    if elig is not None:
        ev = ev[ev['station_id'].isin(elig)]
    return ev


def load_episodes(rdirs, elig):
    """Reads observed episode onsets (frozen 12 h rule) from every region's
    warning_lead.csv; gives frame (station_id, threshold, onset)."""
    frames = []
    for rd in rdirs:
        df = pd.read_csv(os.path.join(rd, 'warning_lead.csv'),
                         usecols=['station_id', 'threshold', 'episode_start'])
        frames.append(df)
    epi = pd.concat(frames, ignore_index=True)
    epi['station_id'] = epi['station_id'].astype(str)
    epi['onset'] = pd.to_datetime(epi['episode_start'])
    if elig is not None:
        epi = epi[epi['station_id'].isin(elig)]
    return epi[['station_id', 'threshold', 'onset']]


# =============================================================================
# Primary rule — episode coverage + false-alert time
# =============================================================================
def episode_leads(onsets, flag_times):
    """PRIMARY hit rule. Gets episode onsets + times of flagged cycles (both
    sorted); gives per-episode warning lead in hours (from the EARLIEST
    flagged cycle c with c < onset <= c + 72 h), NaN if never warned."""
    leads = np.full(len(onsets), np.nan)
    if len(flag_times) == 0:
        return leads
    ft = flag_times.values
    lo = np.searchsorted(ft, onsets - np.timedelta64(WINDOW_H, 'h'), 'left')
    hi = np.searchsorted(ft, onsets, 'left')
    ok = hi > lo
    leads[ok] = (onsets[ok] - ft[lo[ok]]) / np.timedelta64(1, 'h')
    return leads


def has_valid_in_window(onsets, valid_times):
    """Gets episode onsets + times of valid cycles (both sorted); gives a
    bool per onset: does any valid cycle sit in [onset - 72 h, onset)?"""
    if len(valid_times) == 0:
        return np.zeros(len(onsets), bool)
    vt = valid_times.values
    lo = np.searchsorted(vt, onsets - np.timedelta64(WINDOW_H, 'h'), 'left')
    hi = np.searchsorted(vt, onsets, 'left')
    return hi > lo


# =============================================================================
# Ratified-rule variant (SI sensitivity) — episode matching
# =============================================================================
def warning_episodes(times):
    """Gets sorted flagged cycle times; gives a list of arrays — each array
    one warning episode (flags separated by > EPISODE_GAP_H belong apart)."""
    if len(times) == 0:
        return []
    t = times.values
    gap_h = np.diff(t).astype('timedelta64[m]').astype(float) / 60.0
    splits = np.where(gap_h > EPISODE_GAP_H)[0] + 1
    return np.split(t, splits)


def match_events(onsets, weps):
    """One-to-one chronological matching (the ratified rule): observed
    episodes in time order each claim the earliest-starting unused warning
    episode holding a flagged cycle c with c < onset <= c + 72 h.
    Gives (leads with NaN for misses, unmatched warning-episode starts)."""
    used = np.zeros(len(weps), bool)
    leads = np.full(len(onsets), np.nan)
    w72 = np.timedelta64(WINDOW_H, 'h')
    for j, t0 in enumerate(onsets):
        for i, w in enumerate(weps):
            if used[i]:
                continue
            if w[0] >= t0:
                break
            c = w[(w < t0) & (w >= t0 - w72)]
            if len(c):
                used[i] = True
                leads[j] = (t0 - c.min()) / np.timedelta64(1, 'h')
                break
    fa = [w[0] for i, w in enumerate(weps) if not used[i]]
    return leads, np.array(fa, dtype='datetime64[ns]')


# =============================================================================
# Bootstrap helpers (paired 10-day blocks, one pick matrix for everything)
# =============================================================================
def block_of(times, year=EVAL_YEAR):
    t0 = pd.Timestamp(f'{year}-01-01')
    return ((pd.DatetimeIndex(times) - t0).days // BLOCK_DAYS).to_numpy()


def block_sums(blocks, values, nb):
    out = np.zeros(nb)
    np.add.at(out, blocks, values.astype(float))
    return out


def boot_ci(num_b, den_b, picks, den_const=None):
    """Gets per-block numerator (and denominator, or a constant denominator);
    gives (lo, hi) of the resampled ratio."""
    num = num_b[picks].sum(axis=1)
    if den_const is not None:
        r = num / den_const
    else:
        den = den_b[picks].sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            r = num / den
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(lo), float(hi)


# =============================================================================
# Referendum supporting tests (Brier + reliability, ens vs dressed AI)
# =============================================================================
def fit_isotonic(p_fit, y_fit):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                             out_of_bounds='clip')
    iso.fit(p_fit, y_fit)
    return iso


def brier_stats(p, y):
    b = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    b_clim = float(np.mean((base - y) ** 2))
    return dict(brier=b,
                bss_vs_climatology=1 - b / b_clim if b_clim > 0 else np.nan,
                n=int(len(y)), event_rate=base)


def bin_reliability(p, y):
    idx = np.clip(np.digitize(p, PROB_BIN_EDGES[1:-1]), 0, 9)
    rows = []
    for b in range(10):
        m = idx == b
        n = int(m.sum())
        rows.append(dict(bin_left=PROB_BIN_EDGES[b],
                         bin_right=PROB_BIN_EDGES[b + 1], n=n,
                         pred_mean=float(p[m].mean()) if n else np.nan,
                         obs_freq=float(y[m].mean()) if n else np.nan))
    return rows


# =============================================================================
# Main analysis
# =============================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open('paper_fig1_config_v2.json') as f:
        fig_cfg = json.load(f)
    with open('x_ensemble_calibrate_v2_config.json') as f:
        uq_cfg = json.load(f)
    rdirs = sorted(glob.glob(os.path.join(uq_cfg['BASE_DIR'], SPAN,
                                          'region_*')))
    elig = eligible_station_ids(fig_cfg)
    print(f'{len(rdirs)} regions; eligibility gate: '
          f'{"none" if elig is None else len(elig)} stations')

    probs = build_probs(rdirs)
    dstofs_check = validate_dstofs(rdirs, probs)
    print(f'dressed-STOFS forward-direction check (region_01, minor): '
          f'max |diff| = {dstofs_check:.4g}')

    ev = load_events(rdirs, elig)
    ev = ev.merge(probs, on=['cycle_time', 'station_id', 'threshold'],
                  how='left')
    epi = load_episodes(rdirs, elig)
    span_end = ev['cycle_time'].max() + pd.Timedelta(hours=CYCLE_H)
    yr_len = {FIT_YEAR: (pd.Timestamp(f'{FIT_YEAR + 1}-01-01')
                         - pd.Timestamp(f'{FIT_YEAR}-01-01')).days / 365.25,
              EVAL_YEAR: (span_end
                          - pd.Timestamp(f'{EVAL_YEAR}-01-01')).days / 365.25}
    n_st = {t: ev[ev['threshold'] == t]['station_id'].nunique()
            for t in THRESHOLDS}
    print('stations per threshold:', n_st, '| year lengths:', yr_len)

    grids = {'ens': GRID_ENS, 'dai': GRID_DRESSED, 'dstofs': GRID_DRESSED}
    pcol = {'ens': 'ai_prob_raw', 'dai': 'dai_prob', 'dstofs': 'dstofs_prob'}
    epi_by = {k: g['onset'].sort_values().to_numpy()
              for k, g in epi.groupby(['station_id', 'threshold'])}

    # -------- pass 1: sweep both rules --------
    agg_cov = {}     # (system, cutoff_idx|-1, thr) -> {year: [hits, false_cycles, episodes]}
    agg_epi = {}     # same keys, ratified-rule counts [hits, fa_episodes, episodes]
    groups = {}

    def add(agg, key, year, a, b, c):
        d = agg.setdefault(key, {})
        y = d.setdefault(year, [0, 0, 0])
        y[0] += a; y[1] += b; y[2] += c

    print('pass 1: sweep (both rules) ...')
    for (sid, thr), g in ev.groupby(['station_id', 'threshold']):
        g = g.sort_values('cycle_time')
        ct = pd.DatetimeIndex(g['cycle_time'])
        cyear = np.asarray(ct.year)
        onsets = epi_by.get((sid, thr), np.array([], dtype='datetime64[ns]'))
        p = {s: g[pcol[s]].to_numpy(float) for s in PROB_SYSTEMS}
        valid = {s: np.isfinite(p[s]) for s in PROB_SYSTEMS}
        valid['stofs_det'] = g['stofs_valid'].to_numpy(bool)
        valid['ai_det'] = np.isfinite(g['ai_prob_raw'].to_numpy(float))
        flags_det = {
            'stofs_det': g['stofs_event'].to_numpy(bool) & valid['stofs_det'],
            'ai_det': g['ai_det_event'].to_numpy(bool) & valid['ai_det']}
        obs_ok = g['obs_valid'].to_numpy(bool)
        obs_ev_c = g['obs_event'].to_numpy(bool)

        # Matched-availability episode set (frozen rule): every system must
        # have had at least one valid cycle in the 72 h before onset.
        keep = np.ones(len(onsets), bool)
        for s in ALL_SYSTEMS:
            keep &= has_valid_in_window(onsets, ct[valid[s]])
        oy = (pd.DatetimeIndex(onsets[keep]).year.to_numpy()
              if keep.any() else np.array([], dtype=int))
        groups[(sid, thr)] = (ct, cyear, onsets, keep, oy, p, valid,
                              flags_det, obs_ok, obs_ev_c,
                              int(len(onsets) - keep.sum()))

        def sweep_one(key, fl):
            # coverage rule: hits per episode + false-alert cycles
            leads = episode_leads(onsets, ct[fl])[keep]
            false_c = fl & obs_ok & ~obs_ev_c
            # ratified variant: one-to-one matching on warning episodes
            leads_m, fa = match_events(onsets, warning_episodes(ct[fl]))
            leads_m = leads_m[keep]
            fy = pd.DatetimeIndex(fa).year.to_numpy() if len(fa) else []
            for year in (FIT_YEAR, EVAL_YEAR):
                ep_n = int((oy == year).sum())
                add(agg_cov, key, year,
                    int(np.isfinite(leads[oy == year]).sum()),
                    int(false_c[cyear == year].sum()), ep_n)
                add(agg_epi, key, year,
                    int(np.isfinite(leads_m[oy == year]).sum()),
                    int((np.asarray(fy) == year).sum()), ep_n)

        for s in DET_SYSTEMS:
            sweep_one((s, -1, thr), flags_det[s])
        for s in PROB_SYSTEMS:
            for ki, cut in enumerate(grids[s]):
                sweep_one((s, ki, thr), valid[s] & (p[s] >= cut))

    # -------- cutoff selection on 2024, per rule --------
    def burden_cov(key, year):     # false-alert days per station-year
        h, f, e = agg_cov[key].get(year, [0, 0, 0])
        return f * (CYCLE_H / 24.0) / (n_st[key[2]] * yr_len[year])

    def burden_epi(key, year):     # unmatched warning episodes per st-yr
        h, f, e = agg_epi[key].get(year, [0, 0, 0])
        return f / (n_st[key[2]] * yr_len[year])

    frozen = {}
    for rule, burden in [('coverage', burden_cov), ('epimatch', burden_epi)]:
        frozen[rule] = {}
        for thr in THRESHOLDS:
            target = burden(('stofs_det', -1, thr), FIT_YEAR)
            frozen[rule][thr] = dict(target_burden_2024=float(target))
            for s in PROB_SYSTEMS:
                burds = np.array([burden((s, ki, thr), FIT_YEAR)
                                  for ki in range(len(grids[s]))])
                ki = int(np.argmin(np.abs(burds - target) + 1e-9 * burds))
                frozen[rule][thr][s] = dict(
                    cutoff_idx=ki, cutoff=float(grids[s][ki]),
                    burden_2024=float(burds[ki]))
    for thr in THRESHOLDS:
        print(f"frozen cutoffs [{thr}] coverage: " + ', '.join(
            f"{s}={frozen['coverage'][thr][s]['cutoff']:.2f}"
            for s in PROB_SYSTEMS) +
            f" (target {frozen['coverage'][thr]['target_burden_2024']:.1f} "
            f"false-alert days/st-yr)")

    # -------- pass 2: frozen-point detail (primary rule) --------
    print('pass 2: frozen operating points (episode detail) ...')
    detail, false_times = [], {}
    detail_epi = []
    for (sid, thr), (ct, cyear, onsets, keep, oy, p, valid, flags_det,
                     obs_ok, obs_ev_c, n_drop) in groups.items():
        onsets_k = onsets[keep]
        m25 = oy == EVAL_YEAR
        for s in ALL_SYSTEMS:
            if s in DET_SYSTEMS:
                fl = flags_det[s]
            else:
                fl = valid[s] & (p[s] >= frozen['coverage'][thr][s]['cutoff'])
            leads = episode_leads(onsets, ct[fl])[keep]
            for t0, ld in zip(onsets_k[m25], leads[m25]):
                detail.append(dict(system=s, threshold=thr, station_id=sid,
                                   onset=t0, lead_h=ld))
            fc = ct[fl & obs_ok & ~obs_ev_c & (cyear == EVAL_YEAR)]
            false_times.setdefault((s, thr), []).extend(list(fc.values))
            # ratified-rule variant detail (counts only, at its own cutoffs)
            if s in PROB_SYSTEMS:
                fl_m = valid[s] & (p[s] >= frozen['epimatch'][thr][s]['cutoff'])
            else:
                fl_m = fl
            leads_m, _ = match_events(onsets, warning_episodes(ct[fl_m]))
            leads_m = leads_m[keep]
            for t0, ld in zip(onsets_k[m25], leads_m[m25]):
                detail_epi.append(dict(system=s, threshold=thr, onset=t0,
                                       hit=bool(np.isfinite(ld))))

    det = pd.DataFrame(detail)
    det['hit'] = np.isfinite(det['lead_h'])
    det['block'] = block_of(det['onset'])
    det_epi = pd.DataFrame(detail_epi)
    nb = int(block_of(pd.DatetimeIndex(
        [span_end + pd.Timedelta(hours=WINDOW_H)]))[0]) + 1
    rng = np.random.default_rng(0)
    picks = rng.integers(0, nb, size=(N_BOOT, nb))

    # -------- outputs --------
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = dict(
        protocol=dict(
            span=SPAN, fit_year=FIT_YEAR, eval_year=EVAL_YEAR,
            episode_gap_h=EPISODE_GAP_H, window_h=WINDOW_H,
            min_fit_samples=MIN_FIT_SAMPLES, n_pseudo_members=len(Q_LEVELS),
            block_days=BLOCK_DAYS, n_boot=N_BOOT,
            primary_rule='episode caught if ANY pre-onset cycle window covers '
                         'its onset; false-alert burden = days on false alert '
                         'per station-year (flagged cycle with no exceedance '
                         'in window = 6 false hours; unverifiable cycles '
                         'excluded)',
            variant_rule='ratified one-to-one episode matching (12 h warning '
                         'clustering; unmatched warning episode = 1 false '
                         'alarm) — kept as SI sensitivity; see '
                         'correction_note',
            correction_note='one-to-one matching + 12 h clustering punishes '
                            'continuous warnings: 51% of minor episodes have '
                            'another episode within 72 h, so the sweep curve '
                            'was flat and burden matching picked cutoff 0.07. '
                            'Affects every probabilistic system symmetrically; '
                            'panel review decides.',
            selection='per-system cutoff chosen on 2024 only to match '
                      'deterministic STOFS 2024 burden in the rule\'s own '
                      'currency; dressings and isotonic fitted on 2024 only',
            notes=[
                'matched-availability episode set: every system must have '
                'one valid cycle in the 72 h before onset',
                'burden denominator = eligible stations with a defined '
                'threshold x span years',
            ]),
        dstofs_forward_check_max_diff=dstofs_check,
        stations_per_threshold=n_st, year_lengths=yr_len,
        frozen=frozen, episodes={}, referendum={}, decomposition={},
        burdens_2025={}, brier={}, sensitivity_epimatch={})

    # trade-off curves (2025), both rules
    for rule, agg, bfun, fname in [
            ('coverage', agg_cov, burden_cov, 'fig3_tradeoff.csv'),
            ('epimatch', agg_epi, burden_epi, 'fig3_tradeoff_epimatch.csv')]:
        rows = []
        for thr in THRESHOLDS:
            for s in PROB_SYSTEMS:
                for ki, cut in enumerate(grids[s]):
                    h, f, e = agg[(s, ki, thr)].get(EVAL_YEAR, [0, 0, 0])
                    rows.append(dict(system=s, threshold=thr, cutoff=cut,
                                     det_frac=h / e if e else np.nan,
                                     burden=bfun((s, ki, thr), EVAL_YEAR),
                                     n_hits=h, n_false=f, n_episodes=e))
            for s in DET_SYSTEMS:
                h, f, e = agg[(s, -1, thr)].get(EVAL_YEAR, [0, 0, 0])
                rows.append(dict(system=s, threshold=thr, cutoff=np.nan,
                                 det_frac=h / e if e else np.nan,
                                 burden=bfun((s, -1, thr), EVAL_YEAR),
                                 n_hits=h, n_false=f, n_episodes=e))
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, fname), index=False)

    # decomposition + referendum + burdens + leads (primary rule)
    lead_rows, dec_rows = [], []
    for thr in THRESHOLDS:
        d = det[det['threshold'] == thr]
        piv = d.pivot_table(index=['station_id', 'onset'], columns='system',
                            values='hit', aggfunc='first')
        lead_piv = d.pivot_table(index=['station_id', 'onset'],
                                 columns='system', values='lead_h',
                                 aggfunc='first').reindex(piv.index)
        blocks = block_of(piv.index.get_level_values('onset'))
        n_epi = len(piv)
        summary['episodes'][thr] = dict(
            n_eval=n_epi,
            n_dropped_availability=int(sum(v[10] for k, v in groups.items()
                                           if k[1] == thr)))
        den_b = block_sums(blocks, np.ones(n_epi), nb)

        for comp, a, b in [('ens_vs_stofs_det', 'ens', 'stofs_det'),
                           ('ens_vs_dai', 'ens', 'dai')]:
            ha = piv[a].to_numpy(float)
            hb = piv[b].to_numpy(float)
            diff_b = block_sums(blocks, ha - hb, nb)
            lo, hi = boot_ci(diff_b, den_b, picks)
            row = dict(threshold=thr, comparison=comp,
                       both=int(np.nansum(ha * hb)),
                       ens_only=int(np.nansum(ha * (1 - hb))),
                       other_only=int(np.nansum((1 - ha) * hb)),
                       neither=int(np.nansum((1 - ha) * (1 - hb))),
                       n_episodes=n_epi,
                       extra_catches=int(np.nansum(ha) - np.nansum(hb)),
                       diff_rate=float(np.nanmean(ha - hb)),
                       diff_lo=lo, diff_hi=hi)
            dec_rows.append(row)
            if comp == 'ens_vs_dai' and thr in REFERENDUM_THRESHOLDS:
                summary['referendum'][thr] = row
        summary['decomposition'][thr] = [r for r in dec_rows
                                         if r['threshold'] == thr]

        for s in ALL_SYSTEMS:
            hits = piv[s].to_numpy(float)
            leads = lead_piv[s].to_numpy(float)
            hit_b = block_sums(blocks, hits, nb)
            lo, hi = boot_ci(hit_b, den_b, picks)
            lead_rows.append(dict(system=s, threshold=thr, lead_ge_h=0,
                                  frac=float(np.nanmean(hits)),
                                  lo=lo, hi=hi, n_episodes=n_epi))
            for L in LEAD_MARKS:
                ok = (np.nan_to_num(leads) >= L).astype(float)
                ok_b = block_sums(blocks, ok, nb)
                lo, hi = boot_ci(ok_b, den_b, picks)
                lead_rows.append(dict(system=s, threshold=thr, lead_ge_h=L,
                                      frac=float(ok.mean()), lo=lo, hi=hi,
                                      n_episodes=n_epi))
            fc = pd.DatetimeIndex(false_times.get((s, thr), []))
            fc_b = block_sums(block_of(fc), np.ones(len(fc)), nb) \
                if len(fc) else np.zeros(nb)
            denom = n_st[thr] * yr_len[EVAL_YEAR] / (CYCLE_H / 24.0)
            lo, hi = boot_ci(fc_b, None, picks, den_const=denom)
            summary['burdens_2025'].setdefault(thr, {})[s] = dict(
                fa_days_per_sty=len(fc) / denom, lo=lo, hi=hi,
                n_false_cycles=int(len(fc)))

        # ratified-rule variant decomposition (counts only, SI)
        dv = det_epi[det_epi['threshold'] == thr]
        pv = dv.pivot_table(index='onset', columns='system', values='hit',
                            aggfunc='sum')
        sens = {}
        for comp, a, b in [('ens_vs_stofs_det', 'ens', 'stofs_det'),
                           ('ens_vs_dai', 'ens', 'dai')]:
            ha = dv[dv['system'] == a]['hit'].to_numpy(float)
            hb = dv[dv['system'] == b]['hit'].to_numpy(float)
            sens[comp] = dict(hits_a=int(ha.sum()), hits_b=int(hb.sum()),
                              extra=int(ha.sum() - hb.sum()))
        summary['sensitivity_epimatch'][thr] = sens
    pd.DataFrame(dec_rows).to_csv(os.path.join(OUT_DIR, 'fig3_decomp.csv'),
                                  index=False)
    pd.DataFrame(lead_rows).to_csv(os.path.join(OUT_DIR, 'fig3_lead.csv'),
                                   index=False)

    # referendum supporting tests: Brier + reliability (paired rows)
    rel_rows = []
    for thr in REFERENDUM_THRESHOLDS:
        d = ev[(ev['threshold'] == thr) & ev['obs_valid']]
        d = d[np.isfinite(d['ai_prob_raw']) & np.isfinite(d['dai_prob'])]
        yr = pd.DatetimeIndex(d['cycle_time']).year
        y_fit = d[yr == FIT_YEAR]['obs_event'].to_numpy(float)
        y_ev = d[yr == EVAL_YEAR]['obs_event'].to_numpy(float)
        summary['brier'][thr] = {}
        for s, col in [('ens', 'ai_prob_raw'), ('dai', 'dai_prob')]:
            iso = fit_isotonic(d[yr == FIT_YEAR][col].to_numpy(float), y_fit)
            p_ev = iso.predict(d[yr == EVAL_YEAR][col].to_numpy(float))
            summary['brier'][thr][s] = brier_stats(p_ev, y_ev)
            for r in bin_reliability(p_ev, y_ev):
                rel_rows.append(dict(system=s, threshold=thr, **r))
            for rule in ('coverage', 'epimatch'):
                cut = frozen[rule][thr][s]['cutoff']
                frozen[rule][thr][s]['cutoff_isotonic'] = float(
                    iso.predict([cut])[0])
    pd.DataFrame(rel_rows).to_csv(
        os.path.join(OUT_DIR, 'fig3_reliability.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'fig3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)

    print('\n=== headline numbers (2025 forward out-of-sample, primary rule) ===')
    for thr in THRESHOLDS:
        for r in summary['decomposition'][thr]:
            print(f"{thr:9s} {r['comparison']:17s} both {r['both']:4d} "
                  f"ens_only {r['ens_only']:3d} other_only "
                  f"{r['other_only']:3d} neither {r['neither']:3d} "
                  f"extra {r['extra_catches']:+d} "
                  f"[{r['diff_lo']:+.3f},{r['diff_hi']:+.3f}]")
    print('\n(ratified-rule variant, SI):')
    for thr in THRESHOLDS:
        for comp, v in summary['sensitivity_epimatch'][thr].items():
            print(f"{thr:9s} {comp:17s} ens {v['hits_a']:4d} vs "
                  f"{v['hits_b']:4d} -> extra {v['extra']:+d}")
    print('\nDone. Outputs in', OUT_DIR)
