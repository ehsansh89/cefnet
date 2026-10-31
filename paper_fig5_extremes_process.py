"""
paper_fig5_extremes_process.py — computes everything behind the v4 Figure 2
("Skill distribution — where the skill lives"; v3-era name: Figure 5,
extremes statistics). Script keeps the v3 numbering until the renaming sweep.

In plain English: for every gauge we ask two questions against STOFS-2D —
"how much better is the AI on ALL forecast windows?" (the normal case) and
"how much better is it on just the windows where flooding actually happened?"
(the peak-only case, split by minor/moderate/major severity). We also measure,
window by window, how many centimeters each model's predicted 72-h peak missed
the observed peak by (signed: negative = under-called). Paired 10-day-block
bootstrap gives the uncertainty on the medians.

Protocol (frozen in the panel doc): post-seam only (current STOFS product),
common-valid windows (both systems + obs valid), the 112 gauges with all three
NOAA thresholds, AI = GFS-driven deterministic 3-seed mean. Same currency as
Figures 2/4 of the v4 arc.

Severity rule (changed 2026-10-20, Navid's review): each flood window is
counted ONCE, at the highest NOAA threshold it crossed (minor-only /
moderate-only / major), the same grouping Figure 3b-c already used. The
August draft counted a window at every threshold it crossed (a major window
was also a minor and a moderate window); set SEVERITY_RULE = 'exceedance' to
reproduce those numbers (kept as the sensitivity for the response letter).
Warning and probability statistics (Figs 5-6) are unaffected: there the event
at each threshold is, and must be, exceedance of that threshold.

Also computes the two pre-lock checks the figure review asked for (2026-07-22): the
flood-hours tier (panel-a skill recomputed over only the hours the observed
water was above the threshold — did calm hours inside flood windows flatter
the transfer claim?) and the intensity-binned peak errors (median signed peak
error in 10-cm bins of observed surge size and of exceedance above the local
threshold — the fair version of "what happens bin by bin?").

Writes (outputs/paper/v2/):
    fig5_extremes_windows.parquet   cache: per-window MAEs + predicted peaks
                                    + flood-hour error sums + station meta
    fig5_extremes_quadrant.csv      per gauge x severity skill table (panel a
                                    + flood-hours tier + lat/lon)
    fig5_extremes_peaks.parquet     per flood-window signed peak errors (panel b)
    fig5_extremes_peakbins.csv      binned peak errors (SI sensitivity)
    fig5_extremes_summary.json      fractions, medians, CIs, protocol echo
"""

import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from paper_common import eligible_station_ids
from paper_fig2_warn_process import (det_forecast, block_of, SEAM, SPAN,
                                     UQ_BASE, THRESHOLDS, BLOCK_DAYS, N_BOOT)


OUT_DIR = 'outputs/paper/v2'
WIN_CACHE = os.path.join(OUT_DIR, 'fig5_extremes_windows.parquet')
HOURLY_CACHE = os.path.join(OUT_DIR, 'fig5_extremes_hourly_hist.npz')
REBUILD = False
# 'highest' = a flood window belongs to the highest severity it reached (paper);
# 'exceedance' = it belongs to every severity whose threshold it crossed (Aug draft)
SEVERITY_RULE = 'highest'
# minimum flood windows for a gauge to enter a facet's headline fraction
# (declared from the count distribution only; all-gauge numbers also written)
MIN_FLOOD_WINDOWS = 5
# hourly histogram grids: x = observed hourly surge in 10-cm bins (clipped
# ends are open overflow groups), errors counted at 1-cm resolution
HX_LO, HX_N = -100, 50            # bins [-100, 400) cm, 10 cm each
HE_LO, HE_N = -500, 1001          # error grid [-500, 500] cm, 1 cm


def build_window_stats(rdirs):
    """Builds (or loads) the per-window continuous stats: for every
    cycle x gauge, the 72-h MAE of AI-det and STOFS vs observations (residual
    space, jointly valid leads only), each system's predicted water-level
    window maximum, the observed residual maximum, and — per severity — the
    error sums over only the hours the observed water was above the NOAA
    threshold (the flood-hours tier). Gives one DataFrame."""
    if os.path.exists(WIN_CACHE) and not REBUILD:
        print(f'[cache] {WIN_CACHE}')
        return pd.read_parquet(WIN_CACHE)
    with open('paper_fig1_config_v2.json') as f:
        elig_h = set(eligible_station_ids(json.load(f)))
    hx = np.zeros((2, HX_N, HE_N), np.int64)     # [ai/stofs, x-bin, error]
    hx_g, hx_b = set(), set()                    # (x-bin, gauge/block) pairs
    rows = []
    for rd in rdirs:
        rname = os.path.basename(rd)
        ds = xr.open_dataset(os.path.join(rd, 'verify_arrays.nc'))
        sids = [str(s) for s in ds.station.values]
        cyc = pd.DatetimeIndex(ds.cycle_time.values)
        obs = ds['obs_residual'].values.astype(np.float32)      # (C,72,S)
        sto = ds['stofs_residual'].values.astype(np.float32)
        tide_slr = (ds.tide.values + ds.slr_component.values).astype(np.float32)
        thr = {t: ds[f'{t}_threshold'].values.astype(np.float32)
               for t in THRESHOLDS}                             # (S,)
        lat = ds['station_lat'].values.astype(float)
        lon = ds['station_lon'].values.astype(float)
        rid = int(ds.attrs['region_id'])
        ds.close()
        det = det_forecast(rname, sids)
        if det is None:
            print(f'  {rname}: no det archive')
            continue
        det_t, det_a = det
        pos = {t: i for i, t in enumerate(det_t)}
        j = np.array([pos.get(t, -1) for t in cyc])
        ok = j >= 0
        det_r = np.full(obs.shape, np.nan, np.float32)
        det_r[ok] = det_a[j[ok]]

        m = np.isfinite(obs) & np.isfinite(sto) & np.isfinite(det_r)

        # hourly histograms (SI): every jointly-valid post-seam hour at the
        # eligible all-three-thresholds gauges, binned by the OBSERVED
        # hourly surge, errors counted on a 1-cm grid
        thr_ok = np.ones(len(sids), bool)
        for t in THRESHOLDS:
            thr_ok &= np.isfinite(thr[t])
        keep_s = np.array([sids[s] in elig_h and thr_ok[s]
                           for s in range(len(sids))])
        mm = (m & np.asarray(cyc >= SEAM)[:, None, None]
              & keep_s[None, None, :])
        if mm.any():
            o = obs[mm].astype(np.float64)
            ix = np.clip(((o * 100 - HX_LO) // 10).astype(np.int64),
                         0, HX_N - 1)
            for row, pred in ((0, det_r[mm]), (1, sto[mm])):
                e = (np.clip(np.rint((pred - o) * 100), HE_LO, -HE_LO)
                     .astype(np.int64) - HE_LO)
                hx[row] += np.bincount(ix * HE_N + e,
                                       minlength=HX_N * HE_N
                                       ).reshape(HX_N, HE_N)
            sidx = np.broadcast_to(np.arange(len(sids))[None, None, :],
                                   m.shape)[mm]
            hx_g.update(np.unique(ix * 100000 + rid * 1000
                                  + sidx).tolist())
            blk = np.broadcast_to(
                block_of(cyc)[:, None, None], m.shape)[mm]
            hx_b.update(np.unique(ix * 100000 + blk).tolist())

        nlead = m.sum(axis=1)                                   # (C,S)
        err_ai_h = np.where(m, np.abs(det_r - obs), 0)
        err_st_h = np.where(m, np.abs(sto - obs), 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            mae_ai = err_ai_h.sum(1) / nlead
            mae_st = err_st_h.sum(1) / nlead
        det_wl = np.where(m, det_r + tide_slr, np.nan)
        sto_wl = np.where(m, sto + tide_slr, np.nan)
        obs_wl = obs + tide_slr
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN')
            det_max = np.nanmax(det_wl, axis=1)
            sto_max = np.nanmax(sto_wl, axis=1)
            obs_res_max = np.nanmax(np.where(m, obs, np.nan), axis=1)
            # observed WL max on the SAME joint mask as the forecast maxima
            # (the review's panel-b audit: decision_events' obs_max_m spans all
            # obs-valid hours, incl. hours the forecasts aren't scored on)
            obs_wl_max = np.nanmax(np.where(m, obs_wl, np.nan), axis=1)
            # residual-space maxima: the surge-only stress test (the review's
            # panel-c alignment — x and y in the same physical space)
            det_res_max = np.nanmax(np.where(m, det_r, np.nan), axis=1)
            sto_res_max = np.nanmax(np.where(m, sto, np.nan), axis=1)
        # near-peak predicted maxima (Ehsan's timing answer): the forecast's
        # surge max within +/-k h of the hour the OBSERVED surge peaked —
        # event-locked but tolerant of small phase errors. k=3 is our
        # declared tolerance (justified by the 72-h horizon); k=1 matches
        # NOS's +/-1.5 h water-level timing acceptance on the hourly grid;
        # k=2 and 6 are further sensitivities
        has_pk = np.isfinite(obs_res_max)                       # (C,S)
        amax = np.where(m, obs, -np.inf).argmax(axis=1)         # (C,S)
        det_inf = np.where(m, det_r, -np.inf)
        sto_inf = np.where(m, sto, -np.inf)
        near = {}
        for k in (1, 2, 3, 6):
            bd = np.full(amax.shape, -np.inf, np.float32)
            bs = np.full(amax.shape, -np.inf, np.float32)
            for offs in range(-k, k + 1):
                idx = np.clip(amax + offs, 0, det_inf.shape[1] - 1)
                bd = np.maximum(bd, np.take_along_axis(
                    det_inf, idx[:, None, :], axis=1)[:, 0, :])
                bs = np.maximum(bs, np.take_along_axis(
                    sto_inf, idx[:, None, :], axis=1)[:, 0, :])
            bd[~has_pk | ~np.isfinite(bd)] = np.nan
            bs[~has_pk | ~np.isfinite(bs)] = np.nan
            near[k] = (bd, bs)
        # flood-hours tier: error sums over hours with observed WL >= threshold
        fh = {}
        for t in THRESHOLDS:
            mh = m & (obs_wl >= thr[t][None, None, :])
            fh[t] = (mh.sum(1), np.where(mh, err_ai_h, 0).sum(1),
                     np.where(mh, err_st_h, 0).sum(1))
        for s in range(len(sids)):
            d = dict(
                cycle_time=cyc, station_id=sids[s], region_id=rid,
                lat=lat[s], lon=lon[s],
                nlead=nlead[:, s], mae_ai=mae_ai[:, s], mae_st=mae_st[:, s],
                det_max_wl=det_max[:, s], stofs_max_wl=sto_max[:, s],
                obs_res_max=obs_res_max[:, s],
                obs_wl_max=obs_wl_max[:, s],
                det_res_max=det_res_max[:, s],
                stofs_res_max=sto_res_max[:, s])
            for k in (1, 2, 3, 6):
                d[f'det_res_np{k}'] = near[k][0][:, s]
                d[f'stofs_res_np{k}'] = near[k][1][:, s]
            for t in THRESHOLDS:
                d[f'fh_cnt_{t}'] = fh[t][0][:, s]
                d[f'fh_ai_{t}'] = fh[t][1][:, s]
                d[f'fh_st_{t}'] = fh[t][2][:, s]
            rows.append(pd.DataFrame(d))
        print(f'  window stats -> {rname}')
    out = pd.concat(rows, ignore_index=True)
    out = out[out['nlead'] > 0]
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(HOURLY_CACHE, hist=hx,
                        gpairs=np.array(sorted(hx_g), np.int64),
                        bpairs=np.array(sorted(hx_b), np.int64))
    out.to_parquet(WIN_CACHE, index=False)
    return out


def pooled_mae(g):
    """Gets a window group; gives lead-weighted pooled MAE (m) for both
    systems — same pooling as Figure 1 (every valid lead-hour counts once)."""
    w = g['nlead'].to_numpy(float)
    return (np.average(g['mae_ai'], weights=w),
            np.average(g['mae_st'], weights=w))


def highest_severity(ev):
    """Gets the per-threshold event table (one row per window x threshold).
    Gives one row per window with 'sev', the highest severity whose threshold
    the observed water crossed ('none' if it crossed none)."""
    piv = ev.pivot_table(index=['cycle_time', 'station_id'],
                         columns='threshold', values='obs_event',
                         aggfunc='first').fillna(False)
    sev = np.where(piv['major'], 'major',
                   np.where(piv['moderate'], 'moderate',
                            np.where(piv['minor'], 'minor', 'none')))
    out = piv.index.to_frame(index=False)
    out['sev'] = sev
    return out


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open('paper_fig1_config_v2.json') as f:
        fig_cfg = json.load(f)
    elig = eligible_station_ids(fig_cfg)
    rdirs = sorted(glob.glob(os.path.join(UQ_BASE, SPAN, 'region_*')))

    win = build_window_stats(rdirs)

    # window-level event flags + validity + observed peaks, per severity
    ev = pd.concat([pd.read_parquet(
        os.path.join(rd, 'decision_events.parquet'),
        columns=['cycle_time', 'station_id', 'threshold', 'thr_m',
                 'obs_event', 'obs_valid', 'stofs_valid', 'obs_max_m'])
        for rd in rdirs], ignore_index=True)
    ev['station_id'] = ev['station_id'].astype(str)
    ev = ev[ev['station_id'].isin(elig)]
    # 112-gauge rule: all three NOAA thresholds defined
    common = set.intersection(*[
        set(ev.loc[ev['threshold'] == t, 'station_id']) for t in THRESHOLDS])
    ev = ev[ev['station_id'].isin(common)]
    win = win[win['station_id'].isin(common)]

    df = ev.merge(win, on=['cycle_time', 'station_id'], how='inner')
    df = df[(df['cycle_time'] >= SEAM) & df['obs_valid'] & df['stofs_valid']
            & np.isfinite(df['mae_ai']) & np.isfinite(df['mae_st'])]
    # 'flood' marks the rows that count as this threshold's flood windows
    df = df.merge(highest_severity(df), on=['cycle_time', 'station_id'],
                  how='left')
    if SEVERITY_RULE == 'highest':
        df['flood'] = df['sev'] == df['threshold']
    elif SEVERITY_RULE == 'exceedance':
        df['flood'] = df['obs_event']
    else:
        raise ValueError(SEVERITY_RULE)
    print(f'severity rule: {SEVERITY_RULE}; windows by highest severity: '
          f'{df[df["threshold"] == "minor"]["sev"].value_counts().to_dict()}')

    base = df[df['threshold'] == 'minor']       # threshold-independent rows
    n_windows = len(base)
    print(f'common-valid post-seam windows: {n_windows:,} '
          f'at {base["station_id"].nunique()} gauges')

    # ---------------- panel (a): per-gauge quadrant table ----------------
    qrows = []
    for sid, g in base.groupby('station_id'):
        a_all, s_all = pooled_mae(g)
        row = dict(station_id=sid, region_id=int(g['region_id'].iloc[0]),
                   lat=float(g['lat'].iloc[0]), lon=float(g['lon'].iloc[0]),
                   n_windows_all=len(g),
                   mae_all_ai_cm=round(a_all * 100, 3),
                   mae_all_stofs_cm=round(s_all * 100, 3),
                   skill_all=round(1 - a_all / s_all, 4))
        for thr in THRESHOLDS:
            f = df[(df['threshold'] == thr) & (df['station_id'] == sid)
                   & df['flood']]
            row[f'n_flood_{thr}'] = len(f)
            if len(f):
                a_f, s_f = pooled_mae(f)
                row[f'mae_flood_ai_cm_{thr}'] = round(a_f * 100, 3)
                row[f'mae_flood_stofs_cm_{thr}'] = round(s_f * 100, 3)
                row[f'skill_flood_{thr}'] = round(1 - a_f / s_f, 4)
            else:
                row[f'mae_flood_ai_cm_{thr}'] = np.nan
                row[f'mae_flood_stofs_cm_{thr}'] = np.nan
                row[f'skill_flood_{thr}'] = np.nan
            # flood-hours tier (review check 1): same gauge, only the hours
            # the observed water was above this severity's threshold
            n_h = g[f'fh_cnt_{thr}'].sum()
            row[f'n_fh_hours_{thr}'] = int(n_h)
            if n_h > 0 and g[f'fh_st_{thr}'].sum() > 0:
                row[f'skill_fh_{thr}'] = round(
                    1 - g[f'fh_ai_{thr}'].sum() / g[f'fh_st_{thr}'].sum(), 4)
            else:
                row[f'skill_fh_{thr}'] = np.nan
        qrows.append(row)
    quad = pd.DataFrame(qrows)
    quad.to_csv(os.path.join(OUT_DIR, 'fig5_extremes_quadrant.csv'),
                index=False)

    # ---------------- panel (b): signed peak errors on flood windows ------
    # panel-b audit (review): obs peak measured on the joint mask, same as
    # the forecast maxima; decision_events' all-obs-hours obs_max_m kept
    # only as the audit reference
    gap = (df['obs_max_m'] - df['obs_wl_max'])
    print(f'obs-max audit: joint vs all-hours differ >1 cm on '
          f'{(gap > 0.01).mean():.2%} of rows; '
          f'median where differing {gap[gap > 0.01].median()*100:.1f} cm')
    fl = df[df['flood']].copy()
    fl['peak_err_ai_m'] = fl['det_max_wl'] - fl['obs_wl_max']
    fl['peak_err_stofs_m'] = fl['stofs_max_wl'] - fl['obs_wl_max']
    fl = fl[np.isfinite(fl['peak_err_ai_m'])
            & np.isfinite(fl['peak_err_stofs_m'])]
    fl[['cycle_time', 'station_id', 'region_id', 'threshold', 'thr_m',
        'obs_max_m', 'obs_wl_max', 'obs_res_max', 'peak_err_ai_m',
        'peak_err_stofs_m']].to_parquet(
        os.path.join(OUT_DIR, 'fig5_extremes_peaks.parquet'), index=False)

    # ------- intensity-binned peak errors (review check 3 + panel c) --------
    # ALL common-valid windows (one row each, via the minor slice), binned two
    # ways, each in ONE consistent physical space (the review's alignment fix):
    #   - surge bins (main panel c): x = observed residual max, y = forecast
    #     residual max − observed residual max (pure surge stress test)
    #   - exceedance bins (SI): x = observed WL peak above the gauge's minor
    #     threshold, y = WL-peak error (the operational view, as panel b)
    aw = base.copy()
    aw['wl_err_ai_cm'] = (aw['det_max_wl'] - aw['obs_wl_max']) * 100
    aw['wl_err_st_cm'] = (aw['stofs_max_wl'] - aw['obs_wl_max']) * 100
    aw['res_err_ai_cm'] = (aw['det_res_max'] - aw['obs_res_max']) * 100
    aw['res_err_st_cm'] = (aw['stofs_res_max'] - aw['obs_res_max']) * 100
    for k in (1, 2, 3, 6):
        aw[f'np{k}_err_ai_cm'] = (aw[f'det_res_np{k}']
                                  - aw['obs_res_max']) * 100
        aw[f'np{k}_err_st_cm'] = (aw[f'stofs_res_np{k}']
                                  - aw['obs_res_max']) * 100
    aw['res_max_cm'] = aw['obs_res_max'] * 100
    aw['exceed_cm'] = (aw['obs_wl_max'] - aw['thr_m']) * 100
    # predeclared bin-support rule (review): a displayed bin needs >= 20
    # windows from >= 3 gauges spanning >= 3 distinct 10-day blocks; failing
    # 10-cm bins merge upward into their neighbor, the top group is an open
    # ">= X" overflow, and no event is dropped
    MIN_BIN_N, MIN_BIN_GAUGES, MIN_BIN_BLOCKS = 20, 3, 3
    aw['block'] = block_of(aw['cycle_time'])
    surge_edges = np.arange(0, 400, 10)
    brows = []
    for btype, col, eai, est, edges in [
            ('surge_nearpeak_pm1h', 'res_max_cm', 'np1_err_ai_cm',
             'np1_err_st_cm', surge_edges),
            ('surge_nearpeak_pm2h', 'res_max_cm', 'np2_err_ai_cm',
             'np2_err_st_cm', surge_edges),
            ('surge_nearpeak_pm3h', 'res_max_cm', 'np3_err_ai_cm',
             'np3_err_st_cm', surge_edges),
            ('surge_nearpeak_pm6h', 'res_max_cm', 'np6_err_ai_cm',
             'np6_err_st_cm', surge_edges),
            ('obs_residual_max', 'res_max_cm', 'res_err_ai_cm',
             'res_err_st_cm', surge_edges),
            ('exceed_minor_thr', 'exceed_cm', 'wl_err_ai_cm',
             'wl_err_st_cm', np.arange(-40, 400, 10))]:
        ok = aw[np.isfinite(aw[eai]) & np.isfinite(aw[est])
                & (aw[col] >= edges[0])]
        groups, cur_lo, cur = [], None, []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = ok[(ok[col] >= lo) & (ok[col] < hi)]
            if not len(b) and cur_lo is None:
                continue                     # empty leading bin, nothing open
            if cur_lo is None:
                cur_lo = lo
            cur.append(b)
            g = pd.concat(cur)
            if (len(g) >= MIN_BIN_N
                    and g['station_id'].nunique() >= MIN_BIN_GAUGES
                    and g['block'].nunique() >= MIN_BIN_BLOCKS):
                groups.append((cur_lo, hi, g))
                cur_lo, cur = None, []
        if cur and any(len(b) for b in cur):
            # leftover tail that never met the rule: fold into the last
            # group, which becomes the open overflow
            lo0, _, g0 = groups.pop()
            groups.append((lo0, None, pd.concat([g0] + cur)))
        elif groups:
            lo0, _, g0 = groups.pop()        # top group is open-ended
            groups.append((lo0, None, g0))
        # descriptive hurricane-tail group (review): beyond TAIL_LO the
        # windows come from the same 2-3 storms, so within-tail resolution
        # is not supported by independent events — one pooled open group
        # says exactly what the data can
        TAIL_LO = 130
        if col == 'res_max_cm':
            tail = [t for t in groups if t[0] >= TAIL_LO]
            if len(tail) > 1:
                groups = [t for t in groups if t[0] < TAIL_LO]
                groups.append((tail[0][0], None,
                               pd.concat([t[2] for t in tail])))
        for lo, hi, g in groups:
            brows.append(dict(
                bin_type=btype, lo=int(lo),
                hi=(int(hi) if hi is not None else np.nan),
                open_end=hi is None, n=len(g),
                n_gauges=int(g['station_id'].nunique()),
                n_blocks=int(g['block'].nunique()),
                med_ai_cm=round(float(g[eai].median()), 2),
                med_st_cm=round(float(g[est].median()), 2),
                q25_ai=round(float(g[eai].quantile(.25)), 2),
                q75_ai=round(float(g[eai].quantile(.75)), 2),
                q25_st=round(float(g[est].quantile(.25)), 2),
                q75_st=round(float(g[est].quantile(.75)), 2)))
    pd.DataFrame(brows).to_csv(
        os.path.join(OUT_DIR, 'fig5_extremes_peakbins.csv'), index=False)

    # ------- SI: hourly conditional errors (Ehsan, 2026-07-22) -------------
    # every jointly-valid post-seam hour (not just window peaks), binned by
    # the observed hourly surge; same predeclared support rule + overflow
    if os.path.exists(HOURLY_CACHE):
        z = np.load(HOURLY_CACHE)
        hist, gpairs, bpairs = z['hist'], z['gpairs'], z['bpairs']
        from collections import defaultdict
        g_by, b_by = defaultdict(set), defaultdict(set)
        for p in gpairs:
            g_by[int(p) // 100000].add(int(p) % 100000)
        for p in bpairs:
            b_by[int(p) // 100000].add(int(p) % 100000)
        err_val = np.arange(HE_LO, HE_LO + HE_N)

        def hq(h, q):
            """Gets one error histogram + a quantile; gives the error value
            (cm) at that quantile."""
            c = np.cumsum(h)
            return float(err_val[np.searchsorted(c, q * c[-1])])

        hrows, cur_b, cur_lo = [], [], None
        for bi in range(HX_N):
            n_bi = int(hist[0, bi].sum())
            if n_bi == 0 and cur_lo is None:
                continue
            if cur_lo is None:
                cur_lo = bi
            cur_b.append(bi)
            n = int(hist[0, cur_b].sum())
            gset = set().union(*(g_by[b] for b in cur_b))
            bset = set().union(*(b_by[b] for b in cur_b))
            if n >= MIN_BIN_N and len(gset) >= MIN_BIN_GAUGES \
                    and len(bset) >= MIN_BIN_BLOCKS:
                h_ai = hist[0, cur_b].sum(0)
                h_st = hist[1, cur_b].sum(0)
                hrows.append(dict(
                    lo=HX_LO + cur_lo * 10, hi=HX_LO + (cur_b[-1] + 1) * 10,
                    open_start=cur_lo == 0, open_end=cur_b[-1] == HX_N - 1,
                    n=n, n_gauges=len(gset), n_blocks=len(bset),
                    med_ai_cm=hq(h_ai, .5), med_st_cm=hq(h_st, .5),
                    q25_ai=hq(h_ai, .25), q75_ai=hq(h_ai, .75),
                    q25_st=hq(h_st, .25), q75_st=hq(h_st, .75),
                    mae_ai_cm=round(float((np.abs(err_val) * h_ai).sum()
                                          / h_ai.sum()), 2),
                    mae_st_cm=round(float((np.abs(err_val) * h_st).sum()
                                          / h_st.sum()), 2)))
                cur_b, cur_lo = [], None
        if cur_b and hist[0, cur_b].sum() and hrows:
            last = hrows.pop()
            all_b = list(range(int((last['lo'] - HX_LO) // 10),
                               cur_b[-1] + 1))
            h_ai = hist[0, all_b].sum(0)
            h_st = hist[1, all_b].sum(0)
            gset = set().union(*(g_by[b] for b in all_b))
            bset = set().union(*(b_by[b] for b in all_b))
            hrows.append(dict(
                lo=last['lo'], hi=HX_LO + (all_b[-1] + 1) * 10,
                open_start=last['open_start'], open_end=True,
                n=int(h_ai.sum()), n_gauges=len(gset), n_blocks=len(bset),
                med_ai_cm=hq(h_ai, .5), med_st_cm=hq(h_st, .5),
                q25_ai=hq(h_ai, .25), q75_ai=hq(h_ai, .75),
                q25_st=hq(h_st, .25), q75_st=hq(h_st, .75),
                mae_ai_cm=round(float((np.abs(err_val) * h_ai).sum()
                                      / h_ai.sum()), 2),
                mae_st_cm=round(float((np.abs(err_val) * h_st).sum()
                                      / h_st.sum()), 2)))
        pd.DataFrame(hrows).to_csv(
            os.path.join(OUT_DIR, 'fig5_extremes_hourlybins.csv'),
            index=False)
        print(f'hourly bins: {len(hrows)} groups, '
              f'{int(hist[0].sum()):,} hours')
    else:
        print(f'[warn] {HOURLY_CACHE} missing - delete {WIN_CACHE} and '
              'rerun to build the hourly SI table')

    # ---------------- bootstraps ----------------
    rng = np.random.default_rng(0)
    span_end = base['cycle_time'].max()
    nb = int(block_of(pd.DatetimeIndex([span_end]))[0]) + 1
    picks = rng.integers(0, nb, size=(N_BOOT, nb))

    def block_median_boot(vals, blocks):
        """Gets values + their block ids; gives the N_BOOT bootstrap medians
        under the shared 10-day block resampling."""
        by = [vals[blocks == b] for b in range(nb)]
        out = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sel = np.concatenate([by[b] for b in picks[i]])
            out[i] = np.median(sel) if sel.size else np.nan
        return out

    summary = dict(protocol=dict(
        period=f'{SEAM.date()} .. {span_end.date()} (current STOFS product)',
        windows_common_valid=int(n_windows),
        gauges=int(base['station_id'].nunique()),
        ai='GFS-driven deterministic, 3-seed mean',
        mae_space='residual (tide+SLR removed); peaks in water-level space',
        severity_rule=SEVERITY_RULE,
        block_days=BLOCK_DAYS, n_boot=N_BOOT,
        min_flood_windows=MIN_FLOOD_WINDOWS),
        quadrant={}, quadrant_floodhours={}, peaks={})

    for thr in THRESHOLDS:
        q = quad.dropna(subset=[f'skill_flood_{thr}'])
        ur_all = ((q['skill_all'] > 0) & (q[f'skill_flood_{thr}'] > 0))
        qs = q[q[f'n_flood_{thr}'] >= MIN_FLOOD_WINDOWS]
        ur = ((qs['skill_all'] > 0) & (qs[f'skill_flood_{thr}'] > 0))
        # gauge-resample CI on the retained-set fraction
        fb = np.array([ur.sample(len(ur), replace=True,
                                 random_state=int(s)).mean()
                       for s in rng.integers(0, 2**31, N_BOOT)])
        lo, hi = np.percentile(fb, [2.5, 97.5])
        summary['quadrant'][thr] = dict(
            gauges_with_flood=int(len(q)),
            gauges_retained=int(len(qs)),
            frac_upper_right=round(float(ur.mean()), 3),
            frac_lo=round(float(lo), 3), frac_hi=round(float(hi), 3),
            frac_upper_right_allgauges=round(float(ur_all.mean()), 3),
            median_skill_flood=round(float(qs[f'skill_flood_{thr}'].median()),
                                     3),
            flood_window_count_quartiles=[
                int(v) for v in np.percentile(q[f'n_flood_{thr}'],
                                              [25, 50, 75])])

        # check 1 (flood-hours tier): same gauges, y = above-threshold-hours
        # skill — does the transfer conclusion survive the harder conditioning?
        qf = qs.dropna(subset=[f'skill_fh_{thr}'])
        urf = ((qf['skill_all'] > 0) & (qf[f'skill_fh_{thr}'] > 0))
        fb = np.array([urf.sample(len(urf), replace=True,
                                  random_state=int(s)).mean()
                       for s in rng.integers(0, 2**31, N_BOOT)])
        lo, hi = np.percentile(fb, [2.5, 97.5])
        summary['quadrant_floodhours'][thr] = dict(
            gauges=int(len(qf)),
            frac_upper_right=round(float(urf.mean()), 3),
            frac_lo=round(float(lo), 3), frac_hi=round(float(hi), 3),
            median_skill_fh=round(float(qf[f'skill_fh_{thr}'].median()), 3),
            hours_total=int(qf[f'n_fh_hours_{thr}'].sum()))

        f = fl[fl['threshold'] == thr]
        blk = block_of(f['cycle_time'])
        ai = f['peak_err_ai_m'].to_numpy() * 100
        st = f['peak_err_stofs_m'].to_numpy() * 100
        d = dict(n_windows=int(len(f)),
                 n_gauges=int(f['station_id'].nunique()))
        for name, v in [('ai', ai), ('stofs', st)]:
            b = block_median_boot(v, blk)
            d[f'median_signed_{name}_cm'] = round(float(np.median(v)), 2)
            d[f'signed_{name}_lo'] = round(float(np.nanpercentile(b, 2.5)), 2)
            d[f'signed_{name}_hi'] = round(float(np.nanpercentile(b, 97.5)), 2)
        for name, v in [('signed', ai - st), ('abs', np.abs(ai) - np.abs(st))]:
            b = block_median_boot(v, blk)
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            d[f'paired_med_diff_{name}_cm'] = round(float(np.median(v)), 2)
            d[f'paired_{name}_lo'] = round(float(lo), 2)
            d[f'paired_{name}_hi'] = round(float(hi), 2)
            d[f'paired_{name}_excludes_zero'] = bool(lo > 0 or hi < 0)
        summary['peaks'][thr] = d

    with open(os.path.join(OUT_DIR, 'fig5_extremes_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    print(json.dumps(summary['quadrant'], indent=1))
    print(json.dumps(summary['quadrant_floodhours'], indent=1))
    print(json.dumps(summary['peaks'], indent=1))
    print('Done.')
