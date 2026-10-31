"""
paper_fig2_warn_process.py — computes everything behind the deterministic-
warnings figure (new Figure 2 of the big-picture revisit).

In plain English: over the current-STOFS era (post 2024-05-14), we treat the
two deterministic forecasts — our AI (GFS-driven, 3-seed mean) and STOFS — as
warning machines ("warn when the forecast crosses the NOAA threshold within
72 h") and count, per severity: distinct flood episodes caught and missed,
false warning clusters, and days spent on false alert. Paired 10-day-block
bootstrap gives uncertainty on every score and on every AI-minus-STOFS
difference (the table bolds only differences whose interval excludes zero).

Rules (frozen in the figure review of 2026-07-17):
  - episodes: onsets from warning_lead.csv (>12 h gap rule), onset after the
    seam; kept only if BOTH systems had a valid cycle in the 72 h before
    onset (matched availability).
  - common validity: both systems' warnings are read only at cycles where
    BOTH have valid forecasts — same exposure for hits and false alarms.
  - hit: any pre-onset cycle whose 72 h window covers the onset (no
    one-to-one exclusivity — the rule ratified in the Figure 3/4 work).
  - false warning: a 12 h cluster of warned cycles none of which covers any
    real onset. Also expressed as days on false alert per station-year.

Writes (outputs/paper/v2/):
    fig2_warn_detgfs.parquet   cache: GFS-det window-peak events per cycle
    fig2_warn_stats.csv        severity x system scores + CIs + paired diffs
    fig2_warn_summary.json     protocol echo + episode counts
"""

import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from paper_common import eligible_station_ids
from paper_fig3_process import (episode_leads, warning_episodes,
                                has_valid_in_window)


SEAM = pd.Timestamp('2024-05-14T12:00')
SPAN = '20240101_20251201'
UQ_BASE = r'outputs/uq_postprocessing/nn_24lb_72fh_v2/gefs_psfc_v2_raw'
DET_BASE = r'outputs/paper/inference/nn_24lb_72fh_v2'
THRESHOLDS = ['minor', 'moderate', 'major']
WINDOW_H = 72
CYCLE_H = 6
BLOCK_DAYS = 10
N_BOOT = 1000

OUT_DIR = 'outputs/paper/v2'
DET_CACHE = os.path.join(OUT_DIR, 'fig2_warn_detgfs.parquet')
REBUILD_DET = False


def det_forecast(rname, sids):
    """Gets one region name + its verify station order; gives (cycle index,
    residual array (C,72,S)) of the 3-seed-mean GFS deterministic forecast."""
    rid = rname.replace('region_', '')
    parts = []
    for hh in sorted(glob.glob(os.path.join(DET_BASE, f'region_{rid}',
                                            'gfs_*z'))):
        seeds, sids_det = [], None
        for sd in sorted(glob.glob(os.path.join(hh, 'seed_*', 'forecast.nc'))):
            ds = xr.open_dataset(sd)
            seeds.append(ds['residual_mean'].load())
            sids_det = [str(s) for s in ds.station.values]
            ds.close()
        if seeds:
            parts.append((sum(seeds) / len(seeds), sids_det))
    if not parts:
        return None
    da = xr.concat([p[0] for p in parts], dim='cycle_time').sortby('cycle_time')
    sids_det = parts[0][1]
    idx = [sids_det.index(s) if s in sids_det else -1 for s in sids]
    arr = da.values[:, :, [max(i, 0) for i in idx]]
    for j, i in enumerate(idx):
        if i < 0:
            arr[:, :, j] = np.nan
    return pd.DatetimeIndex(da.cycle_time.values), arr


def build_det_events(rdirs):
    """Builds (or loads) GFS-det window-peak warning flags per
    cycle x gauge x threshold, in water-level space."""
    if os.path.exists(DET_CACHE) and not REBUILD_DET:
        print(f'[cache] {DET_CACHE}')
        return pd.read_parquet(DET_CACHE)
    rows = []
    for rd in rdirs:
        rname = os.path.basename(rd)
        ds = xr.open_dataset(os.path.join(rd, 'verify_arrays.nc'))
        sids = [str(s) for s in ds.station.values]
        cyc = pd.DatetimeIndex(ds.cycle_time.values)
        tide_slr = (ds.tide.values + ds.slr_component.values).astype(np.float32)
        thr = {t: ds[f'{t}_threshold'].values.astype(np.float32)
               for t in THRESHOLDS}
        ds.close()
        det = det_forecast(rname, sids)
        if det is None:
            print(f'  {rname}: no det archive')
            continue
        det_t, det_a = det
        pos = {t: i for i, t in enumerate(det_t)}
        j = np.array([pos.get(t, -1) for t in cyc])
        ok = j >= 0
        det_wl = np.full(tide_slr.shape, np.nan, np.float32)
        det_wl[ok] = det_a[j[ok]] + tide_slr[ok]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN')
            det_max = np.nanmax(det_wl, axis=1)
        for t in THRESHOLDS:
            for s in range(len(sids)):
                if not np.isfinite(thr[t][s]):
                    continue
                rows.append(pd.DataFrame(dict(
                    cycle_time=cyc, station_id=sids[s], threshold=t,
                    det_gfs_event=det_max[:, s] >= thr[t][s],
                    det_gfs_valid=np.isfinite(det_max[:, s]))))
        print(f'  det events -> {rname}')
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(DET_CACHE, index=False)
    return out


def block_of(times):
    """Gets datetimes; gives 10-day block index counted from the seam."""
    return ((pd.DatetimeIndex(times) - SEAM).days // BLOCK_DAYS).to_numpy()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open('paper_fig1_config_v2.json') as f:
        fig_cfg = json.load(f)
    elig = eligible_station_ids(fig_cfg)
    rdirs = sorted(glob.glob(os.path.join(UQ_BASE, SPAN, 'region_*')))

    det = build_det_events(rdirs)
    ev = pd.concat([pd.read_parquet(
        os.path.join(rd, 'decision_events.parquet'),
        columns=['cycle_time', 'station_id', 'threshold', 'obs_event',
                 'obs_valid', 'stofs_event', 'stofs_valid']) for rd in rdirs],
        ignore_index=True)
    ev['station_id'] = ev['station_id'].astype(str)
    ev = ev[ev['station_id'].isin(elig)]
    ev = ev.merge(det, on=['cycle_time', 'station_id', 'threshold'],
                  how='left')
    # keep only gauges with all three NOAA thresholds defined, so every
    # severity is scored on the same station set (drops 3 of 115: Santa
    # Barbara no-major, Wauna no-moderate, La Push minor-only)
    common = set.intersection(*[
        set(det.loc[det['threshold'] == t, 'station_id'])
        for t in THRESHOLDS])
    ev = ev[ev['station_id'].isin(common)]

    epi = pd.concat([pd.read_csv(
        os.path.join(rd, 'warning_lead.csv'),
        usecols=['station_id', 'threshold', 'episode_start'])
        for rd in rdirs], ignore_index=True)
    epi['station_id'] = epi['station_id'].astype(str)
    epi['onset'] = pd.to_datetime(epi['episode_start'])
    epi = epi[epi['station_id'].isin(elig) & epi['station_id'].isin(common)
              & (epi['onset'] >= SEAM + pd.Timedelta(hours=1))]

    span_end = ev['cycle_time'].max() + pd.Timedelta(hours=CYCLE_H)
    years = (span_end - SEAM).days / 365.25
    nb = int(block_of(pd.DatetimeIndex(
        [span_end + pd.Timedelta(hours=WINDOW_H)]))[0]) + 1
    w72 = np.timedelta64(WINDOW_H, 'h')

    # per (severity, system): per-block hits, episodes, false clusters,
    # and false-alert cycles (6 h each -> days on false alert)
    hit_b = {}; epi_b = {}; fa_b = {}; fac_b = {}
    # window currency (SI / sensitivity): per-block hit/miss/false WINDOWS
    # + total evaluated windows (common-valid station-cycles, per severity)
    w_hit = {}; w_ev = {}; w_fa = {}; w_tot = {}
    lead_rec = {}          # (thr, sys) -> ([block ids], [leads h]) of hits
    n_st = {}
    epi_by = {k: g['onset'].sort_values().to_numpy()
              for k, g in epi.groupby(['station_id', 'threshold'])}

    print('episode pass ...')
    for (sid, thr), g in ev.groupby(['station_id', 'threshold']):
        g = g.sort_values('cycle_time')
        ct = pd.DatetimeIndex(g['cycle_time'])
        post = np.asarray(ct >= SEAM)
        joint = (post & g['det_gfs_valid'].fillna(False).to_numpy(bool)
                 & g['stofs_valid'].to_numpy(bool))
        if not joint.any():
            continue
        n_st[thr] = n_st.get(thr, 0) + 1
        flags = {'ai': g['det_gfs_event'].fillna(False).to_numpy(bool) & joint,
                 'st': g['stofs_event'].to_numpy(bool) & joint}

        # window currency: contingency per cycle on the common-valid rows
        w_ok = joint & g['obs_valid'].to_numpy(bool)
        obs_f = g['obs_event'].to_numpy(bool)
        cb = block_of(ct)
        np.add.at(w_tot.setdefault(thr, np.zeros(nb)), cb[w_ok], 1.0)
        for s in ('ai', 'st'):
            hb = w_hit.setdefault((thr, s), np.zeros(nb))
            eb = w_ev.setdefault((thr, s), np.zeros(nb))
            fb = w_fa.setdefault((thr, s), np.zeros(nb))
            m = w_ok
            np.add.at(eb, cb[m & obs_f], 1.0)
            np.add.at(hb, cb[m & obs_f & flags[s]], 1.0)
            np.add.at(fb, cb[m & ~obs_f & flags[s]], 1.0)
        onsets_all = epi_by.get((sid, thr),
                                np.array([], dtype='datetime64[ns]'))
        keep = has_valid_in_window(onsets_all, ct[joint])
        onsets = onsets_all[keep]
        ob = block_of(onsets) if len(onsets) else np.array([], int)
        for s in ('ai', 'st'):
            hb = hit_b.setdefault((thr, s), np.zeros(nb))
            eb = epi_b.setdefault((thr, s), np.zeros(nb))
            fb = fa_b.setdefault((thr, s), np.zeros(nb))
            fcb = fac_b.setdefault((thr, s), np.zeros(nb))
            leads = episode_leads(onsets, ct[flags[s]])
            hit = np.isfinite(leads)
            np.add.at(eb, ob, 1.0)
            np.add.at(hb, ob[hit], 1.0)
            lr = lead_rec.setdefault((thr, s), ([], []))
            lr[0].extend(ob[hit].tolist())
            lr[1].extend(leads[hit].tolist())
            for w in warning_episodes(ct[flags[s]]):
                justified = False
                for c in w:
                    lo = np.searchsorted(onsets_all, c)
                    hi = np.searchsorted(onsets_all, c + w72, 'right')
                    if hi > lo:
                        justified = True
                        break
                if not justified:
                    fb[min(int(block_of([w[0]])[0]), nb - 1)] += 1
                    # each false cycle = 6 h on false alert, in its own block
                    np.add.at(fcb, np.minimum(block_of(w), nb - 1), 1.0)

    rng = np.random.default_rng(0)
    picks = rng.integers(0, nb, size=(N_BOOT, nb))

    def scores(h, e, f):
        with np.errstate(invalid='ignore', divide='ignore'):
            pod = h / e
            far = f / (h + f)
            csi = h / (e + f)          # e = hits + misses
        return pod, far, csi

    rows = []
    summary = dict(protocol=dict(
        period=f'{SEAM.date()} .. {span_end.date()} (current STOFS product)',
        years=round(years, 3), block_days=BLOCK_DAYS, n_boot=N_BOOT,
        rules=['gauges with all three NOAA thresholds defined only',
               'matched availability (both systems, 72 h pre-onset)',
               'common validity (warnings read only where both valid)',
               'hit = any pre-onset covering cycle (no exclusivity)',
               'false warning = 12 h cluster covering no onset'],
        ai='GFS-driven deterministic, 3-seed mean'),
        stations_per_threshold=n_st, episodes={}, diffs={})
    def boot_median_leads(thr):
        """Gets a severity; gives per-system bootstrap median-lead
        distributions using the shared block picks."""
        out = {}
        for s in ('ai', 'st'):
            blk = np.asarray(lead_rec[(thr, s)][0], int)
            ld = np.asarray(lead_rec[(thr, s)][1], float)
            by_block = [ld[blk == b] for b in range(nb)]
            meds = np.empty(N_BOOT)
            for i in range(N_BOOT):
                sel = np.concatenate([by_block[b] for b in picks[i]]) \
                    if len(ld) else np.array([np.nan])
                meds[i] = np.nanmedian(sel) if sel.size else np.nan
            out[s] = meds
        return out

    for thr in THRESHOLDS:
        boot = {}
        lead_boot = boot_median_leads(thr)
        sty = n_st[thr] * years
        fad = {s: fac_b[(thr, s)][picks].sum(1) * (CYCLE_H / 24.0) / sty
               for s in ('ai', 'st')}
        for s in ('ai', 'st'):
            h, e, f = hit_b[(thr, s)], epi_b[(thr, s)], fa_b[(thr, s)]
            H, E, Fa = h[picks].sum(1), e[picks].sum(1), f[picks].sum(1)
            boot[s] = scores(H, E, Fa)
            pod, far, csi = scores(h.sum(), e.sum(), f.sum())
            ci = [np.nanpercentile(b, [2.5, 97.5]) for b in boot[s]]
            fci = np.percentile(fad[s], [2.5, 97.5])
            lead_med = float(np.nanmedian(lead_rec[(thr, s)][1])) \
                if lead_rec[(thr, s)][1] else np.nan
            lci = np.nanpercentile(lead_boot[s], [2.5, 97.5])
            rows.append(dict(
                threshold=thr, system=s, episodes=int(e.sum()),
                caught=int(h.sum()), missed=int((e - h).sum()),
                false_warns=int(f.sum()),
                fa_days_per_sty=round(float(fac_b[(thr, s)].sum())
                                      * CYCLE_H / 24.0 / sty, 2),
                fa_lo=round(float(fci[0]), 2), fa_hi=round(float(fci[1]), 2),
                POD=round(float(pod), 3), POD_lo=round(ci[0][0], 3),
                POD_hi=round(ci[0][1], 3),
                FAR=round(float(far), 3), FAR_lo=round(ci[1][0], 3),
                FAR_hi=round(ci[1][1], 3),
                CSI=round(float(csi), 3), CSI_lo=round(ci[2][0], 3),
                CSI_hi=round(ci[2][1], 3),
                lead_med_h=round(lead_med, 1),
                lead_lo=round(float(lci[0]), 1),
                lead_hi=round(float(lci[1]), 1)))
        summary['episodes'][thr] = int(epi_b[(thr, 'ai')].sum())
        d = {}
        for k, name in [(0, 'POD'), (1, 'FAR'), (2, 'CSI')]:
            diff = boot['ai'][k] - boot['st'][k]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            d[name] = dict(lo=round(float(lo), 3), hi=round(float(hi), 3),
                           excludes_zero=bool(lo > 0 or hi < 0))
        fdiff = fad['ai'] - fad['st']
        lo, hi = np.percentile(fdiff, [2.5, 97.5])
        d['FA_DAYS'] = dict(lo=round(float(lo), 2), hi=round(float(hi), 2),
                            excludes_zero=bool(lo > 0 or hi < 0))
        ldiff = lead_boot['ai'] - lead_boot['st']
        lo, hi = np.nanpercentile(ldiff, [2.5, 97.5])
        d['LEAD'] = dict(lo=round(float(lo), 1), hi=round(float(hi), 1),
                         excludes_zero=bool(lo > 0 or hi < 0))
        summary['diffs'][thr] = d
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'fig2_warn_stats.csv'),
                              index=False)

    # window-currency stats + diffs (same bootstrap machinery)
    wrows = []
    summary['diffs_windows'] = {}
    for thr in THRESHOLDS:
        boot = {}
        for s in ('ai', 'st'):
            h, e, f = w_hit[(thr, s)], w_ev[(thr, s)], w_fa[(thr, s)]
            H, E, Fa = h[picks].sum(1), e[picks].sum(1), f[picks].sum(1)
            boot[s] = scores(H, E, Fa)
            pod, far, csi = scores(h.sum(), e.sum(), f.sum())
            ci = [np.nanpercentile(b, [2.5, 97.5]) for b in boot[s]]
            wrows.append(dict(
                threshold=thr, system=s, episodes=int(e.sum()),
                windows_total=int(w_tot[thr].sum()),
                caught=int(h.sum()), missed=int((e - h).sum()),
                false_warns=int(f.sum()), fa_days_per_sty=np.nan,
                POD=round(float(pod), 3), POD_lo=round(ci[0][0], 3),
                POD_hi=round(ci[0][1], 3),
                FAR=round(float(far), 3), FAR_lo=round(ci[1][0], 3),
                FAR_hi=round(ci[1][1], 3),
                CSI=round(float(csi), 3), CSI_lo=round(ci[2][0], 3),
                CSI_hi=round(ci[2][1], 3)))
        d = {}
        for k, name in [(0, 'POD'), (1, 'FAR'), (2, 'CSI')]:
            diff = boot['ai'][k] - boot['st'][k]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            d[name] = dict(lo=round(float(lo), 3), hi=round(float(hi), 3),
                           excludes_zero=bool(lo > 0 or hi < 0))
        summary['diffs_windows'][thr] = d
    pd.DataFrame(wrows).to_csv(
        os.path.join(OUT_DIR, 'fig2_warn_stats_windows.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'fig2_warn_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    print(pd.DataFrame(rows)[['threshold', 'system', 'episodes', 'caught',
                              'missed', 'false_warns', 'fa_days_per_sty',
                              'POD', 'FAR', 'CSI']].to_string(index=False))
    for thr, d in summary['diffs'].items():
        print(thr, {k: (v['lo'], v['hi'], v['excludes_zero'])
                    for k, v in d.items()})
    print('Done.')
