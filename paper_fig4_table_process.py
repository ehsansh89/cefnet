"""
paper_fig4_table_process.py — computes the merged Figure 4 table (v4 arc):
the old deterministic-warnings columns (caught / missed / false warns /
POD / FAR / CSI), now with the probabilistic systems shown as fixed-cutoff
RANGES instead of one fitted operating point.

In plain English: every 72-h forecast window over the whole current-STOFS
era is scored for six warning rules — the deterministic AI, deterministic
STOFS, the AI 93-member ensemble warning when >=10% (and >=90%) of members
cross the threshold, and the dressed-STOFS ensemble at the same two member
fractions. Because 10% and 90% are fixed round numbers (nothing selected
from data), every row can honestly use the full record — one epoch for the
whole figure. Paired 10-day-block bootstrap gives the uncertainty; only the
deterministic AI-vs-STOFS differences are significance-tested (the range
rows sit at different workloads and are descriptive).

Writes (outputs/paper/v2/):
    fig4_table_stats.csv     severity x system scores + CIs
    fig4_table_summary.json  protocol echo + paired det-vs-STOFS verdicts
"""

import os
import json
import glob

import numpy as np
import pandas as pd

from paper_common import eligible_station_ids
from paper_fig2_warn_process import (block_of, SEAM, SPAN, UQ_BASE,
                                     THRESHOLDS, BLOCK_DAYS, N_BOOT,
                                     DET_CACHE)


OUT_DIR = 'outputs/paper/v2'
# fixed a-priori member fractions (not fitted): the main table prints the
# [0.10, 0.90] range; the full ladder feeds the all-cutoffs draft variant
CUTS = tuple(round(0.1 * k, 1) for k in range(1, 10))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open('paper_fig1_config_v2.json') as f:
        elig = set(eligible_station_ids(json.load(f)))
    rdirs = sorted(glob.glob(os.path.join(UQ_BASE, SPAN, 'region_*')))

    det = pd.read_parquet(DET_CACHE)
    ev = pd.concat([pd.read_parquet(
        os.path.join(rd, 'decision_events.parquet'),
        columns=['cycle_time', 'station_id', 'threshold', 'obs_event',
                 'obs_valid', 'ai_prob_raw', 'stofs_event', 'stofs_valid',
                 'dressed_prob']) for rd in rdirs], ignore_index=True)
    ev['station_id'] = ev['station_id'].astype(str)
    ev = ev[ev['station_id'].isin(elig)]
    common = set.intersection(*[
        set(det.loc[det['threshold'] == t, 'station_id'])
        for t in THRESHOLDS])
    ev = ev[ev['station_id'].isin(common)]
    ev = ev.merge(det, on=['cycle_time', 'station_id', 'threshold'],
                  how='left')

    ev = ev[(ev['cycle_time'] >= SEAM) & ev['obs_valid']
            & ev['stofs_valid'] & ev['det_gfs_valid'].fillna(False)
            & ev['ai_prob_raw'].notna() & ev['dressed_prob'].notna()]
    print(f'six-way common-valid post-seam windows (per severity): '
          f'{len(ev[ev["threshold"] == "minor"]):,} at '
          f'{ev["station_id"].nunique()} gauges')

    warn = {
        'ai_det': ev['det_gfs_event'].to_numpy(bool),
        'stofs': ev['stofs_event'].to_numpy(bool)}
    for c in CUTS:
        warn[f'ens{int(c * 100)}'] = (ev['ai_prob_raw'] >= c).to_numpy()
        warn[f'dst{int(c * 100)}'] = (ev['dressed_prob'] >= c).to_numpy()
    obs = ev['obs_event'].to_numpy(bool)
    blk = block_of(ev['cycle_time'])
    thr_arr = ev['threshold'].to_numpy()
    nb = int(blk.max()) + 1
    rng = np.random.default_rng(0)
    picks = rng.integers(0, nb, size=(N_BOOT, nb))

    def scores(h, e, f):
        with np.errstate(invalid='ignore', divide='ignore'):
            return h / e, f / (h + f), h / (e + f)

    rows, summary = [], dict(protocol=dict(
        period=f'{SEAM.date()} .. {ev["cycle_time"].max().date()} '
               '(current STOFS product; full record, one epoch)',
        gauges=int(ev['station_id'].nunique()),
        cutoffs='raw member fractions 0.10 .. 0.90 by 0.10, fixed a priori '
                '(main table prints the [0.10, 0.90] range)',
        note='range rows are descriptive (different workloads); only the '
             'deterministic pair is significance-tested',
        block_days=BLOCK_DAYS, n_boot=N_BOOT), diffs={})
    boot_store = {}
    for thr in THRESHOLDS:
        sel = thr_arr == thr
        o, b = obs[sel], blk[sel]
        for sysk, w in warn.items():
            ws = w[sel]
            hb = np.bincount(b[o & ws], minlength=nb).astype(float)
            eb = np.bincount(b[o], minlength=nb).astype(float)
            fb = np.bincount(b[~o & ws], minlength=nb).astype(float)
            H, E, Fa = hb[picks].sum(1), eb[picks].sum(1), fb[picks].sum(1)
            boot_store[(thr, sysk)] = scores(H, E, Fa)
            pod, far, csi = scores(hb.sum(), eb.sum(), fb.sum())
            ci = [np.nanpercentile(x, [2.5, 97.5])
                  for x in boot_store[(thr, sysk)]]
            rows.append(dict(
                threshold=thr, system=sysk,
                windows_total=int(sel.sum()), events=int(o.sum()),
                caught=int((o & ws).sum()), missed=int((o & ~ws).sum()),
                false_warns=int((~o & ws).sum()),
                POD=round(float(pod), 3), POD_lo=round(ci[0][0], 3),
                POD_hi=round(ci[0][1], 3),
                FAR=round(float(far), 3), FAR_lo=round(ci[1][0], 3),
                FAR_hi=round(ci[1][1], 3),
                CSI=round(float(csi), 3), CSI_lo=round(ci[2][0], 3),
                CSI_hi=round(ci[2][1], 3)))
        d = {}
        for k, name in [(0, 'POD'), (1, 'FAR'), (2, 'CSI')]:
            diff = boot_store[(thr, 'ai_det')][k] - boot_store[(thr,
                                                                'stofs')][k]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            d[name] = dict(lo=round(float(lo), 3), hi=round(float(hi), 3),
                           excludes_zero=bool(lo > 0 or hi < 0))
        summary['diffs'][thr] = d

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'fig4_table_stats.csv'),
                              index=False)
    with open(os.path.join(OUT_DIR, 'fig4_table_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(pd.DataFrame(rows)[['threshold', 'system', 'caught', 'missed',
                              'false_warns', 'POD', 'FAR',
                              'CSI']].to_string(index=False))
    for thr, d in summary['diffs'].items():
        print(thr, {k: (v['lo'], v['hi'], v['excludes_zero'])
                    for k, v in d.items()})
    print('Done.')
