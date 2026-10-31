"""
o_ensemble_decision.py — Decision-value backing data (referendum item R5,
feeds Figure 3): flood-threshold exceedance events over every cycle × gauge,
with calibrated ensemble probabilities vs every deterministic/dressed baseline.

Event definition: window-peak — "does water level exceed the NOAA threshold at
any lead 1..72 h of this forecast cycle?" (the decision-relevant form). For
each (cycle, station, threshold):
    obs_event      max obs_wl >= threshold
    ai_prob_raw    fraction of 93 member trajectories whose max >= threshold
    ai_prob_cal    same, after the held-out year-swap calibration transform
                   (bias + spread inflation around the member median, per
                   lead-bin; parameters from calibration_v2_params.json —
                   every cycle uses the OTHER year's fit)
    ai_det_event   median-trajectory max >= threshold (AI played as a
                   deterministic model — the fair single-answer row)
    stofs_event / ofs_event    deterministic physics window-peak flags
    dressed_prob   from x_dressed_baseline.py (held-out, quantile members)

Also reconstructs the hourly observed series (cycles are 6-hourly, leads 1..72
tile the timeline) to extract observed exceedance EPISODES (>=1 h above
threshold, episodes separated by >12 h), and computes per-episode warning lead
times: how many hours before episode start each system first flagged it
(probability cutoffs 0.1..0.9 for the probabilistic systems).

Writes (per region, next to verify_arrays.nc):
    decision_events.parquet
    warning_lead.csv
    decision_summary.json

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
THRESHOLDS = ['minor', 'moderate', 'major']
CHUNK = 96
EPISODE_GAP_H = 12
WARN_CUTOFFS = [round(p, 1) for p in np.arange(0.1, 0.91, 0.1)]


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def _lead_bin_index(n_lead):
    idx = np.zeros(n_lead, dtype=int)
    for b, (lo, hi) in enumerate(LEAD_BINS):
        idx[lo - 1:hi] = b
    return idx


def _params_arrays(params_path, lead_bin_idx):
    """Per-direction (bias_m, gamma) arrays over lead for the swap transform."""
    with open(params_path) as f:
        p = json.load(f)
    out = {}
    for name, d in p['directions'].items():
        b = np.array(list(d['bias_cm'].values()), np.float32)[lead_bin_idx] / 100
        g = np.array(list(d['gamma'].values()), np.float32)[lead_bin_idx]
        out[name] = (b, g)
    return out, int(p.get('split_year', 2025))


# =============================================================================
# Cycle-level events (streamed)
# =============================================================================
def build_events(ds, params, split_year):
    cycles = pd.to_datetime(ds.cycle_time.values)
    station_ids = [str(s) for s in ds.station.values]
    n_c, n_l = ds.sizes['cycle_time'], ds.sizes['lead_hour']
    n_s = ds.sizes['station']
    thr = {t: ds[f'{t}_threshold'].values.astype(np.float32)
           for t in THRESHOLDS}

    obs_wl = ds.obs_water_level.values.astype(np.float32)
    stofs_wl = ds.stofs_water_level.values.astype(np.float32)
    ofs_wl = ds.regional_ofs_water_level.values.astype(np.float32)
    obs_max = np.nanmax(obs_wl, axis=1)
    stofs_max = np.nanmax(stofs_wl, axis=1)
    ofs_max = np.nanmax(ofs_wl, axis=1)

    # Direction per cycle: 2024 cycles use the fit-2025 params and vice versa.
    is_a = cycles.year < split_year
    b24, g24 = params['fit2025_eval2024']       # applied to 2024 cycles
    b25, g25 = params['fit2024_eval2025']       # applied to 2025 cycles

    ai_med_max = np.full((n_c, n_s), np.nan, np.float32)
    prob_raw = {t: np.full((n_c, n_s), np.nan, np.float32) for t in THRESHOLDS}
    prob_cal = {t: np.full((n_c, n_s), np.nan, np.float32) for t in THRESHOLDS}
    for c0 in range(0, n_c, CHUNK):
        c1 = min(c0 + CHUNK, n_c)
        a = ds.ai_water_level_member.isel(cycle_time=slice(c0, c1)).values
        a = a.reshape(a.shape[0], -1, n_l, n_s)               # (c, 93, L, S)
        med = np.nanmedian(a, axis=1)                          # (c, L, S)
        bias = np.where(is_a[c0:c1, None, None], b24[None, :, None],
                        b25[None, :, None]).astype(np.float32)
        gam = np.where(is_a[c0:c1, None, None], g24[None, :, None],
                       g25[None, :, None]).astype(np.float32)
        cal = (med - bias)[:, None] + gam[:, None] * (a - med[:, None])
        raw_max = np.nanmax(a, axis=2)                         # (c, 93, S)
        cal_max = np.nanmax(cal, axis=2)
        ai_med_max[c0:c1] = np.nanmax(med, axis=1)
        for t in THRESHOLDS:
            th = thr[t][None, None, :]
            prob_raw[t][c0:c1] = np.nanmean(raw_max >= th, axis=1)
            prob_cal[t][c0:c1] = np.nanmean(cal_max >= th, axis=1)

    rows = []
    for t in THRESHOLDS:
        th = thr[t]
        ok_thr = np.isfinite(th)
        for s in range(n_s):
            if not ok_thr[s]:
                continue
            rows.append(pd.DataFrame(dict(
                cycle_time=cycles, station_id=station_ids[s], threshold=t,
                thr_m=th[s],
                obs_event=(obs_max[:, s] >= th[s]),
                obs_valid=np.isfinite(obs_max[:, s]),
                ai_prob_raw=prob_raw[t][:, s],
                ai_prob_cal=prob_cal[t][:, s],
                ai_det_event=(ai_med_max[:, s] >= th[s]),
                stofs_event=(stofs_max[:, s] >= th[s]),
                stofs_valid=np.isfinite(stofs_max[:, s]),
                ofs_event=(ofs_max[:, s] >= th[s]),
                ofs_valid=np.isfinite(ofs_max[:, s]),
                obs_max_m=obs_max[:, s],
            )))
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Observed episodes + warning lead
# =============================================================================
def hourly_obs(ds):
    """Stitch (cycle, lead 1..6) into one hourly obs series per station."""
    cycles = pd.to_datetime(ds.cycle_time.values)
    obs_wl = ds.obs_water_level.isel(lead_hour=slice(0, 6)).values  # (C,6,S)
    times, vals = [], []
    for li in range(6):
        times.append(cycles + pd.Timedelta(hours=li + 1))
        vals.append(obs_wl[:, li, :])
    t = np.concatenate([x.values for x in times])
    v = np.concatenate(vals, axis=0)
    order = np.argsort(t)
    return pd.DatetimeIndex(t[order]), v[order]


def find_episodes(t_index, series, thr):
    """Start times of exceedance episodes (gaps > EPISODE_GAP_H split)."""
    ex = np.where(np.isfinite(series) & (series >= thr))[0]
    if ex.size == 0:
        return []
    starts = [t_index[ex[0]]]
    for i in range(1, len(ex)):
        gap = (t_index[ex[i]] - t_index[ex[i - 1]]).total_seconds() / 3600
        if gap > EPISODE_GAP_H:
            starts.append(t_index[ex[i]])
    return starts


def warning_leads(events, t_index, hourly, station_ids, thr_map):
    """Per observed episode: hours of advance warning per system/cutoff."""
    ev = events.set_index(['station_id', 'threshold'])
    rows = []
    for (sid, t), g in events.groupby(['station_id', 'threshold']):
        s_idx = station_ids.index(sid)
        thr = thr_map[t][s_idx]
        if not np.isfinite(thr):
            continue
        g = g.sort_values('cycle_time')
        ct = pd.DatetimeIndex(g['cycle_time'])
        for t0 in find_episodes(t_index, hourly[:, s_idx], thr):
            w = np.asarray((ct < t0) & (ct >= t0 - pd.Timedelta(hours=72)))
            if not w.any():
                continue
            gw = g[w]
            row = dict(station_id=sid, threshold=t, episode_start=t0)
            for p in WARN_CUTOFFS:
                hit = gw[gw['ai_prob_cal'] >= p]
                row[f'ai_lead_h_p{int(p*100):02d}'] = (
                    (t0 - hit['cycle_time'].iloc[0]).total_seconds() / 3600
                    if len(hit) else np.nan)
            for name, col in [('stofs', 'stofs_event'), ('ofs', 'ofs_event'),
                              ('ai_det', 'ai_det_event')]:
                hit = gw[gw[col] & gw.get(f'{name}_valid',
                                          pd.Series(True, index=gw.index))]
                row[f'{name}_lead_h'] = (
                    (t0 - hit['cycle_time'].iloc[0]).total_seconds() / 3600
                    if len(hit) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Region driver
# =============================================================================
def run_region(verify_dir, cfg):
    out_pq = os.path.join(verify_dir, 'decision_events.parquet')
    if os.path.exists(out_pq):
        print(f"  [exists, skip] {verify_dir}")
        return
    params_path = os.path.join(verify_dir, 'calibration_v2_params.json')
    if not os.path.exists(params_path):
        print(f"  [skip] no calibration params yet: {verify_dir}")
        return
    fc = os.path.join(verify_dir, 'verify_arrays.nc')
    ds = xr.open_dataset(fc, chunks={'cycle_time': CHUNK})
    lead_bin_idx = _lead_bin_index(ds.sizes['lead_hour'])
    params, split_year = _params_arrays(params_path, lead_bin_idx)

    events = build_events(ds, params, split_year)

    # Merge the dressed probabilities where available.
    dp_path = os.path.join(verify_dir, 'dressed_probs.nc')
    if os.path.exists(dp_path):
        dp = xr.open_dataset(dp_path)
        frames = []
        for t in THRESHOLDS:
            df = dp[f'dressed_prob_{t}'].to_dataframe().reset_index()
            df['station_id'] = df['station'].astype(str)
            df['threshold'] = t
            df = df.rename(columns={f'dressed_prob_{t}': 'dressed_prob'})
            frames.append(df[['cycle_time', 'station_id', 'threshold',
                              'dressed_prob']])
        dp.close()
        events = events.merge(pd.concat(frames, ignore_index=True),
                              on=['cycle_time', 'station_id', 'threshold'],
                              how='left')

    t_index, hourly = hourly_obs(ds)
    station_ids = [str(s) for s in ds.station.values]
    thr_map = {t: ds[f'{t}_threshold'].values.astype(np.float32)
               for t in THRESHOLDS}
    wl = warning_leads(events, t_index, hourly, station_ids, thr_map)
    ds.close()

    events.to_parquet(out_pq, index=False)
    wl.to_csv(os.path.join(verify_dir, 'warning_lead.csv'), index=False)
    n_ev = {t: int(events[(events['threshold'] == t)
                          & events['obs_valid']]['obs_event'].sum())
            for t in THRESHOLDS}
    with open(os.path.join(verify_dir, 'decision_summary.json'), 'w') as f:
        json.dump(dict(n_rows=len(events), obs_events_by_threshold=n_ev,
                       n_episodes=len(wl)), f, indent=2)
    print(f"  events: {n_ev} | episodes: {len(wl)}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    spans = [d for d in glob.glob(os.path.join(cfg['BASE_DIR'], '*'))
             if os.path.isdir(d)]
    for span_dir in spans:
        regs = sorted(d for d in glob.glob(os.path.join(span_dir, 'region_*'))
                      if os.path.exists(os.path.join(d, 'verify_arrays.nc')))
        for rd in regs:
            print(f"Decision events -> {rd}")
            run_region(rd, cfg)
