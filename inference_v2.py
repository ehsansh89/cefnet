"""
inference.py — DeepSurge seamless-window inference and visualization.

Given any [START_TIME, END_TIME] range, produce a seamless hourly forecast
archive for the configured region by tiling non-overlapping forecast windows
(stride = forecast_hours by default). Writes:

    outputs/inference/{run}/{region}/{start}_{end}/
        forecast.nc                — xarray archive: (cycle_time, lead_hour,
                                     station) → residual_mean, residual_var
        timeseries_panel.png       — multi-station stacked time-series
        scatter.png                — predicted vs observed hexbin density
        lead_bias.png              — error vs forecast lead hour
        video.mp4                  — sliding-window comparison animation
        meta.json                  — run summary: stations, stats, paths

The forecast archive is the single source of truth consumed by future
downstream scripts (`o_ofs_skill.py`, `o_stofs_skill.py`, `o_storms.py`)
so no one re-runs the model to regenerate a figure.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter


def _register_ffmpeg():
    """Matplotlib can't find ffmpeg on Windows conda when Library/bin isn't
    on PATH. If our env happens to carry one, point matplotlib at it."""
    if FFMpegWriter.isAvailable():
        return
    import sys
    cand = os.path.join(os.path.dirname(sys.executable),
                        'Library', 'bin', 'ffmpeg.exe')
    if os.path.exists(cand):
        mpl.rcParams['animation.ffmpeg_path'] = cand

from nn import DeepSurgeForecaster
from s_sample_batch_v2 import (   # v2: uses samples/scalers_v2 for input scaling
    DatasetCache, get_region_stations, sample_batch,
)
from s_scalers import normalize_data
from train import load_train_config
from o_style import setup, COLORS, FIG, savefig, title


# =============================================================================
# TIMESTAMP GENERATION
# =============================================================================
def generate_cycle_timestamps(start_time, end_time, stride_hours, forecast_hours):
    """Emit cycle-anchor timestamps spaced `stride_hours` apart.

    A cycle timestamp marks the end of lookback / start of forecast +1h
    (matching the sample_batch convention where the forecast spans
    `[t+1, t+forecast_hours]`). The last cycle is included only if its
    forecast window fits within [start_time, end_time].
    """
    stamps = []
    current = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    step = pd.Timedelta(hours=stride_hours)
    while current + pd.Timedelta(hours=forecast_hours) <= end:
        stamps.append(current)
        current = current + step
    return stamps


# =============================================================================
# MODEL + CHECKPOINT
# =============================================================================
def load_model_from_checkpoint(ckpt_path, model_config, num_stations,
                               train_cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DeepSurgeForecaster(
        num_stations=num_stations,
        lookback_hours=model_config['lookback_hours'],
        forecast_hours=model_config['forecast_hours'],
        d_model=train_cfg['D_MODEL'],
        hidden_dim=train_cfg['HIDDEN_DIM'],
        token_grid=train_cfg['TOKEN_GRID'],
        num_fuse_tokens=train_cfg.get('NUM_FUSE_TOKENS', 2),
        dropout=train_cfg['DROPOUT'],
        head_type=train_cfg.get('HEAD_TYPE', 'cross_attn'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


# =============================================================================
# FORECAST LOOP
# =============================================================================
@torch.no_grad()
def run_forecast(model, cycle_times, model_config, station_ids, st_lats, st_lons,
                 cache, device, residual_scaler_path, batch_size,
                 lr_wind_fetcher=None):
    """Run inference over all cycle anchors, return per-cycle (mean, var) arrays.

    `lr_wind_fetcher`, when given, is a callable taking a list of cycle
    timestamps and returning the corresponding (B, T_total, 3, 64, 64)
    raw (unscaled) LR-wind tensor. Used by the GFS-archive atmospheric
    source so the same `sample_batch` contract can serve both data paths.
    """
    forecast_hours = model_config['forecast_hours']
    means, vars_, valid = [], [], []

    n_batches = (len(cycle_times) + batch_size - 1) // batch_size
    for b_start in tqdm(range(0, len(cycle_times), batch_size),
                        total=n_batches, desc='inference'):
        batch_ts = cycle_times[b_start:b_start + batch_size]
        lr_override = lr_wind_fetcher(batch_ts) if lr_wind_fetcher else None
        lr_w, st_t, _res, tm = sample_batch(
            batch_ts, model_config, station_ids, st_lats, st_lons,
            cache, scale=True, lr_wind_override=lr_override,
        )
        inputs = {
            'low_res':  torch.from_numpy(lr_w.astype(np.float32)).to(device, non_blocking=True),
            'stations': torch.from_numpy(st_t.astype(np.float32)).to(device, non_blocking=True),
        }
        out = model(inputs).cpu().numpy()        # (B, F, S, 2)
        means.append(out[..., 0])
        vars_.append(out[..., 1])
        valid.append(tm)                          # (B, S)

        del inputs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    mean = np.concatenate(means, axis=0)          # (C, F, S)
    var  = np.concatenate(vars_, axis=0)
    valid = np.concatenate(valid, axis=0)         # (C, S)

    mean_m = normalize_data(mean, 'descale', residual_scaler_path).astype(np.float32)
    # Variance descale: var_m = var_scaled * std^2 — reuse the scaler std.
    import pickle
    with open(residual_scaler_path, 'rb') as f:
        sp = pickle.load(f)
    std = float(sp['std'])
    var_m = (var.astype(np.float32) * (std * std)).astype(np.float32)
    return mean_m, var_m, valid.astype(bool)


# =============================================================================
# OBSERVED RESIDUALS
# =============================================================================
def load_observed_residuals(h5_path, station_ids, start_time, end_time):
    """Hourly observed residual series for each station across the full window."""
    hourly = pd.date_range(start_time, end_time, freq='h')
    observed = np.full((len(hourly), len(station_ids)), np.nan, dtype=np.float32)
    with pd.HDFStore(h5_path, mode='r') as store:
        keys = set(store.keys())
        for s_idx, sid in enumerate(station_ids):
            key = f'/station_{sid}'
            if key not in keys:
                continue
            df = store.get(key)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep='first')]
            observed[:, s_idx] = df.reindex(hourly)['residual'].values
    return hourly, observed


# =============================================================================
# STITCHING
# =============================================================================
def stitch_to_hourly(cycle_times, mean_m, var_m, valid, hourly_index,
                     forecast_hours):
    """Build hourly continuous prediction time-series from non-overlapping cycles.

    If cycles overlap (stride < forecast_hours), the latest cycle wins —
    operators typically want the most recent model run.
    """
    n_h = len(hourly_index)
    n_s = mean_m.shape[2]
    pred = np.full((n_h, n_s), np.nan, dtype=np.float32)
    pred_var = np.full((n_h, n_s), np.nan, dtype=np.float32)

    time_to_idx = pd.Series(np.arange(n_h), index=hourly_index)
    for c_idx, ts in enumerate(cycle_times):
        lead_times = pd.date_range(ts + pd.Timedelta(hours=1),
                                   periods=forecast_hours, freq='h')
        in_range = lead_times.isin(hourly_index)
        if not in_range.any():
            continue
        idx = time_to_idx.reindex(lead_times[in_range]).values.astype(int)
        # Station mask applies per-cycle.
        for s in range(n_s):
            if not valid[c_idx, s]:
                continue
            pred[idx, s] = mean_m[c_idx, in_range, s]
            pred_var[idx, s] = var_m[c_idx, in_range, s]
    return pred, pred_var


# =============================================================================
# STATION SELECTION
# =============================================================================
def select_viz_stations(config_sel, station_ids, station_names, observed,
                        n_want):
    """Resolve VIZ_STATIONS config into a list of station indices.

    - 'auto': pick n_want stations by data completeness + diversity.
    - list of station_ids: use exactly those stations (in given order).
    """
    n_total = len(station_ids)
    station_ids_str = [str(s) for s in station_ids]

    if isinstance(config_sel, list) and config_sel:
        wanted = [str(s) for s in config_sel]
        idx = []
        for w in wanted:
            if w in station_ids_str:
                idx.append(station_ids_str.index(w))
            else:
                print(f"[warn] viz station {w} not in region - skipped")
        return idx or list(range(min(n_want, n_total)))

    # auto: rank by fraction of hours with valid observation in the window.
    coverage = (~np.isnan(observed)).mean(axis=0)        # (S,)
    # Tie-break by spread in observed values (favour stations with variability).
    spread = np.nanstd(observed, axis=0)
    spread = np.where(np.isfinite(spread), spread, 0.0)
    score = coverage + 0.3 * (spread / (spread.max() + 1e-9))
    order = np.argsort(score)[::-1]
    return order[:n_want].tolist()


# =============================================================================
# FORECAST ARCHIVE
# =============================================================================
def save_forecast_archive(out_path, cycle_times, station_ids, station_names,
                          st_lats, st_lons, mean_m, var_m, valid,
                          forecast_hours, model_config, cfg):
    """Persist per-cycle forecast as a single NetCDF file."""
    ds = xr.Dataset(
        data_vars=dict(
            residual_mean=(('cycle_time', 'lead_hour', 'station'), mean_m),
            residual_var =(('cycle_time', 'lead_hour', 'station'), var_m),
            valid        =(('cycle_time', 'station'), valid.astype(np.uint8)),
            station_name =(('station',), np.asarray(station_names, dtype=object)),
            station_lat  =(('station',), np.asarray(st_lats, dtype=np.float32)),
            station_lon  =(('station',), np.asarray(st_lons, dtype=np.float32)),
        ),
        coords=dict(
            cycle_time=np.array(cycle_times, dtype='datetime64[ns]'),
            lead_hour=np.arange(1, forecast_hours + 1, dtype=np.int16),
            station=np.asarray(station_ids, dtype=object),
        ),
        attrs=dict(
            title='DeepSurge per-cycle forecast archive',
            run_name=cfg.get('RUN_NAME', 'nn'),
            region_id=cfg['REGION_ID'],
            checkpoint=cfg['CHECKPOINT'],
            start_time=cfg['START_TIME'],
            end_time=cfg['END_TIME'],
            stride_hours=cfg['STRIDE_HOURS'],
            forecast_hours=forecast_hours,
            lookback_hours=model_config['lookback_hours'],
            atmospheric_source=cfg.get('ATMOSPHERIC_SOURCE', 'era5'),
            units_mean='meters (residual)',
            units_var='meters^2 (residual)',
            created_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        ),
    )
    comp = dict(zlib=True, complevel=4)
    encoding = {v: comp for v in ds.data_vars if ds[v].dtype.kind in ('f', 'u', 'i')}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path


# =============================================================================
# FIGURES
# =============================================================================
def plot_timeseries_panel(hourly, observed, pred, pred_var, station_ids,
                          station_names, viz_idx, out_path,
                          show_envelope=True):
    """Full-window multi-station stacked time-series plot (publication style)."""
    n = len(viz_idx)
    fig_h = 1.45 * n + 0.6
    fig, axes = plt.subplots(n, 1, figsize=(FIG['hero'][0], fig_h),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for i, s_idx in enumerate(viz_idx):
        ax = axes[i]
        obs = observed[:, s_idx]
        prd = pred[:, s_idx]
        std = np.sqrt(np.maximum(pred_var[:, s_idx], 0))

        # Uncertainty envelope — only draw when the checkpoint actually has a
        # calibrated variance head (two-phase NLL). A deterministic-only run
        # produces softplus variance that's non-zero but meaningless, so the
        # envelope would mislead.
        if show_envelope:
            valid_env = np.isfinite(prd) & np.isfinite(std)
            if valid_env.any() and np.nanmax(std) > 1e-3:
                ax.fill_between(
                    hourly, prd - 1.96 * std, prd + 1.96 * std,
                    where=valid_env,
                    color=COLORS['envelope'], alpha=0.15, linewidth=0,
                    label=r'$\pm 1.96\sigma$',
                )

        ax.plot(hourly, obs, color=COLORS['obs'], linewidth=0.9, alpha=0.9,
                label='observed')
        ax.plot(hourly, prd, color=COLORS['pred'], linewidth=1.0, alpha=0.95,
                label='model')

        mask = np.isfinite(obs) & np.isfinite(prd)
        mae = float(np.mean(np.abs(obs[mask] - prd[mask]))) if mask.any() else np.nan
        bias = float(np.mean(prd[mask] - obs[mask])) if mask.any() else np.nan
        lab = f"{station_names[s_idx][:30]}   ({station_ids[s_idx]})"
        stat = f"MAE={mae*100:.1f} cm   bias={bias*100:+.1f} cm"
        ax.text(0.010, 0.95, lab, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', va='top', ha='left',
                color='#1a1a1a',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.85))
        ax.text(0.990, 0.95, stat, transform=ax.transAxes,
                fontsize=7.5, va='top', ha='right', color='#1a1a1a',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#d0d0d0', linewidth=0.5, alpha=0.95))

        ax.axhline(0, color='#999999', linewidth=0.3)
        ax.set_ylabel('residual (m)')
        # Show y-range centered on data.
        obs_r = obs[np.isfinite(obs)]
        prd_r = prd[np.isfinite(prd)]
        if obs_r.size and prd_r.size:
            lo = min(np.nanpercentile(obs_r, 1), np.nanpercentile(prd_r, 1))
            hi = max(np.nanpercentile(obs_r, 99), np.nanpercentile(prd_r, 99))
            pad = 0.1 * (hi - lo) if (hi - lo) > 0 else 0.1
            ax.set_ylim(lo - pad, hi + pad)

    # Date formatting on the shared x-axis.
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    axes[-1].set_xlabel('time (UTC)')

    fig.suptitle(
        f"DeepSurge forecast — {hourly[0]:%Y-%m-%d} to {hourly[-1]:%Y-%m-%d}",
        fontsize=10.5, fontweight='bold', y=0.995,
    )
    # Single figure-level legend above all panels so it never collides with
    # per-panel stat boxes.
    handles = [
        plt.Line2D([0], [0], color=COLORS['obs'], linewidth=1.2, label='observed'),
        plt.Line2D([0], [0], color=COLORS['pred'], linewidth=1.2, label='model'),
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.995, 0.995),
               fontsize=7.5, ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    savefig(fig, out_path)
    plt.close(fig)


def plot_scatter(observed, pred, out_path):
    mask = np.isfinite(observed) & np.isfinite(pred)
    if not mask.any():
        return
    o = observed[mask]
    p = pred[mask]

    lo = float(np.nanpercentile(np.concatenate([o, p]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([o, p]), 99.5))
    pad = 0.05 * (hi - lo) if (hi - lo) > 0 else 0.1
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=FIG['square'])
    hb = ax.hexbin(o, p, gridsize=55, cmap='viridis', bins='log',
                   extent=(lo, hi, lo, hi), mincnt=1, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=COLORS['peak'], linewidth=0.8,
            linestyle='--', alpha=0.9, label='y=x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('observed residual (m)')
    ax.set_ylabel('predicted residual (m)')
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r'$\log_{10}$ count', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    ss_res = float(np.sum((o - p) ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(p - o)))
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    txt = (f"n = {len(o):,}\n"
           f"R² = {r2:.3f}\n"
           f"MAE = {mae*100:.2f} cm\n"
           f"RMSE = {rmse*100:.2f} cm")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='#cccccc', linewidth=0.5))
    title(ax, 'Predicted vs observed',
          subtitle='all stations · all hours · log density')
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def plot_lead_bias(mean_m, observed, hourly, cycle_times, station_ids,
                   valid, forecast_hours, out_path):
    """MAE and bias as a function of forecast lead hour, aggregated across cycles."""
    n_s = mean_m.shape[2]
    per_lead_mae = np.full(forecast_hours, np.nan)
    per_lead_bias = np.full(forecast_hours, np.nan)
    per_lead_n   = np.zeros(forecast_hours, dtype=int)

    time_to_idx = pd.Series(np.arange(len(hourly)), index=hourly)
    for lh in range(forecast_hours):
        errs = []
        for c_idx, ct in enumerate(cycle_times):
            t = ct + pd.Timedelta(hours=lh + 1)
            if t not in time_to_idx.index:
                continue
            h_idx = int(time_to_idx.loc[t])
            for s in range(n_s):
                if not valid[c_idx, s]:
                    continue
                o = observed[h_idx, s]
                p = mean_m[c_idx, lh, s]
                if np.isfinite(o) and np.isfinite(p):
                    errs.append(p - o)
        if errs:
            e = np.asarray(errs)
            per_lead_mae[lh] = np.mean(np.abs(e))
            per_lead_bias[lh] = np.mean(e)
            per_lead_n[lh] = len(e)

    fig, ax = plt.subplots(figsize=FIG['double'])
    leads = np.arange(1, forecast_hours + 1)
    ax.plot(leads, per_lead_mae * 100, color=COLORS['pred'], linewidth=1.5,
            marker='o', markersize=3.0, label='MAE')
    ax.plot(leads, per_lead_bias * 100, color=COLORS['baseline'],
            linewidth=1.1, linestyle='--', marker='^', markersize=3.0,
            label='bias')
    ax.axhline(0, color='#999999', linewidth=0.4)
    ax.set_xlabel('forecast lead (h)')
    ax.set_ylabel('error (cm)')
    ax.legend(loc='upper left', fontsize=7)
    n_valid = int(np.sum(per_lead_n > 0))
    title(ax, 'Error vs forecast lead',
          subtitle=f"{len(cycle_times)} cycles · aggregated over {n_valid} valid lead hours")
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


# =============================================================================
# PUBLICATION-GRADE PLOTS (Figs 3.5–3.8)
# =============================================================================
def plot_hovmoller(hourly, observed, pred, station_ids, station_names,
                   st_lats, out_path,
                   obs_clip_m=0.6, err_clip_m=0.3,
                   storm_landfalls=None):
    """Fig 3.5 — Hovmöller error stack: time × station.

    Two stacked heatmaps sharing the time axis. Stations sorted N→S by lat.
    Top: observed residual (m). Bottom: AI signed error (pred − obs) (m).
    `storm_landfalls` is an optional list of (datetime, label) tuples annotated
    as dashed vertical lines on the top panel.
    """
    n_st = pred.shape[1]
    if st_lats is not None and len(st_lats) == n_st:
        order = np.argsort(-np.asarray(st_lats, dtype=float))   # north on top
    else:
        order = np.arange(n_st)
    obs_sorted = observed[:, order].T            # (S, T)
    err_sorted = (pred - observed)[:, order].T   # (S, T)
    labels = [str(station_names[i])[:18] for i in order]

    t0, t1 = hourly[0], hourly[-1]
    extent = [mdates.date2num(t0), mdates.date2num(t1), n_st - 0.5, -0.5]

    fig_h = max(4.0, 0.18 * n_st + 1.6)
    fig, axes = plt.subplots(2, 1, figsize=(FIG['hero'][0], fig_h),
                             sharex=True)
    im0 = axes[0].imshow(obs_sorted, aspect='auto', cmap='RdBu_r',
                         vmin=-obs_clip_m, vmax=obs_clip_m, extent=extent,
                         interpolation='nearest')
    axes[0].set_yticks(np.arange(n_st))
    axes[0].set_yticklabels(labels, fontsize=6.5)
    title(axes[0], 'Observed residual (m)',
          subtitle='stations sorted N→S')
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.012)
    cb0.set_label('m', fontsize=7); cb0.ax.tick_params(labelsize=6.5)

    im1 = axes[1].imshow(err_sorted, aspect='auto', cmap='RdBu_r',
                         vmin=-err_clip_m, vmax=err_clip_m, extent=extent,
                         interpolation='nearest')
    axes[1].set_yticks(np.arange(n_st))
    axes[1].set_yticklabels(labels, fontsize=6.5)
    title(axes[1], 'AI signed error (pred − obs, m)',
          subtitle='blue = under, red = over')
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.012)
    cb1.set_label('m', fontsize=7); cb1.ax.tick_params(labelsize=6.5)

    if storm_landfalls:
        for ts, lab in storm_landfalls:
            x = mdates.date2num(pd.Timestamp(ts))
            for ax in axes:
                ax.axvline(x, color='#222222', linewidth=0.7, linestyle='--',
                           alpha=0.7)
            axes[0].text(x, -1.2, lab, fontsize=6, ha='center', va='bottom',
                         color='#222222')

    axes[1].xaxis_date()
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        axes[1].xaxis.get_major_locator()))
    axes[1].set_xlabel('time (UTC)')
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def plot_forecast_curtain(cycle_times, mean_m, observed, hourly,
                          forecast_hours, viz_idx, station_ids, station_names,
                          out_path, err_clip_cm=30.0):
    """Fig 3.6 — Forecast curtain (cycle × lead) for the headline gauge.

    Panel a: cycle_time × lead_hour grid; cell color = signed error (cm).
    Panel b: observed residual (m) at the same gauge as a 1-D context strip.
    Headline gauge = first viz_idx entry.
    """
    if not viz_idx:
        return
    s_idx = int(viz_idx[0])
    n_c = len(cycle_times)
    F = forecast_hours
    err_grid = np.full((n_c, F), np.nan, dtype=np.float32)
    time_to_idx = pd.Series(np.arange(len(hourly)), index=hourly)
    for c_idx, ct in enumerate(cycle_times):
        for lh in range(F):
            valid = pd.Timestamp(ct) + pd.Timedelta(hours=lh + 1)
            if valid not in time_to_idx.index:
                continue
            h_idx = int(time_to_idx.loc[valid])
            o = observed[h_idx, s_idx]
            p = mean_m[c_idx, lh, s_idx]
            if np.isfinite(o) and np.isfinite(p):
                err_grid[c_idx, lh] = (p - o) * 100.0   # cm

    fig = plt.figure(figsize=(FIG['hero'][0], 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ct_num = mdates.date2num([pd.Timestamp(c) for c in cycle_times])
    extent = [0.5, F + 0.5, ct_num[-1], ct_num[0]]
    im = ax0.imshow(err_grid, aspect='auto', cmap='RdBu_r',
                    vmin=-err_clip_cm, vmax=err_clip_cm,
                    extent=extent, interpolation='nearest')
    ax0.set_xlabel('forecast lead (h)')
    ax0.set_ylabel('cycle time (UTC)')
    ax0.yaxis_date()
    ax0.yaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax0.yaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax0.yaxis.get_major_locator()))
    cb = fig.colorbar(im, ax=ax0, fraction=0.04, pad=0.02)
    cb.set_label('signed error pred − obs (cm)', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    sid = station_ids[s_idx] if s_idx < len(station_ids) else f"idx={s_idx}"
    nm = station_names[s_idx] if s_idx < len(station_names) else ""
    title(ax0, f'Forecast curtain — {nm} ({sid})',
          subtitle='blue = under-pred, red = over-pred')

    ax1 = fig.add_subplot(gs[1, 0])
    obs = observed[:, s_idx]
    ax1.plot(hourly, obs, color=COLORS['obs'], linewidth=0.8, alpha=0.9)
    ax1.set_ylabel('observed (m)')
    ax1.axhline(0, color='#999999', linewidth=0.4)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax1.xaxis.get_major_locator()))
    ax1.set_xlabel('time (UTC)')
    ax1.grid(True, linestyle='-', linewidth=0.4, color='#dddddd')
    title(ax1, 'Observed residual context')
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def plot_taylor_diagrams(mean_m, observed, hourly, cycle_times, valid,
                         station_names, out_path, leads=(6, 24, 48)):
    """Fig 3.7 — Taylor diagrams at three lead times.

    For each lead in `leads` (hours), one polar Taylor plot:
        angular = arccos(correlation)
        radial  = sigma_pred / sigma_obs
    Marker color = MAE (cm); marker size ∝ √n_pairs.
    """
    n_st = mean_m.shape[2]
    fig = plt.figure(figsize=(FIG['hero'][0], 3.6))
    gs = fig.add_gridspec(1, len(leads), wspace=0.45)
    time_to_idx = pd.Series(np.arange(len(hourly)), index=hourly)

    all_mae = []
    panels = []
    for j, lh in enumerate(leads):
        per_st = []
        for s_idx in range(n_st):
            obs_v, pred_v = [], []
            for c_idx, ct in enumerate(cycle_times):
                if not valid[c_idx, s_idx]:
                    continue
                t = pd.Timestamp(ct) + pd.Timedelta(hours=lh)
                if t not in time_to_idx.index:
                    continue
                h_idx = int(time_to_idx.loc[t])
                o = observed[h_idx, s_idx]
                p = mean_m[c_idx, lh - 1, s_idx]      # lead h → index h-1
                if np.isfinite(o) and np.isfinite(p):
                    obs_v.append(o); pred_v.append(p)
            if len(obs_v) < 30:
                continue
            o = np.asarray(obs_v); p = np.asarray(pred_v)
            sigma_o = float(np.std(o, ddof=0))
            sigma_p = float(np.std(p, ddof=0))
            if sigma_o < 1e-6:
                continue
            corr = float(np.corrcoef(o, p)[0, 1])
            corr = max(-0.999, min(0.999, corr if np.isfinite(corr) else 0.0))
            mae = float(np.mean(np.abs(p - o))) * 100.0
            per_st.append((s_idx, corr, sigma_p / sigma_o, mae, len(o)))
            all_mae.append(mae)
        panels.append(per_st)

    vmax = max(20.0, float(np.percentile(all_mae, 95))) if all_mae else 20.0

    for j, lh in enumerate(leads):
        ax = fig.add_subplot(gs[0, j], projection='polar')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_thetamin(0); ax.set_thetamax(90)
        per_st = panels[j]
        if per_st:
            theta = np.arccos([r[1] for r in per_st])
            r = np.array([r[2] for r in per_st])
            mae = np.array([r[3] for r in per_st])
            sz = np.array([np.sqrt(r[4]) for r in per_st])
            sz = 18.0 + 25.0 * (sz / max(sz.max(), 1.0))
            sc = ax.scatter(theta, r, c=mae, s=sz, cmap='viridis',
                            vmin=0, vmax=vmax,
                            edgecolors='white', linewidths=0.3, alpha=0.95)
        # reference point at (corr=1, σ=1)
        ax.scatter([0], [1.0], marker='*', s=80, color=COLORS['peak'], zorder=5,
                   edgecolors='white', linewidths=0.5)
        # corr ticks
        corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
        ax.set_xticks([np.arccos(c) for c in corr_ticks])
        ax.set_xticklabels([f'{c:.2g}' for c in corr_ticks], fontsize=6.5)
        ax.set_rmax(max(1.6, float(np.nanmax(r)) if per_st else 1.6))
        ax.set_rticks([0.5, 1.0, 1.5])
        ax.set_rlabel_position(135)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.set_title(f'lead {lh} h', fontsize=9, pad=6)
        ax.grid(True, linestyle='-', linewidth=0.4, color='#dddddd')
    if all_mae:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(0, vmax), cmap='viridis')
        sm.set_array([])
        cb = fig.colorbar(sm, ax=fig.axes, orientation='horizontal',
                          fraction=0.03, pad=0.10, aspect=40, shrink=0.7)
        cb.set_label('MAE (cm)', fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle('Taylor diagrams · per-station skill at fixed lead times',
                 fontsize=10.5, fontweight='bold', y=1.05)
    savefig(fig, out_path)
    plt.close(fig)


def plot_continuity_strip(hourly, observed, pred, pred_var, viz_idx,
                          station_names, station_ids, out_path,
                          show_envelope=False, storm_landfalls=None):
    """Fig 3.8 — Operational-continuity strip: 3 stacked thin panels.

    Panel a: observed residual (m).
    Panel b: AI median residual (m), optional p05/p95 envelope.
    Panel c: error (m); zero line.
    Headline gauge = first viz_idx entry.
    """
    if not viz_idx:
        return
    s_idx = int(viz_idx[0])
    obs = observed[:, s_idx]
    prd = pred[:, s_idx]
    err = prd - obs
    std = np.sqrt(np.maximum(pred_var[:, s_idx], 0.0))

    fig, axes = plt.subplots(3, 1, figsize=(FIG['hero'][0], 4.5),
                             sharex=True, gridspec_kw={'hspace': 0.18})
    ax = axes[0]
    ax.plot(hourly, obs, color=COLORS['obs'], linewidth=0.9)
    ax.set_ylabel('obs (m)')
    ax.axhline(0, color='#999999', linewidth=0.4)
    title(ax, f'{station_names[s_idx][:32]} ({station_ids[s_idx]})',
          subtitle='observed residual')

    ax = axes[1]
    if show_envelope and np.nanmax(std) > 1e-3:
        ax.fill_between(hourly, prd - 1.96 * std, prd + 1.96 * std,
                        color=COLORS['envelope'], alpha=0.15, linewidth=0,
                        label=r'$\pm 1.96\sigma$')
    ax.plot(hourly, prd, color=COLORS['pred'], linewidth=0.9, label='AI median')
    ax.set_ylabel('AI (m)')
    ax.axhline(0, color='#999999', linewidth=0.4)
    if show_envelope and np.nanmax(std) > 1e-3:
        ax.legend(loc='upper right', fontsize=7)

    ax = axes[2]
    ax.plot(hourly, err, color=COLORS['peak'], linewidth=0.7, alpha=0.85)
    ax.fill_between(hourly, 0, err, where=np.isfinite(err),
                    color=COLORS['peak'], alpha=0.10)
    ax.set_ylabel('error (m)')
    ax.axhline(0, color='#999999', linewidth=0.5)
    ax.set_xlabel('time (UTC)')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))

    if storm_landfalls:
        for ts, lab in storm_landfalls:
            x = pd.Timestamp(ts)
            for a in axes:
                a.axvline(x, color='#222222', linewidth=0.5, linestyle='--',
                          alpha=0.6)
            axes[0].text(x, axes[0].get_ylim()[1], lab, fontsize=6,
                         ha='center', va='bottom', color='#222222')

    fig.suptitle('Operational continuity', fontsize=10.5, fontweight='bold',
                 y=1.0)
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


# =============================================================================
# VIDEO
# =============================================================================
def _fmt_dt(ts):
    return pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M')


def _build_map_axis(ax, region_boundary_path, st_lats, st_lons, viz_idx,
                    station_names):
    """Render a static station-location base map. Returns the viz-stations
    scatter artist whose face colors are updated per frame."""
    import cartopy.feature as cfeature
    import numpy as _np

    st_lats = _np.asarray(st_lats, dtype=float)
    st_lons = _np.asarray(st_lons, dtype=float)

    lat_min, lat_max = float(st_lats.min()), float(st_lats.max())
    lon_min, lon_max = float(st_lons.min()), float(st_lons.max())
    pad_lat = max(0.4, 0.10 * (lat_max - lat_min))
    pad_lon = max(0.4, 0.10 * (lon_max - lon_min))

    if region_boundary_path and os.path.exists(region_boundary_path):
        try:
            import geopandas as gpd
            bnd = gpd.read_file(region_boundary_path)
            if bnd.crs and bnd.crs.to_epsg() != 4326:
                bnd = bnd.to_crs(epsg=4326)
            rx_min, ry_min, rx_max, ry_max = bnd.total_bounds
            lon_min = min(lon_min, rx_min)
            lon_max = max(lon_max, rx_max)
            lat_min = min(lat_min, ry_min)
            lat_max = max(lat_max, ry_max)
            pad_lat = max(pad_lat, 0.05 * (lat_max - lat_min))
            pad_lon = max(pad_lon, 0.05 * (lon_max - lon_min))
            ax.add_geometries(
                bnd.geometry, crs=ax.projection,
                facecolor='#f4f0e6', edgecolor='#a99f80', linewidth=0.7,
                alpha=0.35, zorder=0,
            )
        except Exception as e:
            print(f"[warn] region boundary overlay failed: {e}")

    ax.set_extent([lon_min - pad_lon, lon_max + pad_lon,
                   lat_min - pad_lat, lat_max + pad_lat],
                  crs=ax.projection)

    ax.add_feature(cfeature.LAND.with_scale('10m'),
                   facecolor='#f2efe9', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),
                   facecolor='#e8f1f7', edgecolor='none', zorder=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                   edgecolor='#8a8a8a', linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale('10m'),
                   edgecolor='#c4c4c4', linewidth=0.3, zorder=2)

    ax.scatter(st_lons, st_lats, s=11, color='#7a7a7a', edgecolor='white',
               linewidth=0.4, zorder=4, transform=ax.projection, alpha=0.75)

    viz_lats = st_lats[viz_idx]
    viz_lons = st_lons[viz_idx]
    viz_pts = ax.scatter(
        viz_lons, viz_lats, s=60, facecolor='#cccccc', edgecolor='#222222',
        linewidth=0.8, zorder=5, transform=ax.projection,
    )

    for s_idx, lat, lon in zip(viz_idx, viz_lats, viz_lons):
        name = str(station_names[s_idx])[:16]
        ax.annotate(
            name, xy=(lon, lat), xytext=(4, 3),
            textcoords='offset points', fontsize=6.5,
            color='#1a1a1a', zorder=6,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.75),
        )

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color('#333333')
    return viz_pts


def make_video(hourly, observed, pred, station_ids, station_names, viz_idx,
               st_lats, st_lons, region_boundary_path,
               out_path, fps=10, dpi=120, window_days=10, frame_stride=6,
               error_limit_cm=30.0):
    """Sliding-window animation: map (left) + 2-col time-series grid (right).

    Station dots on the map are colour-coded by the current-hour signed
    error (model - observed, m). Grey = missing observation or prediction.
    """
    try:
        import cartopy.crs as ccrs
    except Exception as e:
        raise RuntimeError('cartopy is required for the video map') from e

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm

    n_times = len(hourly)
    viz_idx = list(viz_idx)
    n_st = len(viz_idx)
    ncols = 2
    nrows = int(np.ceil(n_st / ncols))

    fig = plt.figure(figsize=(18.0, max(9.0, 0.95 * nrows + 2.0)))
    outer = fig.add_gridspec(
        2, 2,
        height_ratios=[0.07, 1.0],
        width_ratios=[1.0, 1.5],
        left=0.035, right=0.985, top=0.935, bottom=0.055,
        hspace=0.12, wspace=0.10,
    )

    ax_nav = fig.add_subplot(outer[0, :])
    ax_nav.set_xlim(hourly[0], hourly[-1])
    ax_nav.set_yticks([])
    for s in ('top', 'left', 'right'):
        ax_nav.spines[s].set_visible(False)
    ax_nav.spines['bottom'].set_linewidth(0.5)
    ax_nav.spines['bottom'].set_color('#999999')
    ax_nav.grid(False)
    ax_nav.axhspan(0.35, 0.65, facecolor='#eeeeee', edgecolor='none',
                   transform=ax_nav.get_xaxis_transform())
    ax_nav.text(0.003, 0.5, 'timeline', transform=ax_nav.transAxes,
                va='center', ha='left', fontsize=7.5, fontweight='bold',
                color='#555555')
    nav_rect = patches.Rectangle(
        (hourly[0], 0.30), pd.Timedelta(days=2 * window_days), 0.40,
        transform=ax_nav.get_xaxis_transform(),
        facecolor=COLORS['pred'], alpha=0.28, edgecolor=COLORS['pred'],
        linewidth=1.0,
    )
    ax_nav.add_patch(nav_rect)
    ax_nav.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax_nav.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_nav.xaxis.get_major_locator()))
    ax_nav.tick_params(axis='x', labelsize=7, pad=1, length=3)

    ax_map = fig.add_subplot(outer[1, 0], projection=ccrs.PlateCarree())
    viz_pts = _build_map_axis(
        ax_map, region_boundary_path, st_lats, st_lons, viz_idx, station_names,
    )
    elim = error_limit_cm / 100.0
    err_norm = TwoSlopeNorm(vmin=-elim, vcenter=0.0, vmax=elim)
    err_cmap = plt.get_cmap('RdBu_r')
    sm = ScalarMappable(norm=err_norm, cmap=err_cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, orientation='horizontal',
                        fraction=0.045, pad=0.02, aspect=30, shrink=0.85)
    cbar.set_label('model minus observed residual (m)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    inner = outer[1, 1].subgridspec(nrows, ncols, hspace=0.55, wspace=0.10)

    axes, vlines, text_stats = [], [], []
    for i, s_idx in enumerate(viz_idx):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(inner[r, c])
        axes.append(ax)
        obs = observed[:, s_idx]
        prd = pred[:, s_idx]
        ax.plot(hourly, obs, color=COLORS['obs'], linewidth=0.8,
                alpha=0.9, label='observed')
        ax.plot(hourly, prd, color=COLORS['pred'], linewidth=1.0,
                alpha=0.95, label='model')
        vl = ax.axvline(hourly[0], color=COLORS['peak'], linewidth=0.9,
                        linestyle='--', alpha=0.7)
        vlines.append(vl)

        obs_r = obs[np.isfinite(obs)]
        prd_r = prd[np.isfinite(prd)]
        if obs_r.size and prd_r.size:
            lo = min(np.nanpercentile(obs_r, 1), np.nanpercentile(prd_r, 1))
            hi = max(np.nanpercentile(obs_r, 99), np.nanpercentile(prd_r, 99))
        else:
            lo, hi = -0.5, 0.5
        pad = 0.15 * (hi - lo) if (hi - lo) > 0 else 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0, color='#999999', linewidth=0.3)

        lab = f"{station_names[s_idx][:24]}  ({station_ids[s_idx]})"
        ax.text(0.012, 0.94, lab, transform=ax.transAxes,
                fontsize=7.5, fontweight='bold', va='top', ha='left',
                color='#1a1a1a',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                          edgecolor='none', alpha=0.85))
        stat_t = ax.text(0.988, 0.94, '',
                         transform=ax.transAxes, fontsize=7,
                         va='top', ha='right', color='#1a1a1a',
                         bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                   edgecolor='#d0d0d0', linewidth=0.4,
                                   alpha=0.95))
        text_stats.append(stat_t)

        if c == 0:
            ax.set_ylabel('residual (m)', fontsize=7.5)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(labelsize=7)

    sup = fig.suptitle('', fontsize=13, fontweight='bold', y=0.985)
    handles = [
        plt.Line2D([0], [0], color=COLORS['obs'], linewidth=1.2, label='observed'),
        plt.Line2D([0], [0], color=COLORS['pred'], linewidth=1.3, label='model'),
    ]
    fig.legend(handles=handles, loc='upper right',
               bbox_to_anchor=(0.985, 0.985), fontsize=9, ncol=2,
               frameon=False)

    writer = FFMpegWriter(fps=fps, bitrate=3800)
    frames = list(range(0, n_times, frame_stride))
    print(f"Generating video: {out_path}  frames={len(frames)}  fps={fps}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    viz_obs = observed[:, viz_idx]
    viz_prd = pred[:, viz_idx]
    viz_err = viz_prd - viz_obs
    nan_color = np.array([0.80, 0.80, 0.80, 1.0])

    with writer.saving(fig, out_path, dpi=dpi):
        for t_idx in tqdm(frames, desc='video frames'):
            ct = hourly[t_idx]
            t_lo = ct - pd.Timedelta(days=window_days)
            t_hi = ct + pd.Timedelta(days=window_days)
            nav_rect.set_x(t_lo)
            nav_rect.set_width(t_hi - t_lo)

            err_row = viz_err[t_idx]
            fc = np.zeros((len(viz_idx), 4))
            for j, e in enumerate(err_row):
                if np.isfinite(e):
                    fc[j] = err_cmap(err_norm(float(e)))
                else:
                    fc[j] = nan_color
            viz_pts.set_facecolor(fc)

            for i, s_idx in enumerate(viz_idx):
                ax = axes[i]
                ax.set_xlim(t_lo, t_hi)
                vlines[i].set_xdata([ct])

                win = (hourly >= t_lo) & (hourly <= t_hi)
                o = observed[win, s_idx]
                p = pred[win, s_idx]
                ok = np.isfinite(o) & np.isfinite(p)
                if ok.sum() >= 2:
                    mae = float(np.mean(np.abs(o[ok] - p[ok])))
                    bias = float(np.mean(p[ok] - o[ok]))
                    text_stats[i].set_text(
                        f"MAE={mae*100:.1f} cm  bias={bias*100:+.1f} cm"
                    )
                else:
                    text_stats[i].set_text('MAE=N/A')

            sup.set_text(f"DeepSurge -- {_fmt_dt(ct)} UTC")
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {out_path}")



# =============================================================================
# CONFIG
# =============================================================================
CONFIG_PATH = os.environ.get('X_INFER_CONFIG', 'inference_config_v2.json')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    setup()
    _register_ffmpeg()

    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    # env overrides: drive many per-region/seed runs off one config
    for _k in ('START_TIME', 'END_TIME'):
        if os.environ.get(_k):
            cfg[_k] = os.environ[_k]
    if os.environ.get('REGION_ID'):
        cfg['REGION_ID'] = int(os.environ['REGION_ID'])
        _seed = int(os.environ.get('SEED', 1))
        cfg['CHECKPOINT'] = (f"models/{cfg['RUN_NAME']}/region_{cfg['REGION_ID']:02d}/"
                             f"seed_{_seed}/best_phase1.pt")
    train_cfg = load_train_config(cfg['TRAIN_CONFIG'])
    with open(cfg['MODEL_CONFIG'], 'r') as f:
        model_config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    region_id = cfg['REGION_ID']
    region_name = f"region_{region_id:02d}"
    run_name = cfg.get('RUN_NAME', train_cfg.get('RUN_NAME', 'nn'))

    start_tag = pd.Timestamp(cfg['START_TIME']).strftime('%Y%m%d')
    end_tag   = pd.Timestamp(cfg['END_TIME']).strftime('%Y%m%d')
    # Tag non-ERA5 runs so AI/ERA5 and AI/GFS-archive sit side-by-side without
    # overwriting; legacy ERA5-only outputs keep their existing path layout.
    _atm_for_path = cfg.get('ATMOSPHERIC_SOURCE', 'era5')
    date_tag = (f"{start_tag}_{end_tag}" if _atm_for_path == 'era5'
                else f"{start_tag}_{end_tag}_{_atm_for_path}")
    seed_tag = cfg.get('SEED', None)
    if seed_tag is not None:
        out_dir = os.path.join(cfg['OUTPUT_DIR'], run_name, region_name,
                               date_tag, f"seed_{int(seed_tag)}")
    else:
        out_dir = os.path.join(cfg['OUTPUT_DIR'], run_name, region_name, date_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Station list for the region ──────────────────────────────────────
    station_ids, station_names, st_lats, st_lons = get_region_stations(
        cfg['STATIONS_CSV'], region_name,
    )
    num_stations = len(station_ids)
    print(f"Stations: {num_stations}")

    # ── Cycle timestamps ─────────────────────────────────────────────────
    forecast_hours = model_config['forecast_hours']
    stride = cfg.get('STRIDE_HOURS', forecast_hours)
    cycle_times = generate_cycle_timestamps(
        cfg['START_TIME'], cfg['END_TIME'], stride, forecast_hours,
    )
    if not cycle_times:
        raise RuntimeError('No cycle timestamps fit inside the requested window.')
    print(f"Cycles: {len(cycle_times)}  stride={stride}h  forecast={forecast_hours}h")
    print(f"Range:  {cycle_times[0]}  ->  {cycle_times[-1] + pd.Timedelta(hours=forecast_hours)}")

    # ── Model ────────────────────────────────────────────────────────────
    model, _ckpt = load_model_from_checkpoint(
        cfg['CHECKPOINT'], model_config, num_stations, train_cfg, device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['CHECKPOINT']}  ({n_params:,} params)")

    # ── Atmospheric source (era5 = reanalysis oracle, gfs_archive = §5) ──
    atm_source = cfg.get('ATMOSPHERIC_SOURCE', 'era5')
    lr_wind_fetcher = None
    if atm_source == 'gfs_archive':
        from s_gfs_archive import GFSArchiveCache, fetch_lr_wind_gfs
        gfs_cache = GFSArchiveCache(
            region_name=region_name,
            gfs_base_dir=cfg.get('GFS_BASE_DIR', 'data/atmospheric_forecasts/gfs'),
        )
        _lookback = model_config['lookback_hours']
        _forecast = model_config['forecast_hours']
        def lr_wind_fetcher(batch_ts):
            return np.stack([
                fetch_lr_wind_gfs(gfs_cache, ts, _lookback, _forecast)
                for ts in batch_ts
            ], axis=0)
        print(f"Atmospheric source: gfs_archive  "
              f"({cfg.get('GFS_BASE_DIR', 'data/atmospheric_forecasts/gfs')}/{region_name}/)")
    elif atm_source == 'era5':
        print("Atmospheric source: era5 (reanalysis oracle - skill ceiling)")
    else:
        raise ValueError(f"Unknown ATMOSPHERIC_SOURCE: {atm_source!r}")

    # ── Inference ────────────────────────────────────────────────────────
    cache = DatasetCache(region_name, cfg['WATER_LEVELS_H5'])
    try:
        t0 = time.time()
        mean_m, var_m, valid = run_forecast(
            model, cycle_times, model_config, station_ids, st_lats, st_lons,
            cache, device, cfg['RESIDUAL_SCALER'], cfg.get('BATCH_SIZE', 4),
            lr_wind_fetcher=lr_wind_fetcher,
        )
        print(f"Inference done in {time.time() - t0:.1f}s")
    finally:
        cache.close()

    # ── Archive (single source of truth for downstream) ──────────────────
    archive_path = os.path.join(out_dir, 'forecast.nc')
    save_forecast_archive(
        archive_path, cycle_times, station_ids, station_names, st_lats, st_lons,
        mean_m, var_m, valid, forecast_hours, model_config, cfg,
    )
    print(f"Archive: {archive_path}")

    # ── Load observed residuals + stitch to continuous hourly ────────────
    hourly, observed = load_observed_residuals(
        cfg['WATER_LEVELS_H5'], station_ids, cfg['START_TIME'], cfg['END_TIME'],
    )
    pred, pred_var = stitch_to_hourly(
        cycle_times, mean_m, var_m, valid, hourly, forecast_hours,
    )

    # ── Select viz stations ──────────────────────────────────────────────
    viz_idx = select_viz_stations(
        cfg.get('VIZ_STATIONS', 'auto'), station_ids, station_names, observed,
        cfg.get('N_VIZ_STATIONS', 6),
    )
    print(f"Viz stations: " + ', '.join(
        f"{station_ids[i]} ({station_names[i][:18]})" for i in viz_idx
    ))

    # ── Plots ────────────────────────────────────────────────────────────
    has_variance = train_cfg.get('TRAINING_MODE', 'deterministic_only') != 'deterministic_only'
    if not has_variance:
        print('[info] checkpoint is deterministic-only - suppressing '
              'uncertainty envelope in plots (variance head uncalibrated).')

    ts_path = os.path.join(out_dir, 'timeseries_panel.png')
    plot_timeseries_panel(hourly, observed, pred, pred_var,
                          station_ids, station_names, viz_idx, ts_path,
                          show_envelope=has_variance)

    sc_path = os.path.join(out_dir, 'scatter.png')
    plot_scatter(observed, pred, sc_path)

    lb_path = os.path.join(out_dir, 'lead_bias.png')
    plot_lead_bias(mean_m, observed, hourly, cycle_times, station_ids,
                   valid, forecast_hours, lb_path)

    # ── Publication-grade additions (Figs 3.5–3.8) ───────────────────────
    storm_landfalls = cfg.get('STORM_LANDFALLS')          # optional list
    if isinstance(storm_landfalls, list):
        storm_landfalls = [(s.get('time'), s.get('label', '')) for s in storm_landfalls]

    hov_path = os.path.join(out_dir, 'fig_3_5_hovmoller.png')
    try:
        plot_hovmoller(hourly, observed, pred, station_ids, station_names,
                       st_lats, hov_path, storm_landfalls=storm_landfalls)
    except Exception as e:
        print(f"[warn] Hovmoeller failed: {e}")

    cur_path = os.path.join(out_dir, 'fig_3_6_forecast_curtain.png')
    try:
        plot_forecast_curtain(cycle_times, mean_m, observed, hourly,
                              forecast_hours, viz_idx, station_ids, station_names,
                              cur_path)
    except Exception as e:
        print(f"[warn] forecast curtain failed: {e}")

    tay_path = os.path.join(out_dir, 'fig_3_7_taylor_diagrams.png')
    try:
        plot_taylor_diagrams(mean_m, observed, hourly, cycle_times, valid,
                             station_names, tay_path,
                             leads=tuple(cfg.get('TAYLOR_LEADS', (6, 24, 48))))
    except Exception as e:
        print(f"[warn] Taylor diagrams failed: {e}")

    cs_path = os.path.join(out_dir, 'fig_3_8_continuity_strip.png')
    try:
        plot_continuity_strip(hourly, observed, pred, pred_var, viz_idx,
                              station_names, station_ids, cs_path,
                              show_envelope=has_variance,
                              storm_landfalls=storm_landfalls)
    except Exception as e:
        print(f"[warn] continuity strip failed: {e}")

    # ── Summary stats (archive-derived, computed from stitched hourly) ───
    mask_any = np.isfinite(observed) & np.isfinite(pred)
    errs = pred[mask_any] - observed[mask_any]
    meta = dict(
        region_name=region_name,
        run_name=run_name,
        checkpoint=cfg['CHECKPOINT'],
        atmospheric_source=cfg.get('ATMOSPHERIC_SOURCE', 'era5'),
        start_time=cfg['START_TIME'],
        end_time=cfg['END_TIME'],
        stride_hours=stride,
        forecast_hours=forecast_hours,
        n_cycles=len(cycle_times),
        n_stations=num_stations,
        viz_station_ids=[station_ids[i] for i in viz_idx],
        mae_cm=float(np.mean(np.abs(errs)) * 100) if errs.size else None,
        rmse_cm=float(np.sqrt(np.mean(errs ** 2)) * 100) if errs.size else None,
        bias_cm=float(np.mean(errs) * 100) if errs.size else None,
        files=dict(
            archive=os.path.basename(archive_path),
            timeseries=os.path.basename(ts_path),
            scatter=os.path.basename(sc_path),
            lead_bias=os.path.basename(lb_path),
        ),
        generated_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    )

    # ── Video ────────────────────────────────────────────────────────────
    if cfg.get('EMIT_VIDEO', True):
        video_path = os.path.join(out_dir, 'video.mp4')
        try:
            region_boundary_path = os.path.join(
                'GIS', 'regions', region_name,
                f'{region_name}_lr_boundary.shp',
            )
            make_video(
                hourly, observed, pred, station_ids, station_names, viz_idx,
                st_lats, st_lons, region_boundary_path,
                video_path,
                fps=cfg.get('VIDEO_FPS', 10),
                dpi=cfg.get('VIDEO_DPI', 120),
                window_days=cfg.get('VIDEO_WINDOW_DAYS', 10),
                frame_stride=cfg.get('VIDEO_FRAME_STRIDE', 6),
                error_limit_cm=cfg.get('VIDEO_ERROR_LIMIT_CM', 30.0),
            )
            meta['files']['video'] = os.path.basename(video_path)
        except Exception as e:
            print(f"[warn] video generation failed: {e}")

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print('\n' + '=' * 56)
    print(f"{meta['n_cycles']} cycles over {meta['start_time']} -> {meta['end_time']}")
    if meta['mae_cm'] is not None:
        print(f"Overall stitched metrics:  "
              f"MAE={meta['mae_cm']:.2f} cm  "
              f"RMSE={meta['rmse_cm']:.2f} cm  "
              f"bias={meta['bias_cm']:+.2f} cm")
    print('=' * 56)
    print(f"Meta: {os.path.join(out_dir, 'meta.json')}")
