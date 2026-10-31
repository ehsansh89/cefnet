"""
paper_fig6_tier3.py — interpretability Tier 3: does the model even look at
the storm? (attention + occlusion at the Helene cycle)

In plain English: two checks on the same forecast (Helene, issued
2024-09-25 12Z, Cedar Key). (1) Attention — the encoder's spatial-attention
maps say where in the 64x64 weather grid the network focuses around the
peak. (2) Occlusion — we erase the storm from one tile of the grid at a
time (wind to zero, pressure to the hour's background) and measure how
much the predicted Cedar Key peak drops; tiles that matter light up. If
the storm region lights up in both, the failure is NOT mislocated
attention — the model sees Helene and still cannot produce the amplitude
(which Tier 2 showed is a saturated response curve).

Reads:  ERA5 LR zarrs + v2 water levels, v2 region-5 checkpoints
Gives:  outputs/paper/v2/fig6_tier3_occlusion.npz
        docs/paper/figures/v4/fig6_tier3_attn_draft.png
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from o_style import setup, savefig, PAPER_FONT as F
from s_sample_batch_v2 import (DatasetCache, get_region_stations,
                               sample_batch, get_grid_coords)
from s_scalers import normalize_data
from train import load_train_config
from inference_v2 import load_model_from_checkpoint

STATIONS_CSV = r'data/stations_processed_v2/stations_processed.csv'
H5 = r'data/water_levels_processed_v2/processed_water_levels.h5'
SCALER = r'samples/scalers_v2/water_level_residuals.pkl'
OUT_NPZ = r'outputs/paper/v2/fig6_tier3_occlusion.npz'
OUT_PNG = r'docs/paper/figures/v4/fig6_tier3_attn_draft.png'
MM = 1 / 25.4

RNAME, SID = 'region_05', '8727520'
CYCLE = pd.Timestamp('2024-09-25 12:00')
PEAK = pd.Timestamp('2024-09-27 05:00')
LOCK_H = 3
LOOKBACK = 24
TILE = 8                      # occlusion tile size (px) -> 8x8 grid
SEEDS = (1, 2, 3)


class CBAMCapture:
    """Hooks the SpatialEncoder's CBAM spatial-attention modules and stores
    each block's sigmoid attention map per processed frame (adapted from
    o_attention.py for the v2 run)."""

    def __init__(self, model, layer_names=('layer1', 'layer2', 'layer3',
                                           'layer4')):
        self.layer_names = layer_names
        self.handles = []
        self.spatial = [[] for _ in layer_names]
        for i, ln in enumerate(layer_names):
            block = getattr(model.enc_low_res, ln)

            def make_hook(idx):
                def hook(module, inputs, output):
                    x = inputs[0]
                    avg_out = torch.mean(x, dim=1, keepdim=True)
                    max_out, _ = torch.max(x, dim=1, keepdim=True)
                    attn = torch.sigmoid(
                        module.conv(torch.cat([avg_out, max_out], dim=1)))
                    self.spatial[idx].append(attn.detach().cpu().numpy())
                return hook
            self.handles.append(block.sa.register_forward_hook(make_hook(i)))

    def close(self):
        for h in self.handles:
            h.remove()


def neutral_tile(raw, ty, tx):
    """Erases the storm inside one tile for every hour: wind to zero,
    pressure to that hour's domain median. Gets raw (T,3,64,64) + tile
    indices; gives the modified copy."""
    out = raw.astype(np.float32).copy()
    ys, xs = slice(ty * TILE, (ty + 1) * TILE), slice(tx * TILE,
                                                      (tx + 1) * TILE)
    out[:, 0, ys, xs] = 0.0
    out[:, 1, ys, xs] = 0.0
    med = np.nanmedian(out[:, 2], axis=(1, 2))
    out[:, 2, ys, xs] = med[:, None, None]
    return out


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    mpl.use('Agg')
    setup()
    device = torch.device('cpu')
    train_cfg = load_train_config('train_config_v2.json')
    with open('nn_config.json') as f:
        model_config = json.load(f)

    j = int((PEAK - CYCLE).total_seconds() // 3600) - 1
    w = slice(max(0, j - LOCK_H), min(72, j + LOCK_H + 1))
    sids, snames, lats, lons = get_region_stations(STATIONS_CSV, RNAME)
    s_idx = [str(s) for s in sids].index(SID)
    cache = DatasetCache(RNAME, H5)
    raw, _, _, _ = sample_batch([CYCLE], model_config, sids, lats, lons,
                                cache, scale=False)

    # ---------------- occlusion: 64 tiles + baseline ----------------
    n_t = 64 // TILE
    variants = [raw[0]] + [neutral_tile(raw[0], ty, tx)
                           for ty in range(n_t) for tx in range(n_t)]
    peaks = np.zeros((len(SEEDS), len(variants)))
    models = {}
    for si, seed in enumerate(SEEDS):
        ckpt = f'models/nn_24lb_72fh_v2/{RNAME}/seed_{seed}/best_phase1.pt'
        models[seed], _ = load_model_from_checkpoint(
            ckpt, model_config, len(sids), train_cfg, device)
        for lo in range(0, len(variants), 8):
            chunk = variants[lo:lo + 8]
            lr_w, st_t, _, _ = sample_batch(
                [CYCLE] * len(chunk), model_config, sids, lats, lons,
                cache, scale=True,
                lr_wind_override=np.stack(chunk, axis=0))
            with torch.no_grad():
                out = models[seed]({
                    'low_res': torch.from_numpy(lr_w.astype(np.float32)),
                    'stations': torch.from_numpy(st_t.astype(np.float32)),
                }).numpy()[..., 0]
            pm = normalize_data(out, 'descale', SCALER)
            peaks[si, lo:lo + len(chunk)] = np.nanmax(
                pm[:, w, s_idx], axis=1)
        print(f'seed {seed}: baseline peak {peaks[si, 0] * 100:.0f} cm')

    base = peaks[:, 0].mean()
    dpk = (base - peaks[:, 1:].mean(axis=0)).reshape(n_t, n_t)  # + = matters
    print(f'occlusion: baseline {base * 100:.0f} cm, max tile drop '
          f'{dpk.max() * 100:.0f} cm')

    # ---------------- attention: one forward, seed 1 ----------------
    cap = CBAMCapture(models[SEEDS[0]])
    lr_w, st_t, _, _ = sample_batch([CYCLE], model_config, sids, lats, lons,
                                    cache, scale=True,
                                    lr_wind_override=raw.copy())
    with torch.no_grad():
        models[SEEDS[0]]({
            'low_res': torch.from_numpy(lr_w.astype(np.float32)),
            'stations': torch.from_numpy(st_t.astype(np.float32))})
    cap.close()
    cache.close()

    # frames are (B*T); pick the frames around the observed peak hour
    frames = slice(LOOKBACK + j - 6, LOOKBACK + j + 1)
    attn = {}
    for li, ln in enumerate(cap.layer_names):
        a = np.concatenate(cap.spatial[li], axis=0)[:, 0]   # (T, h, w)
        am = a[frames].mean(axis=0)
        attn[ln] = np.kron(am, np.ones((64 // am.shape[0],
                                        64 // am.shape[1])))

    glat, glon = get_grid_coords(RNAME, (64, 64))
    if glat.ndim == 1:
        glon2, glat2 = np.meshgrid(glon, glat)
    else:
        glat2, glon2 = glat, glon
    np.savez(OUT_NPZ, dpk=dpk, base=base,
             attn_l3=attn['layer3'], attn_l4=attn['layer4'],
             psfc_peak=raw[0, LOOKBACK + j, 2].astype(np.float32),
             u_peak=raw[0, LOOKBACK + j, 0].astype(np.float32),
             v_peak=raw[0, LOOKBACK + j, 1].astype(np.float32),
             glat=glat2, glon=glon2)
    print('wrote', OUT_NPZ)

    # ---------------- draft figure ----------------
    extent = [glon2.min(), glon2.max(), glat2.min(), glat2.max()]
    ck_lat, ck_lon = lats[s_idx], lons[s_idx]
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.10,
                        wspace=0.25)

    ax = axes[0]
    im = ax.imshow(raw[0, LOOKBACK + j, 2].astype(np.float32),
                   origin='upper', extent=extent, cmap='viridis')
    step = 6
    ax.quiver(glon2[::step, ::step], glat2[::step, ::step],
              raw[0, LOOKBACK + j, 0, ::step, ::step].astype(np.float32),
              raw[0, LOOKBACK + j, 1, ::step, ::step].astype(np.float32),
              color='white', width=0.004, scale=450)
    ax.set_title(f'input at obs-peak hour · PSFC + wind',
                 fontsize=F['label'])
    plt.colorbar(im, ax=ax, fraction=0.04).set_label('hPa',
                                                     fontsize=F['small'])

    ax = axes[1]
    im = ax.imshow(attn['layer3'], origin='upper', extent=extent,
                   cmap='magma')
    ax.set_title('CBAM spatial attention (layer 3,\navg peak-7h..peak)',
                 fontsize=F['label'])
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[2]
    im = ax.imshow(dpk * 100, origin='upper', extent=extent, cmap='Reds')
    ax.set_title('occlusion: drop in predicted Cedar Key\npeak when tile '
                 'erased (cm)', fontsize=F['label'])
    plt.colorbar(im, ax=ax, fraction=0.04)

    for ax in axes:
        ax.plot(ck_lon, ck_lat, marker='*', ms=9, color='#EE6677',
                mec='black', mew=0.5)
        ax.tick_params(labelsize=F['tick'])
    with mpl.rc_context({'savefig.bbox': None}):
        savefig(fig, OUT_PNG, dpi=450)
    plt.close(fig)
    print('wrote', OUT_PNG)
