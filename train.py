"""
train.py — DeepSurge trainer over precomputed zarr samples.

Consumes the store written by s_generate_samples.py (scaled fp16 tensors) via
a plain torch Dataset + DataLoader, drops the old prefetcher/shim/scale_batch
machinery, and runs the nn model through a two-phase deterministic → NLL
training schedule driven by train_config.json.

Artifacts written under models/{RUN_NAME}/{region_name}/:
  - train_log.csv      per-step loss / lr
  - epoch_log.csv      per-epoch train + val metrics
  - checkpoint_{phase}_epoch{N}.pt   last epoch weights per phase
  - best_{phase}.pt    lowest-val-loss weights per phase
  - config.json        snapshot of train_config.json used for the run

Smoke mode (set SMOKE_BATCHES below to an int) runs that many training batches
and asserts that the average loss over the last N//3 batches is lower than over
the first N//3, so you can verify the whole pipeline end-to-end in a minute.
"""

import csv
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from torch.utils.data import DataLoader, Dataset

from nn import (
    DeepSurgeForecaster,
    SurgeDeterministicLoss,
    SurgeProbabilisticLoss,
)
from s_sample_batch import get_region_stations
from s_scalers import _load_scaler, normalize_data


# =============================================================================
# DATASET
# =============================================================================
class PrecomputedSurgeDataset(Dataset):
    """Reads one sample at a time from the zarr store written by s_generate_samples."""

    def __init__(self, zarr_path, indices=None, station_drop_rate=0.0, train=True):
        self.zarr_path = zarr_path
        self.root = zarr.open(zarr_path, mode='r')
        self.lr_wind = self.root['lr_wind']
        self.station = self.root['station']
        self.residual = self.root['residual']
        self.target_mask = self.root['target_mask']

        self.n_total = self.lr_wind.shape[0]
        self.indices = np.asarray(indices) if indices is not None else np.arange(self.n_total)
        self.num_stations = self.lr_wind.attrs.get('num_stations', self.station.shape[2])
        self.station_drop_rate = station_drop_rate
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        lr = np.asarray(self.lr_wind[i], dtype=np.float32)
        st = np.asarray(self.station[i], dtype=np.float32)
        res = np.asarray(self.residual[i], dtype=np.float32)
        tm = np.asarray(self.target_mask[i], dtype=np.float32)

        if self.train and self.station_drop_rate > 0.0:
            tide_avail = st[:, :, 2]
            obs_avail = st[:, :, 3]
            has_data = (tide_avail.max(axis=0) > 0.5) | (obs_avail.max(axis=0) > 0.5)
            active = np.where(has_data)[0]
            if len(active) > 1:
                n_drop = max(1, int(len(active) * self.station_drop_rate))
                n_drop = min(n_drop, len(active) - 1)
                drop_idx = np.random.choice(active, n_drop, replace=False)
                st[:, drop_idx, 0:4] = 0.0

        return {
            'low_res': torch.from_numpy(lr),
            'stations': torch.from_numpy(st),
            'residual': torch.from_numpy(res),
            'target_mask': torch.from_numpy(tm),
        }


def _select_indices(catalog_parquet, period):
    df = pd.read_parquet(catalog_parquet)
    df['dt'] = pd.to_datetime(df['dt'])
    mask = (df['dt'] >= pd.Timestamp(period[0])) & (df['dt'] <= pd.Timestamp(period[1]))
    return df.loc[mask, 'sample_index'].to_numpy()


def split_indices_by_period(catalog_parquet, train_period, val_period, test_period,
                            train_period_phase1=None, train_period_phase2=None):
    out = {
        'train': _select_indices(catalog_parquet, train_period),
        'val':   _select_indices(catalog_parquet, val_period),
        'test':  _select_indices(catalog_parquet, test_period),
    }
    if train_period_phase1 is not None:
        out['train_phase1'] = _select_indices(catalog_parquet, train_period_phase1)
    if train_period_phase2 is not None:
        out['train_phase2'] = _select_indices(catalog_parquet, train_period_phase2)
    return out


# =============================================================================
# CONFIG
# =============================================================================
def load_train_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    for key in ('TRAIN_PERIOD', 'VAL_PERIOD', 'TEST_PERIOD', 'VAR_TRAIN_PERIOD'):
        if key in cfg and isinstance(cfg[key], list):
            cfg[key] = tuple(cfg[key])
    return cfg


# =============================================================================
# LOGGING + CHECKPOINTS
# =============================================================================
class CSVLogger:
    """Append-only CSV logger. Creates the header on first write."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self._fh = open(path, 'a', newline='')
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, **row):
        self._writer.writerow({k: row.get(k, '') for k in self.fieldnames})
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def save_checkpoint(path, model, optimizer, scheduler, phase, epoch, global_step, val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'phase': phase,
        'epoch': epoch,
        'global_step': global_step,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


# =============================================================================
# LR SCHEDULE
# =============================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(min_lr_ratio, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(preds, targets, target_mask, residual_scaler_path):
    with torch.no_grad():
        pred_mean = preds[..., 0].cpu().numpy()
        tgt = targets.cpu().numpy()
        mask = target_mask.cpu().numpy()

    pred_phys = normalize_data(pred_mean, 'descale', residual_scaler_path)
    tgt_phys = normalize_data(tgt, 'descale', residual_scaler_path)

    mask_exp = np.broadcast_to(mask[:, np.newaxis, :], pred_phys.shape)
    valid = mask_exp > 0.5
    if not valid.any():
        return {k: float('nan') for k in ['rmse', 'mae', 'bias']}

    errs = (pred_phys - tgt_phys)[valid]
    return {
        'rmse': float(np.sqrt(np.mean(errs ** 2))),
        'mae':  float(np.mean(np.abs(errs))),
        'bias': float(np.mean(errs)),
    }


# =============================================================================
# VALIDATION
# =============================================================================
@torch.no_grad()
def run_validation(model, criterion, loader, device, residual_scaler_path, max_batches=None):
    model.eval()
    losses, all_preds, all_tgts, all_masks = [], [], [], []
    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        inputs = {
            'low_res': batch['low_res'].to(device, non_blocking=True),
            'stations': batch['stations'].to(device, non_blocking=True),
        }
        targets = batch['residual'].to(device, non_blocking=True)
        mask = batch['target_mask'].to(device, non_blocking=True)

        preds = model(inputs)
        loss = criterion(preds, targets, mask)
        losses.append(loss.item())
        all_preds.append(preds.cpu())
        all_tgts.append(targets.cpu())
        all_masks.append(mask.cpu())

    if not losses:
        return {'loss': float('nan'), 'rmse': float('nan'), 'mae': float('nan'), 'bias': float('nan')}

    metrics = compute_metrics(
        torch.cat(all_preds, dim=0), torch.cat(all_tgts, dim=0),
        torch.cat(all_masks, dim=0), residual_scaler_path,
    )
    metrics['loss'] = float(np.mean(losses))
    return metrics


# =============================================================================
# MAIN TRAIN LOOP
# =============================================================================
def train(cfg, smoke_batches=None):
    seed = cfg.get('SEED', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    region_name = f"region_{cfg['REGION_ID']:02d}"
    run_name = cfg.get('RUN_NAME', 'nn')
    seed_tag = f"seed_{seed}"
    run_dir = os.path.join('models', run_name, region_name, seed_tag)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"Run dir: {run_dir}")

    zarr_path = os.path.join(cfg['PRECOMPUTED_DIR'], region_name, 'samples.zarr')
    catalog_path = os.path.join(cfg['PRECOMPUTED_DIR'], region_name, 'catalog.parquet')
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Precomputed store not found: {zarr_path}. Run s_generate_samples.py first.")

    # Station list (needed for model construction; must match precompute).
    station_ids, _station_names, _st_lats, _st_lons = get_region_stations(cfg['STATIONS_CSV'], region_name)
    num_stations = len(station_ids)
    print(f"[{region_name}] {num_stations} stations")

    splits = split_indices_by_period(
        catalog_path, cfg['TRAIN_PERIOD'], cfg['VAL_PERIOD'], cfg['TEST_PERIOD'],
    )
    for k, v in splits.items():
        print(f"  {k}: {len(v):,} samples")
    if len(splits['train']) == 0:
        # Smoke mode fallback: if the limited precompute covers no train period,
        # use the full store as train so the loop still runs end-to-end.
        print("  [smoke-fallback] no train-period samples in precompute; using all samples as train.")
        root = zarr.open(zarr_path, mode='r')
        splits['train'] = np.arange(root['lr_wind'].shape[0])

    with open(cfg['MODEL_CONFIG'], 'r') as f:
        model_config = json.load(f)
    lookback = model_config['lookback_hours']
    forecast = model_config['forecast_hours']

    model = DeepSurgeForecaster(
        num_stations=num_stations,
        lookback_hours=lookback,
        forecast_hours=forecast,
        d_model=cfg['D_MODEL'],
        hidden_dim=cfg['HIDDEN_DIM'],
        token_grid=cfg['TOKEN_GRID'],
        num_fuse_tokens=cfg.get('NUM_FUSE_TOKENS', 2),
        dropout=cfg['DROPOUT'],
        head_type=cfg.get('HEAD_TYPE', 'cross_attn'),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    train_ds = PrecomputedSurgeDataset(
        zarr_path, indices=splits['train'], station_drop_rate=cfg.get('STATION_DROP_RATE', 0.0), train=True,
    )
    val_ds = PrecomputedSurgeDataset(
        zarr_path, indices=splits['val'], station_drop_rate=0.0, train=False,
    ) if len(splits['val']) > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True, drop_last=True,
        num_workers=cfg.get('NUM_WORKERS', 0), pin_memory=(device.type == 'cuda'),
        persistent_workers=bool(cfg.get('NUM_WORKERS', 0)),
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False, drop_last=False,
            num_workers=cfg.get('NUM_WORKERS', 0), pin_memory=(device.type == 'cuda'),
        )

    residual_scaler_path = cfg['RESIDUAL_SCALER']

    phases = []
    mode = cfg.get('TRAINING_MODE', 'deterministic_only')
    if mode in ('deterministic_only', 'two_phase'):
        phases.append(('phase1', cfg['PHASE1_EPOCHS'], SurgeDeterministicLoss, cfg['LEARNING_RATE']))
    if mode in ('probabilistic_only', 'two_phase'):
        phases.append(('phase2', cfg['PHASE2_EPOCHS'], SurgeProbabilisticLoss, cfg['PHASE2_LEARNING_RATE']))

    global_step = 0
    smoke_losses = []

    step_logger = CSVLogger(
        os.path.join(run_dir, 'train_log.csv'),
        fieldnames=['phase', 'epoch', 'batch', 'global_step', 'loss', 'lr', 'time'],
    )
    epoch_logger = CSVLogger(
        os.path.join(run_dir, 'epoch_log.csv'),
        fieldnames=['phase', 'epoch', 'global_step', 'train_loss', 'val_loss',
                    'val_rmse', 'val_mae', 'val_bias', 'epoch_time', 'is_best'],
    )

    try:
        for phase_name, num_epochs, LossClass, lr in phases:
            print(f"\n{'='*60}\nPHASE {phase_name}  lr={lr}  epochs={num_epochs}\n{'='*60}")

            phase_train_loader = train_loader
            two_phase_mode = mode == 'two_phase'
            if phase_name == 'phase2' and two_phase_mode:
                # Disjoint variance-training window (03_project_plan.md §3.4).
                var_period = cfg.get('VAR_TRAIN_PERIOD')
                if var_period is None:
                    print("  [WARN] VAR_TRAIN_PERIOD not set; reusing TRAIN_PERIOD for the "
                          "variance phase. This violates the disjoint-data requirement and "
                          "will let the variance head absorb residual mean error from the "
                          "same samples used to fit the mean.")
                else:
                    phase_indices = _select_indices(catalog_path, var_period)
                    if len(phase_indices) == 0:
                        raise ValueError(
                            f"VAR_TRAIN_PERIOD {var_period} selected zero samples"
                        )
                    print(f"  [phase2] variance window {var_period[0]} -> {var_period[1]} "
                          f"({len(phase_indices):,} samples)")
                    phase_train_ds = PrecomputedSurgeDataset(
                        zarr_path, indices=phase_indices,
                        station_drop_rate=cfg.get('STATION_DROP_RATE', 0.0), train=True,
                    )
                    phase_train_loader = DataLoader(
                        phase_train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True, drop_last=True,
                        num_workers=cfg.get('NUM_WORKERS', 0), pin_memory=(device.type == 'cuda'),
                        persistent_workers=bool(cfg.get('NUM_WORKERS', 0)),
                    )

                # Freeze trunk + proj_mean; train only proj_var.
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.output_head.proj_var.parameters():
                    p.requires_grad = True
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                n_trainable = sum(p.numel() for p in trainable_params)
                print(f"  [phase2] frozen trunk + proj_mean; trainable params = {n_trainable:,}")
            else:
                # probabilistic_only or deterministic_only: train all params end-to-end.
                if phase_name == 'phase2':
                    print("  [phase2] probabilistic_only mode - trunk NOT frozen, training end-to-end.")
                trainable_params = list(model.parameters())

            criterion = LossClass(
                scaler_path=residual_scaler_path,
                use_weighted_loss=cfg.get('USE_WEIGHTED_LOSS', False),
            ).to(device)
            optimizer = optim.AdamW(
                trainable_params, lr=lr, weight_decay=cfg.get('WEIGHT_DECAY', 1e-5),
            )

            steps_per_epoch = max(1, len(phase_train_loader))
            total_steps = steps_per_epoch * num_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, cfg.get('WARMUP_STEPS', 100), total_steps,
            )

            best_val_loss = float('inf')
            best_ckpt_path = os.path.join(run_dir, f'best_{phase_name}.pt')
            # Rolling buffer of (val_loss, epoch, path) for the K best epoch
            # checkpoints in this phase. When the buffer overflows we delete the
            # weakest file on disk. `best_{phase}.pt` is tracked separately.
            keep_top_k = int(cfg.get('KEEP_TOP_K_CHECKPOINTS', 3))
            top_k_ckpts = []

            for epoch in range(num_epochs):
                model.train()
                t0 = time.time()
                losses = []
                for b_idx, batch in enumerate(phase_train_loader):
                    inputs = {
                        'low_res': batch['low_res'].to(device, non_blocking=True),
                        'stations': batch['stations'].to(device, non_blocking=True),
                    }
                    targets = batch['residual'].to(device, non_blocking=True)
                    mask = batch['target_mask'].to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    preds = model(inputs)
                    loss = criterion(preds, targets, mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('GRAD_CLIP', 1.0))
                    optimizer.step()
                    scheduler.step()

                    losses.append(loss.item())
                    smoke_losses.append(loss.item())
                    global_step += 1

                    current_lr = scheduler.get_last_lr()[0]
                    step_logger.log(
                        phase=phase_name, epoch=epoch + 1, batch=b_idx + 1,
                        global_step=global_step, loss=loss.item(),
                        lr=current_lr, time=time.time() - t0,
                    )

                    if global_step % cfg.get('LOG_EVERY_N_STEPS', 5) == 0:
                        print(f"  [{phase_name} e{epoch+1} b{b_idx+1}/{steps_per_epoch}] "
                              f"step={global_step}  loss={loss.item():.4f}  lr={current_lr:.2e}")

                    if smoke_batches is not None and global_step >= smoke_batches:
                        break

                epoch_dur = time.time() - t0
                train_loss = float(np.mean(losses)) if losses else float('nan')
                print(f"[{phase_name} epoch {epoch+1}] avg_loss={train_loss:.4f}  time={epoch_dur:.1f}s")

                val_metrics = {'loss': float('nan'), 'rmse': float('nan'),
                               'mae': float('nan'), 'bias': float('nan')}
                if val_loader is not None:
                    val_metrics = run_validation(
                        model, criterion, val_loader, device, residual_scaler_path,
                        max_batches=cfg.get('VAL_MAX_BATCHES'),
                    )
                    print(f"  [val] loss={val_metrics['loss']:.4f}  rmse={val_metrics['rmse']:.4f}  "
                          f"mae={val_metrics['mae']:.4f}  bias={val_metrics['bias']:.4f}")

                is_best = (not math.isnan(val_metrics['loss'])
                           and val_metrics['loss'] < best_val_loss)
                if is_best:
                    best_val_loss = val_metrics['loss']

                if smoke_batches is None:
                    ckpt_path = os.path.join(
                        run_dir, f'checkpoint_{phase_name}_epoch{epoch+1}.pt',
                    )
                    save_checkpoint(
                        ckpt_path, model, optimizer, scheduler,
                        phase_name, epoch + 1, global_step, val_metrics['loss'],
                    )
                    rank_loss = (val_metrics['loss']
                                 if not math.isnan(val_metrics['loss'])
                                 else float('inf'))
                    top_k_ckpts.append((rank_loss, epoch + 1, ckpt_path))
                    top_k_ckpts.sort(key=lambda x: x[0])
                    while len(top_k_ckpts) > keep_top_k:
                        _, drop_epoch, drop_path = top_k_ckpts.pop()
                        try:
                            os.remove(drop_path)
                            print(f"  [prune] dropped epoch {drop_epoch} "
                                  f"checkpoint (kept top {keep_top_k})")
                        except OSError as e:
                            print(f"  [prune] failed to remove {drop_path}: {e}")
                    if is_best:
                        save_checkpoint(
                            best_ckpt_path, model, optimizer, scheduler,
                            phase_name, epoch + 1, global_step, val_metrics['loss'],
                        )
                        print(f"  [best] val_loss={best_val_loss:.4f} -> {best_ckpt_path}")

                epoch_logger.log(
                    phase=phase_name, epoch=epoch + 1, global_step=global_step,
                    train_loss=train_loss, val_loss=val_metrics['loss'],
                    val_rmse=val_metrics['rmse'], val_mae=val_metrics['mae'],
                    val_bias=val_metrics['bias'], epoch_time=epoch_dur,
                    is_best=int(is_best),
                )

                if smoke_batches is not None and global_step >= smoke_batches:
                    print(f"\n[smoke] reached {global_step} batches - stopping early.")
                    break
            if smoke_batches is not None and global_step >= smoke_batches:
                break
    finally:
        step_logger.close()
        epoch_logger.close()

    # Sentinel for the sequential drivers' skip-if-done logic. Only written
    # for full runs that completed all configured phases without raising.
    if smoke_batches is None:
        with open(os.path.join(run_dir, 'done.txt'), 'w') as f:
            f.write(pd.Timestamp.now(tz='UTC').isoformat() + '\n')

    if smoke_batches is not None and len(smoke_losses) >= 6:
        third = len(smoke_losses) // 3
        early = float(np.mean(smoke_losses[:third]))
        late = float(np.mean(smoke_losses[-third:]))
        print(f"\n[smoke] early-third avg loss = {early:.4f}")
        print(f"[smoke] late-third  avg loss = {late:.4f}")
        if late < early:
            print(f"[smoke] OK - loss dropped by {(early - late):.4f} ({100*(early-late)/max(early,1e-8):.1f}%)")
        else:
            print(f"[smoke] WARNING - loss did not drop (delta={late-early:+.4f}).")
        return {'early': early, 'late': late, 'n_batches': len(smoke_losses)}
    return {'n_batches': len(smoke_losses)}


# ==========================================
# CONFIGURATION (edit here; no CLI args)
# ==========================================
CONFIG_PATH = os.environ.get('X_TRAIN_CONFIG', 'train_config_v2.json')  # env hook for v2 config
SMOKE_BATCHES = None  # int to cap training steps for a smoke test, None for full run


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cfg = load_train_config(CONFIG_PATH)
    # Optional env overrides so one shared train_config.json can drive many
    # parallel single-(region, seed) jobs (e.g. per-seed A100 jobs on HPC)
    # without each job needing its own config file. Mirrors the REGIONS env
    # override used by s_generate_samples.py.
    if os.environ.get('REGION_ID'):
        cfg['REGION_ID'] = int(os.environ['REGION_ID'])
    if os.environ.get('SEED'):
        cfg['SEED'] = int(os.environ['SEED'])
    train(cfg, smoke_batches=SMOKE_BATCHES)
