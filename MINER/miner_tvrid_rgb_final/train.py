"""MINER training script for TVRID RGB track.

Trains EVA02-Large with the MINER contrastive loss (CLIPReIDLoss,
bipartite/cross_only, Equations 9-11) using centroid-based hard-negative
PID mining (Algorithm 1).

Usage:
    python release/train.py --config release/configs/rgb.yml
    python release/train.py --config release/configs/rgb.yml SOLVER.MAX_EPOCHS 4
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import time

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import get_default_cfg, set_seed
from model import build_model
from dataset import TVRID
from frame_dataset import TVRIDFrameDataset
from sampler import MINERBatchSampler
from transforms import build_transforms
from losses import CLIPReIDLoss
from utils import setup_logger, AverageMeter, load_checkpoint

# Optional: wandb
try:
    import wandb as _wandb_module
except ImportError:
    _wandb_module = None


def setup_experiment(cfg, config_path: str):
    """Set up model, data, optimizer, and scheduler for training."""
    set_seed(cfg.SOLVER.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve relative paths against the config file's directory
    config_dir = osp.dirname(osp.abspath(config_path))
    release_dir = osp.dirname(config_dir)  # release/ is parent of configs/

    def _resolve(path: str) -> str:
        if not path or osp.isabs(path) or osp.exists(path):
            return path
        return osp.join(release_dir, path)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("TVRID.train", output_dir, if_train=True)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Device:      {device}")

    # wandb (optional)
    wandb_run = None
    try:
        if _wandb_module is not None:
            with open(config_path) as f:
                wandb_config = yaml.safe_load(f)
            wandb_run = _wandb_module.init(
                project="TVRID",
                name=f"miner-rgb-bs{cfg.DATA.IDS_PER_BATCH}",
                config=wandb_config,
                reinit="finish_previous",
            )
            logger.info(f"wandb run: {wandb_run.name}")
    except Exception as e:
        logger.warning(f"wandb init failed: {e}")

    # Transforms
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    transform_train, transform_test = build_transforms(cfg_dict)

    # Dataset + MINER sampler
    tvrid_dataset = TVRID(
        root=cfg.DATA.ROOT,
        val_ratio=cfg.DATA.VAL_RATIO,
        seed=cfg.SOLVER.SEED,
        eval_frames_per_passage=cfg.DATA.EVAL_FRAMES,
    )

    # Dataset (flat, one frame per index)
    frame_dataset = TVRIDFrameDataset(tvrid_dataset, transform=transform_train)

    # MINER batch sampler (Algorithm 1)
    batch_sampler = MINERBatchSampler(
        frame_dataset=frame_dataset,
        num_ids=cfg.DATA.IDS_PER_BATCH,
        nq=cfg.DATA.REPS_PER_ID,
        ng=cfg.DATA.REPS_PER_ID,
        embeddings_path=_resolve(cfg.DATA.EMBEDDINGS_PATH) or None,
        mining_temperature=cfg.SOLVER.MINING_TEMPERATURE,
        seed=cfg.SOLVER.SEED,
    )

    data_loader = DataLoader(
        frame_dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        persistent_workers=cfg.DATA.NUM_WORKERS > 0,
        prefetch_factor=2 if cfg.DATA.NUM_WORKERS > 0 else None,
        pin_memory=True,
    )

    # Model
    model = build_model(
        num_classes=tvrid_dataset.num_train_pids,
        num_cameras=tvrid_dataset.num_camera,
    )

    pretrained_path = _resolve(cfg.MODEL.PRETRAINED_PATH)
    if pretrained_path and osp.isfile(pretrained_path):
        load_checkpoint(model, pretrained_path, logger=logger)
    else:
        logger.info("No pretrained checkpoint -- using timm init only")

    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    model.to(device)

    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing enabled")

    # MINER Loss
    clip_loss = CLIPReIDLoss(
        temperature=cfg.LOSS.CLIP_TEMPERATURE,
        learn_temperature=cfg.LOSS.CLIP_LEARN_TEMPERATURE,
    ).to(device)
    logger.info(f"Loss: CLIPReIDLoss (bipartite/cross_only), "
                f"temp={cfg.LOSS.CLIP_TEMPERATURE}, learn={cfg.LOSS.CLIP_LEARN_TEMPERATURE}")

    # Optimizer
    base_lr = cfg.SOLVER.LR
    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

    clip_loss_params = list(clip_loss.parameters())
    if clip_loss_params:
        temp_lr = cfg.LOSS.CLIP_TEMPERATURE_LR or base_lr * cfg.SOLVER.TEMP_LR_SCALE
        param_groups.append({"params": clip_loss_params, "lr": temp_lr, "weight_decay": 0.0})
        logger.info(f"CLIPReIDLoss temperature LR: {temp_lr}")

    if cfg.SOLVER.OPTIMIZER.lower() == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=base_lr,
            momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER.lower() == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.SOLVER.OPTIMIZER}")
    logger.info(f"Optimizer: {cfg.SOLVER.OPTIMIZER}, base LR: {base_lr}")

    # LR Scheduler: linear warmup + cosine annealing
    steps_per_epoch = len(data_loader)
    total_steps = cfg.SOLVER.MAX_EPOCHS * steps_per_epoch
    warmup_steps = cfg.SOLVER.WARMUP_EPOCHS * steps_per_epoch
    logger.info(f"Scheduler: {steps_per_epoch} steps/epoch, {total_steps} total, {warmup_steps} warmup")

    if warmup_steps > 0:
        scheduler = SequentialLR(optimizer, schedulers=[
            LinearLR(optimizer, start_factor=cfg.SOLVER.WARMUP_FACTOR, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=cfg.SOLVER.MIN_LR),
        ], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=cfg.SOLVER.MIN_LR)

    return {
        "model": model, "data_loader": data_loader, "clip_loss": clip_loss,
        "optimizer": optimizer, "scheduler": scheduler,
        "batch_sampler": batch_sampler,
        "transform_test": transform_test, "device": device,
        "output_dir": output_dir, "logger": logger, "wandb_run": wandb_run,
    }


def train(cfg, config_path: str) -> None:
    """Main training loop."""
    ctx = setup_experiment(cfg, config_path)
    model, data_loader, clip_loss = ctx["model"], ctx["data_loader"], ctx["clip_loss"]
    optimizer, scheduler = ctx["optimizer"], ctx["scheduler"]
    device, output_dir = ctx["device"], ctx["output_dir"]
    logger, wandb_run = ctx["logger"], ctx["wandb_run"]

    batch_sampler = ctx["batch_sampler"]

    scaler = torch.amp.GradScaler()
    loss_meter = AverageMeter()
    best_loss = float("inf")
    global_step = 0

    logger.info("=" * 60)
    logger.info("Starting MINER training")
    logger.info(f"  Epochs:              {cfg.SOLVER.MAX_EPOCHS}")
    logger.info(f"  Batch IDs:           {cfg.DATA.IDS_PER_BATCH}")
    logger.info(f"  Reps per ID:         {cfg.DATA.REPS_PER_ID}")
    logger.info(f"  Mining temperature:  {cfg.SOLVER.MINING_TEMPERATURE}")
    logger.info("=" * 60)

    start_time = time.time()

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        model.train()
        loss_meter.reset()

        batch_sampler.set_epoch(epoch)

        epoch_start = time.time()

        for step, batch in enumerate(data_loader):
            images, labels, _camids = batch
            images = images.to(device)
            labels = labels.to(device)
            nq = batch_sampler.nq_per_batch

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                combined_feat = model(images)

            combined_feat = combined_feat.float()
            out = clip_loss(combined_feat[:nq], combined_feat[nq:], labels[:nq], labels[nq:])
            total_loss = out["loss"]

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)

            # Only step scheduler when optimizer actually stepped (skipped on inf/nan grads)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            global_step += 1

            loss_meter.update(total_loss.item())

            if global_step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.SOLVER.MAX_EPOCHS}] Step [{global_step}] "
                    f"Loss: {total_loss.item():.4f} (avg: {loss_meter.avg:.4f}) "
                    f"LR: {lr_now:.6f} Temp: {out['temperature'].item():.4f}"
                )
                if wandb_run is not None:
                    _wandb_module.log({
                        "train/loss": total_loss.item(),
                        "train/loss_avg": loss_meter.avg,
                        "train/lr": lr_now,
                        "train/temperature": out["temperature"].item(),
                    }, step=global_step)

        logger.info(f"Epoch {epoch + 1} complete in {time.time() - epoch_start:.1f}s, "
                    f"avg loss: {loss_meter.avg:.4f}")

        torch.save(model.state_dict(), osp.join(output_dir, f"epoch_{epoch + 1}.pth"))

        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            best_path = osp.join(output_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model (loss={best_loss:.4f}) saved: {best_path}")

    torch.save(model.state_dict(), osp.join(output_dir, "final.pth"))
    logger.info(f"Training complete in {(time.time() - start_time) / 60:.1f} min. "
                f"Best loss: {best_loss:.4f}")

    if wandb_run is not None:
        _wandb_module.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MINER training for TVRID RGB track")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("opts", nargs="*", default=[], help="Override config keys")
    args = parser.parse_args()

    cfg = get_default_cfg()
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    train(cfg, args.config)
