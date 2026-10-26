"""Configuration defaults and experiment seeding for MINER training."""

from __future__ import annotations

import random

import numpy as np
import torch
from yacs.config import CfgNode as CN

# --- ImageNet normalization (used across transforms and embedding scripts) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- TVRID constants ---
TVRID_NUM_CAMERAS = 4        # flat, upward, downward, upsideDown
TVRID_DEFAULT_NUM_CLASSES = 62  # number of identities in TVRID training set

# --- EVA02-L architecture constants ---
EVA02_EMBED_DIM = 1024       # backbone output dimension (feat_bio)

# --- ColorJitter defaults (standard values for ReID augmentation) ---
JITTER_BRIGHTNESS = 0.2
JITTER_CONTRAST = 0.2
JITTER_SATURATION = 0.2
JITTER_HUE = 0.1


def get_default_cfg() -> CN:
    """Return a CfgNode with all default keys for MINER RGB training."""
    cfg = CN()

    cfg.DATA = CN()
    cfg.DATA.DATASET = "tvrid_rgb"
    cfg.DATA.ROOT = "./data"
    cfg.DATA.IDS_PER_BATCH = 24
    cfg.DATA.REPS_PER_ID = 4
    cfg.DATA.IMG_SIZE = 224
    cfg.DATA.EVAL_FRAMES = 5
    cfg.DATA.NUM_WORKERS = 4
    cfg.DATA.EMBEDDINGS_PATH = ""
    cfg.DATA.VAL_RATIO = 0.0          # 0 = use all IDs for training

    cfg.MODEL = CN()
    cfg.MODEL.PRETRAINED_PATH = ""

    cfg.LOSS = CN()
    cfg.LOSS.CLIP_TEMPERATURE = 0.07
    cfg.LOSS.CLIP_LEARN_TEMPERATURE = True
    cfg.LOSS.CLIP_TEMPERATURE_LR = 0.0  # 0 = use base_lr * TEMP_LR_SCALE

    cfg.SOLVER = CN()
    cfg.SOLVER.OPTIMIZER = "SGD"
    cfg.SOLVER.LR = 1e-4
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WARMUP_EPOCHS = 1
    cfg.SOLVER.MAX_EPOCHS = 2
    cfg.SOLVER.EVAL_STEPS = 100
    cfg.SOLVER.GRAD_CLIP = 10.0
    cfg.SOLVER.SEED = 1234
    cfg.SOLVER.MINING_TEMPERATURE = 2.0
    cfg.SOLVER.WARMUP_FACTOR = 0.01
    cfg.SOLVER.MIN_LR = 1e-8
    cfg.SOLVER.TEMP_LR_SCALE = 0.01    # temperature LR = base_lr * this

    cfg.AUG = CN()
    cfg.AUG.HFLIP = 0.5
    cfg.AUG.VFLIP = 0.3
    cfg.AUG.RANDOM_CROP = 0.5
    cfg.AUG.COLOR_JITTER = 0.5
    cfg.AUG.RANDOM_ERASING = 0.25

    cfg.OUTPUT_DIR = "./logs"

    return cfg


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
