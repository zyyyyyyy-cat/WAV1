"""Training utilities: metrics tracker, logging, checkpoint I/O."""

import logging
import os
import sys

import torch


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "train_log.txt" if if_train else "test_log.txt"
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, logger=None) -> None:
    """Load a .pth checkpoint into the model, handling common mismatches.

    Handles:
      - DataParallel 'module.' prefix stripping
      - camera_embed dimension mismatches (pad or slice)
      - Classification head shape mismatches (skipped silently)
    """
    log = logger.info if logger else print

    log(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    load_state = {}
    skipped = []

    for k, v in clean_state.items():
        if k not in model_state:
            continue

        if model_state[k].shape != v.shape:
            if k in ("camera_embed", "camera_embedding"):
                target_n = model_state[k].shape[0]
                src_n = v.shape[0]
                if src_n < target_n:
                    pad = torch.zeros(target_n - src_n, *v.shape[1:], dtype=v.dtype)
                    load_state[k] = torch.cat([v, pad], dim=0)
                else:
                    load_state[k] = v[:target_n]
                log(f"  {k}: resized {src_n} -> {target_n}")
                continue

            skipped.append(f"{k}: {list(v.shape)} vs {list(model_state[k].shape)}")
            continue

        load_state[k] = v

    model.load_state_dict(load_state, strict=False)
    if skipped:
        log(f"  Skipped {len(skipped)} mismatched keys: {skipped[:5]}")
    log(f"  Loaded {len(load_state)}/{len(clean_state)} keys")
