"""Deterministic training-split reference extraction and resumable state helpers."""
import hashlib
import io
import inspect
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F


def load_trusted_artifact(path):
    """Load our own training artifacts on both old and new PyTorch releases."""
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model):
    stream = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def manifest_hash(loaders):
    digest = hashlib.sha256()
    for loader in loaders:
        for entry in loader.dataset.dataset[loader.dataset.modality]:
            digest.update(json.dumps(entry, ensure_ascii=False).encode())
            digest.update(b'\n')
    return digest.hexdigest()


@torch.no_grad()
def extract_centers(model, loaders, n_class, dim, device):
    """Rank zero calls the unwrapped model on every training image once."""
    state = rng_state()
    was_training = model.training
    model.eval()
    sums = torch.zeros(3, n_class, dim, device=device)
    counts = torch.zeros(3, n_class, device=device)
    try:
        with torch.inference_mode():
            for m, loader in enumerate(loaders):
                for image, labels, *_ in loader:
                    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                    feat = F.normalize(model(image.to(device), mode=m+1).float(), dim=1)
                    sums[m].index_add_(0, labels, feat)
                    counts[m].index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        valid = counts > 0
        global_centers = F.normalize(sums.sum(0), dim=1)
        sums[valid] = F.normalize(sums[valid] / counts[valid].unsqueeze(1), dim=1)
        return sums, valid, counts, global_centers
    finally:
        model.train(was_training)
        restore_rng(state)


def hard_matrix(centers, valid, counts, global_centers):
    global_valid = valid.any(0)
    hg = global_centers @ global_centers.T
    tir = torch.zeros_like(hg)
    n = torch.zeros_like(hg)
    # Include both orders for all TIR-involving modality pairs.
    for a in range(3):
        for b in range(3):
            if a != 2 and b != 2:
                continue
            sim = centers[a] @ centers[b].T
            mask = valid[a, :, None] & valid[b, None, :]
            tir += sim * mask
            n += mask.float()
    ht = torch.where(n > 0, tir / n.clamp_min(1), hg)
    result = 0.5 * (hg + ht)
    result = 0.5 * (result + result.T)
    result.fill_diagonal_(0)
    if not torch.isfinite(result).all() or not global_valid.all():
        raise RuntimeError('Invalid fixed hardness matrix')
    return result


def broadcast_tensor(tensor, active):
    if active:
        dist.broadcast(tensor, 0)
    return tensor
