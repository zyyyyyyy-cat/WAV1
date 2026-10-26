"""A-V1 losses on the original UAD pre-BN CLS embedding."""
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist


DIRECTIONS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))


class ModalityPrototypeBank(nn.Module):
    def __init__(self, num_classes, dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('prototypes', torch.zeros(3, num_classes, dim, device=device))
        self.register_buffer('valid', torch.zeros(3, num_classes, dtype=torch.bool, device=device))

    @torch.no_grad()
    def update(self, features, labels, distributed=False):
        """Aggregate every (modality, PID) once across ranks before EMA."""
        b = labels.numel()
        f = F.normalize(features.detach().float(), dim=1).reshape(3, b, -1)
        sums = torch.zeros_like(self.prototypes)
        counts = torch.zeros_like(self.valid, dtype=torch.float32)
        for m in range(3):
            sums[m].index_add_(0, labels, f[m])
            counts[m].index_add_(0, labels, torch.ones(b, device=labels.device))
        # The local K-group center is normalized before cross-rank aggregation.
        local = counts > 0
        sums[local] = F.normalize(sums[local] / counts[local].unsqueeze(1), dim=1)
        counts[local] = 1
        if distributed:
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        mask = counts > 0
        centers = F.normalize(sums[mask] / counts[mask].unsqueeze(1), dim=1)
        old = self.valid & mask
        new = mask & ~self.valid
        full = torch.zeros_like(self.prototypes)
        full[mask] = centers
        self.prototypes[old] = F.normalize(
            (1 - self.momentum) * self.prototypes[old] + self.momentum * full[old], dim=1)
        self.prototypes[new] = full[new]
        self.valid[mask] = True


def modality_prototype_loss(features, labels, bank, temperature=0.1):
    b = labels.numel()
    u = F.normalize(features.float(), dim=1).reshape(3, b, -1)
    zero = features.float().sum() * 0
    values, counts = [], {}
    for source, target in (*DIRECTIONS, (2, 2)):
        key = f'{source}->{target}'
        valid = bank.valid[target]
        query = valid[labels]
        counts[key] = int(query.sum().item())
        if query.any():
            class_ids = torch.where(valid)[0]
            logits = u[source][query] @ bank.prototypes[target, valid].float().T / temperature
            mapped = torch.searchsorted(class_ids, labels[query])
            values.append((F.cross_entropy(logits, mapped), key))
    cross = [v for v, k in values if k != '2->2']
    tt = [v for v, k in values if k == '2->2']
    result = (torch.stack(cross).mean() if cross else zero) + (0.1 * tt[0] if tt else zero)
    return result, counts


def candidate_mask(q_cam, g_cam, q_id, g_id):
    """Eligible image pairs: evaluator camera rule plus exact-image exclusion."""
    return (q_cam[:, None] != g_cam[None, :]) & (q_id[:, None] != g_id[None, :])


def _contrast_direction(q, g, q_pid, g_pid, q_cam, g_cam, q_id, g_id, temperature):
    logits = q.float() @ g.float().T / temperature
    valid = candidate_mask(q_cam, g_cam, q_id, g_id)
    positive = valid & (q_pid[:, None] == g_pid[None, :])
    negative = valid & (q_pid[:, None] != g_pid[None, :])
    active = positive.any(1) & negative.any(1)
    if not active.any():
        return q.sum() * 0 + g.sum() * 0, 0
    logits = logits[active]
    valid = valid[active]
    positive = positive[active]
    den = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
    uni = den - (logits * positive).sum(1) / positive.sum(1)
    set_loss = den - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    return (0.3 * uni + 0.7 * set_loss).mean(), int(active.sum().item())


def image_contrast_loss(features, labels, cameras, sample_ids, temperature=0.1):
    b = labels.numel()
    u = F.normalize(features.float(), dim=1).reshape(3, b, -1)
    ids = labels.repeat(3)
    cams = torch.cat(cameras)
    sample_ids = torch.cat(sample_ids)
    chunks = lambda v: [v[i*b:(i+1)*b] for i in range(3)]
    pid, cam, sid = chunks(ids), chunks(cams), chunks(sample_ids)
    zero = features.float().sum() * 0
    pair, mixed, counts = [], [], {}
    for source, target in DIRECTIONS:
        val, count = _contrast_direction(u[source], u[target], pid[source], pid[target],
                                         cam[source], cam[target], sid[source], sid[target], temperature)
        counts[f'pair_{source}->{target}'] = count
        if count:
            pair.append(val)
    gallery = u.reshape(3*b, -1)
    for source in range(3):
        val, count = _contrast_direction(u[source], gallery, pid[source], ids,
                                         cam[source], cams, sid[source], sample_ids, temperature)
        counts[f'mixed_{source}'] = count
        if count:
            mixed.append(val)
    lp = torch.stack(pair).mean() if pair else zero
    lm = torch.stack(mixed).mean() if mixed else zero
    return 0.75 * lp + 0.25 * lm, lp, lm, counts
