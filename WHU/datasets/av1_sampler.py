"""Rearrange only the PID groups actually consumed by the original PKM sampler."""
from collections import Counter, defaultdict
import hashlib
import random
import numpy as np
from torch.utils.data import Sampler


def _rng(seed, *parts):
    token = ':'.join(map(str, (seed, *parts))).encode()
    return random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], 'big'))


class AV1Sampler(Sampler):
    def __init__(self, original, data, modalities, k, batch_size, seed):
        self.original = original
        self.data = data
        self.modalities = list(modalities)
        self.k = k
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 1
        self.hardness = None
        self.last_stats = {}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_hardness(self, matrix):
        self.hardness = np.asarray(matrix, dtype=np.float64)

    def __len__(self):
        return len(self.original)

    def __iter__(self):
        raw = list(iter(self.original))
        usable = len(raw) // self.batch_size * self.batch_size
        raw = raw[:usable]
        groups = []
        for start in range(0, usable, self.k):
            rows = raw[start:start+self.k]
            pid = self.data[self.modalities[0]][rows[0][0]][1]
            assert all(self.data[m][row[j]][1] == pid
                       for row in rows for j, m in enumerate(self.modalities))
            groups.append([pid, [[row[j] for row in rows] for j in range(3)]])
        before = self._ledger(groups)
        slots = defaultdict(list)
        for slot, (pid, _) in enumerate(groups):
            slots[pid].append(slot)
        camera_before, camera_after = [], []
        for pid, positions in slots.items():
            for m, name in enumerate(self.modalities):
                pool = [i for pos in positions for i in groups[pos][1][m]]
                camera_before.extend(len({self.data[name][i][2] for i in groups[pos][1][m]}) for pos in positions)
                rng = _rng(self.seed, self.epoch, getattr(self.original, 'rank', 0), pid, m, 'camera')
                by_cam = defaultdict(list)
                for i in pool:
                    by_cam[self.data[name][i][2]].append(i)
                for values in by_cam.values():
                    rng.shuffle(values)
                for pos in positions:
                    chosen = []
                    available = [c for c, values in by_cam.items() if values]
                    rng.shuffle(available)
                    for c in available[:self.k]:
                        chosen.append(by_cam[c].pop())
                    while len(chosen) < self.k:
                        remaining = [c for c, values in by_cam.items() if values]
                        chosen.append(by_cam[rng.choice(remaining)].pop())
                    groups[pos][1][m] = chosen
                    camera_after.append(len({self.data[name][i][2] for i in chosen}))
        assert self._ledger(groups) == before
        swaps = 0
        if self.epoch >= 11:
            if self.hardness is None:
                raise RuntimeError('A-V1 requires the fixed epoch-10 hardness reference')
            rng = _rng(self.seed, self.epoch, getattr(self.original, 'rank', 0), 'hard_swap')
            candidate_slots = list(range(len(groups)))
            rng.shuffle(candidate_slots)
            used = set()
            n_per_batch = self.batch_size // self.k
            for a in candidate_slots[:len(groups)//2]:
                if a in used:
                    continue
                ba = a // n_per_batch
                proposals = rng.sample(range(len(groups)), min(32, len(groups)))
                feasible = []
                for z in proposals:
                    if z in used or z == a or z // n_per_batch == ba:
                        continue
                    bz = z // n_per_batch
                    ya, yz = groups[a][0], groups[z][0]
                    pa = [groups[i][0] for i in range(ba*n_per_batch, (ba+1)*n_per_batch) if i != a]
                    pz = [groups[i][0] for i in range(bz*n_per_batch, (bz+1)*n_per_batch) if i != z]
                    if yz in pa or ya in pz:
                        continue
                    delta = sum(self.hardness[yz, p] - self.hardness[ya, p] for p in pa)
                    delta += sum(self.hardness[ya, p] - self.hardness[yz, p] for p in pz)
                    delta /= max(1, n_per_batch*(n_per_batch-1)/2)
                    if delta > 0:
                        feasible.append((z, delta))
                if feasible:
                    weights = np.exp(np.array([d for _, d in feasible])/0.2)
                    weights /= weights.sum()
                    pick = min(len(feasible)-1, int(np.searchsorted(np.cumsum(weights), rng.random(), side='right')))
                    z = feasible[pick][0]
                    groups[a], groups[z] = groups[z], groups[a]
                    used.update((a, z))
                    swaps += 1
            assert self._ledger(groups) == before
        for b in range(0, len(groups), self.batch_size // self.k):
            pids = [entry[0] for entry in groups[b:b+self.batch_size//self.k]]
            assert len(pids) == len(set(pids))
        self.last_stats = {'groups': len(groups), 'swaps': swaps,
                           'cameras_before': sum(camera_before)/max(1, len(camera_before)),
                           'cameras_after': sum(camera_after)/max(1, len(camera_after))}
        for _, modal in groups:
            for row in zip(*modal):
                yield tuple(row)

    def _ledger(self, groups):
        return (Counter(pid for pid, _ in groups),
                tuple(Counter(i for _, modal in groups for i in modal[m]) for m in range(3)),
                len(groups))
