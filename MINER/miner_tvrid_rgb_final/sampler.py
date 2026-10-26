"""MINER batch sampler for TVRID (Algorithm 1 from the paper).

Implements centroid-based hard-negative PID mining with hard camera pair
selection. Yields flat index lists structured as [q_indices..., g_indices...]
for use with PyTorch DataLoader's batch_sampler parameter.
"""
from __future__ import annotations

import json
import os.path as osp
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


def _load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """Load pre-computed embeddings from a JSON file.

    Expected format: {"image_path": [float, ...], ...}
    Returns dict mapping normalized path suffix -> L2-normalized 1D tensor.
    Path keys are normalized to the portion after 'train/' so embeddings
    generated on one machine match datasets loaded on another.
    """
    with open(path, "r") as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        t = torch.tensor(v, dtype=torch.float32)
        t = F.normalize(t.unsqueeze(0), p=2, dim=1).squeeze(0)
        result[_path_suffix(k)] = t
    return result


def _path_suffix(path: str) -> str:
    """Extract the path suffix after 'train/' for machine-independent matching."""
    path = path.replace("\\", "/")
    marker = "/train/"
    idx = path.rfind(marker)
    if idx >= 0:
        return path[idx + len(marker):]
    return osp.basename(path)


class MINERBatchSampler(Sampler[List[int]]):
    """BatchSampler implementing MINER Algorithm 1.

    Each iteration yields a flat list of dataset indices structured as:
        [q_idx_0, ..., q_idx_{nq_total-1}, g_idx_0, ..., g_idx_{ng_total-1}]

    where nq_total = num_ids * nq, ng_total = num_ids * ng.
    The training loop splits at nq_total to separate query/gallery.

    Args:
        frame_dataset: TVRIDFrameDataset with index mappings.
        num_ids: number of identities per batch (m in the paper).
        nq: query frames per identity.
        ng: gallery frames per identity.
        embeddings_path: JSON file with pre-computed embeddings for centroids.
        mining_temperature: softmax temperature for hard ID sampling.
        seed: random seed for reproducibility.
    """

    def __init__(
        self,
        frame_dataset,
        num_ids: int,
        nq: int,
        ng: int,
        embeddings_path: Optional[str] = None,
        mining_temperature: float = 2.0,
        seed: int = 1234,
    ):
        self.dataset = frame_dataset
        self.num_ids = num_ids
        self.nq = nq
        self.ng = ng
        self.mining_temperature = mining_temperature
        self._seed = seed
        self._epoch = 0

        # Centroid state
        self._centroid_matrix: Optional[torch.Tensor] = None
        self._centroid_pids: List[int] = []
        self._cam_centroids: Dict[int, Dict[int, torch.Tensor]] = {}

        if embeddings_path and osp.isfile(embeddings_path):
            print(f"Loading MINER embeddings from: {embeddings_path}")
            emb_map = _load_embeddings(embeddings_path)
            self._build_centroids(emb_map)
        else:
            print("No embeddings path — using random sampling (no MINER mining)")

        # Batches per epoch
        denom = num_ids * nq
        self._length = max(1, len(frame_dataset) // max(denom, 1))

    @property
    def nq_per_batch(self) -> int:
        """Total query frames per batch = num_ids * nq."""
        return self.num_ids * self.nq

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._length

    def _build_centroids(self, emb_map: Dict[str, torch.Tensor]) -> None:
        """Compute per-PID and per-PID-per-camera centroids from embeddings."""
        pid_embs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        pid_cam_embs: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for idx in range(len(self.dataset)):
            path = _path_suffix(self.dataset._paths[idx])
            pid = self.dataset.get_pid(idx)
            camid = self.dataset.get_camid(idx)
            emb = emb_map.get(path)
            if emb is None:
                continue
            pid_embs[pid].append(emb)
            pid_cam_embs[pid][camid].append(emb)

        centroid_list = []
        centroid_pids = []
        for pid in self.dataset.valid_pids:
            if pid not in pid_embs:
                continue
            embs = torch.stack(pid_embs[pid], dim=0)
            centroid = F.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
            centroid_list.append(centroid)
            centroid_pids.append(pid)

        if centroid_list:
            self._centroid_matrix = torch.stack(centroid_list, dim=0)
            self._centroid_pids = centroid_pids
            print(f"  MINER centroids: {len(centroid_pids)} PIDs, dim={self._centroid_matrix.shape[1]}")
        else:
            print("  Warning: no embeddings matched training images")

        for pid in self.dataset.valid_pids:
            if pid not in pid_cam_embs:
                continue
            cam_centroids = {}
            for camid, embs in pid_cam_embs[pid].items():
                e = torch.stack(embs, dim=0)
                cam_centroids[camid] = F.normalize(
                    e.mean(dim=0, keepdim=True), p=2, dim=1
                ).squeeze(0)
            if len(cam_centroids) >= 2:
                self._cam_centroids[pid] = cam_centroids

    def _pick_hard_id_set(self, rng: random.Random) -> List[int]:
        """Algorithm 1, Steps 1-4: select num_ids PIDs via centroid mining."""
        k = self.num_ids
        candidates = self._centroid_pids if self._centroid_matrix is not None else self.dataset.valid_pids

        if len(candidates) <= k:
            result = list(candidates)
            while len(result) < k:
                result.append(rng.choice(candidates))
            return result

        if self._centroid_matrix is None:
            return rng.sample(candidates, k)

        seed_idx = rng.randrange(len(candidates))
        sims = torch.matmul(self._centroid_matrix, self._centroid_matrix[seed_idx])
        probs = torch.softmax(sims / self.mining_temperature, dim=0)
        probs[seed_idx] = 0.0

        if probs.sum().item() > 0:
            probs = probs / probs.sum()
            sampled = torch.multinomial(probs, k - 1, replacement=False).tolist()
            return [candidates[seed_idx]] + [candidates[i] for i in sampled]

        return rng.sample(candidates, k)

    def _pick_hard_camera_pair(self, pid: int, rng: random.Random) -> Tuple[int, int]:
        """Algorithm 1, Step 5: select most dissimilar camera pair."""
        if pid not in self._cam_centroids or len(self._cam_centroids[pid]) < 2:
            return -1, -1

        cam_centroids = self._cam_centroids[pid]
        cams = list(cam_centroids.keys())

        best_pair = None
        best_sim = float("inf")
        for i, c1 in enumerate(cams):
            for c2 in cams[i + 1:]:
                sim = (cam_centroids[c1] @ cam_centroids[c2]).item()
                if sim < best_sim:
                    best_sim = sim
                    best_pair = (c1, c2)

        if best_pair is None:
            return -1, -1

        if rng.random() < 0.5:
            return best_pair
        return best_pair[1], best_pair[0]

    def _sample_indices_from_camera(self, pid: int, camid: int, n: int, rng: random.Random) -> List[int]:
        """Sample n frame indices from a (pid, camera) combination."""
        indices = self.dataset.pid_cam_to_indices.get((pid, camid), [])
        if not indices:
            return []
        if len(indices) >= n:
            return rng.sample(indices, n)
        return [rng.choice(indices) for _ in range(n)]

    def _sample_flat(self, pid: int, nq: int, ng: int, rng: random.Random) -> Tuple[List[int], List[int]]:
        """Fallback: sample Q/G indices from flat per-PID pool."""
        indices = self.dataset.pid_to_indices.get(pid, [])
        if not indices:
            return [], []
        if len(indices) >= nq + ng:
            sampled = rng.sample(indices, nq + ng)
            return sampled[:nq], sampled[nq:]
        q = [rng.choice(indices) for _ in range(nq)]
        g = [rng.choice(indices) for _ in range(ng)]
        return q, g

    def _sample_qg_for_pid(self, pid: int, nq: int, ng: int, rng: random.Random) -> Tuple[List[int], List[int]]:
        """Sample Q/G indices for a PID using hard camera pair."""
        cam_q, cam_g = self._pick_hard_camera_pair(pid, rng)
        if cam_q == -1:
            return self._sample_flat(pid, nq, ng, rng)

        q_indices = self._sample_indices_from_camera(pid, cam_q, nq, rng)
        g_indices = self._sample_indices_from_camera(pid, cam_g, ng, rng)

        if not q_indices or not g_indices:
            return self._sample_flat(pid, nq, ng, rng)

        return q_indices, g_indices

    def __iter__(self) -> Iterator[List[int]]:
        for batch_idx in range(self._length):
            rng = random.Random(self._seed + batch_idx + self._epoch * self._length)

            chosen_pids = self._pick_hard_id_set(rng)

            q_all: List[int] = []
            g_all: List[int] = []

            for pid in chosen_pids:
                q_idx, g_idx = self._sample_qg_for_pid(pid, self.nq, self.ng, rng)
                if not q_idx or not g_idx:
                    # Fallback: substitute a random valid PID to keep batch size fixed
                    fallback = rng.choice(self.dataset.valid_pids)
                    q_idx, g_idx = self._sample_flat(fallback, self.nq, self.ng, rng)
                q_all.extend(q_idx)
                g_all.extend(g_idx)

            # Structured: [all query indices, then all gallery indices]
            yield q_all + g_all
