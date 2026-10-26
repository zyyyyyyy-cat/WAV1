"""Flat frame-level Dataset for TVRID training images.

Each __getitem__ returns a single (image, pid, camid) tuple.
Used with MINERBatchSampler to construct structured training batches.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class TVRIDFrameDataset(Dataset):
    """Map-style dataset returning single TVRID frames.

    Builds index mappings from a TVRID dataset object:
      - idx -> (image_path, pid, camid)
      - pid -> [indices]
      - (pid, camid) -> [indices]

    Args:
        tvrid_dataset: TVRID object with .train list of dicts.
        transform: torchvision transform to apply to each image.
    """

    def __init__(self, tvrid_dataset, transform=None):
        self.transform = transform
        self._paths: List[str] = []
        self._pids: List[int] = []
        self._camids: List[int] = []

        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        self.pid_cam_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for idx, entry in enumerate(tvrid_dataset.train):
            path = entry["image_path"]
            pid = entry["pid"]
            camid = entry.get("camid", 0)

            self._paths.append(path)
            self._pids.append(pid)
            self._camids.append(camid)

            self.pid_to_indices[pid].append(idx)
            self.pid_cam_to_indices[(pid, camid)].append(idx)

        self.valid_pids: List[int] = [
            pid for pid, idxs in self.pid_to_indices.items() if len(idxs) >= 2
        ]

    def __len__(self) -> int:
        return len(self._paths)

    def get_pid(self, idx: int) -> int:
        return self._pids[idx]

    def get_camid(self, idx: int) -> int:
        return self._camids[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img = Image.open(self._paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        return img, self._pids[idx], self._camids[idx]
