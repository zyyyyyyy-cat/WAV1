"""TVRID Top-View Re-IDentification dataset (RGB only).

Parses the TVRID data directory structure and provides:
  - Training frames with PID labels and camera IDs
  - Test passages for submission ranking
  - Frame listing and evenly-spaced selection utilities
"""

from __future__ import annotations

import csv
import os
import os.path as osp
from typing import Dict, List

import numpy as np

from config import TVRID_NUM_CAMERAS

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Camera name -> 0-based camera id
CAM_NAME_TO_ID = {
    "flat": 0,
    "upward": 1,
    "downward": 2,
    "upsidedown": 3,
    "upsideDown": 3,
}


def _list_frames(passage_dir: str) -> List[str]:
    """Return sorted absolute paths for RGB frames in a passage dir."""
    if not osp.isdir(passage_dir):
        return []
    return sorted(
        osp.join(passage_dir, f)
        for f in os.listdir(passage_dir)
        if "_RGB" in f and osp.splitext(f)[1].lower() in _IMG_EXTS
    )


def _select_frames_evenly(paths: List[str], n: int = 5) -> List[str]:
    """Return *n* evenly-spaced frame paths from *paths*."""
    if not paths:
        return []
    if n == 1:
        return [paths[len(paths) // 2]]
    if len(paths) <= n:
        return list(paths)
    idxs = np.linspace(0, len(paths) - 1, num=n, dtype=int).tolist()
    return [paths[i] for i in idxs]


class TVRID:
    """TVRID Top-View Re-IDentification dataset (RGB only).

    Reads TVRID data and populates ``self.train`` with per-frame dicts:
    ``{image_path, pid, camid, gallery_id}``.

    The TVRID dataset contains top-view surveillance passages from 4 camera
    angles (flat, upward, downward, upsideDown). Each passage is a sequence
    of RGB frames for one person crossing a corridor.
    """

    dataset_dir = "DB_extracted"

    def __init__(
        self,
        root: str = "data",
        val_ratio: float = 0.0,
        seed: int = 1234,
        eval_frames_per_passage: int = 5,
    ):
        self.root = root
        self.dataset_dir_abs = osp.join(root, self.dataset_dir)

        train_csv = osp.join(self.dataset_dir_abs, "train_labels.csv")
        test_csv = osp.join(self.dataset_dir_abs, "public_test_labels.csv")

        if not osp.isfile(train_csv):
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")

        # Parse training CSV
        all_rows = self._read_csv(train_csv)
        person_ids = sorted(set(r["person_id"] for r in all_rows))

        # Train / val split by identity
        rng = np.random.RandomState(seed)
        shuffled_pids = list(person_ids)
        rng.shuffle(shuffled_pids)
        n_val = max(0 if val_ratio == 0 else 1, int(len(shuffled_pids) * val_ratio))
        train_pids = set(shuffled_pids[: len(shuffled_pids) - n_val])

        # Contiguous relabel: train PIDs -> 0..N-1
        train_pid2label = {
            pid: idx for idx, pid in enumerate(sorted(train_pids))
        }

        train_rows = [r for r in all_rows if r["person_id"] in train_pids]

        # Explode train passages into individual frames
        self.train: List[dict] = []
        self.passage_metadata: Dict[str, dict] = {}
        train_base = osp.join(self.dataset_dir_abs, "train")

        for row in train_rows:
            pid_label = train_pid2label[row["person_id"]]
            cam_name = row["cam_name"].strip().lower()
            camid = CAM_NAME_TO_ID.get(cam_name, 0)
            passage_dir = osp.join(train_base, row["path"].replace("\\", "/"))
            frames = _list_frames(passage_dir)
            if not frames:
                continue

            gid = str(row["gallery_id"])
            self.passage_metadata[gid] = {
                "person_id": row["person_id"],
                "cam_name": cam_name,
                "passage_name": row.get("passage_name", ""),
                "cam_id": camid,
            }
            for frame_path in frames:
                self.train.append(
                    {
                        "image_path": frame_path,
                        "pid": pid_label,
                        "camid": camid,
                        "gallery_id": gid,
                    }
                )

        self.num_train_pids: int = len(train_pids)
        self.num_train_imgs: int = len(self.train)
        self.num_camera: int = TVRID_NUM_CAMERAS

        # Parse test CSV (query = gallery = every passage)
        self.query: List[dict] = []
        self.gallery: List[dict] = []

        if osp.isfile(test_csv):
            test_rows = self._read_csv(test_csv)
            test_base = osp.join(self.dataset_dir_abs, "test_public")
            for row in test_rows:
                passage_dir = osp.join(test_base, row["path"].replace("\\", "/"))
                frames = _list_frames(passage_dir)
                selected = _select_frames_evenly(frames, eval_frames_per_passage)
                if not selected:
                    continue

                cam_name = self._infer_cam_name(row["path"])
                camid = CAM_NAME_TO_ID.get(cam_name, 0)
                gid = str(row["gallery_id"])
                for frame_path in selected:
                    entry = {
                        "image_path": frame_path,
                        "pid": -1,
                        "camid": camid,
                        "gallery_id": gid,
                    }
                    self.query.append(entry)
                    self.gallery.append(entry)

        print(
            f"=> TVRID (rgb): "
            f"{self.num_train_pids} train IDs, "
            f"{self.num_train_imgs} train frames, "
            f"{len(self.query)} test frames"
        )

    @staticmethod
    def _read_csv(csv_path: str) -> List[dict]:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]

    @staticmethod
    def _infer_cam_name(path: str) -> str:
        for part in path.replace("\\", "/").split("/"):
            lower = part.strip().lower()
            if lower in CAM_NAME_TO_ID:
                return lower
        return "flat"
