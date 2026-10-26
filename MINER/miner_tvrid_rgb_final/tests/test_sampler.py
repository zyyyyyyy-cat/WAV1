import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import torch
from sampler import MINERBatchSampler

def _make_mock_dataset():
    class MockDS:
        def __init__(self):
            self.valid_pids = [0, 1, 2, 3]
            self.pid_to_indices = {
                0: [0,1,2,3,4,5], 1: [6,7,8,9,10,11],
                2: [12,13,14,15,16,17], 3: [18,19,20,21,22,23],
            }
            self.pid_cam_to_indices = {
                (0,0): [0,1,2], (0,1): [3,4,5],
                (1,0): [6,7,8], (1,1): [9,10,11],
                (2,0): [12,13,14], (2,1): [15,16,17],
                (3,0): [18,19,20], (3,1): [21,22,23],
            }
            self._pids = [i // 6 for i in range(24)]
            self._camids = [(i % 6) // 3 for i in range(24)]
            self._paths = [f"fake_{i}.png" for i in range(24)]
        def __len__(self):
            return 24
        def get_pid(self, idx):
            return self._pids[idx]
        def get_camid(self, idx):
            return self._camids[idx]
    return MockDS()

def test_batch_sampler_yields_correct_size():
    ds = _make_mock_dataset()
    sampler = MINERBatchSampler(frame_dataset=ds, num_ids=2, nq=2, ng=2, seed=42)
    batch = next(iter(sampler))
    assert len(batch) == 8  # 2 IDs * (2q + 2g)

def test_batch_sampler_q_g_split():
    ds = _make_mock_dataset()
    sampler = MINERBatchSampler(frame_dataset=ds, num_ids=2, nq=2, ng=2, seed=42)
    batch = next(iter(sampler))
    assert len(batch[:sampler.nq_per_batch]) == 4
    assert len(batch[sampler.nq_per_batch:]) == 4

def test_batch_sampler_respects_epoch():
    ds = _make_mock_dataset()
    sampler = MINERBatchSampler(frame_dataset=ds, num_ids=2, nq=2, ng=2, seed=42)
    sampler.set_epoch(0)
    batches_e0 = [b for b in sampler]
    sampler.set_epoch(1)
    batches_e1 = [b for b in sampler]
    assert batches_e0 != batches_e1

def test_batch_sampler_length():
    ds = _make_mock_dataset()
    sampler = MINERBatchSampler(frame_dataset=ds, num_ids=2, nq=2, ng=2, seed=42)
    assert len(sampler) >= 1

def test_batch_sampler_fixed_size_with_fallback():
    ds = _make_mock_dataset()
    sampler = MINERBatchSampler(frame_dataset=ds, num_ids=2, nq=2, ng=2, seed=42)
    for batch in sampler:
        assert len(batch) == 8, f"Expected 8 indices, got {len(batch)}"

def test_nq_per_batch():
    ds = _make_mock_dataset()
    sampler = MINERBatchSampler(frame_dataset=ds, num_ids=3, nq=4, ng=4, seed=42)
    assert sampler.nq_per_batch == 12  # 3 * 4

def test_dataloader_integration():
    from torch.utils.data import DataLoader, Dataset

    class FakeImageDataset(Dataset):
        def __init__(self):
            self.valid_pids = [0, 1, 2, 3]
            self.pid_to_indices = {
                0: [0,1,2,3,4,5], 1: [6,7,8,9,10,11],
                2: [12,13,14,15,16,17], 3: [18,19,20,21,22,23],
            }
            self.pid_cam_to_indices = {
                (0,0): [0,1,2], (0,1): [3,4,5],
                (1,0): [6,7,8], (1,1): [9,10,11],
                (2,0): [12,13,14], (2,1): [15,16,17],
                (3,0): [18,19,20], (3,1): [21,22,23],
            }
            self._pids = [i // 6 for i in range(24)]
            self._camids = [(i % 6) // 3 for i in range(24)]
            self._paths = [f"fake_{i}.png" for i in range(24)]
        def __len__(self): return 24
        def get_pid(self, idx): return self._pids[idx]
        def get_camid(self, idx): return self._camids[idx]
        def __getitem__(self, idx):
            return torch.randn(3, 8, 8), self._pids[idx], self._camids[idx]

    ds = FakeImageDataset()
    bs = MINERBatchSampler(ds, num_ids=2, nq=2, ng=2, seed=42)
    dl = DataLoader(ds, batch_sampler=bs, num_workers=0)
    images, labels, camids = next(iter(dl))
    assert images.shape == (8, 3, 8, 8)
    assert labels.shape == (8,)
    assert camids.shape == (8,)
