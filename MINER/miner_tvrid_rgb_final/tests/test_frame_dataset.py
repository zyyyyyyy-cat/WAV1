import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from frame_dataset import TVRIDFrameDataset

def test_frame_dataset_length_and_accessors():
    class MockTVRID:
        train = [
            {"image_path": "fake/img1.png", "pid": 0, "camid": 1},
            {"image_path": "fake/img2.png", "pid": 1, "camid": 2},
        ]
    ds = TVRIDFrameDataset(MockTVRID(), transform=None)
    assert len(ds) == 2
    assert ds.get_pid(0) == 0
    assert ds.get_pid(1) == 1
    assert ds.get_camid(0) == 1

def test_frame_dataset_pid_to_indices():
    class MockTVRID:
        train = [
            {"image_path": "a.png", "pid": 0, "camid": 0},
            {"image_path": "b.png", "pid": 0, "camid": 1},
            {"image_path": "c.png", "pid": 1, "camid": 0},
        ]
    ds = TVRIDFrameDataset(MockTVRID(), transform=None)
    assert ds.pid_to_indices[0] == [0, 1]
    assert ds.pid_to_indices[1] == [2]

def test_frame_dataset_pid_cam_to_indices():
    class MockTVRID:
        train = [
            {"image_path": "a.png", "pid": 0, "camid": 0},
            {"image_path": "b.png", "pid": 0, "camid": 1},
            {"image_path": "c.png", "pid": 0, "camid": 0},
        ]
    ds = TVRIDFrameDataset(MockTVRID(), transform=None)
    assert sorted(ds.pid_cam_to_indices[(0, 0)]) == [0, 2]
    assert ds.pid_cam_to_indices[(0, 1)] == [1]

def test_valid_pids_filters_singles():
    class MockTVRID:
        train = [
            {"image_path": "a.png", "pid": 0, "camid": 0},
            {"image_path": "b.png", "pid": 0, "camid": 1},
            {"image_path": "c.png", "pid": 1, "camid": 0},  # only 1 frame
        ]
    ds = TVRIDFrameDataset(MockTVRID(), transform=None)
    assert 0 in ds.valid_pids
    assert 1 not in ds.valid_pids  # only 1 frame, needs >= 2
