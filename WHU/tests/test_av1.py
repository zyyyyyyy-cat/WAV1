import unittest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import patch

from loss.av1 import ModalityPrototypeBank, modality_prototype_loss, image_contrast_loss, candidate_mask
from datasets.av1_sampler import AV1Sampler
from processor.av1_state import hard_matrix, load_trusted_artifact
from datasets.bases import ImageDataset
from datasets.make_dataloader import train_collate_fn


class TestAV1(unittest.TestCase):
    def test_artifact_loader_accepts_legacy_torch_load_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'reference.pt')
            torch.save({'epoch': 10}, path)
            self.assertEqual(load_trusted_artifact(path)['epoch'], 10)
            actual_load = torch.load

            def legacy_load(file_path, map_location='cpu'):
                return actual_load(file_path, map_location=map_location, weights_only=False)

            with patch('processor.av1_state.torch.load', legacy_load):
                self.assertEqual(load_trusted_artifact(path)['epoch'], 10)

    def test_sample_id_tracks_original_image_not_pkm_tuple(self):
        names = ('RGB', 'IR', 'Thermal')
        data = {
            name: [(f'{name}_{i}.jpg', pid, cam, m)
                   for i, (pid, cam) in enumerate(entries)]
            for m, (name, entries) in enumerate(zip(names, (
                ((0, 0), (0, 1), (1, 2)),
                ((0, 1), (0, 0), (1, 2)),
                ((0, 1), (0, 0), (1, 2)),
            )))
        }
        dataset = ImageDataset(data, names)
        with patch('datasets.bases.read_image', side_effect=lambda path: torch.ones(2, 2)):
            first = dataset[(0, 0, 0)]
            repeated = dataset[(0, 0, 0)]
            other = dataset[(1, 1, 1)]
            negative = dataset[(2, 2, 2)]
        self.assertEqual(first[3], [0, 3, 6])
        self.assertEqual(repeated[3], first[3])
        self.assertEqual(other[3], [1, 4, 7])
        self.assertEqual(len(set(first[3] + other[3] + negative[3])), 9)

        images, labels, cameras, sample_ids = train_collate_fn([first, other, negative])
        self.assertEqual([ids.tolist() for ids in sample_ids],
                         [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        # RGB query 0 and IR gallery 0 came from the same PKM tuple, but are
        # different images in different cameras: Mixed must keep that positive.
        self.assertNotEqual(sample_ids[0][0], sample_ids[1][0])
        self.assertNotEqual(cameras[0][0], cameras[1][0])
        self.assertEqual(sample_ids[0][0], sample_ids[0][0])
        gallery_cameras = torch.cat(cameras)
        gallery_ids = torch.cat(sample_ids)
        mask = candidate_mask(cameras[0], gallery_cameras, sample_ids[0], gallery_ids)
        self.assertFalse(mask[0, 0])  # exact RGB image
        self.assertTrue(mask[0, 3])   # IR image in the same PKM tuple
        self.assertTrue(mask[0, 6])   # Thermal image in the same PKM tuple
        features = torch.randn(9, 8, requires_grad=True)
        _, _, mixed, coverage = image_contrast_loss(
            features, labels, cameras, sample_ids)
        self.assertEqual(coverage['mixed_0'], 2)
        self.assertTrue(torch.isfinite(mixed))

    def test_bank_and_gradients(self):
        torch.manual_seed(7)
        labels = torch.tensor([0, 0, 1, 1])
        features = torch.randn(12, 16, requires_grad=True)
        bank = ModalityPrototypeBank(3, 16, device='cpu')
        bank.update(features, labels)
        self.assertFalse(bank.valid[:, 2].any())
        self.assertTrue(torch.allclose(bank.prototypes[:, :2].norm(dim=-1), torch.ones(3, 2)))
        mp, counts = modality_prototype_loss(features, labels, bank)
        self.assertEqual(counts['2->2'], 4)
        cameras = [torch.tensor([0, 1, 0, 1])] * 3
        samples = [torch.arange(4) + 4 * m for m in range(3)]
        image, pair, mixed, coverage = image_contrast_loss(features, labels, cameras, samples)
        self.assertEqual(coverage['mixed_0'], 4)
        for loss in (mp, pair, mixed):
            gradient = torch.autograd.grad(loss, features, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertFalse(bank.prototypes.requires_grad)

    def test_no_usable_cross_camera_pair(self):
        labels = torch.tensor([0, 0, 1, 1])
        features = torch.randn(12, 8, requires_grad=True)
        cameras = [torch.zeros(4, dtype=torch.long)] * 3
        samples = [torch.arange(4) + 4 * m for m in range(3)]
        loss, _, _, coverage = image_contrast_loss(features, labels, cameras, samples)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(sum(coverage.values()), 0)
        self.assertTrue(loss.requires_grad)

    def test_sampler_preserves_occurrences(self):
        class Original:
            rank = 0

            def __iter__(self):
                for batch in ((0, 1), (2, 3), (0, 2), (1, 3)):
                    for pid in batch:
                        for k in range(2):
                            yield (pid * 4 + k,) * 3

            def __len__(self):
                return 16

        names = ('RGB', 'IR', 'Thermal')
        data = {name: [('x', i // 4, i % 2, m) for i in range(16)]
                for m, name in enumerate(names)}
        sampler = AV1Sampler(Original(), data, names, 2, 4, 1234)
        expected = sorted(list(Original()))
        for epoch in (1, 11):
            sampler.set_epoch(epoch)
            if epoch == 11:
                sampler.set_hardness(np.random.RandomState(1).rand(4, 4))
            result = list(sampler)
            for m in range(3):
                self.assertEqual(sorted(row[m] for row in result),
                                 sorted(row[m] for row in expected))
            self.assertEqual(len(result), 16)
            for start in range(0, 16, 4):
                pids = [data['RGB'][result[i][0]][1] for i in (start, start + 2)]
                self.assertEqual(len(set(pids)), 2)

    def test_hard_reference_symmetric_and_finite(self):
        centers = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1)
        valid = torch.ones(3, 4, dtype=torch.bool)
        valid[2, 0] = False
        counts = valid.float()
        global_centers = torch.nn.functional.normalize(centers.sum(0), dim=-1)
        result = hard_matrix(centers, valid, counts, global_centers)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.allclose(result, result.T))
        self.assertTrue(torch.allclose(result.diag(), torch.zeros(4)))


if __name__ == '__main__':
    unittest.main()
