import unittest
import os
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from tools.retrieval_diagnostics import analyze_model, aggregate, compare_models
from utils.metrics import eval_func
from tools.diagnose_av1_retrieval import _extract, write_report


class RetrievalDiagnosisTest(unittest.TestCase):
    def setUp(self):
        self.queries = [
            {'image_id': 'query/RGB/q0.jpg', 'pid': 0, 'camera': 0, 'modality': 1},
            {'image_id': 'query/IR/q1.jpg', 'pid': 1, 'camera': 1, 'modality': 2},
            {'image_id': 'query/Thermal/q2.jpg', 'pid': 0, 'camera': 1, 'modality': 3},
        ]
        self.gallery = [
            {'image_id': 'test/RGB/g0.jpg', 'pid': 0, 'camera': 1, 'modality': 1},
            {'image_id': 'test/RGB/g1.jpg', 'pid': 0, 'camera': 2, 'modality': 1},
            {'image_id': 'test/IR/g2.jpg', 'pid': 1, 'camera': 0, 'modality': 2},
            {'image_id': 'test/IR/g3.jpg', 'pid': 1, 'camera': 2, 'modality': 2},
            {'image_id': 'test/Thermal/g4.jpg', 'pid': 2, 'camera': 0, 'modality': 3},
            {'image_id': 'test/Thermal/g5.jpg', 'pid': 2, 'camera': 2, 'modality': 3},
        ]
        torch.manual_seed(7)
        self.qf = F.normalize(torch.randn(3, 768), dim=1).numpy()
        self.gf = F.normalize(torch.randn(6, 768), dim=1).numpy()

    def test_full_ranking_matches_official_eval_func(self):
        rows, top, competition, geometry, skipped = analyze_model(
            self.qf, self.gf, self.queries, self.gallery, 'UAD', query_chunk=2)
        self.assertEqual(skipped, [])
        self.assertEqual(len(top), sum(r['gallery_count'] for r in rows))
        self.assertEqual(len(competition), 12)
        self.assertTrue(geometry)
        qf, gf = torch.from_numpy(self.qf), torch.from_numpy(self.gf)
        distance = qf.square().sum(1, keepdim=True) + gf.square().sum(1)[None, :]
        distance.addmm_(qf, gf.T, beta=1, alpha=-2)
        indices = torch.argsort(distance, dim=1).numpy()
        official_cmc, official_map, official_minp = eval_func(
            indices,
            np.array([r['pid'] for r in self.queries]),
            np.array([r['pid'] for r in self.gallery]),
            np.array([r['camera'] for r in self.queries]),
            np.array([r['camera'] for r in self.gallery]),
            max_rank=3, return_minp=True)
        metrics = aggregate(rows)
        self.assertAlmostEqual(metrics['rank1'], official_cmc[0])
        self.assertAlmostEqual(metrics['mAP'], official_map)
        self.assertAlmostEqual(metrics['mINP'], official_minp)
        self.assertEqual(rows[0]['gallery_count'], 4)
        self.assertEqual(rows[2]['positive_count'], 1)

    def test_paired_accounting_reconstructs_metric_deltas(self):
        old, _, old_comp, _, _ = analyze_model(
            self.qf, self.gf, self.queries, self.gallery, 'UAD')
        new_features = F.normalize(torch.from_numpy(self.qf) + .08 * torch.randn(3, 768), dim=1).numpy()
        new, _, new_comp, _, _ = analyze_model(
            new_features, self.gf, self.queries, self.gallery, 'A1')
        rows, transitions, _, competition, groups, summary = compare_models(
            old, new, old_comp, new_comp)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(transitions), 3)
        self.assertEqual(len(competition), 12)
        self.assertTrue(groups)
        self.assertAlmostEqual(summary['delta']['mAP'],
                               np.mean([r['delta_AP'] for r in rows]))
        self.assertAlmostEqual(summary['transitions']['1']['net_delta'],
                               summary['delta']['rank1'])
        self.assertEqual(len(summary['modality_pair_summary']), 12)
        with tempfile.TemporaryDirectory() as directory:
            report = os.path.join(directory, 'report.md')
            write_report(report, summary, groups)
            with open(report, encoding='utf-8') as stream:
                content = stream.read()
            self.assertIn('Rank transitions', content)
            self.assertIn('Directed modality-pair retrieval', content)

    def test_single_modality_similarity_uses_only_its_gallery(self):
        queries = [{'image_id': 'q.jpg', 'pid': 1, 'camera': 0, 'modality': 1}]
        gallery = [
            {'image_id': 'rgb_positive.jpg', 'pid': 1, 'camera': 1, 'modality': 1},
            {'image_id': 'rgb_negative.jpg', 'pid': 2, 'camera': 1, 'modality': 1},
            {'image_id': 'ir_positive.jpg', 'pid': 1, 'camera': 1, 'modality': 2},
            {'image_id': 'ir_negative.jpg', 'pid': 2, 'camera': 1, 'modality': 2},
        ]
        qf = np.zeros((1, 768), dtype=np.float32)
        gf = np.zeros((4, 768), dtype=np.float32)
        qf[0, 0] = 1
        gf[0, :2] = [.8, .6]
        gf[1, :2] = [.6, .8]
        gf[2, :2] = [1, 0]
        gf[3, :2] = [.9, np.sqrt(.19)]
        _, _, competition, _, _ = analyze_model(qf, gf, queries, gallery, 'UAD')
        rgb = next(row for row in competition if row['gallery_modality'] == 'RGB')
        mixed = next(row for row in competition if row['gallery_modality'] == 'Mixed')
        self.assertAlmostEqual(rgb['best_positive_similarity'], .8)
        self.assertAlmostEqual(rgb['best_negative_similarity'], .6)
        self.assertAlmostEqual(rgb['margin'], .2)
        self.assertAlmostEqual(mixed['best_positive_similarity'], 1.)
        self.assertAlmostEqual(mixed['best_negative_similarity'], .9)

    def test_feature_extraction_checks_manifest_order(self):
        class Model(torch.nn.Module):
            def forward(self, images, mode):
                return torch.ones(len(images), 768) * mode

        def loader(name):
            return [(torch.ones(1, 3, 2, 2), (0,), (1,), torch.tensor([1]),
                     (1,), (name,))]

        query = [{'path': '/dataset/query/RGB/q.jpg', 'pid': 0,
                  'camera': 1, 'modality': 1}]
        gallery = [{'path': '/dataset/test/RGB/g.jpg', 'pid': 0,
                    'camera': 1, 'modality': 1}]
        qf, gf = _extract(Model(), [(loader('q.jpg'), loader('g.jpg'))],
                          torch.device('cpu'), query, gallery)
        self.assertEqual(qf.shape, (1, 768))
        self.assertEqual(gf.shape, (1, 768))
        with self.assertRaisesRegex(RuntimeError, 'loader order'):
            _extract(Model(), [(loader('wrong.jpg'), loader('g.jpg'))],
                     torch.device('cpu'), query, gallery)


if __name__ == '__main__':
    unittest.main()
