"""Read-only paired diagnosis of a trained UAD checkpoint and an A-V1 checkpoint.

Example from WHU/:
  python tools/diagnose_av1_retrieval.py --config-file configs/UAD.yml \
      --uad logs/UAD_1000_DDP/transformer_120.pth \
      --a1 logs/A1_1000_DDP/transformer_120.pth \
      --output logs/diagnosis_1000
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg as default_cfg
from datasets.bases import ImageDatasetTest
from datasets.make_dataloader import val_collate_fn
from datasets.whu_mars import WHU_MARS
from model import make_model
from processor.av1_state import file_hash, load_trusted_artifact
from tools.retrieval_diagnostics import (MODALITIES, aggregate, analyze_model,
                                         compare_models, write_csv, write_json)


EXPECTED_UAD = {'mAP': .1102, 'rank1': .2971, 'rank5': .4632, 'rank10': .5455}
EXPECTED_A1 = {'mAP': .1060, 'rank1': .2765, 'rank5': .4426, 'rank10': .5252}


def parse_expected(text, defaults):
    if not text:
        return defaults
    parsed = {}
    for part in text.split(','):
        key, value = part.split('=', 1)
        parsed[key.strip()] = float(value)
    required = set(defaults)
    if set(parsed) != required:
        raise ValueError('Expected metrics must contain exactly '+', '.join(sorted(required)))
    return parsed


def assert_expected(actual, expected, tolerance, name):
    mismatches = {key: {'actual': actual[key], 'expected': value}
                  for key, value in expected.items() if abs(actual[key]-value) > tolerance}
    if mismatches:
        raise RuntimeError('%s official metric reproduction failed: %s. '
                           'Check checkpoints, split, camera IDs, normalization and TOP_K_EVAL.'
                           % (name, mismatches))


def build_records(dataset, modalities, role):
    source = dataset.query if role == 'query' else dataset.gallery
    records = []
    for modality in modalities:
        for path, pid, camera, modality_id in source.get(modality, []):
            image_id = os.path.relpath(path, dataset.dataset_dir).replace('\\', '/')
            records.append({'index': len(records), 'image_id': image_id,
                            'path': os.path.abspath(path), 'pid': int(pid),
                            'camera': int(camera), 'modality': int(modality_id),
                            'platform': 'Ground' if camera < 5 else 'Aerial',
                            'role': role})
    ids = [record['image_id'] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate evaluation image ID in '+role)
    return records


def records_hash(queries, gallery):
    payload = json.dumps({'query': queries, 'gallery': gallery},
                         sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint(model, path):
    state = load_trusted_artifact(path)
    if 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    if 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    state = {key[7:] if key.startswith('module.') else key: value
             for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return file_hash(path)


def _loaders(dataset, modalities, transform, batch_size, workers):
    loaders = []
    for modality in modalities:
        query = ImageDatasetTest({modality: dataset.query.get(modality, [])}, modality, transform)
        gallery = ImageDatasetTest({modality: dataset.gallery.get(modality, [])}, modality, transform)
        loaders.append((
            DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=workers,
                       collate_fn=val_collate_fn),
            DataLoader(gallery, batch_size=batch_size, shuffle=False, num_workers=workers,
                       collate_fn=val_collate_fn)))
    return loaders


def _extract(model, loaders, device, query_records, gallery_records):
    model.eval()
    query_parts, gallery_parts = [], []
    query_cursor, gallery_cursor = 0, 0
    with torch.inference_mode():
        for modality_index, (query_loader, gallery_loader) in enumerate(loaders, 1):
            for loader, parts, records, cursor in (
                    (query_loader, query_parts, query_records, 'query'),
                    (gallery_loader, gallery_parts, gallery_records, 'gallery')):
                for images, pids, cameras, _, modalities, filenames in loader:
                    position = query_cursor if cursor == 'query' else gallery_cursor
                    expected = records[position:position+len(pids)]
                    if len(expected) != len(pids) or any(
                            int(pid) != record['pid'] or int(camera) != record['camera'] or
                            int(modality) != record['modality'] or
                            filename != os.path.basename(record['path'])
                            for pid, camera, modality, filename, record in zip(
                                pids, cameras, modalities, filenames, expected)):
                        raise RuntimeError('Feature loader order differs from stable image manifest')
                    if cursor == 'query':
                        query_cursor += len(pids)
                    else:
                        gallery_cursor += len(pids)
                    feature = model(images.to(device), mode=modality_index)
                    feature = F.normalize(feature.float(), dim=1)
                    parts.append(feature.cpu().numpy())
    qf = np.concatenate(query_parts, axis=0).astype(np.float32)
    gf = np.concatenate(gallery_parts, axis=0).astype(np.float32)
    if qf.shape != (len(query_records), 768) or gf.shape != (len(gallery_records), 768):
        raise RuntimeError('Extracted feature counts do not match the evaluation manifest')
    return qf, gf


def features_for_checkpoint(label, checkpoint, output, cfg, n_class, n_camera,
                            n_view, loaders, device, query_records, gallery_records,
                            manifest_hash, reuse_cache):
    destination = os.path.join(output, label)
    os.makedirs(destination, exist_ok=True)
    query_path = os.path.join(destination, 'query_features.npy')
    gallery_path = os.path.join(destination, 'gallery_features.npy')
    cache_meta_path = os.path.join(destination, 'cache_manifest.json')
    checksum = file_hash(checkpoint)
    if reuse_cache and all(os.path.isfile(path) for path in
                           (query_path, gallery_path, cache_meta_path)):
        with open(cache_meta_path, encoding='utf-8') as stream:
            meta = json.load(stream)
        if meta.get('checkpoint_sha256') == checksum and meta.get('evaluation_manifest_sha256') == manifest_hash:
            return np.load(query_path), np.load(gallery_path), meta
    model = make_model(cfg, num_class=n_class, camera_num=n_camera, view_num=n_view)
    loaded_checksum = _load_checkpoint(model, checkpoint)
    if loaded_checksum != checksum:
        raise RuntimeError('Checkpoint changed during loading')
    model.to(device)
    qf, gf = _extract(model, loaders, device, query_records, gallery_records)
    np.save(query_path, qf)
    np.save(gallery_path, gf)
    meta = {'checkpoint': os.path.abspath(checkpoint), 'checkpoint_sha256': checksum,
            'evaluation_manifest_sha256': manifest_hash, 'query_count': len(query_records),
            'gallery_count': len(gallery_records), 'feature': 'pre-BN CLS, L2 normalized, 768D',
            'query_features': query_path, 'gallery_features': gallery_path}
    write_json(cache_meta_path, meta)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return qf, gf, meta


def _run_one(model_name, checkpoint, subfolder, args, cfg, dataset, loaders,
             queries, gallery, manifest_hash, geometry=True):
    qf, gf, meta = features_for_checkpoint(
        subfolder, checkpoint, args.output, cfg, dataset.num_train_pids,
        dataset.num_train_cams, dataset.num_train_vids, loaders, args.device,
        queries, gallery, manifest_hash, args.reuse_cache)
    rows, top, competition, geometry_rows, skipped = analyze_model(
        qf, gf, queries, gallery, model_name, args.query_chunk, args.top_k, geometry)
    return rows, top, competition, geometry_rows, skipped, meta


def _checkpoint_dynamics(args, cfg, dataset, loaders, queries, gallery, manifest_hash):
    rows = []
    for epoch_text, uad_path, a1_path in args.checkpoint_pair:
        epoch = int(epoch_text)
        old, _, old_comp, _, _, _ = _run_one('UAD', uad_path,
            'epoch_%03d/UAD' % epoch, args, cfg, dataset, loaders,
            queries, gallery, manifest_hash, geometry=False)
        new, _, new_comp, _, _, _ = _run_one('A1', a1_path,
            'epoch_%03d/A1' % epoch, args, cfg, dataset, loaders,
            queries, gallery, manifest_hash, geometry=False)
        if set(r['query_id'] for r in old) != set(r['query_id'] for r in new):
            raise RuntimeError('Checkpoint pair at epoch %d has different valid Queries' % epoch)
        comparison = compare_models(old, new, old_comp, new_comp)[-1]
        for name, current in (('UAD', old), ('A1', new)):
            metrics = aggregate(current)
            margins = [r['margin'] for r in current if r['margin'] is not None]
            negatives = [r['best_negative_similarity'] for r in current
                         if r['best_negative_similarity'] is not None]
            rows.append({'epoch': epoch, 'model': name,
                         **{key: metrics[key] for key in ('rank1', 'rank5', 'rank10', 'mAP', 'mINP')},
                         'mean_margin': float(np.mean(margins)) if margins else '',
                         'mean_best_positive_similarity': float(np.mean([
                             r['best_positive_similarity'] for r in current])),
                         'mean_best_negative_similarity': float(np.mean(negatives)) if negatives else '',
                         'rank1_regressed_against_matched_epoch':
                             comparison['transitions']['1']['regressed']['count'],
                         'rank10_regressed_against_matched_epoch':
                             comparison['transitions']['10']['regressed']['count']})
    return rows


def write_report(path, summary, groups, dynamics=None):
    n = summary['UAD']['valid_queries']
    lines = ['# A-V1 vs UAD read-only retrieval diagnosis', '',
             'All figures below come from the saved evaluation features. '
             'They describe associations, not causal effects of an individual training module.', '',
             '## Official-protocol reproduction', '',
             '| Model | Valid Q | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ('UAD', 'A1'):
        item = summary[name]
        lines.append('| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f |' % (
            name, item['valid_queries'], item['rank1'], item['rank5'],
            item['rank10'], item['mAP'], item['mINP']))
    lines.extend(['', '## Rank transitions', ''])
    for k in (1, 5, 10):
        t = summary['transitions'][str(k)]
        lines.append('- Rank-%d: %d regressed, %d improved, net %d/%d = %+.4f.' % (
            k, t['regressed']['count'], t['improved']['count'],
            t['improved']['count']-t['regressed']['count'], n, t['net_delta']))
    lines.append('- %d Queries regressed simultaneously at Rank-1/5/10; '
                 '%d crossed outward over Rank-10 and %d crossed inward.' % (
                     summary['regressed_at_rank1_5_10'], summary['crossed_top10_outward'],
                     summary['crossed_top10_inward']))
    lines.extend(['', '## AP and INP accounting by Rank-1 transition', ''])
    for name in ('both_correct', 'regressed', 'improved', 'both_wrong'):
        item = summary['rank_k_AP_INP_contributions']['1'].get(name)
        if item:
            lines.append('- %s: n=%d, mean ΔAP=%+.6f, contribution to total ΔmAP=%+.6f, '
                         'mean ΔINP=%+.6f.' % (
                             name, item['count'], item['mean_delta_AP'],
                             item['contribution_to_delta_mAP'], item['mean_delta_INP']))
    lines.extend(['', '## Closest positive and negative for Rank-1 regressions', ''])
    for item in summary['rank1_regressed_similarity_signs']:
        lines.append('- %s / %s: %d Queries.' % (
            item['positive_change'], item['negative_change'], item['count']))
    source = sorted(summary['rank1_regressed_wrong_sources'],
                    key=lambda row: row['count'], reverse=True)
    lines.extend(['', '## Mixed-gallery wrong top-1 sources', ''])
    for item in source[:20]:
        lines.append('- %s %s → %s %s camera %s: %d Queries.' % (
            item['query_modality'], item['query_platform'], item['gallery_modality'],
            item['gallery_platform'], item['gallery_camera'], item['count']))
    conflict = summary['rank1_regressed_single_gallery_correct_but_mixed_wrong']
    lines.append('- %d/%d Rank-1 regressions were correct in at least one single-modality '
                 'Gallery but wrong in Mixed Gallery.' % (
                     conflict.get('any_single_gallery_correct', 0),
                     summary['transitions']['1']['regressed']['count']))
    lines.extend(['', '## Directed modality-pair retrieval', '',
                  '| Query → Gallery | Valid Q | ΔRank-1 | ΔmAP |',
                  '| --- | ---: | ---: | ---: |'])
    for item in summary['modality_pair_summary']:
        lines.append('| %s → %s | %d | %s | %s |' % (
            item['query_modality'], item['gallery_modality'], item['valid_query_count'],
            ('%+.4f' % item['delta_rank1_hit']) if item['valid_query_count'] else '',
            ('%+.4f' % item['delta_AP']) if item['valid_query_count'] else ''))
    lines.extend(['', '## Group counts and deltas', ''])
    for item in groups:
        if item['group_type'] in ('query_modality', 'query_platform', 'modality_platform',
                                  'uad_first_rank'):
            lines.append('- %s=%s: n=%d, ΔRank-1=%+.4f, ΔmAP=%+.4f, Δmargin=%s.' % (
                item['group_type'], item['group'], item['valid_query_count'],
                item['delta_rank1'], item['delta_mAP'], item['mean_delta_margin']))
    if dynamics:
        lines.extend(['', '## Available intermediate checkpoints', ''])
        for item in dynamics:
            lines.append('- Epoch %d %s: Rank-1=%.4f, mAP=%.4f, mean margin=%s.' % (
                item['epoch'], item['model'], item['rank1'], item['mAP'], item['mean_margin']))
    lines.extend(['', 'Interpretation should use the per-Query CSVs and subgroup counts. '
                  'A sign of similarity change alone does not establish why a module caused it.', ''])
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', default='configs/UAD.yml')
    parser.add_argument('--uad', required=True, help='Trained 120-epoch UAD checkpoint')
    parser.add_argument('--a1', required=True, help='Trained 120-epoch A-V1 checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='', help='Override DATASETS.ROOT_DIR')
    parser.add_argument('--dataset-dir', default='', help='Override DATASETS.DATASET_DIR')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--query-chunk', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--reuse-cache', action='store_true')
    parser.add_argument('--expected-uad', default='')
    parser.add_argument('--expected-a1', default='')
    parser.add_argument('--metric-tolerance', type=float, default=.0002)
    parser.add_argument('--checkpoint-pair', nargs=3, action='append', default=[],
                        metavar=('EPOCH', 'UAD_CHECKPOINT', 'A1_CHECKPOINT'))
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.query_chunk <= 0 or args.top_k < 10:
        parser.error('batch-size/query-chunk must be positive and top-k >= 10')
    for path in (args.uad, args.a1):
        if not os.path.isfile(path):
            parser.error('Checkpoint is missing: '+path)
    for epoch_text, uad_path, a1_path in args.checkpoint_pair:
        if int(epoch_text) <= 0:
            parser.error('checkpoint-pair epoch must be positive')
        for path in (uad_path, a1_path):
            if not os.path.isfile(path):
                parser.error('Intermediate checkpoint is missing: '+path)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = default_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    if args.data_root:
        cfg.DATASETS.ROOT_DIR = args.data_root
    if args.dataset_dir:
        cfg.DATASETS.DATASET_DIR = args.dataset_dir
    if cfg.TEST.RE_RANKING or cfg.TEST.TOP_K_EVAL != 0 or cfg.TEST.FEAT_NORM != 'yes':
        raise ValueError('Diagnosis requires official L2/full-gallery/no-rerank evaluation')
    if list(cfg.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal']:
        raise ValueError('Expected RGB/IR/Thermal in the official order')
    if cfg.DATASETS.FILTER_GD:
        raise ValueError('Use the full WHU-MARS-1000 test split, not FILTER_GD')
    cfg.TEST.NECK_FEAT = 'before'
    cfg.MODEL.PRETRAIN_CHOICE = ''  # Load every model parameter from the supplied checkpoint.
    cfg.MODEL.DIST_TRAIN = False
    cfg.freeze()
    dataset = WHU_MARS(root=cfg.DATASETS.ROOT_DIR,
                       modalities=list(cfg.DATASETS.MODALITIES),
                       dataset_dir=cfg.DATASETS.DATASET_DIR,
                       filter_gd=False)
    queries = build_records(dataset, list(cfg.DATASETS.MODALITIES), 'query')
    gallery = build_records(dataset, list(cfg.DATASETS.MODALITIES), 'gallery')
    if not queries or not gallery:
        raise ValueError('Empty Query or Gallery split')
    manifest_hash = records_hash(queries, gallery)
    transform = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                           T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    loaders = _loaders(dataset, list(cfg.DATASETS.MODALITIES), transform,
                       args.batch_size, args.workers)
    os.makedirs(args.output, exist_ok=True)
    write_csv(os.path.join(args.output, 'query_metadata.csv'), queries)
    write_csv(os.path.join(args.output, 'gallery_metadata.csv'), gallery)
    old = _run_one('UAD', args.uad, 'UAD', args, cfg, dataset,
                   loaders, queries, gallery, manifest_hash)
    new = _run_one('A1', args.a1, 'A1', args, cfg, dataset,
                   loaders, queries, gallery, manifest_hash)
    old_rows, old_top, old_comp, old_geometry, old_skipped, old_meta = old
    new_rows, new_top, new_comp, new_geometry, new_skipped, new_meta = new
    old_metrics, new_metrics = aggregate(old_rows), aggregate(new_rows)
    # Stop before attribution if the feature/evaluation path cannot reproduce logs.
    assert_expected(old_metrics, parse_expected(args.expected_uad, EXPECTED_UAD),
                    args.metric_tolerance, 'UAD')
    assert_expected(new_metrics, parse_expected(args.expected_a1, EXPECTED_A1),
                    args.metric_tolerance, 'A1')
    if old_skipped != new_skipped:
        raise RuntimeError('The two models excluded different Query sets')
    query_rows, transitions, regressed, competition, grouped, summary = compare_models(
        old_rows, new_rows, old_comp, new_comp)
    summary['excluded_queries'] = len(old_skipped)
    summary['protocol'] = {'same_camera_gallery_excluded': True,
                           'full_gallery_sort': True, 'reranking': False,
                           'tta': False, 'feature': 'pre-BN CLS 768D L2',
                           'distance': 'squared Euclidean, official torch.argsort'}
    manifest = {'UAD': old_meta, 'A1': new_meta,
                'evaluation_manifest_sha256': manifest_hash,
                'query_count': len(queries), 'gallery_count': len(gallery),
                'valid_query_count': old_metrics['valid_queries'],
                'excluded_query_count': len(old_skipped),
                'dataset_root': os.path.abspath(dataset.dataset_dir),
                'config_file': os.path.abspath(args.config_file),
                'preprocessing': {'size_test': list(cfg.INPUT.SIZE_TEST),
                                  'mean': list(cfg.INPUT.PIXEL_MEAN),
                                  'std': list(cfg.INPUT.PIXEL_STD)},
                'metrics_reproduced_with_tolerance': args.metric_tolerance}
    write_json(os.path.join(args.output, 'diagnostic_manifest.json'), manifest)
    write_json(os.path.join(args.output, 'ranking_summary.json'), summary)
    write_csv(os.path.join(args.output, 'retrieval_queries.csv'), query_rows)
    write_csv(os.path.join(args.output, 'query_rank_transitions.csv'), transitions)
    write_csv(os.path.join(args.output, 'regressed_queries.csv'), regressed,
              fields=list(dict.fromkeys(list(query_rows[0]) + list(transitions[0]))))
    write_csv(os.path.join(args.output, 'excluded_queries.csv'), old_skipped,
              fields=['query_index', 'query_id', 'query_pid', 'query_camera',
                      'query_modality', 'reason'])
    write_csv(os.path.join(args.output, 'topk_gallery.csv'), old_top+new_top)
    write_csv(os.path.join(args.output, 'modality_competition.csv'), competition)
    write_csv(os.path.join(args.output, 'modality_pair_summary.csv'),
              summary['modality_pair_summary'])
    write_csv(os.path.join(args.output, 'group_comparison.csv'), grouped)
    write_csv(os.path.join(args.output, 'feature_geometry_summary.csv'), old_geometry+new_geometry)
    dynamics = None
    if args.checkpoint_pair:
        dynamics = _checkpoint_dynamics(args, cfg, dataset, loaders,
                                        queries, gallery, manifest_hash)
        write_csv(os.path.join(args.output, 'checkpoint_dynamics.csv'), dynamics)
    write_report(os.path.join(args.output, 'diagnostic_report.md'), summary, grouped, dynamics)
    print(json.dumps(summary['delta'], indent=2))
    print('Diagnosis written to '+os.path.abspath(args.output))


if __name__ == '__main__':
    main()
