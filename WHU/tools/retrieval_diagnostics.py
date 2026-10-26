"""Read-only, full-gallery WHU-MARS retrieval diagnostics.

The distance, sorting, same-camera exclusion, AP and INP definitions mirror
utils.metrics.eval_func with TOP_K_EVAL=0. Rows are streamed in query chunks.
"""
import csv
import json
from collections import Counter, defaultdict

import numpy as np
import torch


MODALITIES = {1: 'RGB', 2: 'NIR', 3: 'TIR'}
RANK_KEYS = (1, 5, 10)
HIST_EDGES = np.linspace(-1.0, 1.0, 401)


def platform(camera):
    return 'Ground' if int(camera) < 5 else 'Aerial'


def rank_bucket(rank):
    if rank == 1:
        return '1'
    if rank <= 5:
        return '2-5'
    if rank <= 10:
        return '6-10'
    if rank <= 20:
        return '11-20'
    if rank <= 50:
        return '21-50'
    if rank <= 100:
        return '51-100'
    return '>100'


def positive_bucket(count):
    if count <= 1:
        return '1'
    if count <= 4:
        return '2-4'
    if count <= 10:
        return '5-10'
    if count <= 50:
        return '11-50'
    return '>50'


def status(old, new):
    if old and new:
        return 'both_correct'
    if old and not new:
        return 'regressed'
    if not old and new:
        return 'improved'
    return 'both_wrong'


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _metadata(records):
    return {
        'pid': np.asarray([int(r['pid']) for r in records], dtype=np.int64),
        'camera': np.asarray([int(r['camera']) for r in records], dtype=np.int64),
        'modality': np.asarray([int(r['modality']) for r in records], dtype=np.int64),
        'platform': np.asarray([platform(r['camera']) for r in records]),
    }


def _ranking(order, similarity, q, gallery, gallery_records, top_k):
    """One Query's full valid order, using the official evaluator's rule."""
    valid = gallery['camera'][order] != q['camera']
    order = order[valid]
    relevant = gallery['pid'][order] == q['pid']
    positive_count = int(relevant.sum())
    if positive_count == 0:
        return None, []
    positions = np.flatnonzero(relevant) + 1
    first = int(positions[0])
    ap = float(np.mean(np.arange(1, positive_count + 1) / positions))
    inp = float(positive_count / positions[-1])
    candidate_mask = np.zeros(len(gallery['pid']), dtype=bool)
    candidate_mask[order] = True
    pos_mask = candidate_mask & (gallery['pid'] == q['pid'])
    neg_mask = candidate_mask & (gallery['pid'] != q['pid'])
    positive_similarity = float(np.max(similarity[pos_mask]))
    best_negative_index = int(np.argmax(np.where(neg_mask, similarity, -np.inf))) if neg_mask.any() else None
    negative_similarity = float(similarity[best_negative_index]) if best_negative_index is not None else None
    top = order[:top_k]
    top_wrong = gallery['pid'][top] != q['pid']
    result = {
        'first_positive_rank': first, 'rank1_hit': int(first <= 1),
        'rank5_hit': int(first <= 5), 'rank10_hit': int(first <= 10),
        'AP': ap, 'INP': inp, 'positive_count': positive_count,
        'gallery_count': len(order),
        'best_positive_similarity': positive_similarity,
        'best_negative_similarity': negative_similarity,
        'margin': positive_similarity - negative_similarity if negative_similarity is not None else None,
        'negative_ahead': first - 1,
        'negative_strictly_above_best_positive': int(np.sum(similarity[neg_mask] > positive_similarity)),
        'top5_wrong': int(top_wrong[:5].sum()),
        'top10_wrong': int(top_wrong[:10].sum()),
        'top10_wrong_pid_count': len(set(gallery['pid'][top[:10]][top_wrong[:10]].tolist())),
        'top1_gallery_id': gallery_records[int(order[0])]['image_id'],
        'top1_gallery_pid': int(gallery['pid'][order[0]]),
        'top1_gallery_modality': MODALITIES[int(gallery['modality'][order[0]])],
        'top1_gallery_platform': str(gallery['platform'][order[0]]),
        'top1_gallery_camera': int(gallery['camera'][order[0]]),
        'top1_similarity': float(similarity[order[0]]),
        'best_positive_modality': MODALITIES[int(gallery['modality'][order[positions[0] - 1]])],
        'best_negative_gallery_id': gallery_records[best_negative_index]['image_id']
            if best_negative_index is not None else '',
        'best_negative_gallery_pid': int(gallery['pid'][best_negative_index])
            if best_negative_index is not None else '',
        'best_negative_modality': MODALITIES[int(gallery['modality'][best_negative_index])]
            if best_negative_index is not None else '',
        'best_negative_platform': str(gallery['platform'][best_negative_index])
            if best_negative_index is not None else '',
        'best_negative_camera': int(gallery['camera'][best_negative_index])
            if best_negative_index is not None else '',
    }
    top_rows = []
    for position, j in enumerate(top, 1):
        g = gallery_records[int(j)]
        top_rows.append({
            'rank': position, 'gallery_id': g['image_id'], 'gallery_pid': int(g['pid']),
            'gallery_modality': MODALITIES[int(g['modality'])],
            'gallery_camera': int(g['camera']), 'gallery_platform': platform(g['camera']),
            'similarity': float(similarity[j]), 'correct': int(g['pid'] == q['pid']),
        })
    return result, top_rows


def _geometry_add(hist, similarity, q, gallery):
    valid = gallery['camera'] != q['camera']
    same_pid = gallery['pid'] == q['pid']
    same_modality = gallery['modality'] == q['modality']
    same_platform = gallery['platform'] == q['platform']
    groups = (
        ('positive_same_modality', valid & same_pid & same_modality),
        ('positive_cross_modality', valid & same_pid & ~same_modality),
        ('negative_same_modality', valid & ~same_pid & same_modality),
        ('negative_cross_modality', valid & ~same_pid & ~same_modality),
        ('positive_same_platform', valid & same_pid & same_platform),
        ('positive_cross_platform', valid & same_pid & ~same_platform),
        ('negative_same_platform', valid & ~same_pid & same_platform),
        ('negative_cross_platform', valid & ~same_pid & ~same_platform),
    )
    for name, mask in groups:
        if mask.any():
            hist[name] += np.histogram(similarity[mask], bins=HIST_EDGES)[0]


def _geometry_rows(hist, model_name):
    rows = []
    mids = (HIST_EDGES[:-1] + HIST_EDGES[1:]) / 2
    for name, counts in hist.items():
        n = int(counts.sum())
        if n == 0:
            continue
        cumulative = np.cumsum(counts)
        quantile = lambda p: float(mids[min(len(mids)-1, np.searchsorted(cumulative, p*n, side='left'))])
        rows.append({'model': model_name, 'group': name, 'pair_count': n,
                     'mean_approx': float((counts*mids).sum()/n),
                     'p01_approx': quantile(.01), 'p10_approx': quantile(.10),
                     'p25_approx': quantile(.25), 'median_approx': quantile(.50),
                     'p75_approx': quantile(.75), 'p90_approx': quantile(.90),
                     'p99_approx': quantile(.99),
                     'histogram_bin_width': float(HIST_EDGES[1]-HIST_EDGES[0])})
    return rows


def analyze_model(query_features, gallery_features, queries, gallery_records,
                  model_name, query_chunk=16, top_k=20, geometry=True):
    """Return official-equivalent full rankings and derived Query diagnostics."""
    if query_features.shape != (len(queries), 768) or gallery_features.shape != (len(gallery_records), 768):
        raise ValueError('Feature count/dimension does not match the evaluation manifest')
    if not np.isfinite(query_features).all() or not np.isfinite(gallery_features).all():
        raise ValueError('Non-finite feature cache')
    qnorm = np.linalg.norm(query_features, axis=1)
    gnorm = np.linalg.norm(gallery_features, axis=1)
    if not np.allclose(qnorm, 1, atol=1e-4) or not np.allclose(gnorm, 1, atol=1e-4):
        raise ValueError('Feature cache must contain L2-normalized pre-BN CLS')
    gallery = _metadata(gallery_records)
    qf = torch.from_numpy(np.asarray(query_features, dtype=np.float32))
    gf = torch.from_numpy(np.asarray(gallery_features, dtype=np.float32))
    gf_t = gf.T.contiguous()
    gf_sq = gf.square().sum(1)[None, :]
    rows, top_rows, competition = [], [], []
    hist = defaultdict(lambda: np.zeros(len(HIST_EDGES)-1, dtype=np.int64))
    skipped = []
    for start in range(0, len(queries), query_chunk):
        stop = min(start + query_chunk, len(queries))
        current = qf[start:stop]
        distances = current.square().sum(1, keepdim=True) + gf_sq
        distances.addmm_(current, gf_t, beta=1, alpha=-2)
        full_orders = torch.argsort(distances, dim=1).numpy()
        similarities = (current @ gf_t).numpy()
        for local, i in enumerate(range(start, stop)):
            q = dict(queries[i])
            q['platform'] = platform(q['camera'])
            sim = similarities[local]
            if geometry:
                _geometry_add(hist, sim, q, gallery)
            result, top = _ranking(full_orders[local], sim, q, gallery, gallery_records, top_k)
            if result is None:
                skipped.append({'query_index': i, 'query_id': q['image_id'],
                                'query_pid': int(q['pid']), 'query_camera': int(q['camera']),
                                'query_modality': MODALITIES[int(q['modality'])],
                                'reason': 'no_cross_camera_positive'})
                continue
            rows.append({'query_index': i, 'query_id': q['image_id'],
                         'query_pid': int(q['pid']), 'query_camera': int(q['camera']),
                         'query_modality': MODALITIES[int(q['modality'])],
                         'query_platform': q['platform'], **result})
            top_rows.extend({'model': model_name, 'query_index': i, 'query_id': q['image_id'], **item}
                            for item in top)
            for target in (1, 2, 3):
                target_order = full_orders[local][gallery['modality'][full_orders[local]] == target]
                sub, _ = _ranking(target_order, sim, q, gallery, gallery_records, 1)
                competition.append({
                    'model': model_name, 'query_index': i, 'query_id': q['image_id'],
                    'query_modality': MODALITIES[int(q['modality'])],
                    'gallery_modality': MODALITIES[target],
                    'valid': int(sub is not None),
                    'first_positive_rank': sub['first_positive_rank'] if sub else '',
                    'rank1_hit': sub['rank1_hit'] if sub else '',
                    'rank5_hit': sub['rank5_hit'] if sub else '',
                    'rank10_hit': sub['rank10_hit'] if sub else '',
                    'AP': sub['AP'] if sub else '',
                    'INP': sub['INP'] if sub else '',
                    'positive_count': sub['positive_count'] if sub else 0,
                    'best_positive_similarity': sub['best_positive_similarity'] if sub else '',
                    'best_negative_similarity': sub['best_negative_similarity'] if sub else '',
                    'margin': sub['margin'] if sub else '',
                    'top1_gallery_id': sub['top1_gallery_id'] if sub else '',
                    'top1_gallery_pid': sub['top1_gallery_pid'] if sub else '',
                    'top1_gallery_platform': sub['top1_gallery_platform'] if sub else '',
                })
            competition.append({
                'model': model_name, 'query_index': i, 'query_id': q['image_id'],
                'query_modality': MODALITIES[int(q['modality'])],
                'gallery_modality': 'Mixed', 'valid': 1,
                'first_positive_rank': result['first_positive_rank'],
                'rank1_hit': result['rank1_hit'],
                'rank5_hit': result['rank5_hit'],
                'rank10_hit': result['rank10_hit'],
                'AP': result['AP'], 'INP': result['INP'],
                'positive_count': result['positive_count'],
                'best_positive_similarity': result['best_positive_similarity'],
                'best_negative_similarity': result['best_negative_similarity'],
                'margin': result['margin'], 'top1_gallery_id': result['top1_gallery_id'],
                'top1_gallery_pid': result['top1_gallery_pid'],
                'top1_gallery_platform': result['top1_gallery_platform'],
                'top1_gallery_modality': result['top1_gallery_modality'],
            })
    return rows, top_rows, competition, _geometry_rows(hist, model_name), skipped


def aggregate(rows):
    if not rows:
        raise ValueError('No valid Query after same-camera filtering')
    return {'valid_queries': len(rows), 'rank1': float(np.mean([r['rank1_hit'] for r in rows])),
            'rank5': float(np.mean([r['rank5_hit'] for r in rows])),
            'rank10': float(np.mean([r['rank10_hit'] for r in rows])),
            'mAP': float(np.mean([r['AP'] for r in rows])),
            'mINP': float(np.mean([r['INP'] for r in rows]))}


def _group_metrics(pairs, name, value):
    selected = [p for p in pairs if value(p) == name]
    n = len(selected)
    if not n:
        return None
    old, new = [p[0] for p in selected], [p[1] for p in selected]
    a, b = aggregate(old), aggregate(new)
    return {'group': name, 'valid_query_count': n,
            **{'uad_'+k: a[k] for k in ('rank1', 'rank5', 'rank10', 'mAP', 'mINP')},
            **{'a1_'+k: b[k] for k in ('rank1', 'rank5', 'rank10', 'mAP', 'mINP')},
            **{'delta_'+k: b[k]-a[k] for k in ('rank1', 'rank5', 'rank10', 'mAP', 'mINP')},
            'mean_delta_best_positive_similarity': float(np.mean([
                p[1]['best_positive_similarity']-p[0]['best_positive_similarity'] for p in selected])),
            'mean_delta_best_negative_similarity': float(np.mean([
                p[1]['best_negative_similarity']-p[0]['best_negative_similarity']
                for p in selected if p[0]['best_negative_similarity'] is not None and
                p[1]['best_negative_similarity'] is not None])) if any(
                p[0]['best_negative_similarity'] is not None and p[1]['best_negative_similarity'] is not None
                for p in selected) else '',
            'mean_delta_margin': float(np.mean([
                p[1]['margin']-p[0]['margin'] for p in selected
                if p[0]['margin'] is not None and p[1]['margin'] is not None])) if any(
                p[0]['margin'] is not None and p[1]['margin'] is not None for p in selected) else ''}


def compare_models(old_rows, new_rows, old_competition, new_competition):
    old = {r['query_id']: r for r in old_rows}
    new = {r['query_id']: r for r in new_rows}
    if set(old) != set(new):
        raise ValueError('The two checkpoints have different valid Query sets')
    pairs = [(old[key], new[key]) for key in sorted(old, key=lambda key: old[key]['query_index'])]
    overall_old, overall_new = aggregate(old_rows), aggregate(new_rows)
    queries, transitions, regressed = [], [], []
    counts = {k: Counter() for k in RANK_KEYS}
    rank_matrix = Counter()
    contribution = {k: defaultdict(lambda: {'count': 0, 'sum_delta_ap': 0.,
                                             'sum_delta_inp': 0.}) for k in RANK_KEYS}
    n = len(pairs)
    for a, b in pairs:
        key = a['query_id']
        delta_ap = b['AP'] - a['AP']
        delta_inp = b['INP'] - a['INP']
        query = {'query_index': a['query_index'], 'query_id': key,
                 'query_pid': a['query_pid'], 'query_camera': a['query_camera'],
                 'query_modality': a['query_modality'], 'query_platform': a['query_platform']}
        for prefix, row in (('uad', a), ('a1', b)):
            query.update({prefix+'_'+k: v for k, v in row.items()
                          if k not in ('query_index', 'query_id', 'query_pid', 'query_camera',
                                       'query_modality', 'query_platform')})
        query.update(delta_AP=delta_ap, delta_INP=delta_inp,
                     delta_positive_similarity=b['best_positive_similarity']-a['best_positive_similarity'],
                     delta_negative_similarity=(b['best_negative_similarity']-a['best_negative_similarity'])
                     if b['best_negative_similarity'] is not None and a['best_negative_similarity'] is not None else '',
                     delta_margin=(b['margin']-a['margin'])
                     if b['margin'] is not None and a['margin'] is not None else '')
        queries.append(query)
        t = {'query_index': a['query_index'], 'query_id': key,
             'uad_first_positive_rank': a['first_positive_rank'],
             'a1_first_positive_rank': b['first_positive_rank'],
             'uad_rank_bucket': rank_bucket(a['first_positive_rank']),
             'a1_rank_bucket': rank_bucket(b['first_positive_rank']),
             'crossed_top10_outward': int(a['first_positive_rank'] <= 10 < b['first_positive_rank']),
             'crossed_top10_inward': int(b['first_positive_rank'] <= 10 < a['first_positive_rank'])}
        rank_matrix[(t['uad_rank_bucket'], t['a1_rank_bucket'])] += 1
        for k in RANK_KEYS:
            name = status(a['rank%d_hit' % k], b['rank%d_hit' % k])
            t['rank%d_status' % k] = name
            counts[k][name] += 1
            group = contribution[k][name]
            group['count'] += 1
            group['sum_delta_ap'] += delta_ap
            group['sum_delta_inp'] += delta_inp
        transitions.append(t)
        if t['rank1_status'] == 'regressed':
            regressed.append({**query, **t})
    comp = {(r['model'], r['query_id'], r['gallery_modality']): r
            for r in old_competition + new_competition}
    competition = []
    for a, b in pairs:
        for target in ('RGB', 'NIR', 'TIR', 'Mixed'):
            ca = comp[('UAD', a['query_id'], target)]
            cb = comp[('A1', a['query_id'], target)]
            row = {'query_id': a['query_id'], 'query_index': a['query_index'],
                                'query_modality': a['query_modality'], 'gallery_modality': target,
                                **{'uad_'+k: v for k, v in ca.items() if k not in
                                   ('model', 'query_id', 'query_index', 'query_modality', 'gallery_modality')},
                                **{'a1_'+k: v for k, v in cb.items() if k not in
                                   ('model', 'query_id', 'query_index', 'query_modality', 'gallery_modality')}}
            if ca['valid'] and cb['valid']:
                row['rank1_status'] = status(ca['rank1_hit'], cb['rank1_hit'])
                row['delta_positive_similarity'] = cb['best_positive_similarity']-ca['best_positive_similarity']
                row['delta_negative_similarity'] = (
                    cb['best_negative_similarity']-ca['best_negative_similarity']
                    if cb['best_negative_similarity'] is not None and
                    ca['best_negative_similarity'] is not None else '')
                row['delta_margin'] = (cb['margin']-ca['margin']) if cb['margin'] is not None and ca['margin'] is not None else ''
            competition.append(row)
    grouped = []
    group_specs = (
        ('query_modality', lambda p: p[0]['query_modality']),
        ('query_platform', lambda p: p[0]['query_platform']),
        ('modality_platform', lambda p: p[0]['query_modality']+'_'+p[0]['query_platform']),
        ('query_camera', lambda p: str(p[0]['query_camera'])),
        ('positive_count', lambda p: positive_bucket(p[0]['positive_count'])),
        ('uad_first_rank', lambda p: rank_bucket(p[0]['first_positive_rank'])),
        ('a1_top1_wrong_source', lambda p: p[1]['top1_gallery_modality']+'_'+p[1]['top1_gallery_platform']
         if not p[1]['rank1_hit'] else 'A1_correct'),
        ('query_modality_x_wrong_gallery_modality', lambda p:
         p[0]['query_modality']+'->'+p[1]['top1_gallery_modality']
         if not p[1]['rank1_hit'] else 'A1_correct'),
        ('query_platform_x_wrong_gallery_platform', lambda p:
         p[0]['query_platform']+'->'+p[1]['top1_gallery_platform']
         if not p[1]['rank1_hit'] else 'A1_correct'),
    )
    for group_type, fn in group_specs:
        for name in sorted(set(fn(p) for p in pairs)):
            result = _group_metrics(pairs, name, fn)
            if result:
                grouped.append({'group_type': group_type, **result})
    pair_summary = []
    for query_modality in ('RGB', 'NIR', 'TIR'):
        for target in ('RGB', 'NIR', 'TIR', 'Mixed'):
            selected = [r for r in competition if r['query_modality'] == query_modality
                        and r['gallery_modality'] == target and r['uad_valid'] and r['a1_valid']]
            if not selected:
                pair_summary.append({'query_modality': query_modality,
                                     'gallery_modality': target, 'valid_query_count': 0})
                continue
            row = {'query_modality': query_modality, 'gallery_modality': target,
                   'valid_query_count': len(selected)}
            for metric in ('rank1_hit', 'rank5_hit', 'rank10_hit', 'AP', 'INP'):
                row['uad_'+metric] = float(np.mean([r['uad_'+metric] for r in selected]))
                row['a1_'+metric] = float(np.mean([r['a1_'+metric] for r in selected]))
                row['delta_'+metric] = row['a1_'+metric]-row['uad_'+metric]
            pair_summary.append(row)
    transition_summary = {}
    for k in RANK_KEYS:
        item = counts[k]
        transition_summary[str(k)] = {
            **{name: {'count': item[name], 'fraction': item[name]/n}
               for name in ('both_correct', 'regressed', 'improved', 'both_wrong')},
            'net_delta': (item['improved']-item['regressed'])/n,
            'actual_delta': overall_new['rank%d' % k]-overall_old['rank%d' % k]}
        if not np.isclose(transition_summary[str(k)]['net_delta'],
                          transition_summary[str(k)]['actual_delta'], atol=1e-10):
            raise AssertionError('Rank transition accounting disagrees with aggregate')
    contribution_summary = {}
    for k, groups in contribution.items():
        contribution_summary[str(k)] = {}
        for name, item in groups.items():
            contribution_summary[str(k)][name] = {
                'count': item['count'], 'fraction': item['count']/n,
                'mean_delta_AP': item['sum_delta_ap']/item['count'],
                'mean_delta_INP': item['sum_delta_inp']/item['count'],
                'contribution_to_delta_mAP': item['sum_delta_ap']/n,
                'contribution_to_delta_mINP': item['sum_delta_inp']/n,
                'contribution_to_delta_Rank': (int(name == 'improved')-int(name == 'regressed'))*item['count']/n}
        if not np.isclose(sum(x['contribution_to_delta_mAP'] for x in contribution_summary[str(k)].values()),
                          overall_new['mAP']-overall_old['mAP'], atol=1e-10):
            raise AssertionError('AP contributions disagree with aggregate')
        if not np.isclose(sum(x['contribution_to_delta_mINP'] for x in contribution_summary[str(k)].values()),
                          overall_new['mINP']-overall_old['mINP'], atol=1e-10):
            raise AssertionError('INP contributions disagree with aggregate')
    rank1_regressed_pairs = [(a, b) for a, b in pairs if a['rank1_hit'] and not b['rank1_hit']]
    signs = Counter()
    wrong_sources = Counter()
    mixed_conflict = Counter()
    for a, b in rank1_regressed_pairs:
        delta_pos = b['best_positive_similarity']-a['best_positive_similarity']
        delta_neg = (b['best_negative_similarity']-a['best_negative_similarity'])
        signs[('positive_decreased' if delta_pos < -1e-6 else 'positive_not_decreased',
               'negative_increased' if delta_neg > 1e-6 else 'negative_not_increased')] += 1
        wrong_sources[(a['query_modality'], b['top1_gallery_modality'],
                       a['query_platform'], b['top1_gallery_platform'],
                       str(b['top1_gallery_camera']))] += 1
        for target in ('RGB', 'NIR', 'TIR'):
            row = comp[('A1', a['query_id'], target)]
            if row['valid'] and row['rank1_hit']:
                mixed_conflict[target] += 1
        if any(comp[('A1', a['query_id'], target)]['valid'] and
               comp[('A1', a['query_id'], target)]['rank1_hit']
               for target in ('RGB', 'NIR', 'TIR')):
            mixed_conflict['any_single_gallery_correct'] += 1
    if len(rank1_regressed_pairs) != counts[1]['regressed']:
        raise AssertionError('Rank-1 regressed Query count mismatch')
    summary = {'UAD': overall_old, 'A1': overall_new,
               'delta': {key: overall_new[key]-overall_old[key]
                         for key in ('rank1', 'rank5', 'rank10', 'mAP', 'mINP')},
               'transitions': transition_summary,
               'rank_bucket_matrix': [{'uad_bucket': a, 'a1_bucket': b, 'count': count}
                                      for (a, b), count in sorted(rank_matrix.items())],
               'rank_k_AP_INP_contributions': contribution_summary,
               'modality_pair_summary': pair_summary,
               'crossed_top10_outward': sum(t['crossed_top10_outward'] for t in transitions),
               'crossed_top10_inward': sum(t['crossed_top10_inward'] for t in transitions),
               'regressed_at_rank1_5_10': sum(all(t['rank%d_status' % k] == 'regressed'
                                                  for k in RANK_KEYS) for t in transitions),
               'rank1_regressed_similarity_signs': [
                   {'positive_change': p, 'negative_change': q, 'count': c}
                   for (p, q), c in sorted(signs.items())],
               'rank1_regressed_wrong_sources': [
                   {'query_modality': qm, 'gallery_modality': gm,
                    'query_platform': qp, 'gallery_platform': gp,
                    'gallery_camera': gc, 'count': c}
                   for (qm, gm, qp, gp, gc), c in sorted(wrong_sources.items())],
               'rank1_regressed_single_gallery_correct_but_mixed_wrong': dict(mixed_conflict)}
    return queries, transitions, regressed, competition, grouped, summary
