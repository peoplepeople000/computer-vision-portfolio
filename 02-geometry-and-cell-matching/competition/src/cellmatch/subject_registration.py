"""Label-free affine search using reliable sections of the same subject.

Only predicted transforms supply the affine bank. Target-region transforms are
excluded from the bank. No ground-truth masks or correspondences are read here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .data import read_rows, parse_instances, image_path, image_shape, split_id
from .density_registration import raster
from .matching import centroids, mutual_neighbors


def linear_bank(records, sample_id):
    subject = split_id(sample_id)[0]
    eligible = [r for r in records if r['sample_id'] != sample_id
                and split_id(r['sample_id'])[0] == subject
                and r.get('status') == 'registered' and r.get('votes', 0) >= 4
                and r.get('matrix') is not None]
    eligible.sort(key=lambda r: r.get('support', 0), reverse=True)
    bank = []
    for r in eligible:
        linear = np.asarray(r['matrix'], dtype=float)[:, :2]
        if not np.isfinite(linear).all() or np.linalg.det(linear) <= 0:
            continue
        if all(np.linalg.norm(linear - a) > .025 for _, a in bank):
            bank.append((r['sample_id'], linear))
    return bank


def support(source, target, matrix):
    a, b, errors = mutual_neighbors(source @ matrix[:, :2].T + matrix[:, 2], target, 5.)
    return float(np.exp(-.5 * (errors / 2.5) ** 2).sum()), len(a)


def search(source, target, shape, bank, grid=2.):
    if min(len(source), len(target)) < 8 or not bank:
        return []
    field = raster(target / grid, tuple(int(np.ceil(n / grid)) + 1 for n in shape), 1.5)
    candidates = []
    for anchor_id, anchor in bank:
        for angle in (-4., -2., 0., 2., 4.):
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            for scale in (.98, 1., 1.02):
                linear = rotation @ anchor * scale
                points = source @ linear.T / grid
                lower = np.floor(points.min(0)) - 6
                points -= lower
                size = np.ceil(points.max(0)).astype(int) + 7
                template = raster(points, (int(size[1]), int(size[0])), 1.5)
                if any(a > b for a, b in zip(template.shape, field.shape)):
                    continue
                scores = cv2.matchTemplate(field, template, cv2.TM_CCOEFF_NORMED)
                for _ in range(3):
                    _, score, _, xy = cv2.minMaxLoc(scores)
                    matrix = np.column_stack([linear, (np.asarray(xy) - lower) * grid])
                    candidates.append((float(score), matrix, anchor_id, anchor))
                    x, y = xy
                    scores[max(0, y-8):y+9, max(0, x-8):x+9] = -1
    candidates.sort(key=lambda v: v[0], reverse=True)
    results = []
    for correlation, matrix, anchor_id, anchor in candidates[:60]:
        for radius in (8., 6., 5.):
            a, b, _ = mutual_neighbors(source @ matrix[:, :2].T + matrix[:, 2], target, radius)
            if len(a) < 6:
                break
            refined, inliers = cv2.estimateAffine2D(
                source[a].astype(np.float32), target[b].astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=3.5, maxIters=3000, confidence=.995)
            if refined is None or inliers.sum() < 6:
                break
            if np.linalg.det(refined[:, :2]) <= 0 or np.linalg.norm(refined[:, :2] - anchor) > .12:
                break
            matrix = refined
        value, count = support(source, target, matrix)
        results.append({'matrix': matrix.tolist(), 'support': value, 'inliers': count,
                        'correlation': correlation, 'anchor_id': anchor_id})
    return sorted(results, key=lambda r: r['support'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='outputs/stage8_iv_quality_capacity/joint/predicted_all.csv')
    parser.add_argument('--geometry', default='outputs/stage7_consensus/predicted_all_manifest.json')
    parser.add_argument('--split', choices=['training', 'hidden_test'], default='training')
    parser.add_argument('--output', default='outputs/stage9_subject_geometry')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    root = Path('data/raw/Project_2_Dataset')
    rows = read_rows(args.predictions)
    manifest = json.loads(Path(args.geometry).read_text())
    records = manifest['regions'] if isinstance(manifest, dict) else manifest
    if {r['sample_id'] for r in records} != set(rows):
        raise ValueError('Prediction and registration IDs differ')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'prediction_sha256': hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
              'geometry_sha256': hashlib.sha256(Path(args.geometry).read_bytes()).hexdigest(),
              'policy': 'Keep >=4-vote transforms; otherwise same-subject predicted affine bank, >=10 inliers, >1.2x previous support and >=7 support. No labels.'}
    config_path = out / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Output configuration changed')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    cache = out / 'regions'
    cache.mkdir(exist_ok=True)
    updated = []
    for r in records:
        sid = r['sample_id']
        saved = cache / f'{sid}.json'
        if saved.exists():
            updated.append(json.loads(saved.read_text()))
            continue
        result = copy.deepcopy(r)
        result['original_registration'] = r
        result['replaced'] = False
        bank = linear_bank(records, sid)
        result['anchor_ids'] = [key for key, _ in bank]
        if r.get('votes', 0) < 4 and bank:
            masks = [parse_instances(rows[sid], m, image_shape(image_path(root, args.split, sid, m)))
                     for m in ('invivo', 'exvivo')]
            points = [centroids(m)[1] for m in masks]
            cv2.setRNGSeed(2026)
            candidates = search(*points, image_shape(image_path(root, args.split, sid, 'exvivo')), bank)
            result['candidates'] = candidates
            old_support = support(*points, np.asarray(r['matrix']))[0] if r.get('matrix') else 0.
            result['current_mask_original_support'] = old_support
            if candidates and candidates[0]['support'] > max(7., 1.2 * old_support) and candidates[0]['inliers'] >= 10:
                result.update(candidates[0])
                result.update(status='registered', selected_source='subject_linear_bank', replaced=True)
        saved.write_text(json.dumps(result, indent=2)+'\n')
        updated.append(result)
        print(sid, 'replaced' if result['replaced'] else 'preserved', len(bank), flush=True)
    result_manifest = {'regions': updated, 'source_manifest': args.geometry, 'configuration': config}
    (out / 'predicted_all_manifest.json').write_text(json.dumps(result_manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
