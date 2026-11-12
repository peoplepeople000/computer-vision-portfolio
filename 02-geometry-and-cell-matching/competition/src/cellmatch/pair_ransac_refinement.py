"""Conservative affine refitting from predicted cell pairs, without test labels.

Motivated by the Lecture 3 consensus-refit workflow and Lecture 5 cell matching
example. Unlike the demonstration, candidate correspondences are predicted.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from .data import read_rows, write_rows, split_id
from .stage3 import FOLDS
from .star_pairing import describe
from .joint_pair_assignment import candidate_features, probability, assign_edges
from .preserve_pair_masks import merge_row
from .cli import evaluate


def refine(source, target, initial):
    source, target = np.asarray(source, float), np.asarray(target, float)
    report = {'accepted': False, 'candidate_pairs': len(source)}
    if initial is None or len(source) < 8:
        return initial, dict(report, reason='insufficient predicted pairs')
    initial = np.asarray(initial, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('Expected matching Nx2 arrays')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite candidate points')
    cv2.setRNGSeed(2069)
    matrix, mask = cv2.estimateAffine2D(source.astype(np.float32), target.astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=3., maxIters=10000,
                confidence=.999, refineIters=20)
    if matrix is None or mask is None or int(mask.sum()) < 8:
        return initial, dict(report, reason='insufficient RANSAC consensus')
    inliers = mask.ravel().astype(bool)
    spread = np.linalg.svd(source[inliers]-source[inliers].mean(0), compute_uv=False)/np.sqrt(inliers.sum())
    if spread.min() < 8.:
        return initial, dict(report, reason='poor spatial support')
    if np.linalg.det(matrix[:, :2]) <= 0 or np.linalg.norm(matrix[:, :2]-initial[:, :2]) > .15:
        return initial, dict(report, reason='incompatible affine change')
    before = source@initial[:, :2].T+initial[:, 2]
    after = source@matrix[:, :2].T+matrix[:, 2]
    displacement = float(np.quantile(np.linalg.norm(after-before, axis=1), .95))
    if displacement > 8.:
        return initial, dict(report, reason='excessive displacement', displacement=displacement)
    errors = [np.linalg.norm(points-target, axis=1) for points in (before, after)]
    support = [int((e < 3.).sum()) for e in errors]
    costs = [float(np.minimum(e**2, 9.).sum()) for e in errors]
    report.update(support_before=support[0], support_after=support[1],
                  cost_before=costs[0], cost_after=costs[1], displacement=displacement)
    if support[1] < support[0] or costs[1] >= costs[0]-.01:
        return initial, dict(report, reason='no robust residual improvement')
    return matrix, dict(report, accepted=True, reason='predicted-pair consensus improved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', choices=list(FOLDS), required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/stage25_pair_ransac'))
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(2)
    subject = FOLDS[args.fold]
    reference_path = Path('outputs/stage13_native_relative/preserved/predicted_all.csv')
    reference = read_rows(reference_path)
    expected = {s for s in reference if split_id(s)[0] == subject}
    source = Path('outputs/stage22_star_pairing')/args.fold
    rows = read_rows(source/'raw.csv')
    if set(rows) != expected:
        raise ValueError('Incomplete radial pairing fold')
    state = torch.load(source/'pairs.pt', map_location='cpu', weights_only=False)
    if state['held_out'] != subject or set(state['train_ids']) != set(reference)-expected:
        raise ValueError('Invalid pair model training partition')
    model = torch.nn.Linear(state['feature_count'], 1); model.load_state_dict(state['state_dict'])
    fitted = model.eval(), state['mean'], state['sd']; policy = state['selected']
    geometry = Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    manifest = json.loads(geometry.read_text())
    matrices = {r['sample_id']: np.asarray(r['matrix']) for r in manifest['regions']
                if r.get('matrix') and r.get('status') == 'registered'}
    output = args.output/args.fold
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output to preserve the prior experiment')
    output.mkdir(parents=True, exist_ok=True)
    raw, merged, diagnostics = [], [], []
    for sid, original in rows.items():
        row = copy.deepcopy(original)
        masks, features = describe(row, Path('outputs/stage8_iv')/args.fold/'maps'/f'{sid}_invivo.npz',
                                   Path('outputs/stage20_star_ex')/args.fold/'maps'/f'{sid}_exvivo.npz')
        centers = [{k: v.coordinates().mean(0) for k, v in m.items()} for m in masks]
        matrix = matrices.get(sid); history = []
        for iteration in range(2):
            pairs, x = candidate_features(masks, features, matrix)
            chosen = assign_edges(pairs, probability(fitted, x), policy['threshold'], x[:, 18], policy['radius'])
            a = np.asarray([centers[0][i] for i, j in chosen]).reshape(-1, 2)
            b = np.asarray([centers[1][j] for i, j in chosen]).reshape(-1, 2)
            matrix, report = refine(a, b, matrix); history.append(report)
            if not report['accepted']:
                break
        pairs, x = candidate_features(masks, features, matrix)
        row['match_pairs'] = assign_edges(pairs, probability(fitted, x), policy['threshold'], x[:, 18], policy['radius'])
        raw.append(row); merged.append(merge_row(row, reference[sid]))
        diagnostics.append({'sample_id': sid, 'matrix': None if matrix is None else matrix.tolist(),
                            'iterations': history, 'pairs_before': len(original['match_pairs']),
                            'pairs_after': len(row['match_pairs'])})
    config = {'fold': args.fold, 'held_out': subject, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [source/'raw.csv', source/'pairs.pt', geometry, reference_path]},
              'regions': diagnostics, 'annotation_use': 'None in transform refinement, assignment, or merge. Labels are read only by the evaluator.'}
    (output/'manifest.json').write_text(json.dumps(config, indent=2)+'\n')
    for name, values in [('raw', raw), ('merged', merged)]:
        path = output/f'{name}.csv'; write_rows(values, path)
        evaluate(SimpleNamespace(data=Path('data/raw/Project_2_Dataset'), predictions=path,
                                 subject=subject, output=output/f'{name}_metrics.json'))


if __name__ == '__main__':
    main()
