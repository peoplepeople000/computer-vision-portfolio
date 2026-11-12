"""Train-only size and quality calibration for adapted CPDINO instances."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi
import torch

from .data import read_rows, write_rows, parse_instances, split_id
from .cellpose_finetune import instance_labels, FOLDS
from .rle import labels_to_instances
from .relative_ex_quality import relative_features
from .ex_joint_calibration import morphology_features
from .metrics import segmentation_score
from .stage3 import calibrate_quality, quality_probabilities
from .cli import evaluate

OFFSETS = (0., 1., 1.5, 2., 2.5)


def grow_labels(labels, radius):
    """Assign nearby background to its nearest original cell, without overlaps."""
    if labels.ndim != 2 or labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('Expected a nonnegative 2D instance label image')
    if radius < 0 or not np.isfinite(radius):
        raise ValueError('Expected a finite nonnegative radius')
    if radius == 0 or not labels.any():
        return labels.copy()
    distance, nearest = ndi.distance_transform_edt(labels == 0, return_indices=True)
    result = labels.copy()
    add = (labels == 0) & (distance <= radius)
    result[add] = labels[tuple(nearest[:, add])]
    return result


def proposals(row, maps, radius):
    labels = instance_labels(row['exvivo_instances'], maps.shape[1:])
    encoded = labels_to_instances(grow_labels(labels, radius), 'CPG')
    masks = parse_instances({'exvivo_instances': encoded}, 'exvivo', maps.shape[1:])
    keys, features = morphology_features(masks, maps)
    return encoded, masks, keys, relative_features(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/stage33_cpdino_adapted'))
    parser.add_argument('--output', type=Path, default=Path('outputs/stage36_cpdino_calibrated'))
    parser.add_argument('--fold', choices=list(FOLDS), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    folder = args.root/args.fold
    config = json.loads((folder/'configuration.json').read_text())
    complete = json.loads((folder/'completion.json').read_text())
    complete = dict(checkpoint=str(folder/'model.pt'), sha256=complete['checkpoint_sha256'])
    subject = FOLDS[args.fold]
    ids = sorted(s for s in truth if split_id(s)[0] != subject)
    if config['held_out'] != subject or config['train_ids'] != ids:
        raise ValueError('Invalid outer-fold training partition')
    if hashlib.sha256(Path(complete['checkpoint']).read_bytes()).hexdigest() != complete['sha256']:
        raise ValueError('Trained checkpoint changed')
    missing = [s for s in truth if not (folder/'regions'/f'{s}.json').exists()
               or not (folder/'maps'/f'{s}_exvivo.npz').exists()]
    if missing:
        raise ValueError(f'Incomplete development/heldout predictions: {len(missing)} missing')
    out = args.output/args.fold; out.mkdir(parents=True, exist_ok=True)
    if (out/'quality.pt').exists():
        raise FileExistsError('Preserve existing calibration; choose a new output')
    records = {r: [] for r in OFFSETS}
    for sid in ids:
        row = json.loads((folder/'regions'/f'{sid}.json').read_text())
        maps = np.load(folder/'maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
        actual = parse_instances(truth[sid], 'exvivo', maps.shape[1:])
        for radius in OFFSETS:
            _, masks, keys, features = proposals(row, maps, radius)
            mapping = segmentation_score(masks, actual)['pred_to_gt']
            records[radius].append({'sample_id': sid, 'subject': split_id(sid)[0], 'features': [None, features],
                'quality': np.array([k in mapping for k in keys]), 'gt_count': len(actual),
                'ious': np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0. for k in keys])})
        print('Cellpose calibration records', sid, flush=True)
    candidates = []
    for radius in OFFSETS:
        fitted, threshold, measurements = calibrate_quality(records[radius])
        score = next(r['mean_pq'] for r in measurements if r['threshold'] == threshold)
        candidates.append((score, radius, threshold, fitted, measurements))
        print('Cellpose calibration', radius, threshold, score, flush=True)
    score, radius, threshold, fitted, measurements = max(candidates, key=lambda r: r[0])
    torch.save({'quality': fitted[0].state_dict(), 'mean': fitted[1], 'sd': fitted[2],
                'threshold': threshold, 'radius': radius, 'held_out': subject, 'train_ids': ids}, out/'quality.pt')
    report = {'held_out': subject, 'train_ids': ids, 'checkpoint_sha256': complete['sha256'],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_root': str(folder), 'radius': radius, 'quality_threshold': threshold,
              'options': [{'inner_pq': s, 'radius': r, 'threshold': t, 'measurements': m}
                          for s, r, t, _, m in candidates],
              'limitation': 'Inner quality mouse holdout only; segmentation trained on both development mice. Outer labels excluded from calibration.'}
    (out/'calibration.json').write_text(json.dumps(report, indent=2)+'\n')
    results = []
    for sid in sorted(set(truth) - set(ids)):
        row = json.loads((folder/'regions'/f'{sid}.json').read_text())
        maps = np.load(folder/'maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
        encoded, _, keys, features = proposals(row, maps, radius)
        keep = {k for k, p in zip(keys, quality_probabilities(fitted, features)) if p >= threshold}
        value = copy.deepcopy(row); value['match_pairs'] = []
        value['exvivo_instances'] = {k: v for k, v in encoded.items() if k in keep}
        results.append(value)
    write_rows(results, out/'heldout.csv')
    evaluate(SimpleNamespace(data=root, predictions=out/'heldout.csv', subject=subject, output=out/'metrics.json'))


if __name__ == '__main__':
    main()
