"""Development-mouse-only confidence calibration for radial EX proposals."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .star_geometry import render_candidates
from .cellpose_baseline import confidence_maps
from .rle import labels_to_instances
from .data import read_rows, write_rows, parse_instances, split_id
from .stage3 import FOLDS, calibrate_quality, quality_probabilities
from .relative_ex_quality import relative_features
from .ex_joint_calibration import morphology_features
from .metrics import segmentation_score
from .cli import evaluate

THRESHOLDS = (.15, .25, .35, .45, .55)


def proposals(path, threshold, radius_scale=1.):
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError('Expected a finite positive radial scale')
    with np.load(path) as values:
        probability = values['probability'].astype(np.float32)
        labels = render_candidates(values['points'], values['radii'].astype(np.float32)*radius_scale,
                                   values['scores'].astype(np.float32), probability.shape,
                                   threshold=threshold)
    clipped = np.clip(probability, 1e-6, 1-1e-6)
    maps = confidence_maps(labels, np.log(clipped/(1-clipped)))
    encoded = labels_to_instances(labels, 'STAR')
    masks = parse_instances({'exvivo_instances': encoded}, 'exvivo', labels.shape)
    keys, features = morphology_features(masks, maps)
    return encoded, masks, keys, relative_features(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/stage20_star_ex'))
    parser.add_argument('--output', type=Path, default=Path('outputs/stage21_star_calibrated'))
    parser.add_argument('--fold', choices=list(FOLDS), required=True)
    parser.add_argument('--radius-scales', nargs='+', type=float, default=[1.])
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in args.radius_scales):
        parser.error('Radial scales must be finite and positive')
    torch.set_num_threads(2)
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    subject = FOLDS[args.fold]
    ids = sorted(s for s in truth if split_id(s)[0] != subject)
    folder = args.root/args.fold
    config = json.loads((folder/'training_config.json').read_text())
    state = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
    if any(r['held_out'] != subject or r['train_ids'] != ids for r in (config, state)):
        raise ValueError('Invalid outer training partition')
    if state['config'] != config or state['step'] != config['steps']:
        raise ValueError('Incomplete or mismatched training')
    if state['architecture'] != 'resnet18_star32':
        raise ValueError('Expected radial model')
    geometry_hash = hashlib.sha256(Path(__file__).with_name('star_geometry.py').read_bytes()).hexdigest()
    if config['geometry_sha256'] != geometry_hash:
        raise ValueError('Radial decoder changed since candidate generation')
    cache = {s: folder/'candidates'/f'{s}.npz' for s in truth}
    if any(not p.exists() for p in cache.values()):
        raise ValueError('Complete development and held-out candidate caches required')
    output = args.output/args.fold
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve previous calibration; choose a new output')
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        'held_out': subject, 'train_ids': ids, 'input_root': str(folder),
        'model_sha256': hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),
        'candidate_sha256': {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in cache.items()},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'geometry_sha256': geometry_hash,
        'radius_scales': args.radius_scales, 'probability_thresholds': THRESHOLDS,
        'limitation': 'Outer mouse excluded. Quality uses inner mouse holdout; segmentation sees both development mice.'}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    options = []
    for radius_scale, threshold in [(scale, threshold) for scale in args.radius_scales for threshold in THRESHOLDS]:
        records = []
        for sid in ids:
            _, masks, keys, features = proposals(cache[sid], threshold, radius_scale)
            with np.load(cache[sid]) as values:
                shape = values['probability'].shape
            actual = parse_instances(truth[sid], 'exvivo', shape)
            mapping = segmentation_score(masks, actual)['pred_to_gt']
            records.append({'sample_id': sid, 'subject': split_id(sid)[0], 'features': [None, features],
                            'quality': np.array([k in mapping for k in keys]), 'gt_count': len(actual),
                            'ious': np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0. for k in keys])})
        if any(not sum(len(r['quality']) for r in records if r['subject'] == mouse)
               for mouse in {r['subject'] for r in records}):
            print('Skip empty development proposal threshold', threshold, flush=True)
            continue
        fitted, quality, measurements = calibrate_quality(records)
        score = next(r['mean_pq'] for r in measurements if r['threshold'] == quality)
        options.append((score, threshold, quality, radius_scale, fitted, measurements))
        print('Star calibration', radius_scale, threshold, quality, score, flush=True)
    if not options:
        raise ValueError('No usable development proposals')
    score, threshold, quality, radius_scale, fitted, _ = max(options, key=lambda r: r[0])
    torch.save({'quality': fitted[0].state_dict(), 'mean': fitted[1], 'sd': fitted[2],
                'threshold': quality, 'probability_threshold': threshold,
                'radius_scale': radius_scale,
                'held_out': subject, 'train_ids': ids}, output/'quality.pt')
    report = {'probability_threshold': threshold, 'quality_threshold': quality, 'radius_scale': radius_scale, 'inner_pq': score,
              'options': [{'inner_pq': s, 'probability_threshold': t, 'quality_threshold': q,
                           'radius_scale': scale, 'measurements': m} for s, t, q, scale, _, m in options]}
    (output/'calibration.json').write_text(json.dumps(report, indent=2)+'\n')
    reference = read_rows('outputs/stage13_native_relative/preserved/predicted_all.csv')
    rows = []
    for sid in sorted(set(truth)-set(ids)):
        encoded, _, keys, features = proposals(cache[sid], threshold, radius_scale)
        keep = {k for k, p in zip(keys, quality_probabilities(fitted, features)) if p >= quality}
        row = copy.deepcopy(reference[sid]); row['match_pairs'] = []
        row['exvivo_instances'] = {k: v for k, v in encoded.items() if k in keep}
        rows.append(row)
    write_rows(rows, output/'heldout.csv')
    evaluate(SimpleNamespace(data=root, predictions=output/'heldout.csv', subject=subject,
                             output=output/'metrics.json'))


if __name__ == '__main__':
    main()
