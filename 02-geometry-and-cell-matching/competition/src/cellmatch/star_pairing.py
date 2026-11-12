"""Learn radial-mask pair credit on development mice, then preserve paired masks.

Development transforms use verified development pairs. Evaluation transforms are
the already audited image-only geometry, never evaluation annotations.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .data import read_rows, write_rows, parse_instances, split_id
from .stage3 import FOLDS, quality_probabilities
from .ex_joint_calibration import morphology_features
from .joint_pair_assignment import candidate_features, fit, probability, assign_edges
from .preserve_pair_masks import merge_row
from .metrics import segmentation_score
from .cli import evaluate


def describe(row, iv_path, ex_path):
    masks, features = [], []
    for modality, path in zip(('invivo', 'exvivo'), (iv_path, ex_path)):
        with np.load(path) as values:
            maps = values['maps'].astype(np.float32)
        masks.append(parse_instances(row, modality, maps.shape[1:]))
        features.append(morphology_features(masks[-1], maps)[1])
    return masks, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/stage20_star_ex'))
    parser.add_argument('--output', type=Path, default=Path('outputs/stage22_star_pairing'))
    parser.add_argument('--fold', choices=list(FOLDS), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    subject = FOLDS[args.fold]
    ids = sorted(s for s in truth if split_id(s)[0] != subject)
    source = args.root/args.fold
    iv_folder = Path('outputs/stage8_iv')/args.fold
    iv_quality_path = Path('outputs/stage8_iv_quality_capacity')/f'{args.fold}.pt'
    model_paths = [source/'model.pt', iv_folder/'model.pt', iv_quality_path]
    for path in model_paths:
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['held_out'] != subject or set(state['train_ids']) != set(ids):
            raise ValueError(f'Wrong training partition: {path}')
    qstate = state
    qmodel = nn.Linear(30, 1); qmodel.load_state_dict(qstate['quality'])
    quality = qmodel.eval(), qstate['mean'], qstate['sd']
    base = read_rows(iv_folder/'development.csv')
    if set(base) != set(ids):
        raise ValueError('Wrong development prediction coverage')
    reference_path = Path('outputs/stage13_native_relative/preserved/predicted_all.csv')
    reference = read_rows(reference_path)
    geometry = Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    manifest = json.loads(geometry.read_text())
    matrices = {r['sample_id']: np.asarray(r['matrix']) for r in manifest['regions']
                if r.get('matrix') and r.get('status') == 'registered'}
    files = model_paths+[reference_path, geometry, iv_folder/'development.csv']
    for sid in truth:
        files.extend([source/'maps'/f'{sid}_exvivo.npz', source/'regions'/f'{sid}.json',
                      iv_folder/'maps'/f'{sid}_invivo.npz'])
    if any(not p.exists() for p in files):
        raise ValueError('Incomplete fold caches')
    output = args.output/args.fold
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve previous pairing result; choose a new output')
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'held_out': subject, 'train_ids': ids,
                  'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ['star_pairing.py', 'joint_pair_assignment.py', 'pair_quality.py',
                                                 'ex_joint_calibration.py', 'preserve_pair_masks.py']},
                  'policy': 'Fixed raw radial proposals; learn pair credit, then preserve their paired EX masks and add non-overlapping baseline segmentation masks.',
                  'limitation': 'Inner mouse holdout for pair classifier. Segmentation sees both development mice; development transforms use development annotations.'}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    records = []
    for sid in ids:
        row = copy.deepcopy(base[sid])
        row['exvivo_instances'] = json.loads((source/'regions'/f'{sid}.json').read_text())['exvivo_instances']
        iv_path = iv_folder/'maps'/f'{sid}_invivo.npz'
        ex_path = source/'maps'/f'{sid}_exvivo.npz'
        masks, features = describe(row, iv_path, ex_path)
        keep = {k for k, p in zip(masks[0], quality_probabilities(quality, features[0]))
                if p >= qstate['threshold']}
        row['invivo_instances'] = {k: v for k, v in row['invivo_instances'].items() if k in keep}
        masks, features = describe(row, iv_path, ex_path)
        if len(truth[sid]['match_pairs']) < 5:
            continue
        actual = []
        for modality, path in zip(('invivo', 'exvivo'), (iv_path, ex_path)):
            with np.load(path) as values:
                shape = values['maps'].shape[1:]
            actual.append(parse_instances(truth[sid], modality, shape))
        verified = set(map(tuple, truth[sid]['match_pairs']))
        a = np.array([actual[0][i].coordinates().mean(0) for i, j in sorted(verified)])
        b = np.array([actual[1][j].coordinates().mean(0) for i, j in sorted(verified)])
        matrix = np.linalg.lstsq(np.c_[a, np.ones(len(a))], b, rcond=None)[0].T
        pairs, x = candidate_features(masks, features, matrix)
        mapping = [segmentation_score(m, g)['pred_to_gt'] for m, g in zip(masks, actual)]
        y = np.array([(mapping[0].get(i), mapping[1].get(j)) in verified for i, j in pairs])
        records.append({'sample_id': sid, 'subject': split_id(sid)[0], 'pairs': pairs, 'x': x, 'y': y})
        print('Star pair development', sid, len(pairs), int(y.sum()), flush=True)
    inner = []
    mice = sorted({r['subject'] for r in records})
    if len(mice) != 2:
        raise ValueError('Expected two development mice')
    for mouse in mice:
        fitted = fit([r for r in records if r['subject'] != mouse])
        inner.extend((r, probability(fitted, r['x'])) for r in records if r['subject'] == mouse)
    verified_count = sum(len(truth[s]['match_pairs']) for s in ids)
    measurements = []
    for radius in (4., 5., 6., 8.):
        for threshold in (.05, .1, .15, .2, .25, .3, .35, .4, .45, .5):
            correct = count = 0
            for record, p in inner:
                chosen = assign_edges(record['pairs'], p, threshold, record['x'][:, 18], radius)
                positive = {tuple(pair) for pair, y in zip(record['pairs'], record['y']) if y}
                correct += sum(tuple(pair) in positive for pair in chosen); count += len(chosen)
            measurements.append({'radius': radius, 'threshold': threshold,
                                 'f1': 2*correct/(count+verified_count)})
    selected = max(measurements, key=lambda r: r['f1']); fitted = fit(records)
    torch.save({'state_dict': fitted[0].state_dict(), 'mean': fitted[1], 'sd': fitted[2],
                'held_out': subject, 'train_ids': ids, 'selected': selected,
                'feature_count': 33}, output/'pairs.pt')
    (output/'calibration.json').write_text(json.dumps({'selected': selected, 'measurements': measurements}, indent=2)+'\n')
    raw, merged = [], []
    for sid in sorted(set(truth)-set(ids)):
        row = copy.deepcopy(reference[sid])
        row['exvivo_instances'] = json.loads((source/'regions'/f'{sid}.json').read_text())['exvivo_instances']
        masks, features = describe(row, iv_folder/'maps'/f'{sid}_invivo.npz', source/'maps'/f'{sid}_exvivo.npz')
        pairs, x = candidate_features(masks, features, matrices.get(sid))
        row['match_pairs'] = assign_edges(pairs, probability(fitted, x), selected['threshold'], x[:, 18], selected['radius'])
        raw.append(row); merged.append(merge_row(row, reference[sid]))
        print('Star pairs held-out', sid, len(row['match_pairs']), flush=True)
    for name, rows in [('raw', raw), ('merged', merged)]:
        path = output/f'{name}.csv'; write_rows(rows, path)
        evaluate(SimpleNamespace(data=root, predictions=path, subject=subject, output=output/f'{name}_metrics.json'))


if __name__ == '__main__':
    main()
