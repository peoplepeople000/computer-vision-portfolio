"""Training-only EX quality calibration with image-relative cell-size features."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .data import read_rows, write_rows, parse_instances, split_id
from .learning import instances_from_maps
from .rle import labels_to_instances
from .ex_joint_calibration import morphology_features
from .stage3 import FOLDS, calibrate_quality, quality_probabilities
from .metrics import segmentation_score
from .cli import evaluate


def relative_features(features):
    result = features.copy()
    if len(result):
        # These are log area and log major covariance eigenvalue. Subtracting
        # their per-image medians makes uniform changes in cell size invariant.
        result[:, :2] -= np.median(result[:, :2], axis=0)
    return result


def build(maps, threshold):
    encoded = labels_to_instances(instances_from_maps(maps, threshold, .35), 'EXP')
    masks = parse_instances({'exvivo_instances': encoded}, 'exvivo', maps.shape[1:])
    keys, features = morphology_features(masks, maps)
    return encoded, masks, keys, relative_features(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, help='Completed native EX calibration root')
    parser.add_argument('--output', type=Path, default=Path('outputs/stage10_relative_ex_quality'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    baseline = read_rows('outputs/stage9_subject_geometry/joint/predicted_all.csv')
    calibration = json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    out = args.output
    if any((out/f'{tag}.pt').exists() for tag in FOLDS):
        raise FileExistsError('Use a new output directory to preserve earlier quality models')
    out.mkdir(parents=True, exist_ok=True)
    rows, reports = [], {}
    for tag, subject in FOLDS.items():
        ids = {s for s in truth if split_id(s)[0] != subject}
        cache = Path(f'outputs/stage6_ex/development_{tag}')
        provenance = json.loads((cache/'manifest.json').read_text())
        if set(provenance['train_ids']) != ids or provenance['held_out'] != subject:
            raise ValueError('Invalid EX development partition')
        if set(calibration[subject]['train_ids']) != ids:
            raise ValueError('Invalid foreground calibration partition')
        foreground = calibration[subject]['threshold']
        if args.native_root:
            native = args.native_root/tag
            provenance = json.loads((native/'training_config.json').read_text())
            state = torch.load(native/'model.pt', map_location='cpu', weights_only=False)
            if any(set(v['train_ids']) != ids or v['held_out'] != subject for v in [provenance, state]):
                raise ValueError('Invalid native EX development partition')
            if provenance['tile'] != 128:
                raise ValueError('Expected native 128-pixel maps')
            cache = native/'maps'
        records = []
        for sid in sorted(ids):
            maps = np.load(cache/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
            _, masks, keys, features = build(maps, foreground)
            actual = parse_instances(truth[sid], 'exvivo', maps.shape[1:])
            mapping = segmentation_score(masks, actual)['pred_to_gt']
            records.append({'sample_id': sid, 'subject': split_id(sid)[0], 'features': [None, features],
                            'quality': np.array([k in mapping for k in keys]), 'gt_count': len(actual),
                            'ious': np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0. for k in keys])})
        fitted, threshold, measurements = calibrate_quality(records)
        torch.save({'quality': fitted[0].state_dict(), 'mean': fitted[1], 'sd': fitted[2],
                    'threshold': threshold, 'foreground': foreground, 'train_ids': sorted(ids), 'held_out': subject,
                    'features': 'Morphology confidence with log area and log major eigenvalue centered per image'}, out/f'{tag}.pt')
        reports[subject] = {'foreground': foreground, 'quality_threshold': threshold, 'measurements': measurements,
                            'train_ids': sorted(ids), 'map_root': str(cache),
                            'limitation': 'Outer labels excluded; development images seen by segmentation network.',
                            'development_tile': 128 if args.native_root else 256}
        for sid, row in baseline.items():
            if split_id(sid)[0] != subject:
                continue
            maps = np.load(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')['maps'].astype(np.float32)
            encoded, _, keys, features = build(maps, foreground)
            p = quality_probabilities(fitted, features)
            keep = {k for k, v in zip(keys, p) if v >= threshold}
            value = copy.deepcopy(row); value['match_pairs'] = []
            value['exvivo_instances'] = {k: v for k, v in encoded.items() if k in keep}
            rows.append(value)
        print('Relative EX quality', tag, threshold, flush=True)
    write_rows(rows, out/'segmentation_oof.csv')
    (out/'calibration.json').write_text(json.dumps(reports, indent=2)+'\n')
    evaluate(SimpleNamespace(data=root, predictions=out/'segmentation_oof.csv', subject=None, output=out/'segmentation_oof_metrics.json'))


if __name__ == '__main__':
    main()
