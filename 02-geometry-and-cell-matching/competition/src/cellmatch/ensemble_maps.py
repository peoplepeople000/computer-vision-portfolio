"""Equal-weight out-of-fold probability ensemble with provenance checks."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .data import read_rows, write_rows
from .rle import labels_to_instances
from .learning import instances_from_maps
from .stage3 import FOLDS
from .cli import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/raw/Project_2_Dataset')
    parser.add_argument('--output', default='outputs/stage5_ensemble')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    truth = read_rows(Path(args.data)/'training/train_ground_truth.csv')
    rows, sources = [], []
    for tag, subject in FOLDS.items():
        files = [Path(f'outputs/{stage}/unet_holdout_{tag}.csv')
                 for stage in ('stage2', 'stage5_refinement')]
        ids = {sid for sid in truth if sid.startswith(subject+'__')}
        for path in files:
            manifest = json.loads(Path(str(path)+'.manifest.json').read_text())
            if manifest['held_out'] != subject or ids & set(manifest['train_ids']):
                raise ValueError('Ensemble source contains held-out training samples')
            if set(read_rows(path)) != ids:
                raise ValueError('Ensemble source has incorrect region IDs')
            sources.append({'path': str(path), 'manifest': manifest,
                            'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for sid in sorted(ids):
            row = {'sample_id': sid, 'match_pairs': []}
            for mode, prefix in [('invivo', 'IVP'), ('exvivo', 'EXP')]:
                maps = [np.load(path.parent/(path.stem+'_maps')/f'{sid}_{mode}.npz')['maps'].astype(np.float32)
                        for path in files]
                if maps[0].shape != maps[1].shape:
                    raise ValueError('Ensemble map shape mismatch')
                labels = instances_from_maps((maps[0]+maps[1])/2, .5, .35)
                row[mode+'_instances'] = labels_to_instances(labels, prefix)
            rows.append(row)
    path = output/'segmentation_oof.csv'
    write_rows(rows, path)
    (output/'manifest.json').write_text(json.dumps({'weights': [.5, .5], 'threshold': .5,
        'seed_threshold': .35, 'sources': sources}, indent=2)+'\n')
    evaluate(SimpleNamespace(data=Path(args.data), predictions=path, subject=None,
                             output=output/'segmentation_oof_metrics.json'))


if __name__ == '__main__':
    main()
