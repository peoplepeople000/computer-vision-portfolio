"""Apply frozen training-derived calibration and conservative rejection to TTA.

Thresholds and quality decisions were fitted before the TTA experiment. Mask
identity transfer uses predicted-mask IoU only. No held-out labels select them.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .data import read_rows, write_rows, parse_instances, split_id
from .rle import labels_to_instances
from .learning import instances_from_maps
from .transfer_filter import transfer_rejections
from .cli import evaluate


def main():
    root = Path('data/raw/Project_2_Dataset')
    original = read_rows('outputs/stage2/unet_oof.csv')
    retained = read_rows('outputs/stage4_conservative/quality_oof.csv')
    tta = read_rows('outputs/stage5_tta/segmentation_oof.csv')
    calibration = json.loads(Path('outputs/stage4_calibrated/calibration.json').read_text())
    if not set(original) == set(retained) == set(tta): raise ValueError('Region mismatch')
    rows, diagnostics = [], []
    for sid, row in tta.items():
        subject = split_id(sid)[0]; config = calibration[subject]
        if any(split_id(s)[0] == subject for s in config['train_ids']):
            raise ValueError('Calibration contains held-out mouse')
        new = {'sample_id': sid, 'match_pairs': []}
        for mode, modality in enumerate(('invivo', 'exvivo')):
            maps = np.load(f'outputs/stage5_tta/maps/{sid}_{modality}.npz')['maps'].astype(np.float32)
            threshold = config['thresholds'][modality]
            new[modality+'_instances'] = (row[modality+'_instances'] if threshold == .5 else
                labels_to_instances(instances_from_maps(maps, threshold, .35), 'IVP' if mode == 0 else 'EXP'))
            if modality == 'exvivo':
                before = parse_instances(original[sid], modality, maps.shape[1:])
                after = parse_instances(new, modality, maps.shape[1:])
                removed = transfer_rejections(before, retained[sid]['exvivo_instances'], after)
                new['exvivo_instances'] = {k:v for k,v in new['exvivo_instances'].items() if k not in removed}
                diagnostics.append({'sample_id': sid, 'removed': len(removed)})
        rows.append(new)
    output = Path('outputs/stage5_tta_combined'); output.mkdir(exist_ok=True)
    path = output/'segmentation_oof.csv'; write_rows(rows, path)
    (output/'manifest.json').write_text(json.dumps({'calibration': calibration,
        'quality_decisions': 'outputs/stage4_conservative/quality_oof.csv',
        'identity_iou': .5, 'regions': diagnostics,
        'policy': 'Frozen prior-stage thresholds and predicted-mask rejection transfer'}, indent=2)+'\n')
    evaluate(SimpleNamespace(data=root, predictions=path, subject=None, output=output/'segmentation_oof_metrics.json'))


if __name__ == '__main__': main()
