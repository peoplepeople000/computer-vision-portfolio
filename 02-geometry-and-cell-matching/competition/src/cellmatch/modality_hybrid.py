"""Combine refined IV masks and frozen previous-best EX masks for every mouse."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from .data import read_rows, write_rows
from .cli import evaluate


def main():
    iv_path = Path('outputs/stage5_refinement/segmentation_oof.csv')
    ex_path = Path('outputs/stage4_combined/segmentation_oof.csv')
    iv, ex = read_rows(iv_path), read_rows(ex_path)
    if set(iv) != set(ex): raise ValueError('Region mismatch')
    rows = [{'sample_id': sid, 'invivo_instances': row['invivo_instances'],
             'exvivo_instances': ex[sid]['exvivo_instances'], 'match_pairs': []}
            for sid, row in iv.items()]
    out = Path('outputs/stage5_hybrid'); out.mkdir(exist_ok=True)
    path = out/'segmentation_oof.csv'; write_rows(rows, path)
    (out/'manifest.json').write_text(json.dumps({'recipe': 'Refined IV; previous calibrated conservative EX, identical policy for all folds',
        'selection': 'Exploratory modality-level selection after observing refinement IV gain and EX regression',
        'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [iv_path, ex_path]]}, indent=2)+'\n')
    evaluate(SimpleNamespace(data=Path('data/raw/Project_2_Dataset'), predictions=path,
                             subject=None, output=out/'segmentation_oof_metrics.json'))


if __name__ == '__main__': main()
