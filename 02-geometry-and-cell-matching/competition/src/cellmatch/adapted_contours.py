"""Fixed EX contour consensus with adapted CPDINO, retaining existing pair IDs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .cellpose_contour_consensus import blend
from .data import read_rows, write_rows, image_path, image_shape, split_id
from .stage3 import FOLDS
from .cli import evaluate


def blend_ex(base, candidate, shape):
    a, b = copy.deepcopy(base), copy.deepcopy(candidate)
    a['invivo_instances'] = a['exvivo_instances']
    b['invivo_instances'] = b['exvivo_instances']
    result, count = blend(a, b, shape)
    row = copy.deepcopy(base)
    row['exvivo_instances'] = result['invivo_instances']
    return row, count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', choices=list(FOLDS), required=True)
    p.add_argument('--source', type=Path, default=Path('outputs/stage33_cpdino_adapted'))
    p.add_argument('--output', type=Path, default=Path('outputs/stage34_adapted_contours'))
    a = p.parse_args()
    baseline_path = Path('outputs/stage32_contour_consensus/predicted_all.csv')
    candidate_path = a.source/a.fold/'predictions.csv'
    base, candidate = read_rows(baseline_path), read_rows(candidate_path)
    expected = {s for s in base if split_id(s)[0] == FOLDS[a.fold]}
    if set(candidate) != expected:
        raise ValueError('Require complete held-out mouse')
    cfg = json.loads((a.source/a.fold/'configuration.json').read_text())
    if cfg['held_out'] != FOLDS[a.fold] or set(cfg['train_ids']) & expected:
        raise ValueError('Invalid training partition')
    root = Path('data/raw/Project_2_Dataset')
    rows, counts = [], {}
    for sid in sorted(candidate):
        row, counts[sid] = blend_ex(base[sid], candidate[sid], image_shape(image_path(root, 'training', sid, 'exvivo')))
        rows.append(row)
    out = a.output/a.fold
    out.mkdir(parents=True, exist_ok=True)
    write_rows(rows, out/'predictions.csv')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(changed=counts, held_out=FOLDS[a.fold],
        source_sha256=sha(__file__), blend_sha256=sha(Path(__file__).with_name('cellpose_contour_consensus.py')),
        inputs={str(path):sha(path) for path in [baseline_path, candidate_path, a.source/a.fold/'configuration.json']},
        selection='Fixed half-weight signed-distance consensus; no held-out tuning'), indent=2)+'\n')
    evaluate(SimpleNamespace(data=root, predictions=out/'predictions.csv', subject=FOLDS[a.fold], output=out/'metrics.json'))


if __name__ == '__main__':
    main()
