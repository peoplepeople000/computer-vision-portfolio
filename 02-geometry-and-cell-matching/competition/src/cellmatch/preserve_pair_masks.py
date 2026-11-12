"""Apply a new segmentation filter while preserving masks used by existing pairs.

Keep the original paired masks, then add new segmentation masks that do not
overlap them. IDs from regenerated watershed proposals are not assumed stable.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .data import read_rows, write_rows
from .cli import evaluate


def pixels(rle):
    values = [int(v) for v in rle.split()]
    if len(values) % 2:raise ValueError('Malformed RLE')
    result = set(); previous = 0
    for start, length in zip(values[::2], values[1::2]):
        if start < previous or length <= 0:raise ValueError('Invalid RLE runs')
        result.update(range(start, start+length)); previous = start+length
    return result


def merge_row(paired, segmentation):
    if paired['sample_id'] != segmentation['sample_id']:
        raise ValueError('Sample IDs differ')
    if paired['invivo_instances'] != segmentation['invivo_instances']:
        raise ValueError('IV proposals differ')
    old = paired['exvivo_instances']; new = segmentation['exvivo_instances']
    row = copy.deepcopy(segmentation)
    row['match_pairs'] = copy.deepcopy(paired['match_pairs'])
    row['exvivo_instances'] = {}; occupied = set()
    for iv, ex in row['match_pairs']:
        if iv not in row['invivo_instances'] or ex not in old:
            raise ValueError('Invalid original pair')
        row['exvivo_instances'][ex] = old[ex]
        mask = pixels(old[ex])
        if mask & occupied:raise ValueError('Original paired masks overlap')
        occupied.update(mask)
    for n, rle in enumerate(new.values(), 1):
        mask = pixels(rle)
        if mask & occupied:continue
        key = f'EXP_REL_{n:06d}'
        if key in row['exvivo_instances']:raise ValueError('Generated ID collision')
        row['exvivo_instances'][key] = rle; occupied.update(mask)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', default='outputs/stage9_subject_geometry/joint/predicted_all.csv')
    parser.add_argument('--segmentation', default='outputs/stage10_relative_ex_quality/segmentation_oof.csv')
    parser.add_argument('--output', default='outputs/stage11_preserved_pairs')
    args = parser.parse_args()
    paired, segmentation = read_rows(args.paired), read_rows(args.segmentation)
    if set(paired) != set(segmentation):
        raise ValueError('Prediction IDs differ')
    rows = [merge_row(paired[sid], segmentation[sid]) for sid in paired]
    out = Path(args.output); out.mkdir(exist_ok=True)
    write_rows(rows, out/'predicted_all.csv')
    config = {**vars(args), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'paired_sha256': hashlib.sha256(Path(args.paired).read_bytes()).hexdigest(),
              'segmentation_sha256': hashlib.sha256(Path(args.segmentation).read_bytes()).hexdigest(),
              'policy': 'Preserve original paired EX masks and pairs; add non-overlapping relative-quality masks with fresh IDs. No annotation input.'}
    (out/'pipeline.json').write_text(json.dumps(config, indent=2)+'\n')
    manifest = Path(args.paired).parent/'predicted_all_manifest.json'
    (out/'predicted_all_manifest.json').write_text(manifest.read_text())
    evaluate(SimpleNamespace(data=Path('data/raw/Project_2_Dataset'), predictions=out/'predicted_all.csv', subject=None, output=out/'predicted_all_metrics.json'))


if __name__ == '__main__':
    main()
