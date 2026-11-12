"""Shared Stage101 candidate, calibration and contour operations for delivery.

Inference receives images, predicted masks, image-derived geometry and a matched
model bundle. Annotations are optional training labels, never inference inputs.
"""
import copy
import numpy as np
import torch
from .data import read_image, image_path
from .raw_guided_bundle import make_record, separate
from .guided_multicomponent_decode import infer
from .guided_ex_bank import bank_record, choose, assemble
from .guided_appearance_quality import appearance_record
from .guided_endpoint_quality import endpoint_labels, factor_fit, factor_probability
from .guided_large_iv_bank import supplement
from .cellpose_contour_consensus import blend
from .ex_boundary import refine
from .learning import image_features
from .data import parse_instances


def candidates(root, split, sid, banks, maps, matrix, guide, device, truth=None):
    """Use the exact Stage77 original bank followed by Stage89 supplementation."""
    if split != 'training' and truth is not None:
        raise ValueError('Hidden inference cannot receive annotations')
    iv = read_image(image_path(root, split, sid, 'invivo'))
    ex = read_image(image_path(root, split, sid, 'exvivo'))
    records = []
    for positions, only_first in [([0, 1], False), ([2, 1], True)]:
        rows = [banks[n] for n in positions]
        mp = [maps[n] for n in positions]
        raw = make_record(root, split, sid, rows, matrix)
        if only_first:
            raw['cells'] = [c for c in raw['cells'] if c['key'].startswith('0:')]
        generated = infer(guide, raw, device)['proposals']
        rec = bank_record(rows, mp, separate(generated), ex.shape, truth)
        rec = appearance_record(rec, rows, mp, iv, ex)
        if truth is not None:
            rec['endpoint'] = endpoint_labels(rec, truth, [iv.shape, ex.shape])
        records.append(rec)
    return supplement(*records, iv.shape)


def calibrate(records, verified_count):
    """Original fixed grid; inner mouse holdout applies to quality heads only."""
    inner = []
    for mouse in sorted({r['subject'] for r in records}):
        fitted = factor_fit([r for r in records if r['subject'] != mouse])
        inner.extend((r, factor_probability(fitted, r['x'], 'gated_joint'))
                     for r in records if r['subject'] == mouse)
    measures = []
    for threshold in np.arange(.05, 1., .05):
        count = correct = 0
        for rec, scores in inner:
            selected = choose(rec, scores, threshold)
            count += len(selected)
            correct += int(rec['y'][selected].sum())
        measures.append(dict(threshold=float(threshold), correct=correct,
                             predicted=count, f1=2*correct/(count+verified_count)))
    selected = max(measures, key=lambda v: v['f1'])
    return factor_fit(records), selected, measures


def load_heads(state):
    heads = []
    for saved in state['heads']:
        net = torch.nn.Linear(len(saved['mean']), 1)
        net.load_state_dict(saved['state_dict'])
        heads.append((net.eval(), saved['mean'], saved['sd']))
    return heads


def select(base, record, heads, threshold):
    if record is None:
        row = copy.deepcopy(base)
        row['match_pairs'] = []
        return row
    scores = factor_probability(heads, record['x'], 'gated_joint')
    return assemble(base, record, choose(record, scores, threshold))


def contours(row, lora, image, refiner, device):
    mixed, _ = blend(row, lora, image.shape)
    result = copy.deepcopy(mixed)
    result['invivo_instances'], _ = refine(
        parse_instances(mixed, 'invivo', image.shape), image_features(image), refiner, device)
    assert result['match_pairs'] == row['match_pairs']
    assert result['exvivo_instances'] == row['exvivo_instances']
    assert set(result['invivo_instances']) == set(row['invivo_instances'])
    return result
