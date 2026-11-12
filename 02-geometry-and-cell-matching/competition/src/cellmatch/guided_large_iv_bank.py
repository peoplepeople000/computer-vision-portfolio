"""Supplement frozen Stage77 candidates with public ViT-L IV prompts.
Preserve original candidates and73 image features exactly. Keep original
endpoint heads and development-only calibration recipe; no neural fitting.
New features use the public-versus-adapted role, while candidate source2 records
ViT-L provenance. Exact mask-pair duplicates retain the original entry.
"""
import copy
import numpy as np
from .rle import Mask


def supplement(original, additional, iv_shape):
    assert original['sample_id'] == additional['sample_id']
    supervised = 'endpoint' in original
    assert ('endpoint' in additional) == supervised
    if not supervised:
        assert not len(original['y']) and not len(additional['y'])
    output = copy.deepcopy(original)
    keys = {(c['iv_rle'], c['ex_rle']) for c in original['candidates']}
    old_masks = {}
    for c in original['candidates']:
        old_masks.setdefault(c['iv_rle'], (Mask.parse(c['iv_rle'], iv_shape), c['family']))
    associated = {}
    features, labels, endpoints = [original['x']], [original['y']], []
    if supervised:
        endpoints.append(original['endpoint'])
    for n, candidate in enumerate(additional['candidates']):
        key = candidate['iv_rle'], candidate['ex_rle']
        if key in keys:
            continue
        keys.add(key)
        c = copy.deepcopy(candidate)
        c['iv'] = 'LARGE_' + c['iv']
        c['source'] = 2
        if c['iv_rle'] not in associated:
            mask = Mask.parse(c['iv_rle'], iv_shape)
            overlap, family = max(((mask.iou(m), f) for m, f in old_masks.values()), default=(0., ''))
            associated[c['iv_rle']] = family if overlap > .25 else 'LARGE_' + candidate['family']
        c['family'] = associated[c['iv_rle']]
        output['candidates'].append(c)
        features.append(additional['x'][n:n+1])
        if supervised:
            labels.append(additional['y'][n:n+1])
            endpoints.append(additional['endpoint'][n:n+1])
    output['x'] = np.concatenate(features)
    output['y'] = np.concatenate(labels)
    if supervised:
        output['endpoint'] = np.concatenate(endpoints)
    return output
