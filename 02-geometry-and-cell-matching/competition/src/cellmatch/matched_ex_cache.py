"""Lossless NPZ reader for the fixed matched-resolution prerequisite."""
import json
import numpy as np
import torch


def read_cache(path):
    with np.load(path, allow_pickle=False) as z:
        result = json.loads(str(z['metadata']))
        for key in ['tokens', 'target', 'valid', 'positive', 'eligible']:
            result[key] = torch.from_numpy(z[key].copy())
        result['labels'] = z['labels'].copy()
        result['public_logits'] = torch.from_numpy(z['public_logits'].copy())
    return result
