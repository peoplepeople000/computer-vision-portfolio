"""BCE-only focal modulation for the existing native EX specialist.

Keep positive/boundary weights, Dice and normalized-distance losses unchanged.
"""
import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
import json
from .data import read_rows, parse_instances
from .learning import load_training
from .ex_specialist import difficult_centers


def loss(logits, target, gamma=2.):
    fg = target[:, :1]
    prob = logits.sigmoid()
    local = F.avg_pool2d(fg, 3, stride=1, padding=1)
    weight = 1 + ((local > 0) & (local < 1)).float()
    bce = F.binary_cross_entropy_with_logits(logits[:, :1], fg, reduction='none',
        pos_weight=logits.new_tensor(2.))
    pt = prob[:, :1]*fg + (1-prob[:, :1])*(1-fg)
    bce = (bce*(1-pt).pow(gamma)*weight).sum()/weight.sum()
    dims = (1, 2, 3)
    dice = 1 - ((2*(prob[:, :1]*fg).sum(dims)+1)/(prob[:, :1].sum(dims)+fg.sum(dims)+1)).mean()
    distance = (F.smooth_l1_loss(prob[:, 1:], target[:, 1:], reduction='none')*(1+4*fg)).mean()
    return bce + dice + 2*distance


def load_items(root, held):
    """Rebuild the original specialist's actual34-image sampler inputs."""
    items, ids = load_training(root, held)
    items = [r for r in items if r['mode'] == 1]
    dev = Path('outputs/stage3/b2')
    manifest = json.loads((dev/'development_manifest.json').read_text())
    predictions = read_rows(dev/'development.csv')
    truth = read_rows(root/'training/train_ground_truth.csv')
    assert set(predictions) == set(ids)
    assert all(r['subject'] != held for r in items)
    for item in items:
        sid = item['sample_id']
        shape = item['y'].shape[1:]
        actual = parse_instances(truth[sid], 'exvivo', shape)
        predicted = parse_instances(predictions[sid], 'exvivo', shape)
        item['difficult_centers'] = difficult_centers(actual, predicted)
    return items, ids, manifest
