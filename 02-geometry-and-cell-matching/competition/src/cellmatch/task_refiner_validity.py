"""Frozen paired EX refiner features; strict-validity head is separate.

The e block is the final decoder representation, not an encoder layer.
Input priors are the same actual contours that the head will classify.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .ex_boundary import origin, local_mask


def chips(features, masks):
    origins = [origin(mask) for mask in masks]
    x = [np.concatenate([features[:, y:y + 32, x:x + 32],
                         local_mask(mask, x, y)[None]]) for mask, (x, y) in zip(masks, origins)]
    return np.stack(x).astype(np.float32), origins


def embed(model, inputs, retain_fields=False):
    """D4 mean of prior-weighted and complementary-crop decoder means."""
    x = torch.from_numpy(inputs)
    saved = []
    hook = model.e.register_forward_hook(lambda module, arguments, output: saved.append(output))
    sums = None
    fields = []
    try:
        with torch.inference_mode():
            for flip in [False, True]:
                for turn in range(4):
                    view = torch.rot90(x, turn, (-2, -1))
                    if flip:
                        view = view.flip(-1)
                    saved.clear()
                    logits = model(view)
                    assert len(saved) == 1
                    e = saved[0]
                    # The hook does not change the original decoder's output.
                    assert torch.equal(logits, model.out(e))
                    weights = F.interpolate(view[:, 2:3], size=e.shape[-2:], mode='area')
                    inside = (e * weights).sum((-2, -1)) / weights.sum((-2, -1)).clamp_min(1e-6)
                    outside = (e * (1 - weights)).sum((-2, -1)) / (1 - weights).sum((-2, -1)).clamp_min(1e-6)
                    vector = torch.cat([inside, outside], 1).numpy().copy()
                    sums = vector if sums is None else sums + vector
                    if retain_fields:
                        fields.append(dict(e=e.numpy().copy(), weights=weights.numpy().copy(), vector=vector))
    finally:
        hook.remove()
    result = sums / 8
    assert result.shape == (len(inputs), 32) and np.isfinite(result).all()
    return result, fields
