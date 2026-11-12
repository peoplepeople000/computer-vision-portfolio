"""Two-native-pixel contour edit support, derived only from the input guide."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def native_band(prior):
    prior = np.asarray(prior, bool)
    core = ndimage.binary_erosion(prior, structure=CROSS, iterations=2)
    support = ndimage.binary_dilation(prior, structure=CROSS, iterations=2)
    return core, support, support & ~core


def project_decode(probability, prior):
    core, support, _ = native_band(prior)
    resized = F.interpolate(torch.as_tensor(probability)[None, None].float(),
                            size=prior.shape, mode='bilinear', align_corners=False)[0, 0].numpy()
    foreground = (resized >= .5) & support | core
    labels, count = ndimage.label(foreground)
    overlap = np.bincount(labels[prior], minlength=count+1)
    overlap[0] = 0
    result = labels == int(overlap.argmax()) if overlap.max(initial=0)>0 else np.zeros_like(prior)
    if result.sum()<12:
        result[:] = False
    assert not (result & ~support).any()
    # Component/area rejection can still drop a disconnected core; never fallback.
    return result


def constrained_probability(logits, native, band):
    return logits.sigmoid()*band + native*(1-band)


def band_loss(logits, target, native, band):
    positive = target.flatten(1).any(1)
    if not positive.any():
        return logits.sum()*0
    logits, target, native, band = [v[positive] for v in [logits, target, native, band]]
    count = band.sum((1, 2, 3))
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*band).sum((1, 2, 3))/count.clamp_min(1)
    p = constrained_probability(logits, native, band)
    dice = 1-(2*(p*target).sum((1, 2, 3))+1)/(p.sum((1, 2, 3))+target.sum((1, 2, 3))+1)
    return (bce+dice).mean()
