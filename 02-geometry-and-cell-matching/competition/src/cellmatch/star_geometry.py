"""Star-convex target geometry and exclusive polygon decoding.

Inspired by Schmidt et al., Cell Detection with Star-convex Polygons (2018).
This is an independent small implementation, not the StarDist package.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.draw import polygon
from .rle import Mask
from .ex_boundary import patch_rle

RAYS = 32
RAY_SCALE = 16.


def directions(n=RAYS):
    angle = np.arange(n)*2*np.pi/n
    return np.stack([np.sin(angle), np.cos(angle)], 1)


def star_targets(labels, n_rays=RAYS, max_distance=64., step=.25):
    """Dense radial targets; ignore instances truncated by the crop boundary."""
    if labels.ndim != 2 or labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('Expected a nonnegative integer instance image')
    valid = np.ones(labels.shape, np.float32)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border > 0]
    clean = labels.copy()
    if len(border):
        ignored = np.isin(labels, border); valid[ignored] = 0; clean[ignored] = 0
    probability = np.zeros(labels.shape, np.float32)
    distances = np.zeros((n_rays, *labels.shape), np.float32)
    for key, box in enumerate(ndi.find_objects(clean), 1):
        if box is None:
            continue
        mask = clean[box] == key
        distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        probability[box] = np.maximum(probability[box], distance/max(float(distance.max()), 1.))
    yy, xx = np.nonzero(clean)
    if not len(yy):
        return probability, distances, valid
    unit = directions(n_rays)
    identities = clean[yy, xx, None]
    alive = np.ones((len(yy), n_rays), bool)
    values = np.full(alive.shape, max_distance, np.float32)
    h, w = labels.shape
    for radius in np.arange(step, max_distance+step, step):
        y = np.floor(yy[:, None]+unit[None, :, 0]*radius+.5).astype(int)
        x = np.floor(xx[:, None]+unit[None, :, 1]*radius+.5).astype(int)
        inside = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        inside &= clean[np.clip(y, 0, h-1), np.clip(x, 0, w-1)] == identities
        outside = alive & ~inside
        values[outside] = max(radius-step/2, step/2)
        alive &= inside
        if not alive.any():
            break
    distances[:, yy, xx] = values.T
    return probability, distances, valid


def extract_candidates(probability, distances, minimum=.1, limit=2000):
    if distances.shape[1:] != probability.shape:
        raise ValueError('Probability and ray shapes differ')
    points = peak_local_max(probability, min_distance=2, threshold_abs=minimum, exclude_border=False)
    scores = probability[points[:, 0], points[:, 1]]
    order = np.argsort(-scores, kind='stable')[:limit]
    points = points[order]
    return points, distances[:, points[:, 0], points[:, 1]].T, scores[order]


def render_candidates(points, radii, scores, shape, threshold=.35, nms_iou=.3, min_area=12):
    """Score-ordered polygon suppression followed by exclusive pixel ownership."""
    if len(points) != len(radii) or len(points) != len(scores):
        raise ValueError('Candidate lengths differ')
    labels = np.zeros(shape, np.int32); kept = []
    unit = directions(radii.shape[1])
    for index in np.argsort(-scores, kind='stable'):
        if scores[index] < threshold:
            continue
        lengths = np.asarray(radii[index])
        if not np.isfinite(lengths).all():
            raise ValueError('Nonfinite predicted radii')
        vertices = points[index]+np.clip(lengths, .125, 64)[:, None]*unit
        yy, xx = polygon(vertices[:, 0], vertices[:, 1], shape=shape)
        if len(yy) < min_area:
            continue
        ymin, xmin = int(yy.min()), int(xx.min())
        patch = np.zeros((int(yy.max())-ymin+1, int(xx.max())-xmin+1), bool)
        patch[yy-ymin, xx-xmin] = True
        mask = Mask.parse(patch_rle(patch, xmin, ymin, shape), shape)
        if any(mask.iou(previous) > nms_iou for previous in kept):
            continue
        free = labels[yy, xx] == 0
        if free.mean() < .75 or free.sum() < min_area:
            continue
        labels[yy[free], xx[free]] = len(kept)+1
        kept.append(mask)
    return labels
