"""Serial APG postprocessing following the archived official default policy.

Uses SciPy/skimage instead of elf parallel primitives. This is a local port,
not an assertion of bitwise equivalence to the unavailable parallel backend.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


def points_from_maps(maps):
    assert maps.ndim == 3 and maps.shape[0] == 3 and np.isfinite(maps).all()
    fg, center, boundary = maps
    components = label((fg >= .5) & (center < .5) & (boundary < .5), connectivity=1)
    interior = ~find_boundaries(components, mode='outer')
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    distance = distance_transform_edt(interior)
    points = []
    for prop in regionprops(components):
        y0, x0, y1, x1 = prop.bbox
        local = distance[y0:y1, x0:x1].copy()
        local[components[y0:y1, x0:x1] != prop.label] = 0
        y, x = np.unravel_index(local.argmax(), local.shape)
        points.append([x+x0, y+y0])
    return np.asarray(points, np.float32).reshape(-1, 2)


def suppress(masks, scores, shape, min_size=25, threshold=.9):
    """Score NMS, then area-first exclusive painting and connected components."""
    masks = [np.asarray(m, bool) for m in masks]
    eligible = [i for i,m in enumerate(masks) if m.sum() > min_size]
    order = torch.argsort(torch.as_tensor(np.asarray(scores)[eligible]), descending=True).tolist()
    order = [eligible[i] for i in order]
    kept = []
    while order:
        i, *remaining = order
        kept.append(i)
        order = [j for j in remaining if np.logical_and(masks[i],masks[j]).sum() / np.logical_or(masks[i],masks[j]).sum() <= threshold]
    result = np.zeros(shape, np.uint32)
    for index, i in enumerate(sorted(kept, key=lambda j:int(masks[j].sum()), reverse=True), 1):
        result[masks[i] & (result==0)] = index
    result = label(result, connectivity=1)
    for prop in regionprops(result):
        if prop.area < min_size:
            result[result==prop.label] = 0
    return label(result, connectivity=1).astype(np.uint32), kept
