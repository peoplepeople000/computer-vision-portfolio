"""Additional EX candidates from distance-field watershed of merged foreground.

Original guided components are preserved by the caller. This decoder only adds
watershed parts of foreground components with at least two existing center
peaks. Parameters reuse the fixed native watershed and guided area/radius rules.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from .rle import encode


def proposals(prob, presence, x, cell, origin, shape):
    foreground = prob[0] > .5
    components, _ = ndi.label(foreground)
    score = ndi.gaussian_filter(prob[1], .7)
    coordinates = peak_local_max(score, min_distance=3, threshold_abs=.35,
                                 labels=foreground.astype(np.uint8), exclude_border=False)
    if not len(coordinates):
        return [], dict(seeds=0, merged_components=0)
    markers = np.zeros(foreground.shape, np.int32)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates)+1)
    parents = components[tuple(coordinates.T)]
    counts = np.bincount(parents)
    merged = {i for i, n in enumerate(counts) if i and n >= 2}
    if not merged:
        return [], dict(seeds=len(coordinates), merged_components=0)
    labels = watershed(-score, markers, mask=foreground)
    options = []
    for n, parent in enumerate(parents, 1):
        if int(parent) not in merged:
            continue
        mask = labels == n
        area = int(mask.sum())
        if not 6 <= area <= 500:
            continue
        yy, xx = np.where(mask)
        distance = float(np.linalg.norm(np.array([xx.mean(), yy.mean()])-24))
        if distance >= 12:
            continue
        confidence = float(prob[0][mask].mean())*np.exp(-.5*(distance/5)**2)
        options.append((confidence, mask, distance, int(parent)))
    options.sort(key=lambda v: -v[0])
    result = []
    for _, mask, distance, parent in options[:3]:
        yy, xx = np.where(mask); x0, y0 = origin
        gx, gy = xx+x0, yy+y0
        valid = (gx >= 0)&(gy >= 0)&(gx < shape[1])&(gy < shape[0])
        if valid.sum() < 6:
            continue
        canvas = np.zeros(shape, bool); canvas[gy[valid], gx[valid]] = True
        area = float(valid.sum()); height = float(yy.max()-yy.min()+1); width = float(xx.max()-xx.min()+1)
        boundary = mask&~ndi.binary_erosion(mask); prior = x[3] > .5; union = (mask|prior).sum()
        features = [np.log1p(area), area/(width*height), np.log(width/height),
                    float(prob[0][mask].mean()), float(prob[0][mask].max()), float(prob[0][mask].std()),
                    float(prob[0][boundary].mean()), float(prob[1][mask].max()), float(presence), distance,
                    np.log1p(cell['patch'].sum()), np.log(max(area,1)/max(cell['patch'].sum(),1)),
                    float((mask&prior).sum()/max(union,1))]
        result.append(dict(iv=cell['key'], rle=encode(canvas), features=[float(v) for v in features],
                           presence=float(presence), parent_component=parent))
    return result, dict(seeds=len(coordinates), merged_components=len(merged))
