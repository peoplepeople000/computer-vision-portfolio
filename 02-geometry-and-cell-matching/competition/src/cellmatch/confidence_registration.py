"""EX confidence weighting of the unchanged coarse-search/refinement recipe."""
import inspect
import numpy as np
from scipy import ndimage as ndi
from . import geometry_prior
from .matching import centroids


def mask_confidences(masks, maps):
    keys, _ = centroids(masks)
    foreground, center = maps.reshape(2, -1)
    result = []
    for key in keys:
        mask = masks[key]
        indices = np.concatenate([np.arange(s, e) for s, e in zip(mask.starts, mask.ends)])
        result.append(float(foreground[indices].mean()*center[indices].max()))
    result = np.array(result)
    assert np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all()
    return result


def weighted_raster(points, shape, sigma, weights):
    canvas = np.zeros(shape, np.float32); xy = np.rint(points).astype(int)
    good = (xy[:, 0] >= 0) & (xy[:, 0] < shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < shape[0])
    np.add.at(canvas, (xy[good, 1], xy[good, 0]), weights[good])
    return ndi.gaussian_filter(canvas, sigma)


def register(iv, ex, prior, confidence):
    confidence = np.asarray(confidence, np.float64)
    assert confidence.shape == (len(centroids(ex)[0]),)
    assert np.isfinite(confidence).all() and (confidence >= 0).all() and (confidence <= 1).all()
    # Keep archived search geometry and RANSAC implementation frozen. Each
    # replacement is narrowly checked so upstream changes fail explicitly.
    source = inspect.getsource(geometry_prior.register)
    changes = {
        'field=raster(target/grid,tuple(int(np.ceil(n/grid))+1 for n in shape),1.5)':
        'field=_weighted_raster(target/grid,tuple(int(np.ceil(n/grid))+1 for n in shape),1.5,_confidence)',
        'support=float(np.exp(-.5*(errors/2.5)**2).sum())':
        'raw_support=float(np.exp(-.5*(errors/2.5)**2).sum())\n        support=float((np.exp(-.5*(errors/2.5)**2)*_confidence[b]).sum())',
        'float(np.median(errors)) if len(errors) else None)':
        'float(np.median(errors)) if len(errors) else None,raw_support)',
        'support,matrix,count,correlation,residual=best':
        'support,matrix,count,correlation,residual,raw_support=best',
        "'weighted_support':support": "'weighted_support':support,'raw_support':raw_support",
        'if count<10 or support<7:': 'if count<10 or raw_support<7:',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    scope = dict(geometry_prior.register.__globals__)
    scope.update(_weighted_raster=weighted_raster, _confidence=confidence)
    exec(compile(source, '<confidence_registration>', 'exec'), scope)
    return scope['register'](iv, ex, prior)
