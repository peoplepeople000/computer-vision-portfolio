"""Training-only isotropic scale augmentation around the crop pixel center."""
import numpy as np
from scipy import ndimage as ndi


def scale_patch(x, y, scale):
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    assert x.shape == (3, 32, 32) and y.shape == (2, 32, 32)
    assert .8 <= scale <= 1.25
    if scale == 1.:
        return x.copy(), y.copy()
    matrix = np.eye(2) / scale
    offset = np.full(2, 15.5 * (1 - 1 / scale))
    features = [ndi.affine_transform(c, matrix, offset=offset, order=1,
                                    mode='reflect', prefilter=False)
                for c in x[:2]]
    binary = lambda c: ndi.affine_transform(c, matrix, offset=offset, order=0,
                                          mode='constant', cval=0.,
                                          prefilter=False).astype(np.float32)
    prior, foreground = binary(x[2]), binary(y[0])
    distance = ndi.distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    distance = distance.astype(np.float32)
    distance /= max(float(distance.max()), 1.)
    return np.stack(features + [prior]), np.stack([foreground, distance])
