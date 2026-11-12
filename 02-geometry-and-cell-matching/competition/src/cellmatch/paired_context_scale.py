"""Physical-scale paired training with observed context and unchanged identity."""
import numpy as np
from scipy import ndimage as ndi


def sample_context(context, foreground, original_target, k, flip, gamma, gain,
                   scale, eligible):
    x = np.rot90(context.astype(np.float32), k, (1, 2))
    fg = np.rot90(foreground.astype(np.float32), k)
    y = np.rot90(original_target.astype(np.float32), k, (1, 2))
    if flip:
        x, fg, y = x[:, :, ::-1], fg[:, ::-1], y[:, :, ::-1]
    x = x.copy()
    x[:2] = np.clip(x[:2] ** gamma * gain, 0, 1)
    if not eligible or scale == 1.:
        return x[:, 8:40, 8:40].copy(), y.copy()
    assert .8 <= scale <= 1.25
    matrix = np.eye(2) / scale
    offset = np.full(2, 23.5 - 15.5 / scale)

    def warp(a, order):
        return ndi.affine_transform(a, matrix, offset=offset, output_shape=(32, 32),
                                    order=order, mode='constant', cval=0., prefilter=False)

    raw = [warp(v, 1) for v in x[:2]]
    prior, target = warp(x[2], 0), warp(fg, 0)
    distance = ndi.distance_transform_edt(np.pad(target, 1))[1:-1, 1:-1].astype(np.float32)
    distance /= max(float(distance.max()), 1.)
    return np.stack(raw + [prior]), np.stack([target, distance])
