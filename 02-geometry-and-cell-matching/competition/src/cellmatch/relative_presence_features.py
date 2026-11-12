"""Per-image appearance midranks; morphology retains its original units."""
import numpy as np
from scipy.stats import rankdata


def relative_appearance(features):
    x = np.asarray(features, np.float32)
    assert x.ndim == 2 and x.shape[1] == 26 and np.isfinite(x).all()
    result = x.copy()
    if len(x):
        result[:, 5:26] = ((rankdata(x[:, 5:26], axis=0, method='average')-.5)
                           / len(x)).astype(np.float32)
    return result
