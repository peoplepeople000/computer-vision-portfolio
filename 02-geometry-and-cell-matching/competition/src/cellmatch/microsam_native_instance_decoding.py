"""Direct native-pixel decoding of supervised microscopy distance fields.

Unlike the archived AIS port, this preserves native foreground pixels rather
than applying an additional Gaussian filter after bilinear resizing. The
fixed thresholds and marker watershed remain unchanged.
"""
import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed


def native_distance_watershed(maps):
    assert maps.ndim==3 and maps.shape[0]==3 and np.isfinite(maps).all()
    fg=maps[0]>.5
    markers=label(fg&(maps[1]<.5)&(maps[2]<.5),connectivity=1)
    return watershed(maps[2],markers=markers,mask=fg,connectivity=1).astype(np.int32)
