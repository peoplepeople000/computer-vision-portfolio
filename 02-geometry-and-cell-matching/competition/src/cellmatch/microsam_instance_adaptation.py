"""Task targets for a frozen-encoder microscopy UNETR instance decoder.

Target conventions follow torch-em PerObjectDistanceTransform (BSD reference
and provenance archived in Stage418). This is a SciPy implementation, not a
claim of equivalence to every bioimage_cpp backend edge case.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries, watershed


def instance_targets(labels):
    assert labels.ndim==2 and labels.dtype.kind in 'iu' and (labels>=0).all()
    foreground=labels>0
    result=np.ones((3,*labels.shape),np.float32);result[0]=foreground
    boundaries=find_boundaries(labels,mode='inner')
    for prop in regionprops(labels):
        y0,x0,y1,x1=prop.bbox; box=np.s_[y0:y1,x0:x1]
        mask=labels[box]==prop.label
        boundary=ndi.distance_transform_edt(~boundaries[box])
        boundary[~mask]=0
        far=np.unravel_index(np.argmax(boundary),boundary.shape)
        center=np.round(prop.centroid).astype(int)-[y0,x0]
        if not mask[tuple(center)]:center=np.asarray(far)
        yy,xx=np.indices(mask.shape)
        center_distance=np.hypot(yy-center[0],xx-center[1]);center_distance[~mask]=0
        inverted=boundary[far]-boundary;inverted[~mask]=0
        for n,values in enumerate((center_distance,inverted),1):
            values=values/(np.abs(values).max()+1e-7)
            result[n][box][mask]=values[mask]
    assert np.isfinite(result).all() and result.min()>=0 and result.max()<=1
    return result


def distance_watershed(maps):
    """Serial port of official AIS defaults, with SciPy/skimage backends."""
    assert maps.ndim==3 and maps.shape[0]==3 and np.isfinite(maps).all()
    fg=ndi.gaussian_filter(maps[0],1.)>.5
    center=ndi.gaussian_filter(maps[1],1.6)
    boundary=ndi.gaussian_filter(maps[2],1.6)
    markers=label(fg&(center<.5)&(boundary<.5),connectivity=1)
    return watershed(boundary,markers=markers,mask=fg,connectivity=1).astype(np.int32)
