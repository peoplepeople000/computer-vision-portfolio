"""Fixed sparse-mixture coarse contrast; not a calibrated registration model."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .confidence_registration import weighted_raster

SPARSE_FRACTION=.075

def contrast_field(points,shape,grid=2.,weights=None):
    points=np.asarray(points,np.float64)
    weights=np.ones(len(points),np.float64) if weights is None else np.asarray(weights,np.float64)
    assert points.ndim==2 and points.shape[1]==2 and weights.shape==(len(points),)
    assert np.isfinite(points).all() and np.isfinite(weights).all() and (weights>=0).all()
    if not len(points) or weights.sum()==0:return np.zeros(shape,np.float32),dict(empty=True)
    # One fixed image-relative background scale, not fitted from correspondences.
    k=min(17,len(points));distances=cKDTree(points).query(points,k=k)[0]
    scale=float(np.median(distances[:,-1])) if k>1 else 3.
    sigma=max(1.5,scale/grid)
    near=weighted_raster(points/grid,shape,1.5,weights).astype(np.float64)
    broad=weighted_raster(points/grid,shape,sigma,weights).astype(np.float64)
    mass=float(near.sum())
    if mass==0:return np.zeros(shape,np.float32),dict(empty=True)
    near/=mass
    background=np.maximum(broad,weights.sum()/np.prod(shape))
    background/=background.sum()
    ratio=near/background
    field=np.log1p(SPARSE_FRACTION*(ratio-1)).astype(np.float32)
    assert np.isfinite(field).all()
    return field,dict(empty=False,background_sigma_grid=sigma,neighbor_count=k-1,near_mass=float(near.sum()),background_mass=float(background.sum()),max_ratio=float(ratio.max()))

def contrast_score(field,template):
    """Mean log contrast across the unchanged blurred source template."""
    mass=float(template.sum())
    if mass<=0:return np.zeros((field.shape[0]-template.shape[0]+1,field.shape[1]-template.shape[1]+1),np.float32)
    return cv2.matchTemplate(field.astype(np.float32),template.astype(np.float32),cv2.TM_CCORR)/mass
