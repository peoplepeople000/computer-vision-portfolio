"""Fixed-center local polar contour representation; no learned predictor yet."""
import numpy as np
from scipy.ndimage import map_coordinates

ANGLES = 64
STEP = .25
MAX_RADIUS = 24.

def prior_center(prior):
    y, x = np.nonzero(prior>.5)
    if not len(x):
        raise ValueError('Prior must contain foreground')
    return np.array([x.mean(), y.mean()], dtype=np.float64)

def radial_target(target, center):
    """First foreground exit; outside-target centers produce zero radii."""
    theta = np.arange(ANGLES, dtype=np.float64)*2*np.pi/ANGLES
    radius = np.arange(0, MAX_RADIUS+STEP/2, STEP, dtype=np.float64)
    x = center[0]+np.cos(theta)[:,None]*radius
    y = center[1]+np.sin(theta)[:,None]*radius
    values = map_coordinates(np.asarray(target,np.float64),np.stack([y,x]),order=0,mode='constant',cval=0,prefilter=False)
    exits = values<.5
    first = exits.argmax(axis=1)
    inside = bool(np.all(values[:,0]>.5))
    result = np.where(exits.any(axis=1),np.maximum(first*STEP-STEP/2,0),MAX_RADIUS)
    if not inside:
        result.fill(0)
    return result.astype(np.float32), dict(center_inside=inside,max_radius_hit=int((~exits.any(axis=1)).sum()))

def render(radii, center, shape=(32,32)):
    """Interpolate neighboring angular radii at original pixel centers."""
    radii = np.asarray(radii, dtype=np.float64)
    assert radii.shape==(ANGLES,) and np.isfinite(radii).all() and (radii>=0).all()
    y,x = np.indices(shape,dtype=np.float64)
    x -= center[0]
    y -= center[1]
    angular = np.mod(np.arctan2(y,x),2*np.pi)*ANGLES/(2*np.pi)
    lower = np.floor(angular).astype(int)%ANGLES
    fraction = angular-np.floor(angular)
    boundary = radii[lower]*(1-fraction)+radii[(lower+1)%ANGLES]*fraction
    return np.hypot(x,y)<=boundary

def iou(a,b):
    a,b = np.asarray(a,bool),np.asarray(b,bool)
    union = np.count_nonzero(a|b)
    return np.count_nonzero(a&b)/union if union else 0.
