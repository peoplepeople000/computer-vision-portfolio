"""Bounded nearest rounded target-center affinity for coarse translation proposals.

Many source centers can share one target; no one-to-one or partial-FOV guarantee.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def nearest_field(points,shape,grid=2.):
    background=np.ones(shape,dtype=bool)
    xy=np.rint(np.asarray(points).reshape(-1,2)).astype(int)
    good=(xy[:,0]>=0)&(xy[:,0]<shape[1])&(xy[:,1]>=0)&(xy[:,1]<shape[0])
    if not good.any():return np.zeros(shape,np.float32)
    background[xy[good,1],xy[good,0]]=False
    distance=distance_transform_edt(background)
    return np.exp(-.5*(distance/(2.5/grid))**2).astype(np.float32)


def mean_affinity(field,template):
    field=np.asarray(field,np.float32);template=np.asarray(template,np.float32)
    assert field.ndim==template.ndim==2
    total=float(template.sum(dtype=np.float32))
    if total==0:
        return np.zeros((field.shape[0]-template.shape[0]+1,field.shape[1]-template.shape[1]+1),np.float32)
    return cv2.matchTemplate(field,template,cv2.TM_CCORR)/total
