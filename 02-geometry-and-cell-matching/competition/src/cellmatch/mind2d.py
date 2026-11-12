"""Independent four-neighbor2D MIND-style descriptor and affine comparison.

References: Heinrich etal, Medical Image Analysis2012, DOI10.1016/j.media.2012.05.008.
Author-attributed2D source inspected at github.com/cmirfin/BBR/MIND_descriptor2D.m.
This implementation uses reflected borders, stable exponent normalization and an
explicit flat-image guard. It is not MATLAB bitwise equivalence or ConvexAdam.
"""
import numpy as np
from scipy.ndimage import gaussian_filter,binary_dilation
import cv2

OFFSETS=((1,0),(-1,0),(0,1),(0,-1))

def descriptor(image):
    a=np.asarray(image,dtype=np.float32)
    if a.ndim!=2 or min(a.shape)<5 or not np.isfinite(a).all():raise ValueError('Expected finite2Dimage with dimensions>=5')
    padded=np.pad(a,1,mode='reflect');h,w=a.shape
    distances=np.stack([gaussian_filter((a-padded[1+dy:1+dy+h,1+dx:1+dx+w])**2,sigma=.5,radius=1,mode='mirror') for dx,dy in OFFSETS])
    variance=distances.mean(0);mean=float(variance.mean())
    if mean<=np.finfo(np.float32).tiny:return np.ones_like(distances)
    variance=np.clip(variance,.001*mean,1000*mean)
    return np.exp(-(distances-distances.min(0))/variance[None]).astype(np.float32)

def normalize(image):
    a=np.asarray(image,np.float32);lo,hi=np.percentile(a,[1,99.8]);return np.clip((a-lo)/max(float(hi-lo),1e-8),0,1).astype(np.float32)

def compare_affine(iv_descriptor,ex_image,matrix,support):
    """Matrix maps IVxy to EXxy; resample EX onto IV grid before descriptor."""
    h,w=support.shape;m=np.asarray(matrix,np.float64)
    warped=cv2.warpAffine(np.asarray(ex_image,np.float32),m,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    eh,ew=ex_image.shape;yy,xx=np.mgrid[:h,:w];x=m[0,0]*xx+m[0,1]*yy+m[0,2];y=m[1,0]*xx+m[1,1]*yy+m[1,2]
    valid=(x>=4)&(x<ew-4)&(y>=4)&(y<eh-4);valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
    total=int(support.sum());coverage=int((valid&support).sum())/max(total,1)
    if total==0 or coverage<.95:return None,coverage
    features=descriptor(warped);error=np.abs(iv_descriptor-features).mean(0)
    return float(error[support&valid].mean()),coverage
