"""Fixed histogram normalized mutual information for affine candidate ranking.
Uses arithmetic-entropy normalization 2*MI/(H(A)+H(B)), 32 fixed bins.
No SimpleITK dependency or claim of equivalence to Mattes interpolation.
"""
import numpy as np
import cv2
from .mind2d import normalize

def descriptor(image):return np.asarray(image,np.float32)

def nmi(a,b):
 ia=np.minimum((np.clip(a,0,1)*32).astype(np.int64),31);ib=np.minimum((np.clip(b,0,1)*32).astype(np.int64),31)
 counts=np.bincount(ia*32+ib,minlength=1024).reshape(32,32).astype(float)
 if counts.sum()==0:return 0.
 p=counts/counts.sum();pa=p.sum(1);pb=p.sum(0);expected=pa[:,None]*pb[None,:];nz=p>0
 mi=float((p[nz]*np.log(p[nz]/expected[nz])).sum());h=-float((pa[pa>0]*np.log(pa[pa>0])).sum()+(pb[pb>0]*np.log(pb[pb>0])).sum())
 return 2*mi/h if h>1e-12 else 0.

def compare_affine(iv,ex,matrix,support):
 h,w=support.shape;m=np.asarray(matrix,np.float64);warped=cv2.warpAffine(np.asarray(ex,np.float32),m,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 yy,xx=np.mgrid[:h,:w];x=m[0,0]*xx+m[0,1]*yy+m[0,2];y=m[1,0]*xx+m[1,1]*yy+m[1,2];eh,ew=ex.shape;valid=(x>=4)&(x<ew-4)&(y>=4)&(y<eh-4);valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
 total=int(support.sum());coverage=int((valid&support).sum())/max(total,1)
 if not total or coverage<.95:return None,coverage
 return -nmi(iv[valid&support],warped[valid&support]),coverage
