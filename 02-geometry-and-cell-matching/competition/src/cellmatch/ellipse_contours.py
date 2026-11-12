"""Fixed area/moment ellipse rasterization from prediction coordinates only."""
import numpy as np

def ellipse_points(xy,shape):
    xy=np.asarray(xy,np.float64);assert len(xy)>0
    center=xy.mean(0);z=xy-center;cov=z.T@z/len(xy)+np.eye(2)/12
    threshold=len(xy)/(np.pi*np.sqrt(np.linalg.det(cov)))
    radius=np.sqrt(threshold*np.diag(cov));low=np.maximum(np.floor(center-radius).astype(int),0);high=np.minimum(np.ceil(center+radius).astype(int),np.array(shape[::-1])-1)
    xx,yy=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1));points=np.c_[xx.ravel(),yy.ravel()];delta=points-center;inside=np.einsum('ni,ij,nj->n',delta,np.linalg.inv(cov),delta)<=threshold
    return points[inside]

