"""Native-grid center offsets and stable log-SPD ellipse representation."""
import numpy as np

def encode_points(xy):
 xy=np.asarray(xy,np.float64);center=xy.mean(0);z=xy-center
 cov=z.T@z/len(xy)+np.eye(2)/12;matrix=cov/np.sqrt(np.linalg.det(cov))*len(xy)/np.pi
 values,vectors=np.linalg.eigh(matrix);log=(vectors*np.log(values))@vectors.T
 anchor=np.rint(center).astype(np.int64)
 return anchor,np.r_[center-anchor,.5*np.trace(log),.5*(log[0,0]-log[1,1]),log[0,1]].astype(np.float32)

def decode(anchor,values,shape):
 v=np.asarray(values,np.float64);center=np.asarray(anchor)+v[:2];mean,diag,off=v[2:]
 log=np.array([[mean+diag,off],[off,mean-diag]]);eig,vec=np.linalg.eigh(log);matrix=(vec*np.exp(eig))@vec.T
 radius=np.sqrt(np.diag(matrix));lo=np.maximum(np.floor(center-radius).astype(int),0);hi=np.minimum(np.ceil(center+radius).astype(int),np.array(shape[::-1])-1)
 xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));xy=np.c_[xx.ravel(),yy.ravel()];z=xy-center
 distance=np.einsum('ni,ij,nj->n',z,np.linalg.inv(matrix),z);return xy[distance<=1],distance[distance<=1]

def transform_values(values,rotation):
 v=np.asarray(values,np.float64);m,d,o=v[2:];log=np.array([[m+d,o],[o,m-d]]);r=np.asarray(rotation);log=r@log@r.T
 return np.r_[r@v[:2],.5*np.trace(log),.5*(log[0,0]-log[1,1]),log[0,1]]
