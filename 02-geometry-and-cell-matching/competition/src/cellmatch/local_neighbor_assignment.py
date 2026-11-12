"""Query-anchored relative point matching using independent background cells."""
import numpy as np
from scipy.optimize import linear_sum_assignment

RADIUS=128.
EXCLUDE_RADIUS=4.
MAX_NEIGHBORS=32
SIGMA=4.
MIN_CLOSE=4


def neighborhoods(queries, background):
    points=np.unique(np.asarray(background,dtype=np.float64).reshape(-1,2),axis=0)
    result=[]
    for query in np.asarray(queries,dtype=np.float64).reshape(-1,2):
        delta=points-query;radius=np.linalg.norm(delta,axis=1)
        keep=np.flatnonzero((radius>EXCLUDE_RADIUS)&(radius<=RADIUS))
        order=sorted(keep,key=lambda i:(radius[i],delta[i,0],delta[i,1]))[:MAX_NEIGHBORS]
        result.append(delta[order])
    return result


def pair_scores(source, target):
    """Return maximum Gaussian assignment similarity and close-neighbor counts."""
    scores=np.zeros((len(source),len(target)),np.float64)
    counts=np.zeros_like(scores,dtype=np.uint8)
    padded=np.zeros((len(target),MAX_NEIGHBORS,2),np.float64)
    lengths=np.asarray([len(v) for v in target])
    for j,v in enumerate(target):padded[j,:len(v)]=v
    for i,u in enumerate(source):
        if len(u)<MIN_CLOSE:continue
        d2=((u[None,:,None,:]-padded[:,None,:,:])**2).sum(-1)
        weights=np.exp(-d2/(2*SIGMA**2))
        for j,size in enumerate(lengths):
            if size<MIN_CLOSE:continue
            value=weights[j,:,:size];a,b=linear_sum_assignment(-value)
            scores[i,j]=value[a,b].sum()/min(len(u),int(size))
            counts[i,j]=int((d2[j,a,b]<SIGMA**2).sum())
    return scores,counts
