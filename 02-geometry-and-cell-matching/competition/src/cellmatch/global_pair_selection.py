"""Jointly maximize calibrated pair utility under exact family/pixel conflicts."""
from functools import lru_cache
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csc_matrix
from .preserve_pair_masks import pixels

@lru_cache(maxsize=256)
def constraints(keys):
    groups=[]
    families={}
    for n,(_,_,family) in enumerate(keys):families.setdefault(family,[]).append(n)
    groups.extend(tuple(v) for v in families.values() if len(v)>1)
    for modality in [0,1]:
        occupied={}
        for n,c in enumerate(keys):
            for p in pixels(c[modality]):occupied.setdefault(p,[]).append(n)
        groups.extend(tuple(v) for v in occupied.values() if len(v)>1)
    groups=sorted(set(groups));rr=[];cc=[]
    for row,values in enumerate(groups):rr.extend([row]*len(values));cc.extend(values)
    return csc_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(groups),len(keys)))

def choose(record,scores,threshold):
    scores=np.asarray(scores,float);n=len(scores)
    if not n or not (scores>threshold).any():return []
    assert np.isfinite(scores).all()
    keys=tuple((c['iv_rle'],c['ex_rle'],c['family']) for c in record['candidates']);assert len(keys)==n
    matrix=constraints(keys)
    result=milp(c=-(scores-threshold),integrality=np.ones(n),bounds=Bounds(np.zeros(n),(scores>threshold).astype(float)),constraints=LinearConstraint(matrix,np.zeros(matrix.shape[0]),np.ones(matrix.shape[0])),options=dict(time_limit=5.,mip_rel_gap=0.))
    if result.status!=0:raise RuntimeError(f'Pair selection did not prove optimality: {result.message}')
    selected=np.flatnonzero(result.x>.5);assert np.all(matrix[:,selected].sum(axis=1)<=1+1e-8)
    return sorted(selected.tolist(),key=lambda i:(-scores[i],i))
