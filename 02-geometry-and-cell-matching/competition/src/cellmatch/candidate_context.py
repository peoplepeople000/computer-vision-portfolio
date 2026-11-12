"""Label-free local displacement support from distinct candidate families.

This changes neither registration nor masks. Each family contributes one median
position/displacement, preventing duplicate banks from multiplying support.
"""
import numpy as np
from .rle import Mask


def context_features(record, matrix, shapes):
    candidates=record['candidates']
    if not candidates:
        return np.empty((0,6),np.float32)
    matrix=np.asarray(matrix,dtype=np.float64)
    assert matrix.shape==(2,3) and np.isfinite(matrix).all()
    centers=[]
    for key,shape in zip(['iv_rle','ex_rle'],shapes):
        cache={}
        for c in candidates:
            if c[key] not in cache:
                cache[c[key]]=Mask.parse(c[key],shape).coordinates().mean(0)
        centers.append(np.asarray([cache[c[key]] for c in candidates]))
    projected=centers[0]@matrix[:,:2].T+matrix[:,2]
    displacement=centers[1]-projected
    families=sorted({c['family'] for c in candidates})
    positions=[]; offsets=[]
    for family in families:
        ids=[i for i,c in enumerate(candidates) if c['family']==family]
        # Exact duplicate masks must not change family medians.
        unique={tuple(np.r_[projected[i],displacement[i]]) for i in ids}
        values=np.asarray(sorted(unique))
        positions.append(np.median(values[:,:2],axis=0))
        offsets.append(np.median(values[:,2:],axis=0))
    positions=np.asarray(positions); offsets=np.asarray(offsets)
    output=[]
    for i,c in enumerate(candidates):
        distances=np.linalg.norm(positions-projected[i],axis=1)
        order=sorted((j for j,f in enumerate(families) if f!=c['family'] and distances[j]<=128),key=lambda j:(distances[j],families[j]))[:8]
        if not order:
            output.append([0.,0.,0.,0.,0.,0.]);continue
        differences=np.linalg.norm(offsets[order]-displacement[i],axis=1)
        spread=np.linalg.norm(offsets[order]-np.median(offsets[order],axis=0),axis=1)
        output.append([np.median(differences)/4,np.quantile(differences,.9)/4,
                       np.mean(differences<2),np.mean(differences<4),len(order)/8,
                       np.median(spread)/4])
    return np.asarray(output,np.float32)
