"""Binary four-neighbor Potts cut using frozen neural foreground unaries.

Local contrast-sensitive boundary energy, not a reproduction of an interactive
segmentation benchmark. Unit boundary weight is fixed, never fitted to labels.
"""
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

CAPACITY_SCALE = 100000


def terms(probability, image, boundary_weight=1.):
    p=np.clip(np.asarray(probability,dtype=np.float64),1e-6,1-1e-6)
    raw=np.asarray(image,dtype=np.float64)
    assert p.shape==raw.shape and p.ndim==2 and np.isfinite(p).all() and np.isfinite(raw).all()
    h,w=p.shape;indices=np.arange(p.size).reshape(p.shape)
    u=np.concatenate([indices[:,:-1].ravel(),indices[:-1,:].ravel()])
    v=np.concatenate([indices[:,1:].ravel(),indices[1:,:].ravel()])
    differences=(raw.ravel()[u]-raw.ravel()[v])**2
    beta=1/(2*max(float(differences.mean()) if len(differences) else 0.,1e-6))
    pairwise=float(boundary_weight)*np.exp(-beta*differences)
    return -np.log1p(-p).ravel(),-np.log(p).ravel(),u,v,pairwise,beta


def energy(mask, probability, image, boundary_weight=1., quantized=False):
    bg,fg,u,v,pairwise,_=terms(probability,image,boundary_weight)
    if quantized:
        bg,fg,pairwise=[np.rint(a*CAPACITY_SCALE).astype(np.int64) for a in [bg,fg,pairwise]]
    labels=np.asarray(mask,dtype=bool).ravel()
    return np.where(labels,fg,bg).sum()+pairwise[labels[u]!=labels[v]].sum()


def cut(probability,image,boundary_weight=1.,method='dinic'):
    bg,fg,u,v,pairwise,beta=terms(probability,image,boundary_weight)
    n=np.asarray(probability).size;source=n;sink=n+1;nodes=np.arange(n)
    rows=np.concatenate([np.full(n,source),nodes,u,v])
    columns=np.concatenate([nodes,np.full(n,sink),v,u])
    capacities=np.rint(np.concatenate([bg,fg,pairwise,pairwise])*CAPACITY_SCALE).astype(np.int64)
    assert capacities.min(initial=0)>=0 and capacities.max(initial=0)<2**31
    graph=csr_matrix((capacities,(rows,columns)),shape=(n+2,n+2))
    result=maximum_flow(graph,source,sink,method=method)
    residual=(graph-result.flow).tocsr();visited=np.zeros(n+2,bool);visited[source]=True;queue=deque([source])
    while queue:
        a=queue.popleft()
        for b,c in zip(residual.indices[residual.indptr[a]:residual.indptr[a+1]],residual.data[residual.indptr[a]:residual.indptr[a+1]]):
            if c>0 and not visited[b]:visited[b]=True;queue.append(int(b))
    assert not visited[sink]
    mask=visited[:n].reshape(np.asarray(probability).shape)
    objective=int(energy(mask,probability,image,boundary_weight,quantized=True))
    assert objective==result.flow_value
    return mask,dict(quantized_energy=objective,flow_value=int(result.flow_value),
                     energy=float(energy(mask,probability,image,boundary_weight)),beta=beta)
