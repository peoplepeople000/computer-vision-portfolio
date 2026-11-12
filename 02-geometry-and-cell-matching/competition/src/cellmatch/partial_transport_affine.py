"""Small CPU prototype of KL-unbalanced transport with affine barycentric updates.

This is not the authors' RobOT implementation. The transport objective uses
KL(plan, a outer b), KL(row sums,a), and KL(column sums,b).
"""
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

def plan(cost,epsilon,rho,iterations=200):
    n,m=cost.shape;la=np.full(n,-np.log(n));lb=np.full(m,-np.log(m))
    kernel=-cost/epsilon+la[:,None]+lb[None,:];u=np.zeros(n);v=np.zeros(m);power=rho/(rho+epsilon)
    for _ in range(iterations):
        old=u.copy();u=power*(la-logsumexp(kernel+v[None,:],axis=1));v=power*(lb-logsumexp(kernel+u[:,None],axis=0))
        if np.max(np.abs(u-old))<1e-9:break
    return np.exp(kernel+u[:,None]+v[None,:])

def align(source,target,initial):
    matrix=np.array(initial,dtype=float).copy();center=source.mean(0);design=np.c_[source-center,np.ones(len(source))];history=[]
    for sigma in [8.,4.,2.]:
        for _ in range(10):
            transformed=source@matrix[:,:2].T+matrix[:,2];cost=.5*cdist(transformed,target,'sqeuclidean');p=plan(cost,sigma*sigma,20.**2)
            weights=p.sum(1);bary=p@target/np.maximum(weights[:,None],1e-300);w=np.sqrt(weights/max(weights.max(),1e-300));coef=np.linalg.lstsq(design*w[:,None],bary*w[:,None],rcond=None)[0]
            new=np.column_stack([coef[:2].T,coef[2]-center@coef[:2]]);sv=np.linalg.svd(new[:,:2],compute_uv=False)
            if not np.isfinite(new).all() or sv.min()<.7 or sv.max()>1.3 or np.linalg.det(new[:,:2])<=0:return matrix,dict(status='shape_guard',history=history)
            delta=float(np.max(np.linalg.norm(source@(new[:,:2]-matrix[:,:2]).T+new[:,2]-matrix[:,2],axis=1)));matrix=new;history.append(dict(sigma=sigma,delta=delta,mass=float(p.sum())))
            if delta<1e-4:break
    return matrix,dict(status='complete',history=history)
