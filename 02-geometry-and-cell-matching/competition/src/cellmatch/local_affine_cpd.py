"""Local affine CPD with explicit weighted least-squares EM updates.

Equations: Myronenko/Song, arXiv:0905.2635; reference behavior inspected at
https://github.com/siavashk/pycpd (affine_registration.py, emregistration.py).
This is a local implementation, not a pycpd package invocation. Starting
variance is fixed to64 native-pixel squared; stopping uses variance change.
"""
import numpy as np
from scipy.spatial.distance import cdist


def expectation(moving, target, variance, outlier_weight=.5):
    kernel=np.exp(-cdist(moving,target,'sqeuclidean')/(2*variance))
    constant=(2*np.pi*variance)**(moving.shape[1]/2)*outlier_weight/(1-outlier_weight)*len(moving)/len(target)
    return kernel/(np.maximum(kernel.sum(0,keepdims=True),np.finfo(float).eps)+constant)


def affine_mstep(source, target, weights):
    center=source.mean(0)
    design=np.c_[source-center,np.ones(len(source))]
    mass=weights.sum(1)
    gram=design.T@(mass[:,None]*design)
    coefficients=np.linalg.solve(gram,design.T@weights@target)
    matrix=np.c_[coefficients[:2].T,coefficients[2]-center@coefficients[:2]]
    return matrix


def align(source,target,initial):
    source=np.asarray(source,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    matrix=np.array(initial,dtype=np.float64,copy=True);variance=64.;history=[]
    if len(source)<3 or len(target)<3:return matrix,dict(status='insufficient_points',history=[])
    for iteration in range(100):
        moved=source@matrix[:,:2].T+matrix[:,2]
        weights=expectation(moved,target,variance)
        if weights.sum()<1e-12:return np.array(initial,copy=True),dict(status='vanishing_mass',history=history)
        try:updated=affine_mstep(source,target,weights)
        except np.linalg.LinAlgError:return np.array(initial,copy=True),dict(status='singular',history=history)
        moved=source@updated[:,:2].T+updated[:,2]
        newvariance=max(float(np.sum(weights*cdist(moved,target,'sqeuclidean'))/(2*weights.sum())),1e-6)
        if not np.isfinite(updated).all() or not np.isfinite(newvariance):raise ValueError('Nonfinite CPD update')
        history.append(dict(iteration=iteration+1,variance=newvariance,mass=float(weights.sum())))
        matrix=updated
        if abs(variance-newvariance)<1e-6:break
        variance=newvariance
    return matrix,dict(status='complete',history=history)
