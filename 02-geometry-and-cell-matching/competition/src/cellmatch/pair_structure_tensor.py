"""Image-only axial gradient agreement in affine-aligned cell surroundings.

Background: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.structure_tensor
This implementation pools gradients in a fixed annulus, not a Gaussian window.
"""
import numpy as np
from scipy import ndimage as ndi

def axial_tensor(patch):
    patch=ndi.gaussian_filter(np.asarray(patch,np.float64),1.)
    gy,gx=np.gradient(patch)
    yy,xx=np.mgrid[-24:25,-24:25]
    ring=(xx*xx+yy*yy>=8**2)&(xx*xx+yy*yy<23**2)
    xxm=float(np.mean(gx[ring]**2));yym=float(np.mean(gy[ring]**2));xym=float(np.mean(gx[ring]*gy[ring]));energy=xxm+yym
    return np.array([xxm-yym,2*xym])/energy if energy>1e-20 else np.zeros(2)

def pair_features(iv_image,ex_image,iv_center,ex_center,linear):
    linear=np.asarray(linear,np.float64)
    assert linear.shape==(2,2) and np.isfinite(linear).all() and abs(np.linalg.det(linear))>1e-8
    yy,xx=np.mgrid[-24:25,-24:25];offset=np.stack([xx.ravel(),yy.ravel()])
    ci=np.asarray(iv_center)[:,None]+np.linalg.solve(linear,offset)
    ce=np.asarray(ex_center)[:,None]+offset
    patches=[]
    for im,c in [(iv_image,ci),(ex_image,ce)]:
        patches.append(ndi.map_coordinates(im,[c[1],c[0]],order=1,mode='reflect').reshape(49,49))
    a,b=map(axial_tensor,patches)
    result=np.array([np.linalg.norm(a),np.linalg.norm(b),np.dot(a,b)],np.float32)
    assert np.isfinite(result).all()
    return result

def record_features(record,matrix,images):
    from .rle import Mask
    candidates=record['candidates'];centers=[]
    for key,im in zip(['iv_rle','ex_rle'],images):
        cache={c[key]:Mask.parse(c[key],im.shape).coordinates().mean(0) for c in candidates}
        centers.append(cache)
    cache={};result=[]
    for c in candidates:
        key=(c['iv_rle'],c['ex_rle'])
        if key not in cache:cache[key]=pair_features(*images,centers[0][key[0]],centers[1][key[1]],np.asarray(matrix)[:,:2])
        result.append(cache[key])
    return np.asarray(result,np.float32).reshape(-1,3)
