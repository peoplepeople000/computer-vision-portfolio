"""Rotation-invariant cell appearance, morphology, and confidence features."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .segmentation import normalize


def extract_features(image, masks, maps=None):
    keys = [k for k, m in masks.items() if m.area]
    normalized = normalize(image)
    local = normalized - ndi.gaussian_filter(normalized, 8)
    gradient = np.hypot(ndi.sobel(normalized, 0), ndi.sobel(normalized, 1))
    centers = np.array([masks[k].coordinates().mean(0) for k in keys]).reshape(-1, 2)
    vectors = []
    for key, center in zip(keys, centers):
        mask = masks[key]
        xy = mask.coordinates()
        x, y = xy.T
        centered = xy - center
        eigen = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(xy)), 0.01)
        xmin, ymin, xmax, ymax = mask.bbox
        patch = np.zeros((ymax-ymin+3, xmax-xmin+3), bool)
        patch[y-ymin+1, x-xmin+1] = True
        perimeter = (patch & ~ndi.binary_erosion(patch)).sum()
        raw = normalized[y, x]
        features = [np.log1p(mask.area), np.log1p(eigen[1]), np.log(eigen[1]/eigen[0]),
                    mask.area/((xmax-xmin+1)*(ymax-ymin+1)), perimeter/np.sqrt(mask.area)]
        features.extend(np.quantile(raw, [.1, .25, .5, .75, .9]))
        features.extend([raw.std(), local[y,x].mean(), gradient[y,x].mean(), gradient[y,x].std()])
        # Annular context does not encode absolute image position or subject ID.
        cx, cy = center
        x0, x1 = max(0, int(cx)-32), min(image.shape[1], int(cx)+33)
        y0, y1 = max(0, int(cy)-32), min(image.shape[0], int(cy)+33)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        radius = np.hypot(xx-cx, yy-cy)
        context = normalized[y0:y1, x0:x1]
        for lower, upper in zip([0,3,6,10,16,24], [3,6,10,16,24,32]):
            values = context[(radius>=lower)&(radius<upper)]
            features.extend([values.mean() if len(values) else 0, values.std() if len(values) else 0])
        if maps is None:
            confidence = [1., 1., 1., 1.]
        else:
            confidence = [maps[0,y,x].mean(), np.quantile(maps[0,y,x],.1),
                          maps[1,y,x].mean(), maps[1,y,x].max()]
        features.extend(confidence)
        vectors.append(features)
    return keys, centers, np.asarray(vectors, np.float32).reshape(-1, 30)


def geometry_candidates(source, target, similarities, top_k=5):
    """Spectral spatial-consistency weights on appearance-proposed candidates.

    Inspired by the general spatial-consistency principle; not PointDSC.
    A broad scale tolerance allows moderate affine changes and missing cells.
    """
    from scipy.spatial.distance import cdist
    if not len(source) or not len(target):
        return np.empty(0,int), np.empty(0,int), np.empty(0)
    k = min(top_k, len(source))
    a = np.argsort(-similarities, axis=0)[:k].T.ravel()
    b = np.repeat(np.arange(len(target)), k)
    # Bound memory on dense images, while covering every target when possible.
    if len(a) > 1200:
        order = np.argsort(-similarities[a,b])[:1200]
        a,b = a[order],b[order]
    da,db = cdist(source[a],source[a]),cdist(target[b],target[b])
    relative = np.abs(da-db)/(0.12*np.maximum(da,db)+3.)
    adjacency = np.exp(-0.5*relative**2).astype(np.float32)
    adjacency[(a[:,None]==a[None,:]) | (b[:,None]==b[None,:]) | (da<20) | (db<20)] = 0
    appearance = np.exp(np.clip((similarities[a,b]-1)/.25,-20,0))
    weights = np.sqrt(appearance[:,None]*appearance[None,:])
    adjacency *= weights
    vector = np.ones(len(a),np.float32)/max(len(a),1)
    for _ in range(20):
        vector = adjacency @ vector
        vector /= max(float(np.linalg.norm(vector)),1e-12)
    return a,b,vector


def fit_feature_geometry(source, target, similarities, seed=2026):
    """Two-point similarity hypotheses, then robust affine refinement."""
    import cv2
    from .matching import mutual_neighbors
    cv2.setNumThreads(2)
    cv2.setRNGSeed(seed)
    a,b,weight = geometry_candidates(source,target,similarities)
    if len(a)<6 or weight.sum()<=0:
        return None, {'status':'insufficient_candidates'}
    rng=np.random.default_rng(seed)
    probability=(weight+1e-5)/(weight+1e-5).sum()
    chosen=rng.choice(len(a),(6000,2),p=probability)
    sa,ta=source[a[chosen]],target[b[chosen]]
    ds,dt=sa[:,1]-sa[:,0],ta[:,1]-ta[:,0]
    denom=(ds*ds).sum(1)
    real=(ds*dt).sum(1)/np.maximum(denom,1e-9)
    imag=(ds[:,0]*dt[:,1]-ds[:,1]*dt[:,0])/np.maximum(denom,1e-9)
    scale=np.hypot(real,imag)
    valid=(denom>30**2)&(scale>.65)&(scale<1.4)
    matrices=np.zeros((valid.sum(),2,3))
    real,imag=real[valid],imag[valid]
    matrices[:,0,0]=real;matrices[:,1,1]=real
    matrices[:,1,0]=imag;matrices[:,0,1]=-imag
    matrices[:,:,2]=ta[valid,0]-np.einsum('nij,nj->ni',matrices[:,:,:2],sa[valid,0])
    if not len(matrices):
        return None,{'status':'no_valid_hypotheses'}
    # Rank transforms using proposed correspondences, not unrestricted dense neighbors.
    scores=[]
    for offset in range(0,len(matrices),100):
        ms=matrices[offset:offset+100]
        delta=np.einsum('nij,kj->nki',ms[:,:,:2],source[a])+ms[:,:,2][:,None,:]-target[b]
        residual=np.linalg.norm(delta,axis=2)
        scores.extend((np.exp(-.5*(residual/5)**2)*(weight+0.002)).sum(1))
    best=None
    for idx in np.argsort(scores)[-12:]:
        matrix=matrices[idx]
        for radius in (12,8,6):
            aa,bb,_=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,radius)
            if len(aa)<6: break
            proposal,inliers=cv2.estimateAffine2D(source[aa].astype('float32'),target[bb].astype('float32'),
                method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=2000,confidence=.995)
            if proposal is None or inliers.sum()<6: break
            singular=np.linalg.svd(proposal[:,:2],compute_uv=False)
            if singular.min()<.6 or singular.max()>1.5 or np.linalg.det(proposal[:,:2])<=0: break
            matrix=proposal
        aa,bb,err=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,5)
        appearance=np.exp(np.clip((similarities[aa,bb]-1)/.25,-20,0))
        support=float((np.exp(-.5*(err/3)**2)*appearance).sum())
        if best is None or support>best[0]: best=(support,matrix,len(aa),float(np.median(err)) if len(err) else None)
    support,matrix,count,residual=best
    diagnostic={'status':'registered','inliers':count,'support':support,'residual':residual,'matrix':matrix.tolist()}
    if count<8 or support<2:
        return None,dict(diagnostic,status='rejected')
    return matrix,diagnostic
