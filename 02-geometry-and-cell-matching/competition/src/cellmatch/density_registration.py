"""Global point-density correlation followed by robust affine refinement.

The search uses predicted centroids only. No per-region coordinates or
transforms are hardcoded. Defaults are a fixed stage-two development protocol.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from .matching import centroids, mutual_neighbors
from .registration import assign_pairs
from .data import read_rows, image_path, image_shape, parse_instances, split_id, write_rows


def raster(points,shape,sigma):
    result=np.zeros(shape,np.float32)
    xy=np.rint(points).astype(int)
    good=(xy[:,0]>=0)&(xy[:,0]<shape[1])&(xy[:,1]>=0)&(xy[:,1]<shape[0])
    np.add.at(result,(xy[good,1],xy[good,0]),1)
    return ndi.gaussian_filter(result,sigma)


def fit_density(iv_masks,ex_masks,grid=4.0):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(2026)
    _,source=centroids(iv_masks)
    _,target=centroids(ex_masks)
    if min(len(source),len(target))<8:
        return None,{"status":"insufficient_cells"}
    shape=next(iter(ex_masks.values())).shape
    field=raster(target/grid,tuple(int(np.ceil(n/grid))+1 for n in shape),2.0)
    candidates=[]
    # Full angular search avoids assuming the orientation of an unseen mouse.
    for angle in range(-180,180,10):
        theta=np.deg2rad(angle)
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for scale in (0.75,0.9,1.05,1.2):
            matrix=rotation*scale
            transformed=source@matrix.T/grid
            lower=np.floor(transformed.min(0))-7
            points=transformed-lower
            size=np.ceil(points.max(0)).astype(int)+8
            template=raster(points,(int(size[1]),int(size[0])),2.0)
            if any(a>b for a,b in zip(template.shape,field.shape)):
                continue
            scores=cv2.matchTemplate(field,template,cv2.TM_CCOEFF_NORMED)
            _,score,_,xy=cv2.minMaxLoc(scores)
            offset=(np.array(xy)-lower)*grid
            candidates.append((score,np.column_stack([matrix,offset])))
    if not candidates:
        return None,{"status":"no_search_candidates"}
    candidates.sort(key=lambda r:r[0],reverse=True)
    best=None
    for correlation,matrix in candidates[:12]:
        initial=matrix.copy()
        # Coarse reciprocal neighbors initialize RANSAC; tighten gradually.
        for radius in (24,18,12,9,7):
            a,b,_=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,radius)
            if len(a)<6:
                break
            refined,inliers=cv2.estimateAffine2D(source[a].astype(np.float32),target[b].astype(np.float32),
                method=cv2.RANSAC,ransacReprojThreshold=max(3.,radius*0.5),maxIters=3000,confidence=0.995,refineIters=10)
            if refined is None or inliers.sum()<6:
                break
            singular=np.linalg.svd(refined[:,:2],compute_uv=False)
            if singular.min()<0.55 or singular.max()>1.5 or np.linalg.det(refined[:,:2])<=0:
                break
            matrix=refined
        a,b,errors=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,6)
        support=float(np.sum(np.exp(-0.5*(errors/3)**2)))
        if best is None or support>best[0]:
            best=(support,matrix,correlation,len(a),float(np.median(errors)) if len(errors) else None)
    support,matrix,correlation,count,residual=best
    diagnostic={"status":"registered","method":"density_affine","weighted_support":support,
                "inliers":count,"correlation":float(correlation),"median_residual":residual,
                "matrix":matrix.tolist(),"search_candidates":len(candidates)}
    if count<8 or support<5:
        diagnostic['status']='insufficient_registration_support'
        return None,diagnostic
    return matrix,diagnostic


def run(args):
    rows=read_rows(args.predictions)
    predictions,diagnostics=[],[]
    for n,(sid,row) in enumerate(rows.items(),1):
        if args.subject and split_id(sid)[0]!=args.subject:
            continue
        masks={m:parse_instances(row,m,image_shape(image_path(args.data,args.split,sid,m))) for m in ('invivo','exvivo')}
        matrix,diagnostic=fit_density(masks['invivo'],masks['exvivo'])
        row['match_pairs']=assign_pairs(masks['invivo'],masks['exvivo'],matrix,args.radius)
        predictions.append(row)
        diagnostic.update(sample_id=sid,pairs=len(row['match_pairs']))
        diagnostics.append(diagnostic)
        print(f"[{n}/{len(rows)}] {sid}: {diagnostic['status']}, {len(row['match_pairs'])} pairs",flush=True)
    if not predictions:
        raise ValueError('No selected samples')
    write_rows(predictions,args.output)
    manifest=vars(args).copy()
    manifest['regions']=diagnostics
    Path(str(args.output)+'.registration.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--subject')
    parser.add_argument('--split',choices=['training','hidden_test'],default='training')
    parser.add_argument('--radius',type=float,default=6.0)
    run(parser.parse_args())


if __name__=='__main__':
    main()
