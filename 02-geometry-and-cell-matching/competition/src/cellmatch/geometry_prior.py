"""Training-derived affine priors and fine point-density translation search.

All transforms used for prior fitting come only from the two training mice.
Held-out registration uses mask centroids only, never known correspondences.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from .data import read_rows,parse_instances,image_path,image_shape,split_id,write_rows
from .matching import centroids,mutual_neighbors
from .density_registration import raster
from .registration import assign_pairs
from .stage3 import FOLDS
from .cli import evaluate


def fit_prior(root,truth,heldout):
    stretches=[];angles=[];ids=[]
    for sid,row in truth.items():
        if split_id(sid)[0]==heldout or len(row['match_pairs'])<5:continue
        centers=[]
        for mode in ('invivo','exvivo'):
            masks=parse_instances(row,mode,image_shape(image_path(root,'training',sid,mode)))
            centers.append({k:m.coordinates().mean(0) for k,m in masks.items() if m.area})
        a=np.array([centers[0][i] for i,j in row['match_pairs']])
        b=np.array([centers[1][j] for i,j in row['match_pairs']])
        matrix=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0].T[:,:2]
        u,s,vt=np.linalg.svd(matrix);rotation=u@vt
        if np.linalg.det(rotation)<=0:continue
        stretches.append(vt.T@np.diag(s)@vt)
        angles.append(np.degrees(np.arctan2(rotation[1,0],rotation[0,0])))
        ids.append(sid)
    if not stretches:raise ValueError('No training affine fits')
    return {'stretch':np.median(stretches,axis=0).tolist(),
            'angle_low':float(min(angles)-15),'angle_high':float(max(angles)+15),
            'train_ids':ids,'held_out':heldout,'purpose':'Training-derived prior; no per-test-region fitted labels'}


def register(iv,ex,prior,grid=2.):
    _,source=centroids(iv);_,target=centroids(ex)
    if min(len(source),len(target))<8:return None,{'status':'insufficient_cells'}
    shape=next(iter(ex.values())).shape
    field=raster(target/grid,tuple(int(np.ceil(n/grid))+1 for n in shape),1.5)
    stretch=np.asarray(prior['stretch']);candidates=[]
    for angle in np.arange(prior['angle_low'],prior['angle_high']+.01,2.):
        theta=np.deg2rad(angle);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for scale in (.96,1.,1.04):
            matrix=rotation@stretch*scale
            p=source@matrix.T/grid;lower=np.floor(p.min(0))-6;p-=lower
            size=np.ceil(p.max(0)).astype(int)+7
            template=raster(p,(int(size[1]),int(size[0])),1.5)
            if any(a>b for a,b in zip(template.shape,field.shape)):continue
            scores=cv2.matchTemplate(field,template,cv2.TM_CCOEFF_NORMED)
            for _ in range(3):
                _,score,_,xy=cv2.minMaxLoc(scores)
                offset=(np.asarray(xy)-lower)*grid
                candidates.append((float(score),np.column_stack([matrix,offset])))
                x,y=xy;scores[max(0,y-8):y+9,max(0,x-8):x+9]=-1
    if not candidates:return None,{'status':'no_candidates'}
    candidates.sort(key=lambda v:v[0],reverse=True);best=None
    for correlation,matrix in candidates[:30]:
        for radius in (8,6,5):
            a,b,_=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,radius)
            if len(a)<6:break
            refined,inliers=cv2.estimateAffine2D(source[a].astype(np.float32),target[b].astype(np.float32),
                method=cv2.RANSAC,ransacReprojThreshold=3.5,maxIters=3000,confidence=.995)
            if refined is None or inliers.sum()<6:break
            singular=np.linalg.svd(refined[:,:2],compute_uv=False)
            if singular.min()<.7 or singular.max()>1.3 or np.linalg.det(refined[:,:2])<=0:break
            matrix=refined
        a,b,errors=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,5.)
        support=float(np.exp(-.5*(errors/2.5)**2).sum())
        if best is None or support>best[0]:best=(support,matrix,len(a),correlation,float(np.median(errors)) if len(errors) else None)
    support,matrix,count,correlation,residual=best
    diag={'status':'registered','inliers':count,'weighted_support':support,'correlation':correlation,
          'residual':residual,'matrix':matrix.tolist(),'candidate_count':len(candidates)}
    if count<10 or support<7:return None,dict(diag,status='rejected')
    return matrix,diag


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['oracle','predicted'],default='oracle')
    p.add_argument('--output',default='outputs/stage4_geometry')
    p.add_argument('--predictions',help='Optional predicted-mask CSV; never permitted in oracle mode')
    p.add_argument('--fold',choices=['all',*FOLDS],default='all');args=p.parse_args()
    cv2.setNumThreads(2);cv2.setRNGSeed(2026)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    if args.source=='oracle' and args.predictions:raise ValueError('Oracle mode cannot take predicted masks')
    prediction=truth if args.source=='oracle' else read_rows(args.predictions or 'outputs/stage2/unet_oof.csv')
    if set(prediction)!=set(truth):raise ValueError('Predictions must cover all labeled regions exactly')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    selected=list(FOLDS) if args.fold=='all' else [args.fold]
    allrows=[];diagnostics=[];priors={}
    for tag in selected:
        heldout=FOLDS[tag];prior=fit_prior(root,truth,heldout);priors[heldout]=prior
        for sid,row in prediction.items():
            if split_id(sid)[0]!=heldout:continue
            masks=[parse_instances(row,m,image_shape(image_path(root,'training',sid,m))) for m in ['invivo','exvivo']]
            matrix,diag=register(*masks,prior)
            pairs=assign_pairs(*masks,matrix,radius=5.,ambiguity_ratio=.8)
            value=copy.deepcopy(row);value['match_pairs']=pairs;allrows.append(value)
            diag.update(sample_id=sid,pairs=len(pairs));diagnostics.append(diag)
            print(args.source,sid,diag['status'],len(pairs),flush=True)
    path=output/f'{args.source}_{args.fold}.csv';write_rows(allrows,path)
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None if args.fold=='all' else FOLDS[args.fold],output=output/f'{args.source}_{args.fold}_metrics.json'))
    (output/f'{args.source}_{args.fold}_manifest.json').write_text(json.dumps({'priors':priors,'regions':diagnostics,
        'predictions':args.predictions or ('ground_truth_masks' if args.source=='oracle' else 'outputs/stage2/unet_oof.csv'),
        'oracle_warning':'Oracle-mask composite scores are not achieved segmentation performance' if args.source=='oracle' else None,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
