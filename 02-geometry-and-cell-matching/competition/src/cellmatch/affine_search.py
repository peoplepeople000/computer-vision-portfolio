"""Training-derived anisotropic affine search and chance-adjusted consensus."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from .data import read_rows,parse_instances,image_path,image_shape,split_id,write_rows
from .matching import centroids,mutual_neighbors
from .density_registration import raster
from .geometry_prior import fit_prior
from .registration import assign_pairs
from .stage3 import FOLDS
from .cli import evaluate


def stretch_bank(root,truth,heldout):
    values=[];ids=[]
    for sid,row in truth.items():
        if split_id(sid)[0]==heldout or len(row['match_pairs'])<5:continue
        masks=[parse_instances(row,m,image_shape(image_path(root,'training',sid,m))) for m in ['invivo','exvivo']]
        a=np.array([masks[0][i].coordinates().mean(0) for i,j in row['match_pairs']]);b=np.array([masks[1][j].coordinates().mean(0) for i,j in row['match_pairs']])
        matrix=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0].T
        error=np.median(np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1))
        u,s,vt=np.linalg.svd(matrix[:,:2])
        if error>5 or np.linalg.det(u@vt)<=0 or s.min()<.7 or s.max()>1.3:continue
        stretch=vt.T@np.diag(s)@vt;values.append([stretch[0,0],stretch[1,1],stretch[0,1]]);ids.append(sid)
    q=np.quantile(values,[.15,.5,.85],axis=0)
    matrices=[np.array([[x,z],[z,y]]) for x in q[:,0] for y in q[:,1] for z in q[:,2]]
    return matrices,ids


def support_score(source,target,matrix):
    projected=source@matrix[:,:2].T+matrix[:,2]
    a,b,e=mutual_neighbors(projected,target,5.)
    support=float(np.exp(-.5*(e/2.5)**2).sum())
    hull=cv2.convexHull(projected.astype(np.float32));area=max(cv2.contourArea(hull),1.)
    inside=sum(cv2.pointPolygonTest(hull,tuple(map(float,p)),False)>=0 for p in target)
    expected=len(source)*inside*(2*np.pi*2.5**2*(1-np.exp(-2)))/area
    excess=(support-expected)/np.sqrt(expected+1)
    return support,float(excess),len(a),float(expected)


def search(iv,ex,prior,bank):
    _,source=centroids(iv);_,target=centroids(ex);grid=4.
    if min(len(source),len(target))<8:return [],[]
    shape=next(iter(ex.values())).shape;field=raster(target/grid,tuple(int(np.ceil(n/grid))+1 for n in shape),1.2)
    candidates=[]
    for angle in np.arange(prior['angle_low']-5,prior['angle_high']+5,3.):
        theta=np.deg2rad(angle);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for stretch in bank:
            matrix=rotation@stretch;p=source@matrix.T/grid;lower=np.floor(p.min(0))-4;p-=lower
            size=np.ceil(p.max(0)).astype(int)+5;template=raster(p,(int(size[1]),int(size[0])),1.2)
            if any(a>b for a,b in zip(template.shape,field.shape)):continue
            scores=cv2.matchTemplate(field,template,cv2.TM_CCOEFF_NORMED)
            for _ in range(2):
                _,score,_,xy=cv2.minMaxLoc(scores);offset=(np.asarray(xy)-lower)*grid
                candidates.append((float(score),np.column_stack([matrix,offset])))
                x,y=xy;scores[max(0,y-5):y+6,max(0,x-5):x+6]=-1
    candidates.sort(key=lambda v:v[0],reverse=True);chosen=[];projections=[]
    anchor=source[np.linspace(0,len(source)-1,min(20,len(source))).astype(int)]
    for score,matrix in candidates:
        p=anchor@matrix[:,:2].T+matrix[:,2]
        if any(np.median(np.linalg.norm(p-q,axis=1))<8 for q in projections):continue
        chosen.append((score,matrix));projections.append(p)
        if len(chosen)>=80:break
    refined=[];diagnostics=[]
    for correlation,matrix in chosen:
        for radius in (10,7,5):
            a,b,_=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,radius)
            if len(a)<6:break
            proposal,inliers=cv2.estimateAffine2D(source[a].astype(np.float32),target[b].astype(np.float32),method=cv2.RANSAC,ransacReprojThreshold=3.5,maxIters=3000,confidence=.995)
            if proposal is None or inliers.sum()<6:break
            s=np.linalg.svd(proposal[:,:2],compute_uv=False)
            if s.min()<.7 or s.max()>1.3 or np.linalg.det(proposal[:,:2])<=0:break
            matrix=proposal
        support,excess,count,expected=support_score(source,target,matrix)
        if count<10 or support<7:continue
        refined.append(matrix);diagnostics.append({'matrix':matrix.tolist(),'support':support,'excess':excess,'inliers':count,'expected_support':expected,'correlation':correlation})
    return refined,diagnostics


def main():
    cv2.setNumThreads(2);cv2.setRNGSeed(2031);root=Path('data/raw/Project_2_Dataset')
    truth=read_rows(root/'training/train_ground_truth.csv');pred=read_rows('outputs/stage6_specialist_quality_geometry/predicted_all.csv')
    outputs={v:[] for v in ['support','excess']};reports={v:[] for v in outputs};priors={}
    cache=Path('outputs/stage7_affine_candidates');cache.mkdir(exist_ok=True)
    for tag,subject in FOLDS.items():
        prior=fit_prior(root,truth,subject);bank,ids=stretch_bank(root,truth,subject);priors[subject]={'angle_prior':prior,'stretch_train_ids':ids,'bank':[b.tolist() for b in bank]}
        for sid,row in pred.items():
            if split_id(sid)[0]!=subject:continue
            masks=[parse_instances(row,m,image_shape(image_path(root,'training',sid,m))) for m in ['invivo','exvivo']]
            matrices,diags=search(*masks,prior,bank);(cache/f'{sid}.json').write_text(json.dumps(diags)+'\n')
            for variant in outputs:
                index=max(range(len(diags)),key=lambda i:diags[i][variant]) if diags else None
                matrix=matrices[index] if index is not None else None;diag=copy.deepcopy(diags[index]) if index is not None else {}
                value=copy.deepcopy(row);value['match_pairs']=assign_pairs(*masks,matrix,radius=5,ambiguity_ratio=.8)
                diag.update(sample_id=sid,status='registered' if matrix is not None else 'rejected',pairs=len(value['match_pairs']))
                outputs[variant].append(value);reports[variant].append(diag)
            print('Affine bank',sid,len(matrices),flush=True)
    for variant,rows in outputs.items():
        out=Path(f'outputs/stage7_affine_{variant}');out.mkdir(exist_ok=True);path=out/'predicted_all.csv';write_rows(rows,path)
        (out/'predicted_all_manifest.json').write_text(json.dumps({'regions':reports[variant],'priors':priors,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'predicted_all_metrics.json'))


if __name__=='__main__':main()
