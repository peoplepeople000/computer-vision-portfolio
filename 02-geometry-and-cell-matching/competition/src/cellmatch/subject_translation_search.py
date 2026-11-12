"""Image-only registration search with a robust same-subject translation prior.

All prior anchors are OTHER regions with >=4 predicted consensus votes. No
annotation file is read. High-vote target transforms are preserved. Translation
priors act before correlation top-k truncation, so previously omitted hypotheses
can enter the candidate bank.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import cv2,numpy as np
from .data import read_rows,parse_instances,image_path,image_shape,split_id
from .density_registration import raster
from .matching import centroids,mutual_neighbors
from .subject_registration import linear_bank,support


def translation_prior(records,sid):
    eligible=[r for r in records if r['sample_id']!=sid and split_id(r['sample_id'])[0]==split_id(sid)[0] and r.get('status')=='registered' and r.get('votes',0)>=4 and r.get('matrix')]
    if len(eligible)<3:return None
    values=np.array([np.array(r['matrix'])[:,2] for r in eligible]);center=np.median(values,axis=0)
    scale=np.maximum(1.4826*np.median(np.abs(values-center),axis=0),60.)
    return dict(center=center.tolist(),scale=scale.tolist(),anchor_ids=[r['sample_id'] for r in eligible])


def weight(translation,prior):
    z=(np.asarray(translation)-np.asarray(prior['center']))/np.asarray(prior['scale'])
    return np.exp(-.125*np.sum(z*z,axis=-1))


def search(source,target,shape,bank,prior,grid=2.):
    if min(len(source),len(target))<8 or not bank or prior is None:return []
    field=raster(target/grid,tuple(int(np.ceil(n/grid))+1 for n in shape),1.5);candidates=[]
    for anchor_id,anchor in bank:
        for angle in (-4.,-2.,0.,2.,4.):
            theta=np.deg2rad(angle);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            for scale in (.98,1.,1.02):
                linear=rotation@anchor*scale;points=source@linear.T/grid;lower=np.floor(points.min(0))-6;points-=lower;size=np.ceil(points.max(0)).astype(int)+7
                template=raster(points,(int(size[1]),int(size[0])),1.5)
                if any(a>b for a,b in zip(template.shape,field.shape)):continue
                correlation=cv2.matchTemplate(field,template,cv2.TM_CCOEFF_NORMED)
                yy,xx=np.indices(correlation.shape);offsets=(np.stack([xx,yy],-1)-lower)*grid
                scores=np.maximum(correlation,0)*weight(offsets,prior)
                for _ in range(5):
                    _,score,_,xy=cv2.minMaxLoc(scores);x,y=xy
                    matrix=np.column_stack([linear,(np.asarray(xy)-lower)*grid])
                    candidates.append((float(score),float(correlation[y,x]),matrix,anchor_id,anchor))
                    scores[max(0,y-8):y+9,max(0,x-8):x+9]=-1
    candidates.sort(key=lambda v:v[0],reverse=True);results=[]
    for peak,correlation,matrix,anchor_id,anchor in candidates[:90]:
        for radius in (8.,6.,5.):
            a,b,_=mutual_neighbors(source@matrix[:,:2].T+matrix[:,2],target,radius)
            if len(a)<6:break
            refined,inliers=cv2.estimateAffine2D(source[a].astype(np.float32),target[b].astype(np.float32),method=cv2.RANSAC,ransacReprojThreshold=3.5,maxIters=3000,confidence=.995)
            if refined is None or inliers.sum()<6 or np.linalg.det(refined[:,:2])<=0 or np.linalg.norm(refined[:,:2]-anchor)>.12:break
            matrix=refined
        value,count=support(source,target,matrix);prior_weight=float(weight(matrix[:,2],prior))
        results.append(dict(matrix=matrix.tolist(),support=value,inliers=count,correlation=correlation,prior_weight=prior_weight,score=value*prior_weight,anchor_id=anchor_id))
    return sorted(results,key=lambda v:v['score'],reverse=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',default='outputs/stage32_contour_consensus/predicted_all.csv');p.add_argument('--geometry',default='outputs/stage9_subject_geometry/predicted_all_manifest.json');p.add_argument('--split',choices=['training','hidden_test'],default='training');p.add_argument('--output',type=Path,default=Path('outputs/stage52_translation_geometry'));a=p.parse_args()
    cv2.setNumThreads(1);root=Path('data/raw/Project_2_Dataset');rows=read_rows(a.predictions);records=json.loads(Path(a.geometry).read_text())['regions'];assert set(rows)=={r['sample_id'] for r in records}
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'regions';cache.mkdir(exist_ok=True)
    config=dict(source_sha256=sha(__file__),prediction_sha256=sha(a.predictions),geometry_sha256=sha(a.geometry),split=a.split,policy='Other-region >=4-vote anchors; robust translation sigma >=60 pixels; Gaussian scale 2; preserve high votes; replace if >=8 inliers, support >=6, and score >1.1x old.')
    cp=a.output/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');updated=[]
    for r in records:
        sid=r['sample_id'];path=cache/f'{sid}.json'
        if path.exists():updated.append(json.loads(path.read_text()));continue
        result=copy.deepcopy(r);result['translation_replaced']=False;prior=translation_prior(records,sid);result['translation_prior']=prior
        if r.get('votes',0)<4 and prior:
            shapes=[image_shape(image_path(root,a.split,sid,m)) for m in ['invivo','exvivo']]
            points=[centroids(parse_instances(rows[sid],m,shape))[1] for m,shape in zip(['invivo','exvivo'],shapes)]
            cv2.setRNGSeed(2026);candidates=search(*points,shapes[1],linear_bank(records,sid),prior);result['translation_candidates']=candidates
            old=np.array(r['matrix']);value,count=support(*points,old);oldscore=value*float(weight(old[:,2],prior));result['translation_original_score']=oldscore
            if candidates and candidates[0]['inliers']>=8 and candidates[0]['support']>=6 and candidates[0]['score']>1.1*oldscore:
                result.update(candidates[0]);result.update(status='registered',translation_replaced=True,selected_source='subject_translation_prior')
        path.write_text(json.dumps(result,indent=2)+'\n');updated.append(result);print(sid,result['translation_replaced'],flush=True)
    (a.output/'predicted_all_manifest.json').write_text(json.dumps(dict(configuration=config,regions=updated),indent=2)+'\n')

if __name__=='__main__':main()
