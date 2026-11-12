"""Training-only pair-credit calibration using geometry and mask confidence.

Training correspondences fit training image transforms only. Evaluation
transforms are supplied by held-out registration consensus, never by labels.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
from .data import read_rows,parse_instances,write_rows,image_path,read_image,image_shape,split_id
from .matching import centroids
from .registration import assign_pairs
from .cell_features import extract_features
from .metrics import segmentation_score
from .learning import instances_from_maps
from .rle import labels_to_instances
from .stage3 import FOLDS,quality_probabilities,development_signature
from .cli import evaluate


def pair_features(masks,features,pairs,matrix):
    keys,points=zip(*(centroids(m) for m in masks));indices=[{k:i for i,k in enumerate(ks)} for ks in keys]
    costs=cdist(points[0]@matrix[:,:2].T+matrix[:,2],points[1]);rows=[]
    columns=[0,1,2,3,4,26,27,28,29]
    for a,b in pairs:
        i,j=indices[0][a],indices[1][b]
        alternative=min(np.delete(costs[i],j).min(initial=100),np.delete(costs[:,j],i).min(initial=100))
        rows.append(np.r_[features[0][i,columns],features[1][j,columns],costs[i,j],min(alternative,20),
                          costs[i,j]/max(alternative,.01),features[1][j,0]-features[0][i,0]])
    return np.asarray(rows,np.float32).reshape(-1,22)


def fit(records):
    x=np.concatenate([r['x'] for r in records]);y=np.concatenate([r['y'] for r in records]).astype(np.float32)
    mean=x.mean(0);sd=np.maximum(x.std(0),.03);xt=torch.from_numpy(((x-mean)/sd).clip(-8,8));yt=torch.from_numpy(y)
    torch.manual_seed(2030);model=nn.Linear(22,1);optimizer=torch.optim.LBFGS(model.parameters(),max_iter=150,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(xt).ravel(),yt)+.02*model.weight.square().mean();loss.backward();return loss
    optimizer.step(closure);return model.eval(),mean,sd


def probabilities(fitted,x):
    model,mean,sd=fitted
    with torch.inference_mode():return model(torch.from_numpy(((x-mean)/sd).clip(-8,8))).sigmoid().numpy().ravel()


def training_record(root,sid,truth,base,newmaps,threshold,quality,quality_transform=None):
    shapes=[image_shape(image_path(root,'training',sid,m)) for m in ['invivo','exvivo']]
    row=copy.deepcopy(base);row['exvivo_instances']=labels_to_instances(instances_from_maps(newmaps,threshold,.35),'EXP')
    masks=[parse_instances(row,m,s) for m,s in zip(['invivo','exvivo'],shapes)]
    eximage=read_image(image_path(root,'training',sid,'exvivo'));keys,_,xf=extract_features(eximage,masks[1],newmaps)
    quality_x=xf.copy();quality_x[:,5:26]=0
    if quality_transform is not None:quality_x=quality_transform(quality_x)
    p=quality_probabilities(quality[:3],quality_x)
    keep={k for k,v in zip(keys,p) if v>=quality[3]};row['exvivo_instances']={k:v for k,v in row['exvivo_instances'].items() if k in keep}
    masks[1]={k:v for k,v in masks[1].items() if k in keep}
    actual=[parse_instances(truth,m,s) for m,s in zip(['invivo','exvivo'],shapes)]
    pairs=truth['match_pairs']
    if len(pairs)<5:return None
    a=np.array([actual[0][i].coordinates().mean(0) for i,j in pairs]);b=np.array([actual[1][j].coordinates().mean(0) for i,j in pairs])
    matrix=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0].T
    return row,masks,actual,matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refined-iv',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    current=read_rows('outputs/stage7_consensus/predicted_all.csv')
    transforms={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(Path('outputs/stage7_consensus/predicted_all_manifest.json').read_text())['regions'] if r.get('matrix')}
    thresholds=json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    output=Path('outputs/stage7_pair_quality_refined' if args.refined_iv else 'outputs/stage7_pair_quality');output.mkdir(exist_ok=True);allrows=[];reports={}
    for tag,heldout in FOLDS.items():
        folder=Path(f'outputs/stage3/{tag}');base=read_rows(folder/'development.csv')
        if set(base)!={s for s in truth if split_id(s)[0]!=heldout}:raise ValueError('Held-out data in development')
        if json.loads((folder/'development_manifest.json').read_text())!=development_signature(f'outputs/stage2/unet_holdout_{tag}/model.pt'):raise ValueError('Stale baseline maps')
        state=torch.load(f'outputs/stage6_specialist_quality/{tag}/model.pt',map_location='cpu',weights_only=False)
        if state['held_out']!=heldout or set(state['train_ids'])!=set(base):raise ValueError('Invalid EX quality fold')
        q=nn.Linear(30,1);q.load_state_dict(state['quality']);quality=(q.eval(),state['mean'],state['sd'],state['threshold'])
        if args.refined_iv:
            iv_folder=Path(f'outputs/stage7_iv_development/{tag}')
            provenance=json.loads((iv_folder/'manifest.json').read_text())
            expected_hash=hashlib.sha256(Path(f'outputs/stage5_refinement/unet_holdout_{tag}/model.pt').read_bytes()).hexdigest()
            if set(provenance['train_ids'])!=set(base) or provenance['checkpoint_sha256']!=expected_hash:raise ValueError('Wrong refined IV cache')
            iv_rows=read_rows(iv_folder/'predictions.csv')
            for sid in base:base[sid]['invivo_instances']=iv_rows[sid]['invivo_instances']
        records=[]
        for sid,row in base.items():
            newmaps=np.load(f'outputs/stage6_ex/development_{tag}/{sid}_exvivo.npz')['maps'].astype(np.float32)
            record=training_record(root,sid,truth[sid],row,newmaps,thresholds[heldout]['threshold'],quality)
            if record is None:continue
            row,masks,actual,matrix=record;features=[]
            for mode,modality in enumerate(['invivo','exvivo']):
                maps=np.load((iv_folder if args.refined_iv else folder/'development_maps')/f'{sid}_invivo.npz')['maps'].astype(np.float32) if mode==0 else newmaps
                features.append(extract_features(read_image(image_path(root,'training',sid,modality)),masks[mode],maps)[2])
            proposed=assign_pairs(*masks,matrix,radius=5,ambiguity_ratio=.8)
            mapping=[segmentation_score(m,g)['pred_to_gt'] for m,g in zip(masks,actual)];verified=set(map(tuple,truth[sid]['match_pairs']))
            x=pair_features(masks,features,proposed,matrix)
            y=np.array([(mapping[0].get(a),mapping[1].get(b)) in verified for a,b in proposed])
            records.append({'sample_id':sid,'subject':split_id(sid)[0],'x':x,'y':y,'verified':len(verified)})
        subjects=sorted({r['subject'] for r in records});inner=[]
        for subject in subjects:
            fitted=fit([r for r in records if r['subject']!=subject])
            inner.extend((r,probabilities(fitted,r['x'])) for r in records if r['subject']==subject)
        measurements=[]
        for threshold in (0.,.1,.15,.2,.25,.3,.35,.4,.45,.5):
            correct=sum(int(r['y'][p>=threshold].sum()) for r,p in inner);count=sum(int((p>=threshold).sum()) for r,p in inner);verified=sum(r['verified'] for r,p in inner)
            measurements.append({'threshold':threshold,'f1':2*correct/(count+verified)})
        chosen=max(measurements,key=lambda r:r['f1'])['threshold'];fitted=fit(records)
        torch.save({'state_dict':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],'threshold':chosen,'held_out':heldout,'train_ids':list(base)},output/f'{tag}.pt')
        reports[heldout]={'train_ids':list(base),'threshold':chosen,'inner_scores':measurements,'training_alignment':'Affine least squares from verified training pairs only','regions':[]}
        for sid,row in current.items():
            if split_id(sid)[0]!=heldout:continue
            value=copy.deepcopy(row)
            if sid in transforms:
                masks=[];features=[]
                for mode,modality in enumerate(['invivo','exvivo']):
                    image=read_image(image_path(root,'training',sid,modality));m=parse_instances(row,modality,image.shape);masks.append(m)
                    cache=f'outputs/stage5_refinement/unet_holdout_{tag}_maps' if mode==0 else f'outputs/stage6_ex/unet_holdout_{tag}_maps'
                    maps=np.load(f'{cache}/{sid}_{modality}.npz')['maps'].astype(np.float32)
                    features.append(extract_features(image,m,maps)[2])
                p=probabilities(fitted,pair_features(masks,features,row['match_pairs'],transforms[sid]))
                value['match_pairs']=[pair for pair,v in zip(row['match_pairs'],p) if v>=chosen]
                reports[heldout]['regions'].append({'sample_id':sid,'probabilities':p.tolist(),'input_pairs':row['match_pairs']})
            allrows.append(value)
        print('Pair threshold',heldout,chosen,flush=True)
    path=output/'predicted_all.csv';write_rows(allrows,path)
    (output/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=output/'predicted_all_metrics.json'))


if __name__=='__main__':main()
