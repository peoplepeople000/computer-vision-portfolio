"""Predict per-instance EX contour IoU gains using training mice only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy import ndimage as ndi
from .data import read_rows,write_rows,parse_instances,split_id
from .rle import Mask,labels_to_instances
from .ex_boundary import patch_rle
from .ex_joint_calibration import morphology_features
from .ex_contour_calibration import adjust
from .learning import instances_from_maps
from .stage3 import FOLDS,quality_probabilities
from .cli import evaluate

POLICIES=[(0.,0.),(-.1,0.),(-.05,0.),(.05,0.),(.1,0.),(0.,.6)]
COLUMNS=[0,1,2,3,4,26,27,28,29]


def candidates(masks,maps,threshold):
    """All policies share a nearest-original-mask ownership partition."""
    labels=np.zeros(maps.shape[1:],np.int32)
    for i,m in enumerate(masks.values(),1):
        for a,b in zip(m.starts,m.ends):labels.ravel()[a:b]=i
    indices=ndi.distance_transform_edt(labels==0,return_distances=False,return_indices=True)
    owner=labels[tuple(indices)];del indices
    stacks=[];features=[]
    for delta,sigma in POLICIES:
        proposed=adjust(masks,maps,threshold,delta,sigma);parsed={}
        for i,(key,m) in enumerate(masks.items(),1):
            candidate=Mask.parse(proposed[key],m.shape);xy=candidate.coordinates();xy=xy[owner[xy[:,1],xy[:,0]]==i]
            xmin,ymin,xmax,ymax=candidate.bbox;patch=np.zeros((ymax-ymin+1,xmax-xmin+1),bool)
            patch[xy[:,1]-ymin,xy[:,0]-xmin]=True
            new=Mask.parse(patch_rle(patch,xmin,ymin,m.shape),m.shape)
            parsed[key]=new if .65<=new.area/m.area<=1.4 and new.iou(m)>.6 else m
        stacks.append(parsed);features.append(morphology_features(parsed,maps)[1][:,COLUMNS])
    f=np.stack(features);difference=f-f[0];base=np.broadcast_to(f[0],f.shape)
    # Zero change has exactly zero features and hence exactly zero predicted gain.
    x=np.concatenate([difference,base*difference,difference**2],axis=2)
    return stacks,x


def fit(records):
    x=np.concatenate([r['x'].reshape(-1,27) for r in records]);y=np.concatenate([r['gain'].ravel() for r in records])
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),.001);z=x/scale
    coefficient=np.linalg.solve(z.T@z+np.eye(27)*len(z)*.01,z.T@y)
    return coefficient,scale


def choose(fitted,x,margin):
    coefficient,scale=fitted;scores=(x/scale)@coefficient
    scores[1:]-=margin
    return np.argmax(scores,axis=0)


def main():
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    baseline=read_rows('outputs/stage7_tiles/native/predicted_all.csv');calibration=json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    out=Path('outputs/stage8_ex_ranker');out.mkdir(exist_ok=True);results=[];reports={}
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject};state=torch.load(f'outputs/stage6_specialist_quality/{tag}/model.pt',map_location='cpu',weights_only=False)
        config=calibration[subject];manifest=json.loads(Path(f'outputs/stage6_ex/development_{tag}/manifest.json').read_text())
        if set(state['train_ids'])!=ids or state['held_out']!=subject or set(config['train_ids'])!=ids or set(manifest['train_ids'])!=ids:raise ValueError('Invalid training fold')
        model=nn.Linear(30,1);model.load_state_dict(state['quality']);quality=(model.eval(),state['mean'],state['sd']);records=[]
        for sid in sorted(ids):
            maps=np.load(f'outputs/stage6_ex/development_{tag}/{sid}_exvivo.npz')['maps'].astype(np.float32)
            instances=labels_to_instances(instances_from_maps(maps,config['threshold'],.35),'EXP');masks=parse_instances({'exvivo_instances':instances},'exvivo',maps.shape[1:])
            keys,features=morphology_features(masks,maps);p=quality_probabilities(quality,features);masks={k:masks[k] for k,v in zip(keys,p) if v>=state['threshold']}
            stacks,x=candidates(masks,maps,config['threshold']);actual=parse_instances(truth[sid],'exvivo',maps.shape[1:])
            ious=np.asarray([[max((m.iou(g) for g in actual.values()),default=0.) for m in candidates_.values()] for candidates_ in stacks])
            records.append({'sample_id':sid,'subject':split_id(sid)[0],'x':x,'gain':ious-ious[0],'ious':ious,'gt_count':len(actual)})
            print('Contour ranker development',tag,sid,flush=True)
        scored=[]
        for mouse in sorted({r['subject'] for r in records}):
            fitted=fit([r for r in records if r['subject']!=mouse])
            scored.extend((r,fitted) for r in records if r['subject']==mouse)
        measurements=[]
        for margin in (0.,.005,.01,.02,.05,1.):
            values=[]
            for r,fitted in scored:
                selected=choose(fitted,r['x'],margin);ious=r['ious'][selected,np.arange(len(selected))]
                values.append(float(ious[ious>.75].sum()/(.5*(r['gt_count']+len(selected)))))
            measurements.append({'margin':margin,'mean_pq':float(np.mean(values))})
        selected=max(measurements,key=lambda v:(v['mean_pq'],v['margin']));fitted=fit(records)
        np.savez(out/f'{tag}.npz',coefficient=fitted[0],scale=fitted[1])
        reports[subject]={'train_ids':sorted(ids),'selected':selected,'inner_scores':measurements,'policy_counts':[0]*len(POLICIES)}
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            maps=np.load(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')['maps'].astype(np.float32);masks=parse_instances(row,'exvivo',maps.shape[1:])
            stacks,x=candidates(masks,maps,config['threshold']);selected_policy=choose(fitted,x,selected['margin']);instances={}
            for n,(key,index) in enumerate(zip(masks,selected_policy)):
                m=stacks[index][key];instances[key]=' '.join(f'{a} {b-a}' for a,b in zip(m.starts,m.ends));reports[subject]['policy_counts'][index]+=1
            value=copy.deepcopy(row);value['exvivo_instances']=instances;value['match_pairs']=[];results.append(value)
        (out/'calibration.json').write_text(json.dumps({'policies':POLICIES,'folds':reports,'caveat':'Inner quality-model mouse holdout, with segmentation in-sample development predictions at 256px versus 128px held-out tiles.'},indent=2)+'\n')
        print('Contour ranker selection',tag,selected,flush=True)
    path=out/'segmentation_oof.csv';write_rows(results,path)
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))

if __name__=='__main__':main()
