"""Low-capacity quadratic mask-quality models with development-mouse calibration."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .data import read_rows,write_rows,parse_instances,split_id
from .learning import instances_from_maps
from .rle import labels_to_instances
from .stage3 import FOLDS
from .ex_joint_calibration import morphology_features
from .metrics import segmentation_score
from .cli import evaluate

COLUMNS=[0,1,2,3,4,26,27,28,29]
UPPER=np.triu_indices(9)


def expansion(x,mean,sd):
    z=((x-mean)/sd).clip(-5,5)
    return np.c_[z,z[:,UPPER[0]]*z[:,UPPER[1]]].astype(np.float32)


def fit(records):
    x=np.concatenate([r['x'] for r in records]);y=np.concatenate([r['y'] for r in records]).astype(np.float32)
    mean=x.mean(0);sd=np.maximum(x.std(0),.03);z=expansion(x,mean,sd);center=z.mean(0);scale=np.maximum(z.std(0),.03)
    xt=torch.from_numpy((z-center)/scale);yt=torch.from_numpy(y);model=nn.Linear(54,1)
    nn.init.zeros_(model.weight);nn.init.zeros_(model.bias)
    optimizer=torch.optim.LBFGS(model.parameters(),max_iter=150,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(xt).ravel(),yt)+.03*model.weight.square().mean();loss.backward();return loss
    optimizer.step(closure);return model.eval(),mean,sd,center,scale


def probability(fitted,x):
    model,mean,sd,center,scale=fitted;z=expansion(x,mean,sd)
    with torch.inference_mode():return model(torch.from_numpy((z-center)/scale)).sigmoid().numpy().ravel()


def calibrate(records):
    inner=[]
    for mouse in sorted({r['subject'] for r in records}):
        fitted=fit([r for r in records if r['subject']!=mouse]);inner.extend((r,probability(fitted,r['x'])) for r in records if r['subject']==mouse)
    scores=[]
    for threshold in (0.,.15,.25,.35,.45,.55,.65):
        pqs=[]
        for r,p in inner:
            keep=p>=threshold;pqs.append(float(r['ious'][keep].sum()/(.5*(r['gt_count']+keep.sum()))))
        scores.append({'threshold':threshold,'mean_pq':float(np.mean(pqs))})
    selected=max(scores,key=lambda v:v['mean_pq']);return fit(records),selected['threshold'],scores


def main():
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');baseline=read_rows('outputs/stage7_tiles/native/predicted_all.csv')
    ex_calibration=json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    results={mode:[] for mode in ['invivo','exvivo','both']};reports={};out=Path('outputs/stage8_quadratic_quality');out.mkdir(exist_ok=True)
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject};models={};reports[subject]={}
        for modality in ['invivo','exvivo']:
            cache=Path(f'outputs/stage7_iv_development/{tag}' if modality=='invivo' else f'outputs/stage6_ex/development_{tag}')
            manifest=json.loads((cache/'manifest.json').read_text())
            if set(manifest['train_ids'])!=ids or manifest['held_out']!=subject:raise ValueError('Invalid development maps')
            foreground=.5 if modality=='invivo' else ex_calibration[subject]['threshold'];records=[]
            for sid in sorted(ids):
                maps=np.load(cache/f'{sid}_{modality}.npz')['maps'].astype(np.float32)
                instances=labels_to_instances(instances_from_maps(maps,foreground,.35),'IVP' if modality=='invivo' else 'EXP')
                masks=parse_instances({f'{modality}_instances':instances},modality,maps.shape[1:]);keys,x=morphology_features(masks,maps)
                actual=parse_instances(truth[sid],modality,maps.shape[1:]);mapping=segmentation_score(masks,actual)['pred_to_gt']
                records.append({'sample_id':sid,'subject':split_id(sid)[0],'x':x[:,COLUMNS],'y':np.array([k in mapping for k in keys]),
                    'ious':np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0 for k in keys]),'gt_count':len(actual)})
            fitted,threshold,scores=calibrate(records);models[modality]=(fitted,threshold,foreground)
            torch.save({'state_dict':fitted[0].state_dict(),'normalization':fitted[1:],'threshold':threshold,'foreground':foreground,'train_ids':sorted(ids),'held_out':subject,'modality':modality},out/f'{tag}_{modality}.pt')
            reports[subject][modality]={'train_ids':sorted(ids),'threshold':threshold,'foreground':foreground,'inner_scores':scores}
            print('Quadratic calibration',tag,modality,threshold,flush=True)
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            proposed={}
            for modality,(fitted,threshold,foreground) in models.items():
                maps=np.load(f'outputs/stage7_tiles/maps/{sid}_{modality}.npz')['maps'].astype(np.float32)
                instances=labels_to_instances(instances_from_maps(maps,foreground,.35),'IVP' if modality=='invivo' else 'EXP')
                masks=parse_instances({f'{modality}_instances':instances},modality,maps.shape[1:]);keys,x=morphology_features(masks,maps);p=probability(fitted,x[:,COLUMNS])
                keep={k for k,v in zip(keys,p) if v>=threshold};proposed[modality]={k:v for k,v in instances.items() if k in keep}
            for policy in results:
                value=copy.deepcopy(row)
                for modality in ['invivo','exvivo']:
                    if policy in [modality,'both']:value[f'{modality}_instances']=proposed[modality]
                value['match_pairs']=[];results[policy].append(value)
        (out/'calibration.json').write_text(json.dumps({'folds':reports,'caveat':'Development-subject classifier holdout only; segmentation sees training mice. All evaluation mice excluded from their fits.'},indent=2)+'\n')
    for policy,rows in results.items():
        folder=out/policy;folder.mkdir(exist_ok=True);path=folder/'segmentation_oof.csv';write_rows(rows,path)
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=folder/'segmentation_oof_metrics.json'))

if __name__=='__main__':main()
