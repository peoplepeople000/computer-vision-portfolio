"""Joint IV foreground/seed and mask-quality selection on development mice."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows,write_rows,parse_instances,split_id
from .learning import instances_from_maps
from .rle import labels_to_instances
from .stage3 import FOLDS,calibrate_quality,quality_probabilities
from .metrics import segmentation_score
from .ex_joint_calibration import morphology_features
from .cli import evaluate

POLICIES=[(.4,.35),(.5,.35),(.6,.35),(.7,.35),(.5,.25)]


def main():
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    baseline=read_rows('outputs/stage7_tiles/native/predicted_all.csv');out=Path('outputs/stage8_iv_joint_calibration');out.mkdir(exist_ok=True);results=[];reports={}
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject};cache=Path(f'outputs/stage7_iv_development/{tag}')
        manifest=json.loads((cache/'manifest.json').read_text())
        if set(manifest['train_ids'])!=ids or manifest['held_out']!=subject:raise ValueError('Invalid IV development maps')
        records={p:[] for p in POLICIES}
        for sid in sorted(ids):
            maps=np.load(cache/f'{sid}_invivo.npz')['maps'].astype(np.float32);actual=parse_instances(truth[sid],'invivo',maps.shape[1:])
            for threshold,seed in POLICIES:
                instances=labels_to_instances(instances_from_maps(maps,threshold,seed),'IVP')
                masks=parse_instances({'invivo_instances':instances},'invivo',maps.shape[1:]);keys,features=morphology_features(masks,maps)
                mapping=segmentation_score(masks,actual)['pred_to_gt']
                records[(threshold,seed)].append({'sample_id':sid,'subject':split_id(sid)[0],'features':[None,features],'quality':np.array([k in mapping for k in keys]),
                    'gt_count':len(actual),'ious':np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0 for k in keys])})
            print('Joint IV calibration',tag,sid,flush=True)
        fitted_policies={};measurements=[]
        for policy in POLICIES:
            fitted,threshold,scores=calibrate_quality(records[policy]);fitted_policies[policy]=(fitted,threshold)
            measurements.append({'foreground':policy[0],'seed':policy[1],'quality_threshold':threshold,'mean_pq':max(v['mean_pq'] for v in scores),'inner_scores':scores})
        selected=max(measurements,key=lambda v:v['mean_pq']);policy=(selected['foreground'],selected['seed']);fitted,threshold=fitted_policies[policy]
        torch.save({'quality':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],'threshold':threshold,'foreground':policy[0],'seed_threshold':policy[1],'train_ids':sorted(ids),'held_out':subject},out/f'{tag}.pt')
        reports[subject]={'train_ids':sorted(ids),'selected':selected,'policies':measurements,
            'caveat':'Inner subject holdout for quality only; segmentation sees both development mice. Development tiles 256, held-out tiles 128.'}
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            maps=np.load(f'outputs/stage7_tiles/maps/{sid}_invivo.npz')['maps'].astype(np.float32)
            instances=labels_to_instances(instances_from_maps(maps,*policy),'IVP');masks=parse_instances({'invivo_instances':instances},'invivo',maps.shape[1:])
            keys,features=morphology_features(masks,maps);p=quality_probabilities(fitted,features);keep={k for k,v in zip(keys,p) if v>=threshold}
            value=copy.deepcopy(row);value['invivo_instances']={k:v for k,v in instances.items() if k in keep};value['match_pairs']=[];results.append(value)
        (out/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n');print('Joint IV selected',tag,selected,flush=True)
    path=out/'segmentation_oof.csv';write_rows(results,path)
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))

if __name__=='__main__':main()
