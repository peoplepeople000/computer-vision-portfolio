"""Calibrate IV morphology/confidence rejection on each model's training mice."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows,write_rows,parse_instances,split_id
from .ex_joint_calibration import morphology_features
from .stage3 import FOLDS,calibrate_quality,quality_probabilities
from .metrics import segmentation_score
from .cli import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capacity',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    baseline=read_rows('outputs/stage8_iv/segmentation_oof.csv' if args.capacity else 'outputs/stage7_tiles/native/predicted_all.csv')
    out=Path('outputs/stage8_iv_quality_capacity' if args.capacity else 'outputs/stage8_iv_quality');out.mkdir(exist_ok=True);results=[];reports={}
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject}
        folder=Path(f'outputs/stage8_iv/{tag}' if args.capacity else f'outputs/stage7_iv_development/{tag}')
        config=json.loads((folder/('training_config.json' if args.capacity else 'manifest.json')).read_text())
        base=read_rows(folder/('development.csv' if args.capacity else 'predictions.csv'));cache=folder/'maps' if args.capacity else folder
        if set(config['train_ids'])!=ids or set(base)!=ids:raise ValueError('Invalid IV development split')
        records=[]
        for sid,row in base.items():
            maps=np.load(cache/f'{sid}_invivo.npz')['maps'].astype(np.float32)
            masks=parse_instances(row,'invivo',maps.shape[1:]);keys,features=morphology_features(masks,maps)
            actual=parse_instances(truth[sid],'invivo',maps.shape[1:]);mapping=segmentation_score(masks,actual)['pred_to_gt']
            records.append({'sample_id':sid,'subject':split_id(sid)[0],'features':[None,features],'quality':np.array([k in mapping for k in keys]),'gt_count':len(actual),
                'ious':np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0 for k in keys])})
        fitted,threshold,measurements=calibrate_quality(records)
        torch.save({'quality':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],'threshold':threshold,'train_ids':sorted(ids),'held_out':subject},out/f'{tag}.pt')
        reports[subject]={'train_ids':sorted(ids),'threshold':threshold,'inner_scores':measurements}
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            maps=np.load((cache if args.capacity else Path('outputs/stage7_tiles/maps'))/f'{sid}_invivo.npz')['maps'].astype(np.float32)
            masks=parse_instances(row,'invivo',maps.shape[1:]);keys,features=morphology_features(masks,maps);p=quality_probabilities(fitted,features)
            keep={k for k,v in zip(keys,p) if v>=threshold};value=copy.deepcopy(row);value['invivo_instances']={k:v for k,v in row['invivo_instances'].items() if k in keep};value['match_pairs']=[];results.append(value)
        print('IV quality',tag,threshold,flush=True)
    (out/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n');path=out/'segmentation_oof.csv';write_rows(results,path)
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))

if __name__=='__main__':main()
