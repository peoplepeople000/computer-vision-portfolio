"""Refit morphology/confidence quality on calibrated EX development predictions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows,write_rows,parse_instances,read_image,image_path,split_id
from .learning import instances_from_maps
from .rle import labels_to_instances
from .cell_features import extract_features
from .stage3 import FOLDS,calibrate_quality,quality_probabilities
from .metrics import segmentation_score
from .conservative_ex import rejected_keys
from .cli import evaluate


def load_ex_maps(tag,sid,variant,development):
    base=Path(f'outputs/stage6_ex/development_{tag}' if development else f'outputs/stage6_ex/unet_holdout_{tag}_maps')
    maps=np.load(base/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
    if variant=='ensemble':
        old=Path(f'outputs/stage3/{tag}/development_maps' if development else f'outputs/stage2/unet_holdout_{tag}_maps')
        maps=(maps+np.load(old/f'{sid}_exvivo.npz')['maps'].astype(np.float32))/2
    return maps


def record(root,sid,row,maps,truth=None):
    image=read_image(image_path(root,'training',sid,'exvivo'));masks=parse_instances(row,'exvivo',image.shape)
    keys,_,features=extract_features(image,masks,maps);features[:,5:26]=0
    result={'sample_id':sid,'subject':split_id(sid)[0],'keys':keys,'features':[None,features]}
    if truth is not None:
        gt=parse_instances(truth,'exvivo',image.shape);mapping=segmentation_score(masks,gt)['pred_to_gt']
        result.update(quality=np.array([k in mapping for k in keys]),gt_count=len(gt),
            ious=np.array([masks[k].iou(gt[mapping[k]]) if k in mapping else 0 for k in keys]))
    return result


def main():
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    for variant in ['specialist','ensemble']:
        source=Path(f'outputs/stage6_{variant}_calibrated');calibration=json.loads((source/'calibration.json').read_text())
        predictions=read_rows(source/'segmentation_oof.csv');outputs={'quality':[],'quality_conservative':[]};reports={}
        for tag,subject in FOLDS.items():
            config=calibration[subject];expected={s for s in truth if split_id(s)[0]!=subject}
            if set(config['train_ids'])!=expected:raise ValueError('Calibration includes held-out data')
            manifest=json.loads(Path(f'outputs/stage6_ex/development_{tag}/manifest.json').read_text())
            if set(manifest['train_ids'])!=expected or manifest['held_out']!=subject:raise ValueError('Wrong development cache')
            records=[]
            for sid in sorted(expected):
                maps=load_ex_maps(tag,sid,variant,True)
                instances=labels_to_instances(instances_from_maps(maps,config['threshold'],.35),'EXP')
                records.append(record(root,sid,{'exvivo_instances':instances},maps,truth[sid]))
            fitted,threshold,measurements=calibrate_quality(records)
            model_dir=Path(f'outputs/stage6_{variant}_quality')/tag;model_dir.mkdir(parents=True,exist_ok=True)
            torch.save({'quality':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],'zero_columns':list(range(5,26)),
                'threshold':threshold,'train_ids':sorted(expected),'held_out':subject,'variant':variant},model_dir/'model.pt')
            reports[subject]={'threshold':threshold,'inner_subject_scores':measurements,'train_ids':sorted(expected),'regions':[]}
            for sid,row in predictions.items():
                if split_id(sid)[0]!=subject:continue
                value=record(root,sid,row,load_ex_maps(tag,sid,variant,False));prob=quality_probabilities(fitted,value['features'][1])
                policies={'quality':{k for k,p in zip(value['keys'],prob) if p<threshold},
                          'quality_conservative':rejected_keys(value['keys'],prob)}
                for policy,removed in policies.items():
                    result=copy.deepcopy(row);result['exvivo_instances']={k:v for k,v in row['exvivo_instances'].items() if k not in removed}
                    outputs[policy].append(result)
                    reports[subject]['regions'].append({'sample_id':sid,'policy':policy,'removed':len(removed)})
            print('EX quality fitted',variant,subject,threshold,flush=True)
        for policy,rows in outputs.items():
            out=Path(f'outputs/stage6_{variant}_{policy}');out.mkdir(exist_ok=True)
            path=out/'segmentation_oof.csv';write_rows(rows,path)
            (out/'quality_calibration.json').write_text(json.dumps(reports,indent=2)+'\n')
            evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
