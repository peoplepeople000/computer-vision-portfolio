"""Refit pair credit and unpaired IV quality for adapted CPDINO masks.

Every segmentation and calibration model excludes the outer mouse. Development
affines use development labels only; evaluation uses frozen image-only affines.
Pairs are assigned before unpaired IV filtering, preserving their exact masks.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows, write_rows, parse_instances, split_id
from .stage3 import FOLDS, calibrate_quality, quality_probabilities
from .pair_quality import training_record
from .ex_joint_calibration import morphology_features
from .relative_ex_quality import relative_features
from .joint_pair_assignment import candidate_features, fit, probability, assign_edges
from .cellpose4_quality import preserve_iv_pairs
from .star_pairing import describe
from .metrics import segmentation_score
from .cli import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', choices=list(FOLDS), required=True)
    p.add_argument('--source',type=Path,default=Path('outputs/stage35_cpdino_iv_adapted'))
    p.add_argument('--output',type=Path,default=Path('outputs/stage40_adapted_iv_pairing'))
    a=p.parse_args();torch.set_num_threads(1)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    subject=FOLDS[a.fold];ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    source=a.source/a.fold
    cfg_path=source/'configuration.json';cfg=json.loads(cfg_path.read_text())
    if cfg['held_out']!=subject or cfg['train_ids']!=ids:
        raise ValueError('Invalid IV model fold')
    missing=[s for s in truth if not (source/'regions'/f'{s}.json').exists() or not (source/'maps'/f'{s}_invivo.npz').exists()]
    if missing:raise ValueError(f'Missing full development coverage: {len(missing)} images')
    baseline_path=Path('outputs/stage32_contour_consensus/predicted_all.csv');baseline=read_rows(baseline_path)
    quality_path=Path(f'outputs/stage13_native_relative/{a.fold}.pt')
    state=torch.load(quality_path,map_location='cpu',weights_only=False)
    if state['held_out']!=subject or set(state['train_ids'])!=set(ids):raise ValueError('Invalid EX quality fold')
    qm=torch.nn.Linear(30,1);qm.load_state_dict(state['quality'])
    exquality=(qm.eval(),state['mean'],state['sd'],state['threshold'])
    calibration_path=Path('outputs/stage13_native_relative/calibration.json')
    calibration=json.loads(calibration_path.read_text())[subject]
    if set(calibration['train_ids'])!=set(ids):raise ValueError('Invalid EX foreground calibration')
    exdev=Path(f'outputs/stage11_native_ex/{a.fold}/maps')
    devconfig=json.loads((exdev.parent/'training_config.json').read_text())
    if devconfig['held_out']!=subject or set(devconfig['train_ids'])!=set(ids) or devconfig['tile']!=128:raise ValueError('Invalid EX development predictions')
    records=[];quality_records=[]
    files=[cfg_path,source/'model.pt',baseline_path,quality_path,calibration_path,exdev.parent/'training_config.json']
    for sid in ids:
        row_path=source/'regions'/f'{sid}.json';row=json.loads(row_path.read_text())
        iv_path=source/'maps'/f'{sid}_invivo.npz';ex_path=exdev/f'{sid}_exvivo.npz'
        files.extend([row_path,iv_path,ex_path])
        ivmaps=np.load(iv_path)['maps'].astype(np.float32)
        ivmasks=parse_instances(row,'invivo',ivmaps.shape[1:])
        actual_iv=parse_instances(truth[sid],'invivo',ivmaps.shape[1:])
        keys,features=morphology_features(ivmasks,ivmaps)
        mapping=segmentation_score(ivmasks,actual_iv)['pred_to_gt']
        quality_records.append(dict(sample_id=sid,subject=split_id(sid)[0],features=[None,relative_features(features)],
            quality=np.array([k in mapping for k in keys]),gt_count=len(actual_iv),
            ious=np.array([ivmasks[k].iou(actual_iv[mapping[k]]) if k in mapping else 0. for k in keys])))
        exmaps=np.load(ex_path)['maps'].astype(np.float32)
        result=training_record(root,sid,truth[sid],row,exmaps,calibration['foreground'],exquality,quality_transform=relative_features)
        if result is None:continue
        row,masks,actual,matrix=result
        features=[morphology_features(m,v)[1] for m,v in zip(masks,[ivmaps,exmaps])]
        pairs,x=candidate_features(masks,features,matrix)
        mapping=[segmentation_score(m,g)['pred_to_gt'] for m,g in zip(masks,actual)]
        verified=set(map(tuple,truth[sid]['match_pairs']))
        y=np.array([(mapping[0].get(i),mapping[1].get(j)) in verified for i,j in pairs])
        records.append(dict(sample_id=sid,subject=split_id(sid)[0],pairs=pairs,x=x,y=y))
        print('Adapted IV pair training',a.fold,sid,len(pairs),int(y.sum()),flush=True)
    qfit,qthreshold,qmeasurements=calibrate_quality(quality_records)
    inner=[]
    for mouse in sorted({r['subject'] for r in records}):
        model=fit([r for r in records if r['subject']!=mouse])
        inner.extend((r,probability(model,r['x'])) for r in records if r['subject']==mouse)
    measurements=[];verified_count=sum(len(truth[s]['match_pairs']) for s in ids)
    for radius in (4.,5.,6.,8.):
        for threshold in (.05,.1,.15,.2,.25,.3,.35,.4,.45,.5):
            correct=count=0
            for r,prob in inner:
                chosen=assign_edges(r['pairs'],prob,threshold,r['x'][:,18],radius)
                positive={tuple(pair) for pair,y in zip(r['pairs'],r['y']) if y}
                correct+=sum(tuple(pair) in positive for pair in chosen);count+=len(chosen)
            measurements.append(dict(radius=radius,threshold=threshold,f1=2*correct/(count+verified_count)))
    selected=max(measurements,key=lambda v:v['f1']);model=fit(records)
    out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True)
    if (out/'pairs.pt').exists():raise FileExistsError('Preserve previous result; choose another output')
    torch.save(dict(state_dict=model[0].state_dict(),mean=model[1],sd=model[2],held_out=subject,train_ids=ids,selected=selected),out/'pairs.pt')
    torch.save(dict(quality=qfit[0].state_dict(),mean=qfit[1],sd=qfit[2],held_out=subject,train_ids=ids,threshold=qthreshold),out/'quality.pt')
    geometry_path=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    geometry=json.loads(geometry_path.read_text());files.append(geometry_path)
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in geometry['regions'] if r.get('matrix') and r.get('status')=='registered'}
    raw=[];filtered=[]
    for sid in sorted(set(truth)-set(ids)):
        row=copy.deepcopy(baseline[sid]);row_path=source/'regions'/f'{sid}.json'
        row['invivo_instances']=json.loads(row_path.read_text())['invivo_instances'];row['match_pairs']=[]
        iv_path=source/'maps'/f'{sid}_invivo.npz';ex_path=Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')
        files.extend([row_path,iv_path,ex_path])
        masks,features=describe(row,iv_path,ex_path)
        pairs,x=candidate_features(masks,features,matrices.get(sid))
        row['match_pairs']=assign_edges(pairs,probability(model,x),selected['threshold'],x[:,18],selected['radius'])
        raw.append(row)
        keep={k for k,p in zip(masks[0],quality_probabilities(qfit,relative_features(features[0]))) if p>=qthreshold}
        segmentation=copy.deepcopy(row);segmentation['match_pairs']=[]
        segmentation['invivo_instances']={k:v for k,v in row['invivo_instances'].items() if k in keep}
        filtered.append(preserve_iv_pairs(row,segmentation))
    for mode,rows in [('raw',raw),('filtered',filtered)]:
        path=out/f'{mode}.csv';write_rows(rows,path)
        evaluate(SimpleNamespace(data=root,predictions=path,subject=subject,output=out/f'{mode}_metrics.json'))
    dependencies=['adapted_iv_pairing.py','joint_pair_assignment.py','pair_quality.py','cellpose4_quality.py','relative_ex_quality.py','stage3.py','ex_joint_calibration.py']
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(held_out=subject,train_ids=ids,pair_policy=selected,
        pair_calibration=measurements,quality_threshold=qthreshold,quality_calibration=qmeasurements,
        input_sha256={str(p):sha(p) for p in files},source_sha256={n:sha(Path(__file__).with_name(n)) for n in dependencies},
        limitation='Inner classifier mouse holdout only; segmentation has seen both development mice. Outer mouse excluded from all fitting.'),indent=2)+'\n')


if __name__=='__main__':main()
