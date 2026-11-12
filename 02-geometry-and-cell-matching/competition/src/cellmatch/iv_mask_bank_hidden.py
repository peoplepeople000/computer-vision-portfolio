"""Deliver the audited IV mask-bank recipe using one all-training IV network.

Fold pair classifiers are equally averaged; their training-only radius and
threshold are combined by median. This final-training adaptation has no known
hidden score. The original EX masks and image-only registrations are retained.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,torch
from .data import read_rows,write_rows,split_id,validate_submission,parse_instances,image_path,image_shape
from .stage3 import FOLDS
from .iv_mask_bank import bank_record,choose
from .joint_pair_assignment import probability
from .cellpose4_quality import preserve_iv_pairs


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source,heads):
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');hidden=read_rows(root/'sample_submission.csv')
    baseline=Path('outputs/submission_stage32_cellpose_consensus/submission.csv');base=read_rows(baseline)
    baseaudit=json.loads(baseline.with_name('delivery_audit.json').read_text())
    if baseaudit['status']!='passed' or baseaudit['csv_sha256']!=digest(baseline):raise ValueError('Baseline delivery audit is stale')
    candidate=read_rows(source/'predictions.csv');cfg=json.loads((source/'configuration.json').read_text())
    if set(candidate)!=set(hidden) or set(base)!=set(hidden):raise ValueError('Incomplete hidden split')
    if cfg['held_out'] is not None or cfg['train_ids']!=sorted(truth) or cfg['steps']!=1200:raise ValueError('Invalid final training partition/budget')
    if cfg['source_sha256']!=digest(Path(__file__).with_name('cpdino_iv_adaptation.py')):raise ValueError('Changed trained source')
    if cfg['initializer_sha256']!=digest('models/cellpose4/cpdino-vitb'):raise ValueError('Changed initializer')
    checkpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False)
    if checkpoint['config']!=cfg or checkpoint['step']!=1200:raise ValueError('Final model provenance differs')
    del checkpoint
    done=json.loads((source/'completion.json').read_text())
    if done['checkpoint_sha256']!=digest(source/'model.pt') or done['csv_sha256']!=digest(source/'predictions.csv'):raise ValueError('Changed final checkpoint or predictions')
    audit=json.loads((heads/'full/audit.json').read_text())
    if audit['status']!='passed' or audit['csv_sha256']!=digest(heads/'full/predictions.csv'):raise ValueError('Missing/stale validation audit')
    models=[];thresholds=[];radii=[]
    files=[baseline,source/'model.pt',source/'configuration.json',source/'completion.json',source/'predictions.csv',heads/'full/audit.json']
    for tag,subject in FOLDS.items():
        path=heads/tag/'model.pt'
        expected_head=next(r['head_sha256'] for r in audit['folds'] if r['fold']==tag)
        if digest(path)!=expected_head:raise ValueError('Pair head differs from validation audit')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['held_out']!=subject or state['train_ids']!=sorted(s for s in truth if split_id(s)[0]!=subject):raise ValueError('Invalid pair head fold')
        manifest=heads/tag/'manifest.json';metadata=json.loads(manifest.read_text())
        if metadata['source_sha256']!=digest(Path(__file__).with_name('iv_mask_bank.py')):raise ValueError('Changed bank implementation')
        m=torch.nn.Linear(37,1);m.load_state_dict(state['state_dict']);models.append((m.eval(),state['mean'],state['sd']))
        thresholds.append(state['selected']['threshold']);radii.append(state['selected']['radius']);files.extend([path,manifest])
    policy=dict(threshold=float(np.median(thresholds)),radius=float(np.median(radii)))
    geometry=Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json');files.append(geometry)
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(geometry.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    rows=[];report=[]
    for sid in sorted(hidden):
        paths=[Path(f'outputs/submission_ensemble_maps/ensemble/{sid}_invivo.npz'),source/'maps'/f'{sid}_invivo.npz',Path(f'outputs/submission_ensemble_maps/ensemble/{sid}_exvivo.npz')];files.extend(paths)
        om,nm,em=[np.load(p)['maps'].astype(np.float32) for p in paths]
        old=copy.deepcopy(base[sid]);new=copy.deepcopy(old);new['invivo_instances']=candidate[sid]['invivo_instances']
        record=bank_record([old,new],[om,nm],em,matrices.get(sid));prob=np.mean([probability(m,record['x']) for m in models],axis=0)
        chosen=choose(record,prob,policy['threshold'],policy['radius'])
        paired=copy.deepcopy(old);paired['invivo_instances']={f'{b}:{k}':v for b,row in enumerate([old,new]) for k,v in row['invivo_instances'].items()}
        paired['match_pairs']=[record['pairs'][n] for n in chosen]
        row=preserve_iv_pairs(paired,old)
        if row['exvivo_instances']!=old['exvivo_instances']:raise ValueError('EX changed')
        for mode in ('invivo','exvivo'):
            shape=image_shape(image_path(root,'hidden_test',sid,mode));occupied=np.zeros(shape,bool)
            for mask in parse_instances(row,mode,shape).values():
                xy=mask.coordinates()
                if occupied[xy[:,1],xy[:,0]].any():raise ValueError('Overlapping cell masks')
                occupied[xy[:,1],xy[:,0]]=True
        rows.append(row);report.append(dict(sample_id=sid,pairs=len(row['match_pairs']),adapted_iv_pair_masks=sum(iv.startswith('1:') for iv,ex in paired['match_pairs'])))
    return rows,dict(policy=policy,regions=report,input_sha256={str(p):digest(p) for p in files},source_sha256=digest(__file__),training_ids=sorted(truth),hidden_ids=sorted(hidden),
        limitation='Final IV network trains on all three mice; pair classifiers are averaged. The local outer-fold score is not a hidden-test score. EX and legacy IV maps inherit the earlier final ensemble adaptation.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('outputs/stage35_cpdino_iv_adapted/all-training'))
    p.add_argument('--heads',type=Path,default=Path('outputs/stage43_iv_mask_bank'));p.add_argument('--output',type=Path,default=Path('outputs/submission_stage43_iv_mask_bank'));a=p.parse_args();torch.set_num_threads(1)
    rows,manifest=build(a.source,a.heads);a.output.mkdir(parents=True,exist_ok=True);path=a.output/'submission.csv'
    if path.exists():raise FileExistsError('Preserve previous delivery')
    write_rows(rows,path);counts=validate_submission('data/raw/Project_2_Dataset',path)
    if read_rows(path)!={r['sample_id']:r for r in rows}:raise ValueError('CSV roundtrip mismatch')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'validation.json').write_text(json.dumps(dict(status='passed',counts=counts,csv_sha256=digest(path),uploaded=False,hidden_score=None),indent=2)+'\n')
    print(json.dumps(dict(path=str(path),counts=counts,csv_sha256=digest(path)),indent=2),flush=True)


if __name__=='__main__':main()
