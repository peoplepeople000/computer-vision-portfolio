"""Build hidden predictions from all-training CPDINO IV and calibrated fold heads.

This is an explicit final-training adaptation of the audited outer validation:
one IV network sees all 47 training regions, while three pair/quality heads are
averaged and their training-derived decision thresholds are combined by median.
No hidden labels or leaderboard-dependent parameter selection are used.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,torch
from .data import read_rows,write_rows,split_id,validate_submission,parse_instances,image_shape,image_path
from .stage3 import FOLDS,quality_probabilities
from .joint_pair_assignment import candidate_features,probability,assign_edges
from .star_pairing import describe
from .relative_ex_quality import relative_features
from .cellpose4_quality import preserve_iv_pairs


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source,heads,baseline,geometry,exmaps):
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    hidden=read_rows(root/'sample_submission.csv');base=read_rows(baseline)
    candidate=read_rows(source/'predictions.csv');cfg=json.loads((source/'configuration.json').read_text())
    if set(candidate)!=set(hidden) or set(base)!=set(hidden):raise ValueError('Incomplete hidden coverage')
    if cfg['held_out'] is not None or cfg['train_ids']!=sorted(truth) or cfg['steps']!=1200:raise ValueError('Invalid final training partition/budget')
    if cfg['source_sha256']!=digest(Path(__file__).with_name('cpdino_iv_adaptation.py')):raise ValueError('Changed trained source')
    complete=json.loads((source/'completion.json').read_text())
    if complete['checkpoint_sha256']!=digest(source/'model.pt') or complete['csv_sha256']!=digest(source/'predictions.csv'):raise ValueError('Changed model or predictions')
    audit=json.loads((heads/'full/filtered_audit.json').read_text())
    if audit['status']!='passed' or audit['csv_sha256']!=digest(heads/'full/filtered.csv'):raise ValueError('Validation audit missing/stale')
    pm=[];qm=[];thresholds=[];radii=[];qt=[]
    inputs=[baseline,geometry,source/'configuration.json',source/'completion.json',source/'model.pt',source/'predictions.csv',heads/'full/filtered_audit.json']
    for tag,subject in FOLDS.items():
        expected=sorted(s for s in truth if split_id(s)[0]!=subject)
        pairpath=heads/tag/'pairs.pt';qpath=heads/tag/'quality.pt'
        pair=torch.load(pairpath,map_location='cpu',weights_only=False);quality=torch.load(qpath,map_location='cpu',weights_only=False)
        for state in [pair,quality]:
            if state['held_out']!=subject or state['train_ids']!=expected:raise ValueError('Invalid calibration partition')
        m=torch.nn.Linear(33,1);m.load_state_dict(pair['state_dict']);pm.append((m.eval(),pair['mean'],pair['sd']))
        m=torch.nn.Linear(30,1);m.load_state_dict(quality['quality']);qm.append((m.eval(),quality['mean'],quality['sd']))
        thresholds.append(pair['selected']['threshold']);radii.append(pair['selected']['radius']);qt.append(quality['threshold'])
        inputs.extend([pairpath,qpath,heads/tag/'manifest.json'])
    policy=dict(pair_threshold=float(np.median(thresholds)),radius=float(np.median(radii)),quality_threshold=float(np.median(qt)))
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(geometry.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    rows=[];diagnostics=[]
    for sid in sorted(hidden):
        row=copy.deepcopy(base[sid]);row['invivo_instances']=candidate[sid]['invivo_instances'];row['match_pairs']=[]
        ivpath=source/'maps'/f'{sid}_invivo.npz';expath=exmaps/f'{sid}_exvivo.npz';inputs.extend([ivpath,expath])
        masks,features=describe(row,ivpath,expath);pairs,x=candidate_features(masks,features,matrices.get(sid))
        p=np.mean([probability(m,x) for m in pm],axis=0)
        row['match_pairs']=assign_edges(pairs,p,policy['pair_threshold'],x[:,18],policy['radius'])
        q=np.mean([quality_probabilities(m,relative_features(features[0])) for m in qm],axis=0)
        keep={k for k,v in zip(masks[0],q) if v>=policy['quality_threshold']}
        seg=copy.deepcopy(row);seg['match_pairs']=[];seg['invivo_instances']={k:v for k,v in row['invivo_instances'].items() if k in keep}
        row=preserve_iv_pairs(row,seg)
        if row['exvivo_instances']!=base[sid]['exvivo_instances']:raise ValueError('EX segmentation unexpectedly changed')
        for mode in ('invivo','exvivo'):
            shape=image_shape(image_path(root,'hidden_test',sid,mode));occupied=np.zeros(shape,bool)
            for mask in parse_instances(row,mode,shape).values():
                xy=mask.coordinates()
                if occupied[xy[:,1],xy[:,0]].any():raise ValueError('Overlapping cells')
                occupied[xy[:,1],xy[:,0]]=True
        rows.append(row);diagnostics.append(dict(sample_id=sid,iv=len(row['invivo_instances']),ex=len(row['exvivo_instances']),pairs=len(row['match_pairs']),registered=sid in matrices))
    provenance=dict(policy=policy,regions=diagnostics,input_sha256={str(p):digest(p) for p in inputs},
        source_sha256=digest(__file__),training_ids=sorted(truth),hidden_ids=sorted(hidden),
        limitation='Final IV model uses three training mice versus two in validation; fold classifiers are averaged. Hidden performance is unknown; the validation score is not a Kaggle score.')
    return rows,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/stage35_cpdino_iv_adapted/all-training'))
    p.add_argument('--heads',type=Path,default=Path('outputs/stage40_adapted_iv_pairing'))
    p.add_argument('--output',type=Path,default=Path('outputs/submission_stage40_adapted_iv'))
    a=p.parse_args();torch.set_num_threads(1)
    baseline=Path('outputs/submission_stage32_cellpose_consensus/submission.csv')
    geometry=Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json')
    exmaps=Path('outputs/submission_ensemble_maps/ensemble')
    rows,provenance=build(a.source,a.heads,baseline,geometry,exmaps)
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/'submission.csv'
    if path.exists():raise FileExistsError('Preserve existing delivery; choose another output')
    write_rows(rows,path);counts=validate_submission('data/raw/Project_2_Dataset',path)
    if read_rows(path)!={r['sample_id']:r for r in rows}:raise ValueError('CSV roundtrip mismatch')
    (a.output/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (a.output/'validation.json').write_text(json.dumps(dict(status='passed',counts=counts,csv_sha256=digest(path),checks=['29 exact hidden IDs','RLE and pair integrity','nonoverlapping masks','unchanged EX masks','CSV roundtrip','all-training IV provenance','audited validation heads']),indent=2)+'\n')
    print(json.dumps(dict(path=str(path),counts=counts,sha256=digest(path)),indent=2),flush=True)


if __name__=='__main__':main()
