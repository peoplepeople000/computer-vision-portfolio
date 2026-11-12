"""Refine established baseline pairs with predicted-pair RANSAC, then reassign.

All masks remain fixed. No annotations enter geometry or assignment.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows,write_rows,split_id,validate_submission
from .stage3 import FOLDS
from .star_pairing import describe
from .joint_pair_assignment import candidate_features,probability,assign_edges
from .pair_ransac_refinement import refine
from .cli import evaluate


def infer(row,masks,features,matrix,fitted,policy):
    chosen=row['match_pairs'];centers=[{k:v.coordinates().mean(0) for k,v in m.items()} for m in masks];history=[]
    for iteration in range(2):
        a=np.asarray([centers[0][i] for i,j in chosen]).reshape(-1,2)
        b=np.asarray([centers[1][j] for i,j in chosen]).reshape(-1,2)
        matrix,report=refine(a,b,matrix);history.append(report)
        pairs,x=candidate_features(masks,features,matrix)
        chosen=assign_edges(pairs,np.mean([probability(f,x) for f in fitted],axis=0),policy['threshold'],x[:,18],policy['radius'])
        if not report['accepted']:break
    result=copy.deepcopy(row);result['match_pairs']=chosen
    return result,{'matrix':None if matrix is None else matrix.tolist(),'iterations':history,'pairs_before':len(row['match_pairs']),'pairs_after':len(chosen)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['training','hidden_test'],default='training');a=p.parse_args()
    torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    is_test=a.split=='hidden_test'
    reference_path=Path('outputs/submission_stage8/submission.csv' if is_test else 'outputs/stage13_native_relative/preserved/predicted_all.csv')
    geometry_path=Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json' if is_test else 'outputs/stage9_subject_geometry/predicted_all_manifest.json')
    reference=read_rows(reference_path);manifest=json.loads(geometry_path.read_text());matrices={r['sample_id']:np.asarray(r['matrix']) for r in manifest['regions'] if r.get('matrix') and r.get('status')=='registered'}
    if is_test:
        if set(reference)!=set(read_rows(root/'sample_submission.csv')) or set(reference)&set(truth):raise ValueError('Invalid hidden coverage')
    elif set(reference)!=set(truth) or len(reference)!=47:raise ValueError('Incomplete validation')
    fitted={};policies={};files=[reference_path,geometry_path]
    for tag,subject in FOLDS.items():
        path=Path(f'outputs/stage8_joint_iv_quality/{tag}.pt');files.append(path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        if set(state['train_ids'])!={s for s in truth if split_id(s)[0]!=subject} or state['held_out']!=subject:raise ValueError('Wrong fitted partition')
        model=torch.nn.Linear(33,1);model.load_state_dict(state['state_dict']);fitted[tag]=model.eval(),state['mean'],state['sd'];policies[tag]=state['selected']
    out=a.output
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh output')
    out.mkdir(parents=True,exist_ok=True);rows=[];diagnostics=[]
    for sid,row in reference.items():
        if is_test:
            folder=Path('outputs/submission_ensemble_maps/ensemble');iv_path=folder/f'{sid}_invivo.npz';ex_path=folder/f'{sid}_exvivo.npz'
            models=list(fitted.values());policy={k:float(np.median([v[k] for v in policies.values()])) for k in ['threshold','radius']}
        else:
            tag=next(k for k,v in FOLDS.items() if v==split_id(sid)[0]);models=[fitted[tag]];policy=policies[tag]
            iv_path=Path(f'outputs/stage8_iv/{tag}/maps/{sid}_invivo.npz');ex_path=Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')
        files.extend([iv_path,ex_path]);masks,features=describe(row,iv_path,ex_path)
        result,report=infer(row,masks,features,matrices.get(sid),models,policy);rows.append(result);report['sample_id']=sid;diagnostics.append(report)
        print('Baseline RANSAC',sid,report['pairs_before'],report['pairs_after'],flush=True)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    path=out/('submission.csv' if is_test else 'predicted_all.csv');write_rows(rows,path)
    config={'split':a.split,'source_sha256':digest(__file__),'input_sha256':{str(p):digest(p) for p in files},'regions':diagnostics,
        'dependencies_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['pair_ransac_refinement.py','joint_pair_assignment.py','star_pairing.py','pair_quality.py','ex_joint_calibration.py']},
        'policy':'Start with existing predicted baseline pairs, at most two conservative affine RANSAC/refit iterations, frozen pair-model reassignment; identical masks.',
        'annotation_use':'Training IDs only verify fitted model partitions; no annotations enter inference. Evaluation is separate.'}
    (out/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    if is_test:
        (out/'validation.json').write_text(json.dumps({'status':'passed','counts':validate_submission(root,path),'csv_sha256':digest(path),'uploaded':False,'hidden_score':None,'adaptation':'Three-fold hidden ensemble; validation uses one excluded-mouse model.'},indent=2)+'\n')
    else:evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'metrics.json'))

if __name__=='__main__':main()
