"""Apply the selected geometry/native relative EX recipe to hidden ensembles.

Uses only stored image-derived hidden geometry and training-only fold models.
The final three-fold ensemble adaptation has no measured hidden-test score.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .data import read_rows,write_rows,split_id,validate_submission
from .stage3 import FOLDS,quality_probabilities
from .relative_ex_quality import build
from .star_pairing import describe
from .joint_pair_assignment import candidate_features,probability,assign_edges
from .preserve_pair_masks import merge_row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');expected=read_rows(root/'sample_submission.csv')
    original_path=Path('outputs/submission_stage8/submission.csv');original=read_rows(original_path)
    if set(original)!=set(expected) or len(expected)!=29 or set(expected)&set(truth):raise ValueError('Invalid split')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit_path=Path('outputs/stage13_native_relative/preserved/audit.json');audit=json.loads(audit_path.read_text())
    if audit['status']!='passed' or audit['csv_sha256']!=digest('outputs/stage13_native_relative/preserved/predicted_all.csv'):raise ValueError('Selected baseline audit stale')
    geometry_path=Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json');geometry=json.loads(geometry_path.read_text())
    if {r['sample_id'] for r in geometry['regions']}!=set(expected):raise ValueError('Hidden geometry incomplete')
    config_geometry=geometry['configuration']
    if config_geometry['prediction_sha256']!=digest(original_path) or config_geometry['source_sha256']!=digest('src/cellmatch/subject_registration.py'):raise ValueError('Geometry provenance differs')
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in geometry['regions'] if r.get('status')=='registered' and r.get('matrix')}
    models=[];quality=[];policies=[];foreground=[];thresholds=[];files=[original_path,geometry_path,audit_path]
    map_root=Path('outputs/submission_ensemble_maps');ensemble_manifest=json.loads((map_root/'ensemble_manifest.json').read_text())
    if set(ensemble_manifest['hidden_ids'])!=set(expected) or set(ensemble_manifest['modalities_completed'])!={'invivo','exvivo'}:raise ValueError('Missing hidden maps')
    files.append(map_root/'ensemble_manifest.json')
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject}
        pair_path=Path(f'outputs/stage8_joint_iv_quality/{tag}.pt');qpath=Path(f'outputs/stage13_native_relative/{tag}.pt');files.extend([pair_path,qpath])
        pair=torch.load(pair_path,map_location='cpu',weights_only=False);q=torch.load(qpath,map_location='cpu',weights_only=False)
        for state in [pair,q]:
            if state['held_out']!=subject or set(state['train_ids'])!=ids:raise ValueError('Invalid trained partition')
        model=torch.nn.Linear(33,1);model.load_state_dict(pair['state_dict']);models.append((model.eval(),pair['mean'],pair['sd']));policies.append(pair['selected'])
        qm=torch.nn.Linear(30,1);qm.load_state_dict(q['quality']);quality.append((qm.eval(),q['mean'],q['sd']));foreground.append(q['foreground']);thresholds.append(q['threshold'])
        for modality in ['invivo','exvivo']:
            path=map_root/modality/tag/'manifest.json';m=json.loads(path.read_text());files.append(path)
            if set(m['train_ids'])!=ids or set(m['hidden_ids'])!=set(expected) or m['tile']!=128:raise ValueError('Invalid hidden map provenance')
            if digest(m['checkpoint'])!=m['checkpoint_sha256'] or digest('src/cellmatch/learning.py')!=m['learning_sha256']:raise ValueError('Cache model/inference source changed')
    policy={k:float(np.median([v[k] for v in policies])) for k in ['threshold','radius']}
    fg=float(np.median(foreground));qt=float(np.median(thresholds));rows=[];counts=[]
    for sid,old in original.items():
        iv_path=map_root/'ensemble'/f'{sid}_invivo.npz';ex_path=map_root/'ensemble'/f'{sid}_exvivo.npz';files.extend([iv_path,ex_path])
        masks,features=describe(old,iv_path,ex_path);pairs,x=candidate_features(masks,features,matrices.get(sid))
        p=np.mean([probability(f,x) for f in models],axis=0);paired=copy.deepcopy(old);paired['match_pairs']=assign_edges(pairs,p,policy['threshold'],x[:,18],policy['radius'])
        maps=np.load(ex_path)['maps'].astype(np.float32);encoded,_,keys,f=build(maps,fg)
        probabilities=np.mean([quality_probabilities(q,f) for q in quality],axis=0);keep={k for k,v in zip(keys,probabilities) if v>=qt}
        segmentation=copy.deepcopy(old);segmentation['match_pairs']=[];segmentation['exvivo_instances']={k:v for k,v in encoded.items() if k in keep}
        row=merge_row(paired,segmentation);rows.append(row);counts.append({'sample_id':sid,'pairs':len(row['match_pairs']),'ex_masks':len(row['exvivo_instances'])})
    out=a.output;out.mkdir(parents=True,exist_ok=True);path=out/'submission.csv';write_rows(rows,path)
    report={'source_sha256':digest(__file__),'input_sha256':{str(p):digest(p) for p in files},'settings':{'foreground':fg,'quality':qt,'pairs':policy},'regions':counts,
            'policy':'Three-fold classifier means and median training thresholds; same-subject image-only geometry; relative native EX quality; exact pair-preserving EX merge.',
            'adaptation':'Equal fold ensembles on hidden mice differ from the single excluded-mouse validation models. Hidden score and rank are unknown.',
            'dependencies_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['relative_ex_quality.py','star_pairing.py','joint_pair_assignment.py','preserve_pair_masks.py','stage3.py']}}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'status':'passed','counts':validate_submission(root,path),'csv_sha256':digest(path),'uploaded':False,'hidden_score':None},indent=2)+'\n')
    print(json.dumps(report['settings']),flush=True)

if __name__=='__main__':main()
