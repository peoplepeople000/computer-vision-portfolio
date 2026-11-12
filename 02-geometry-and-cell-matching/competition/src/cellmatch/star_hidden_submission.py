"""Create a documented radial/RANSAC hidden-test ensemble; never upload it.

The three-fold dense prediction/classifier ensemble is a final-data adaptation.
Its hidden-test score is unknown, and cannot be equated with single-fold CV.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import cv2
import numpy as np
import torch
from .data import read_rows,write_rows,read_image,image_path,split_id,validate_submission
from .stage3 import FOLDS
from .star_ex import make_model,tiled_prediction
from .star_geometry import extract_candidates,render_candidates
from .cellpose_baseline import confidence_maps
from .rle import labels_to_instances
from .star_pairing import describe
from .joint_pair_assignment import candidate_features,probability,assign_edges
from .pair_ransac_refinement import refine
from .preserve_pair_masks import merge_row


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    device=torch.device(args.device)
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    root=Path('data/raw/Project_2_Dataset')
    truth=read_rows(root/'training/train_ground_truth.csv');hidden=read_rows(root/'sample_submission.csv')
    reference_path=Path('outputs/submission_stage8/submission.csv');reference=read_rows(reference_path)
    if set(reference)!=set(hidden) or len(hidden)!=29 or set(hidden)&set(truth):raise ValueError('Invalid test coverage')
    audit_path=Path('outputs/stage25_pair_ransac/complete/audit.json')
    audit=json.loads(audit_path.read_text())
    # Delivery requires an actual complete validation gain, not a pilot gain.
    if audit['status']!='passed' or audit['regions']!=47 or not audit['ransac']:
        raise ValueError('Complete radial/RANSAC validation audit required')
    if audit['summary']['score']<=.4854699974871364:
        raise ValueError('No complete validation improvement; do not deliver this candidate')
    if digest('outputs/stage25_pair_ransac/complete/predicted_all.csv')!=audit['csv_sha256']:
        raise ValueError('Validation CSV changed')
    geometry=json.loads(args.geometry.read_text());matrices={}
    if set(r['sample_id'] for r in geometry['regions'])!=set(hidden):raise ValueError('Geometry coverage differs')
    for r in geometry['regions']:
        matrices[r['sample_id']]=np.asarray(r['matrix']) if r.get('status')=='registered' and r.get('matrix') else None
    models=[];fitted=[];thresholds=[];radii=[];provenance={}
    for tag,subject in FOLDS.items():
        expected={s for s in truth if split_id(s)[0]!=subject}
        checkpoint=Path(f'outputs/stage20_star_ex/{tag}/model.pt')
        pair_path=Path(f'outputs/stage22_star_pairing/{tag}/pairs.pt')
        fold_audit_path=Path(f'outputs/stage25_pair_ransac/{tag}/audit.json')
        fold_audit=json.loads(fold_audit_path.read_text())
        if digest(fold_audit_path)!=audit['fold_audits'][tag]['sha256'] or digest(pair_path)!=fold_audit['pair_model_sha256']:
            raise ValueError('Fold audit or pair model changed')
        original_provenance=json.loads(Path(f'outputs/stage22_star_pairing/{tag}/provenance.json').read_text())
        if digest(checkpoint)!=original_provenance['input_sha256'][str(checkpoint)]:
            raise ValueError('Radial model changed since validation')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        pairs=torch.load(pair_path,map_location='cpu',weights_only=False)
        for value in (state,pairs):
            if set(value['train_ids'])!=expected or value['held_out']!=subject:raise ValueError('Invalid training partition')
        if state['step']!=1500:raise ValueError('Wrong fixed radial budget')
        model=make_model();model.load_state_dict(state['state_dict']);models.append(model.eval())
        classifier=torch.nn.Linear(33,1);classifier.load_state_dict(pairs['state_dict'])
        fitted.append((classifier.eval(),pairs['mean'],pairs['sd']))
        thresholds.append(pairs['selected']['threshold']);radii.append(pairs['selected']['radius'])
        provenance.update({str(p):digest(p) for p in (checkpoint,pair_path)})
    policy={'threshold':float(np.median(thresholds)),'radius':float(np.median(radii))}
    config={'source_sha256':digest(__file__),'models_sha256':provenance,'geometry_sha256':digest(args.geometry),
            'reference_sha256':digest(reference_path),'validation_audit_sha256':digest(audit_path),
            'hidden_ids':sorted(hidden),'device':args.device,'pair_policy':policy,
            'recipe':'Equal average of three radial dense predictions and pair classifiers; median training-selected pair thresholds; two predicted-pair RANSAC iterations; preserve paired radial EX masks and fill with nonoverlapping baseline EX.',
            'adaptation':'Existing hidden IV ensemble and EX baseline retained; CV used single-fold IV and radial models. Ensemble hidden performance is unmeasured.',
            'annotation_use':'No hidden labels. Known training IDs used only to verify model provenance.',
            'dependencies_sha256':{name:digest(Path(__file__).with_name(name)) for name in ['star_ex.py','star_geometry.py','star_pairing.py','pair_ransac_refinement.py','joint_pair_assignment.py','preserve_pair_masks.py','cellpose_baseline.py','ex_joint_calibration.py']}}
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    config_path=out/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Changed output configuration')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    for name in ['candidates','maps','regions']:(out/name).mkdir(exist_ok=True)
    rows=[];diagnostics=[]
    for sid in sorted(hidden):
        partial=out/'regions'/f'{sid}.json';report_path=out/'regions'/f'{sid}.diagnostic.json'
        if partial.exists() and report_path.exists():
            rows.append(json.loads(partial.read_text()));diagnostics.append(json.loads(report_path.read_text()));continue
        candidate_path=out/'candidates'/f'{sid}.npz'
        if not candidate_path.exists():
            image=read_image(image_path(root,'hidden_test',sid,'exvivo'));average=None
            for tag,model in zip(FOLDS,models):
                model.to(device);value=tiled_prediction(model,image,device)
                if average is None:average=value
                else:average+=value
                model.to('cpu');del value
                if args.device=='mps':torch.mps.empty_cache()
                print('Hidden radial inference',sid,tag,flush=True)
            average/=len(models)
            points,distances,scores=extract_candidates(average[0],average[1:])
            temporary=candidate_path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,probability=average[0].astype(np.float16),points=points,radii=distances.astype(np.float16),scores=scores.astype(np.float16))
            temporary.replace(candidate_path);del average
        with np.load(candidate_path) as values:
            foreground=values['probability'].astype(np.float32)
            labels=render_candidates(values['points'],values['radii'].astype(np.float32),values['scores'].astype(np.float32),foreground.shape)
        clipped=np.clip(foreground,1e-6,1-1e-6)
        ex_path=out/'maps'/f'{sid}_exvivo.npz'
        np.savez_compressed(ex_path,maps=confidence_maps(labels,np.log(clipped/(1-clipped))).astype(np.float16))
        row=copy.deepcopy(reference[sid]);row['match_pairs']=[];row['exvivo_instances']=labels_to_instances(labels,'STAR')
        iv_path=Path('outputs/submission_ensemble_maps/ensemble')/f'{sid}_invivo.npz'
        masks,features=describe(row,iv_path,ex_path)
        centers=[{k:v.coordinates().mean(0) for k,v in m.items()} for m in masks]
        matrix=matrices[sid];history=[]
        for iteration in range(2):
            candidates,x=candidate_features(masks,features,matrix)
            p=np.mean([probability(f,x) for f in fitted],axis=0)
            chosen=assign_edges(candidates,p,policy['threshold'],x[:,18],policy['radius'])
            a=np.asarray([centers[0][i] for i,j in chosen]).reshape(-1,2)
            b=np.asarray([centers[1][j] for i,j in chosen]).reshape(-1,2)
            matrix,report=refine(a,b,matrix);history.append(report)
            if not report['accepted']:break
        candidates,x=candidate_features(masks,features,matrix)
        p=np.mean([probability(f,x) for f in fitted],axis=0)
        row['match_pairs']=assign_edges(candidates,p,policy['threshold'],x[:,18],policy['radius'])
        result=merge_row(row,reference[sid]);rows.append(result)
        report={'sample_id':sid,'matrix':None if matrix is None else matrix.tolist(),'iterations':history,
                'pairs':len(row['match_pairs']),'candidate_sha256':digest(candidate_path),
                'iv_maps_sha256':digest(iv_path),'ex_maps_sha256':digest(ex_path)}
        diagnostics.append(report);partial.write_text(json.dumps(result)+'\n');report_path.write_text(json.dumps(report,indent=2)+'\n')
        print('Hidden radial pairs',sid,len(row['match_pairs']),flush=True)
    path=out/'submission.csv';write_rows(rows,path);validation=validate_submission(root,path)
    (out/'delivery_audit.json').write_text(json.dumps({'status':'passed','counts':validation,'csv_sha256':digest(path),'uploaded':False,'hidden_score':None,'validation_score_for_single_fold_recipe':audit['summary']['score'],'ensemble_score_unknown':True},indent=2)+'\n')
    (out/'registration.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__':main()
