"""Build an explicitly documented final ensemble submission without uploading it.

Final inference averages the three fold checkpoints and classifiers, and fits
an affine prior from all labeled training mice. It is an adaptation of the CV
recipe, not a claim that the final ensemble has a measured hidden-test score.
"""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from .data import read_rows,write_rows,parse_instances,split_id,validate_submission,image_shape,image_path
from .learning import instances_from_maps
from .rle import labels_to_instances
from .ex_joint_calibration import morphology_features
from .stage3 import FOLDS,quality_probabilities
from .geometry_prior import fit_prior,register
from .registration_consensus import choose_matrix
from .matching import centroids
from .joint_pair_assignment import candidate_features,probability,assign_edges


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_maps(path,shape):
    maps=np.load(path)['maps'].astype(np.float32)
    if maps.shape!=(2,*shape) or not np.isfinite(maps).all() or maps.min()<0 or maps.max()>1:raise ValueError(f'Invalid probability maps: {path}')
    return maps


def raw_masks(maps,foreground,seed,modality):
    encoded=labels_to_instances(instances_from_maps(maps,foreground,seed),'IVP' if modality=='invivo' else 'EXP')
    parsed=parse_instances({f'{modality}_instances':encoded},modality,maps.shape[1:])
    return encoded,parsed


def estimate_hypothesis(job):
    cv2.setNumThreads(1);cv2.setRNGSeed(2026)
    name,iv,ex,prior=job
    matrix,diagnostic=register(iv,ex,prior)
    return name,matrix,diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--maps',required=True);parser.add_argument('--iv-quality',required=True)
    parser.add_argument('--pairs',required=True);parser.add_argument('--output',required=True);parser.add_argument('--rescue-ex',action='store_true');parser.add_argument('--registration-workers',type=int,choices=[1,2],default=2);args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2);cv2.setRNGSeed(2026)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');hidden=read_rows(root/'sample_submission.csv');cache=Path(args.maps)
    if len(hidden)!=29 or set(truth)&set(hidden):raise ValueError('Invalid training/test split')
    ensemble_manifest=json.loads((cache/'ensemble_manifest.json').read_text())
    if set(ensemble_manifest['hidden_ids'])!=set(hidden) or set(ensemble_manifest['modalities_completed'])!={'invivo','exvivo'}:raise ValueError('Incomplete hidden probability ensemble')
    quality={m:{} for m in ['invivo','exvivo']};pair_models={};model_provenance=[];foreground={m:[] for m in quality};seeds={m:[] for m in quality};thresholds={m:[] for m in quality};pair_thresholds=[];radii=[]
    ex_calibration=json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    for tag,subject in FOLDS.items():
        expected={s for s in truth if split_id(s)[0]!=subject}
        for modality in quality:
            path=Path(args.iv_quality)/f'{tag}.pt' if modality=='invivo' else Path(f'outputs/stage6_specialist_quality/{tag}/model.pt')
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['held_out']!=subject or set(state['train_ids'])!=expected:raise ValueError('Invalid quality training partition')
            model=nn.Linear(30,1);model.load_state_dict(state['quality']);quality[modality][tag]=(model.eval(),state['mean'],state['sd'])
            foreground[modality].append(state.get('foreground',.5) if modality=='invivo' else ex_calibration[subject]['threshold'])
            seeds[modality].append(state.get('seed_threshold',.35));thresholds[modality].append(state['threshold']);model_provenance.append({'path':str(path),'sha256':digest(path),'train_ids':sorted(expected)})
            provenance=json.loads((cache/modality/tag/'manifest.json').read_text())
            if set(provenance['train_ids'])!=expected or set(provenance['hidden_ids'])!=set(hidden) or provenance['held_out']!=subject:raise ValueError('Invalid hidden map provenance')
            if digest(provenance['checkpoint'])!=provenance['checkpoint_sha256']:raise ValueError('Hidden cache checkpoint changed')
            if digest(Path(__file__).with_name('learning.py'))!=provenance['learning_sha256']:raise ValueError('Hidden cache inference implementation changed')
            model_provenance.append({'path':str(cache/modality/tag/'manifest.json'),'sha256':digest(cache/modality/tag/'manifest.json')})
        path=Path(args.pairs)/f'{tag}.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        if state['held_out']!=subject or set(state['train_ids'])!=expected or state['feature_count']!=33:raise ValueError('Invalid pair classifier')
        model=nn.Linear(33,1);model.load_state_dict(state['state_dict']);pair_models[tag]=(model.eval(),state['mean'],state['sd'])
        pair_thresholds.append(state['selected']['threshold']);radii.append(state['selected']['radius']);model_provenance.append({'path':str(path),'sha256':digest(path),'train_ids':sorted(expected)})
    settings={m:{'foreground':float(np.median(foreground[m])),'seed':float(np.median(seeds[m])),'quality':float(np.median(thresholds[m]))} for m in quality}
    settings['pairs']={'threshold':float(np.median(pair_thresholds)),'radius':float(np.median(radii))}
    prior=fit_prior(root,truth,None)
    if set(prior['train_ids'])&set(hidden):raise ValueError('Hidden IDs in geometry fitting')
    config={'arguments':vars(args),'settings':settings,'models':model_provenance,'prior':prior,'source_sha256':digest(__file__),
        'map_manifest_sha256':digest(cache/'ensemble_manifest.json'),'policy':'Equal three-fold probability/classifier ensembles; median training-derived thresholds; six predicted-mask registration hypotheses; all-training affine prior.',
        'score_warning':'Final ensemble differs from single-fold whole-mouse validation. No hidden-test score or leaderboard rank is known.'}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);manifest=out/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Different output configuration; choose a new output directory')
    manifest.write_text(json.dumps(config,indent=2)+'\n');partial=out/'regions';partial.mkdir(exist_ok=True);rows=[];diagnostics=[]
    pool=ProcessPoolExecutor(max_workers=args.registration_workers,mp_context=multiprocessing.get_context('spawn')) if args.registration_workers>1 else None
    for sid in hidden:
        saved=partial/f'{sid}.json';saved_diagnostic=partial/f'{sid}.diagnostic.json'
        if saved.exists() and saved_diagnostic.exists():
            rows.append(json.loads(saved.read_text()));diagnostics.append(json.loads(saved_diagnostic.read_text()));continue
        maps={};raw={};encoded={};filtered={};probability_by_mask={}
        for modality in quality:
            shape=image_shape(image_path(root,'hidden_test',sid,modality));maps[modality]=load_maps(cache/'ensemble'/f'{sid}_{modality}.npz',shape)
            encoded[modality],raw[modality]=raw_masks(maps[modality],settings[modality]['foreground'],settings[modality]['seed'],modality)
            keys,features=morphology_features(raw[modality],maps[modality])
            p=np.mean([quality_probabilities(fitted,features) for fitted in quality[modality].values()],axis=0)
            probability_by_mask[modality]=dict(zip(keys,p.tolist()));filtered[modality]={k:raw[modality][k] for k,v in zip(keys,p) if v>=settings[modality]['quality']}
        hypotheses=[('ensemble_filtered',filtered['invivo'],filtered['exvivo']),('ensemble_raw',raw['invivo'],raw['exvivo']),('ensemble_iv_filtered',filtered['invivo'],raw['exvivo'])]
        for index,tag in enumerate(FOLDS):
            masks=[]
            for modality in quality:
                single=load_maps(cache/modality/tag/f'{sid}.npz',maps[modality].shape[1:]);_,parsed=raw_masks(single,foreground[modality][index],seeds[modality][index],modality)
                keys,features=morphology_features(parsed,single);p=quality_probabilities(quality[modality][tag],features)
                masks.append({k:parsed[k] for k,v in zip(keys,p) if v>=thresholds[modality][index]})
            hypotheses.append((f'fold_{tag}',*masks))
        candidates=[];registration=[]
        jobs=[(name,iv,ex,prior) for name,iv,ex in hypotheses]
        estimates=pool.map(estimate_hypothesis,jobs) if pool is not None else map(estimate_hypothesis,jobs)
        for name,matrix,diag in estimates:
            registration.append({'name':name,**diag})
            if matrix is not None:candidates.append((name,matrix))
        selected_masks=[filtered['invivo'],raw['exvivo'] if args.rescue_ex else filtered['exvivo']]
        points=[centroids(m)[1] for m in selected_masks]
        if all(len(p) for p in points):matrix,diagnostic=choose_matrix(*points,candidates)
        else:matrix,diagnostic=None,{'status':'insufficient_cells'}
        features=[morphology_features(m,maps[modality])[1] for m,modality in zip(selected_masks,quality)]
        pairs,x=candidate_features(selected_masks,features,matrix);p=np.mean([probability(fitted,x) for fitted in pair_models.values()],axis=0)
        matched=assign_edges(pairs,p,settings['pairs']['threshold'],x[:,18],settings['pairs']['radius'])
        keep_ex=set(filtered['exvivo'])|({b for a,b in matched} if args.rescue_ex else set())
        row={'sample_id':sid,'invivo_instances':{k:v for k,v in encoded['invivo'].items() if k in filtered['invivo']},
             'exvivo_instances':{k:v for k,v in encoded['exvivo'].items() if k in keep_ex},'match_pairs':matched}
        diagnostic.update(sample_id=sid,registration_hypotheses=registration,raw_counts={m:len(raw[m]) for m in raw},
            kept_counts={m:len(row[f'{m}_instances']) for m in raw},pairs=len(matched),
            probability_map_sha256={m:digest(cache/'ensemble'/f'{sid}_{m}.npz') for m in raw})
        saved.write_text(json.dumps(row)+'\n');saved_diagnostic.write_text(json.dumps(diagnostic,indent=2)+'\n');rows.append(row);diagnostics.append(diagnostic)
        print('Hidden submission',sid,diagnostic['status'],len(matched),flush=True)
    if pool is not None:pool.shutdown(wait=True)
    path=out/'submission.csv';write_rows(rows,path);validation=validate_submission(root,path)
    (out/'validation.json').write_text(json.dumps({'status':'passed','counts':validation,'csv_sha256':digest(path),'uploaded':False,'hidden_test_score':None},indent=2)+'\n')
    (out/'registration.json').write_text(json.dumps(diagnostics,indent=2)+'\n');print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
