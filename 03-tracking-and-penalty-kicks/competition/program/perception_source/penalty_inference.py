"""Frozen R19/R22 submission ensemble; recover and audit existing checkpoints."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import rgb_gru_data as data
import pose_tree_controls as controls

FOLDS = (0, 2)
SEEDS = (6643, 6644, 6645)


def verify_archive(root):
    root = Path(root)
    for name, digest in json.loads((root/'artifact_manifest.json').read_text()).items():
        assert data.sha(root/name)==digest


def predict(pose, body, valid, fold, models):
    """Fixed seed probabilities. XGBoost CUDA; CatBoost multiclass Kaggle CPU."""
    import torch
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import xgb_probe
    assert torch.cuda.is_available() and fold in FOLDS
    pose,body,valid = np.asarray(pose,np.float32),np.asarray(body,np.float32),np.asarray(valid,bool)
    assert pose.shape==(len(pose),189) and body.shape==(len(pose),123) and valid.shape==(len(pose),)
    assert np.isfinite(pose).all() and np.isfinite(body).all()
    meta=pose[:,186:]; assert np.isin(meta,[0,1]).all() and (meta.sum(1)==1).all()
    models=Path(models); priors=np.load(models/'priors.npy',allow_pickle=False)[FOLDS.index(fold)]
    margin=priors[meta.argmax(1)]
    result=[]
    for seed in SEEDS:
        folder=models/f'fold{fold}'/str(seed); kicks=[]
        for name,x in [('kick2d',pose),('kick3d',np.c_[pose,body].astype(np.float32))]:
            net=CatBoostClassifier();net.load_model(str(folder/(name+'.cbm')))
            assert net.tree_count_==450
            raw=net.predict(x,prediction_type='RawFormulaVal',task_type='CPU',thread_count=4)
            kicks.append(controls.softmax(raw+margin[:,:3]))
        kick=kicks[1].copy();kick[~valid]=kicks[0][~valid]
        keeper=xgb.Booster(model_file=folder/'keeper.ubj');keeper.set_param({'device':'cuda'})
        xgb_probe.gpu_config(keeper);assert keeper.num_boosted_rounds()==19
        kp=keeper.predict(xgb.DMatrix(pose,base_margin=margin[:,3:6]),iteration_range=(0,19))
        ps=[kick,kp,controls.softmax(margin[:,6:8])]
        for p in ps:
            assert np.isfinite(p).all() and (p>=0).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-5)
        result.append(ps)
    return result


def ensemble(fold_probabilities):
    """Predeclared equal fold/seed probability mean, with no private labels."""
    assert len(fold_probabilities)==2 and all(len(p)==3 for p in fold_probabilities)
    return [np.mean([p[k] for group in fold_probabilities for p in group],0) for k in range(3)]


def prepare_models(archive, folds, selection_file, cached_nlf, fold2_features, r16, r19, r22, roles2, inputs, out):
    """Copy classifiers unchanged, recover missing role0, replay saved validation."""
    import torch
    import xgboost as xgb
    import catboost
    import nlf_pose_features as nlf
    import research_loop
    import role_selector as roles
    import role_source_validation as role_validation
    import xgb_probe
    import rgb_gru as metrics
    assert torch.cuda.is_available() and xgb.__version__=='3.4.1' and catboost.__version__=='1.2.10'
    cached_nlf,fold2_features,r16,r19,r22,roles2,inputs,out=map(Path,(cached_nlf,fold2_features,r16,r19,r22,roles2,inputs,out))
    out.mkdir(parents=True,exist_ok=True);models=out/'models';models.mkdir(exist_ok=True)
    roots={'nlf':cached_nlf,'fold2_features':fold2_features,'r16':r16,'r19':r19,'r22':r22,'roles2':roles2}
    for root in roots.values():verify_archive(root)
    manifest=json.loads((inputs/'inference-inputs.json').read_text())
    for name,h in manifest['files'].items():assert data.sha(inputs/name)==h
    protocol={'source_sha256':data.sha(__file__),'source_folds_sha256':data.sha(folds),
        'input_manifest_sha256':data.sha(inputs/'inference-inputs.json'),
        'reference_manifests':{k:data.sha(v/'artifact_manifest.json') for k,v in roots.items()},
        'folds':list(FOLDS),'seeds':list(SEEDS),'ensemble':'Equal mean of six fixed fold/seed probabilities.',
        'classifier_refitting':False,'role0':'Recover missing one-tree model using frozen R9 fit; require all public selections identical.',
        'scope':'Inference preparation only; no new crop, pose or NLF inference. No score selection or tuning.'}
    data.immutable_json(out/'protocol.json',protocol)
    rows,arrays,labels,margins,ix=controls.load_inputs(archive,folds);n=len(rows)
    with np.load(selection_file) as z: selection={k:z[k] for k in z.files}
    np.testing.assert_array_equal(selection['clips'],rows['clip'])
    with np.load(fold2_features/'features.npz') as z:
        np.testing.assert_array_equal(z['clips'],rows['clip']);np.testing.assert_array_equal(z['labels'],labels)
        values2,body2,valid2=z['pose'],z['body'],z['body_valid']
    metadata=torch.from_numpy(arrays['outer'][:,186:])
    # Dummy rows query each foot category, but are never members of fitting indices.
    query_meta=torch.cat([metadata,torch.eye(3)])
    query_y=torch.cat([torch.from_numpy(labels),torch.zeros((3,3),dtype=torch.long)])
    priors=np.stack([research_loop.foot_priors(query_meta,query_y,torch.from_numpy(np.flatnonzero(rows.fold!=f))).numpy()[-3:] for f in FOLDS])
    np.save(models/'priors.npy',priors,allow_pickle=False)
    copied={}
    for fold in FOLDS:
        for seed in SEEDS:
            source_paths = ({'kick2d':r19/'outer/2d_control'/str(seed)/'model.cbm',
                'kick3d':r19/'outer/2d_plus_body3d'/str(seed)/'model.cbm',
                'keeper':r16/'outer/mirror_control'/str(seed)/'target1/model.ubj'} if fold==0 else
                {'kick2d':r22/'catboost_2d'/str(seed)/'target0/model.cbm',
                 'kick3d':r22/'catboost_3d'/str(seed)/'target0/model.cbm',
                 'keeper':r22/'xgb_2d'/str(seed)/'target1/model.ubj'})
            for name,source in source_paths.items():
                destination=models/f'fold{fold}'/str(seed)/(name+source.suffix)
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
                assert data.sha(destination)==data.sha(source)
                copied[str(destination.relative_to(out))]={'sha256':data.sha(source),'origin':str(source)}
    data.dump(out/'copied_models.json',copied)
    (models/'roles').mkdir(exist_ok=True)
    with np.load(inputs/'candidates.npz') as z: x,y=z['features'],z['labels']
    frames=json.loads((inputs/'frames.json').read_text());expected=json.loads((inputs/'expected_roles.json').read_text())
    matrix=xgb.DMatrix(x,label=np.maximum(y,0));fit=role_validation.training_indices(frames,0)
    count=np.bincount(y[fit],minlength=3);weights=np.sqrt(len(fit)/(3*count))
    params={'objective':'multi:softprob','num_class':3,'device':'cuda','tree_method':'hist','max_depth':5,
        'eta':.05,'min_child_weight':3,'lambda':10,'subsample':1.,'colsample_bytree':1.,'seed':6643,'nthread':4,'eval_metric':'mlogloss'}
    role0=models/'roles/fold0.ubj'
    data.immutable_json(out/'role0_recovery_protocol.json',{'params':params,'fit_indices':fit.tolist(),
        'rounds':1,'class_weights':weights.tolist(),'candidate_sha256':data.sha(inputs/'candidates.npz')})
    if role0.exists(): net=xgb.Booster(model_file=role0);net.set_param({'device':'cuda'})
    else:
        train=matrix.slice(fit);train.set_weight(weights[y[fit]])
        net=xgb.train(params,train,num_boost_round=1);net.save_model(role0)
    xgb_probe.gpu_config(net);p=net.predict(matrix)
    pairs=[roles.choose_pair(p[f['start']:f['end']]) for f in frames]
    assert pairs==expected['outer_pairs'],'Recovered model must preserve every saved role choice'
    shutil.copy2(roles2/'model.ubj',models/'roles/fold2.ubj')
    other=xgb.Booster(model_file=models/'roles/fold2.ubj');other.set_param({'device':'cuda'});xgb_probe.gpu_config(other)
    p2=other.predict(matrix);pairs2=[roles.choose_pair(p2[f['start']:f['end']]) for f in frames]
    assert pairs2==[f['pair'] for f in json.loads((roles2/'selections.json').read_text())]
    data.dump(out/'role_recovery.json',{'all_public_anchor_pairs_verified':len(pairs),
        'old_role0_sha256':expected['outer_model_sha256'],'recovered_role0_sha256':data.sha(role0),
        'role0_same_serialized_model':data.sha(role0)==expected['outer_model_sha256'],
        'role2_reused_sha256':data.sha(models/'roles/fold2.ubj'),'new_perception_calls':0})
    replays=[]
    for fold in FOLDS:
        if fold==0:
            values=arrays['outer'];body,valid=nlf.feature_array(selection,cached_nlf/'features','outer','body')
            reference=r19/'outer/2d_plus_body3d'
        else: values,body,valid,reference=values2,body2,valid2,r22/'catboost_3d'
        check=np.flatnonzero(rows.fold==fold)
        probabilities=predict(values[check],body[check],valid[check],fold,models)
        for seed,ps in zip(SEEDS,probabilities):
            path=reference/str(seed)/'predictions.json';records=json.loads(path.read_text())
            assert [r['clip'] for r in records]==rows.iloc[check]['clip'].tolist()
            for k,p in enumerate(ps):np.testing.assert_allclose(p,[r['probabilities'][k] for r in records],atol=1e-6,rtol=1e-6)
            pred=np.column_stack([p.argmax(1) for p in ps]);np.testing.assert_array_equal(pred,[r['prediction'] for r in records])
            replays.append({'fold':fold,'seed':seed,**metrics.metric_report(labels[check],pred),'reference_sha256':data.sha(path)})
    data.dump(out/'replay.json',replays)
    model_checks={str(p.relative_to(models)):data.sha(p) for p in models.rglob('*') if p.is_file()}
    data.dump(models/'model_manifest.json',model_checks)
    report={'state':'complete','models_copied':len(copied),'role_models':2,'heldout_prediction_sets_replayed':len(replays),
        'new_perception_calls':0,'gpu':torch.cuda.get_device_name(0),'ready_for_private_inference':True,
        'ensemble':'Equal folds 0/2 and seeds 6643/44/45; parameters frozen before private inference.'}
    data.dump(out/'summary.json',report);return report
