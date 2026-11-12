"""R13: anatomical mirror augmentation of cached R10 pose descriptors."""
import numpy as np
import hashlib
import json
import time
from pathlib import Path

# RTMW first 23 COCO-WholeBody joints: body, left foot, right foot.
JOINT_SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11,
                      14, 13, 16, 15, 20, 21, 22, 17, 18, 19])
DIRECTION_SWAP = np.array([2, 1, 0])


def mirror_features(values):
    """Reflect x, exchange anatomical sides, preserve ages and swap foot metadata."""
    values = np.asarray(values)
    assert values.ndim == 2 and values.shape[1] == 189
    result = values.copy()
    pose = values[:, :184].reshape(-1, 2, 23, 4)
    reflected = pose[:, :, JOINT_SWAP].copy()
    reflected[..., 0] *= -1
    result[:, :184] = reflected.reshape(-1, 184)
    result[:, 186:189] = values[:, [187, 186, 188]]
    return result


def mirror_labels(labels):
    result = np.asarray(labels).copy()
    assert result.ndim == 2 and result.shape[1] == 3
    result[:, :2] = DIRECTION_SWAP[result[:, :2]]
    return result


def training_rows(indices, count, augmented):
    """Original and synthetic rows share total weight; held-out rows stay excluded."""
    indices = np.asarray(indices, dtype=np.int64)
    assert len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < count))
    if augmented:
        return np.r_[indices, indices + count], np.full(2 * len(indices), .5, np.float32)
    return indices.copy(), np.ones(len(indices), np.float32)


def fit(x, y, margin, indices, check, weights, classes, depth, rounds, seed, out):
    import xgboost as xgb
    import rgb_gru_data as data
    import xgb_probe as previous
    params = {'objective': 'multi:softprob', 'num_class': classes, 'eval_metric': 'merror',
              'device': 'cuda', 'tree_method': 'hist', 'max_depth': depth, 'eta': .05,
              'min_child_weight': 5, 'lambda': 10, 'subsample': .85,
              'colsample_bytree': .85, 'seed': seed, 'nthread': 4}
    config = {'params': params, 'rounds': rounds, 'source_sha256': data.sha(__file__),
              'fit_indices': indices.tolist(), 'fit_weights': weights.tolist(),
              'check_indices': None if check is None else check.tolist(), 'xgboost': xgb.__version__,
              'x_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
              'prior_sha256': hashlib.sha256(margin.tobytes()).hexdigest(),
              'y_sha256': hashlib.sha256(y.tobytes()).hexdigest()}
    data.immutable_json(out/'protocol.json', config)
    matrix = xgb.DMatrix(x, label=y, base_margin=margin)
    training = matrix.slice(indices); training.set_weight(weights)
    evaluation = [(training, 'fit')]
    if check is not None: evaluation.append((matrix.slice(check), 'inner'))
    started = time.monotonic(); history = {}
    if (out/'result.json').exists():
        info = json.loads((out/'result.json').read_text())
        assert data.sha(out/'model.ubj') == info['model_sha256']
        booster = xgb.Booster(model_file=out/'model.ubj'); booster.set_param({'device': 'cuda'})
    else:
        booster = xgb.train(params, training, num_boost_round=rounds, evals=evaluation,
                            early_stopping_rounds=30 if check is not None else None,
                            evals_result=history, verbose_eval=False)
        selected = booster.best_iteration + 1 if check is not None else rounds
        probability = booster.predict(matrix, iteration_range=(0, selected))
        booster.save_model(out/'model.ubj'); data.dump(out/'history.json', history)
        np.save(out/'train_only_prior.npy', margin)
        info = {'selected_rounds': selected, 'trained_rounds': booster.num_boosted_rounds(),
                'inner_accuracy': None if check is None else float((probability[check].argmax(1) == y[check]).mean()),
                'seconds': time.monotonic()-started, 'depth': depth,
                'model_sha256': data.sha(out/'model.ubj')}
        data.dump(out/'result.json', info)
    runtime = previous.gpu_config(booster); data.dump(out/'runtime_config.json', runtime)
    probability = booster.predict(matrix, iteration_range=(0, info['selected_rounds']))
    reloaded = xgb.Booster(model_file=out/'model.ubj'); reloaded.set_param({'device': 'cuda'})
    np.testing.assert_allclose(probability, reloaded.predict(matrix, iteration_range=(0, info['selected_rounds'])), atol=1e-7, rtol=1e-6)
    print(json.dumps({'fit': str(out), **info, 'cuda_reload_verified': True}), flush=True)
    return probability, info


def execute(cache, folds, out, budget=300):
    import pandas as pd
    import torch
    import rgb_gru_data as data
    import rgb_gru as model
    import research_loop as research
    import sparse_role_comparison as sparse
    import xgb_probe as previous
    assert torch.cuda.is_available(), 'CUDA required'
    out.mkdir(parents=True, exist_ok=True)
    source = cache/'results/r10_classification'
    manifest = json.loads((source/'artifact_manifest.json').read_text())
    assert data.sha(source/'features.npz') == manifest['features.npz']
    rows = pd.read_csv(folds); n = len(rows)
    original = json.loads((cache/'r10_matched_manifest.json').read_text())
    assert data.sha(folds) == original['files']['public_folds.csv']
    assert n == 1299 and '0526_public.mp4' not in set(rows['clip'])
    assert rows.groupby('source_id').fold.nunique().max() == 1
    with np.load(source/'features.npz') as z:
        np.testing.assert_array_equal(rows['clip'], z['clips'])
        arrays = {stage: z[stage+'__new_pose'].astype(np.float32) for stage in ['inner','outer']}
    assert all(a.shape == (n, 189) and np.isfinite(a).all() for a in arrays.values())
    meta = torch.nn.functional.one_hot(torch.tensor([{'left':0,'right':1}.get(f,2) for f in rows.preferred_foot]),3).float()
    for a in arrays.values(): np.testing.assert_array_equal(a[:,186:], meta.numpy())
    labels = np.array([[cs.index(row[k]) for k,cs in data.TARGETS.items()] for _,row in rows.iterrows()])
    y = np.concatenate([labels, mirror_labels(labels)])
    arrays = {stage: np.concatenate([a, mirror_features(a)]) for stage,a in arrays.items()}
    inner, check, train, val = [np.flatnonzero(v) for v in (~rows.fold.isin([0,1]),rows.fold==1,rows.fold!=0,rows.fold==0)]
    priors = {stage: research.foot_priors(meta, torch.from_numpy(labels), torch.tensor(ix)).numpy()
              for stage,ix in [('inner',inner),('outer',train)]}
    margins = {}
    for stage,p in priors.items():
        margins[stage] = [np.concatenate([p[:,start:start+count], p[:,start:start+count][:,DIRECTION_SWAP] if count==3 else p[:,start:start+count]])
                          for start,count in [(0,3),(3,3),(6,2)]]
    protocol = {'source_sha256':data.sha(__file__),'previous_tree_sha256':data.sha(previous.__file__),
                'features_sha256':manifest['features.npz'],'folds_sha256':data.sha(folds),
                'variants':['control','mirror'],'seeds':list(previous.SEEDS),'depths':list(previous.DEPTHS),
                'inner_fold':1,'outer_fold':0,'inner_promotion_delta':.01,'joint_permutation':JOINT_SWAP.tolist(),
                'gpu':torch.cuda.get_device_name(0),
                'prior':'Unchanged original train-only foot prior; swap direction logits on synthetic rows.',
                'weights':'Original+mirror each 0.5; same total training mass as control.'}
    data.immutable_json(out/'protocol.json',protocol)
    started=time.monotonic();report={'stage':'running'};choices={}
    def boundary():
        if time.monotonic()-started>budget-30: raise TimeoutError('Budget boundary; completed fits saved for resume')
    try:
        ix,w=training_rows(inner,n,True)
        _,warmup=fit(arrays['inner'],y[:,0],margins['inner'][0],ix,None,w,3,4,20,6643,out/'discarded_warmup')
        _,pilot=fit(arrays['inner'],y[:,0],margins['inner'][0],ix,None,w,3,4,60,6643,out/'discarded_cost_pilot')
        # At most 12 inner + 18 outer fits, each bounded by 200 rounds.
        estimate=warmup['seconds']+pilot['seconds']*100*1.5
        data.dump(out/'profile.json',{'cold_warmup_seconds':warmup['seconds'],
                                   'seconds_60_rounds':pilot['seconds'],'estimated_max_seconds':estimate,
                                   'gate_passed':estimate<180,'cuda_histogram_verified':True})
        if estimate>=180:
            report={'stage':'cost_rejected'};return report
        for variant in ['control','mirror']:
            ix,w=training_rows(inner,n,variant=='mirror');choices[variant]=[]
            for k,count in enumerate([3,3,2]):
                baseline=float((margins['inner'][k][check].argmax(1)==labels[check,k]).mean());candidates=[]
                for depth in previous.DEPTHS:
                    boundary();_,info=fit(arrays['inner'],y[:,k],margins['inner'][k],ix,check,w,count,depth,200,6643,out/variant/f'inner_target{k}_depth{depth}')
                    candidates.append(info)
                choices[variant].append({'selected':previous.choose_candidate(baseline,candidates),'candidates':candidates})
            data.dump(out/'inner_choices.json',choices)
        scores={v:sum(w*c['selected']['inner_accuracy'] for w,c in zip(data.WEIGHTS,cs)) for v,cs in choices.items()}
        report={'stage':'inner_complete','inner_scores':scores,'choices':choices,
                'promoted':scores['mirror']>scores['control']+.01}
        data.dump(out/'inner_summary.json',report)
        if report['promoted']:
            predictions={};trials={}
            for variant in ['control','mirror']:
                trials[variant]=[];predictions[variant]=[];ix,w=training_rows(train,n,variant=='mirror')
                for seed in previous.SEEDS:
                    probs=[]
                    for k,count in enumerate([3,3,2]):
                        chosen=choices[variant][k]['selected'];margin=margins['outer'][k]
                        if chosen['kind']=='foot_rule': p=torch.softmax(torch.from_numpy(margin),1).numpy()
                        else:
                            boundary();p,_=fit(arrays['outer'],y[:,k],margin,ix,None,w,count,chosen['depth'],chosen['selected_rounds'],seed,out/variant/str(seed)/f'target{k}')
                        probs.append(p)
                    pred=np.column_stack([p.argmax(1) for p in probs]);metrics=model.metric_report(labels[val],pred[val]);trials[variant].append({'seed':seed,**metrics});predictions[variant].append(pred[val])
                    data.dump(out/variant/str(seed)/'predictions.json',[{'clip':rows.iloc[int(i)]['clip'],'truth':labels[i].tolist(),'prediction':pred[i].tolist(),'probabilities':[p[i].tolist() for p in probs]} for i in val])
            delta=np.mean([(a==labels[val]).astype(float)-(b==labels[val]).astype(float) for a,b in zip(predictions['mirror'],predictions['control'])],0)@np.array(data.WEIGHTS)
            report['outer']={v:{'trials':ts,'mean':float(np.mean([t['weighted_score'] for t in ts])),'sd':float(np.std([t['weighted_score'] for t in ts],ddof=1))} for v,ts in trials.items()}
            report['paired_vs_control']=sparse.source_interval(delta,rows.iloc[val].source_id)
        report.update(stage='complete',seconds=time.monotonic()-started);return report
    except Exception as error:
        report.update(stage='interrupted' if isinstance(error, TimeoutError) else 'failed',
                      error_type=type(error).__name__, error=str(error),
                      seconds=time.monotonic()-started)
        raise
    finally:
        data.dump(out/'summary.json',report)
        data.dump(out/'artifact_manifest.json',{str(p.relative_to(out)):data.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
