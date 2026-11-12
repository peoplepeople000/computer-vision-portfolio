"""R62 confidence-proxy audit and fixed masking ablation; no identity claims."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ARMS = ('joint_mask', 'role_mask')
SEEDS = (6643, 6644, 6645)
EXPECTED = {
    'baseline.npz': 'b4166927f3e53b8b55c2aac454074c838feaf542728b45edc9fe73a982316f88',
    'references.json': 'a3637ca9d40795c6801586c99db5b543f52162e9e6071f710fb9e4782847d311',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def quality(values):
    """Label-independent, reflection-symmetric quality PROXY, not correctness."""
    values = np.asarray(values)
    assert values.ndim == 2 and values.shape[1] == 435 and np.isfinite(values).all()
    pose = values[:, :184].reshape(-1, 2, 23, 4)
    visible = pose[..., 3] > 0
    strong = visible & (pose[..., 2] >= .5)
    core = strong[..., 11] & strong[..., 12] & (strong[..., 5] | strong[..., 6])
    leg = (strong[..., 13] & strong[..., 15]) | (strong[..., 14] & strong[..., 16])
    return {'visible': visible, 'strong': strong, 'roles': core & leg}


def transform(values, arm):
    assert arm in ARMS
    q = quality(values); result = np.array(values, copy=True)
    pose = result[:, :184].reshape(-1, 2, 23, 4)
    pose[~q['strong']] = 0
    if arm == 'role_mask':
        pose[~q['roles']] = 0
        result[~q['roles'][:, 0], 189:] = 0
    return result


def groups(values):
    q = quality(values); count = q['roles'].sum(1)
    return {f'{n}_roles_pass_proxy': count == n for n in (2, 1, 0)}


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    assert truth.shape == prediction.shape and truth.ndim == 2 and truth.shape[1] == 2
    if not len(truth):
        return {'clips': 0, 'kick_accuracy': None, 'keeper_accuracy': None, 'score_v2': None}
    correct = (truth == prediction).sum(0)
    return {'clips': len(truth), 'kick_accuracy': float(correct[0]/len(truth)),
            'keeper_accuracy': float(correct[1]/len(truth)),
            'score_v2': float((7*int(correct[0])+3*int(correct[1]))/(10*len(truth)))}


def load(inputs):
    inputs = Path(inputs)
    for name, digest in EXPECTED.items():
        assert sha(inputs/name) == digest, name
    with np.load(inputs/'baseline.npz', allow_pickle=False) as a:
        cache = {k: a[k] for k in a.files}
    refs = json.loads((inputs/'references.json').read_text())
    n = len(cache['clips']); train, check, fit = (cache['inner_'+k] for k in ('train','check','fit'))
    assert n == 1299 and '0526_public.mp4' not in cache['clips']
    assert len(train) == 779 and len(check) == 262
    assert set(cache['folds'][train]) == {2,3,4} and set(cache['folds'][check]) == {1}
    assert set(cache['sources'][train]).isdisjoint(cache['sources'][check])
    assert all(len(set(cache['folds'][cache['sources']==s]))==1 for s in set(cache['sources']))
    np.testing.assert_array_equal(np.sort(fit), np.sort(np.r_[train,train+n]))
    np.testing.assert_array_equal(cache['inner_weights'], np.full(len(fit), .5))
    for seed in SEEDS:
        assert [r['clip'] for r in refs[str(seed)]] == cache['clips'][check].tolist()
        np.testing.assert_array_equal([r['truth'] for r in refs[str(seed)]], cache['labels'][check])
        np.testing.assert_array_equal([r['prediction'][:2] for r in refs[str(seed)]],
            [[int(np.argmax(r['probabilities'][j])) for j in range(2)] for r in refs[str(seed)]])
    return cache, refs


def audit(inputs):
    cache, refs = load(inputs); x = cache['original_full']; n = len(cache['clips'])
    result = {'state': 'audit_complete_no_training', 'metric_id': 'direction_accuracy_70_30_v2',
              'input_hashes': EXPECTED, 'groups_are_confidence_proxies_not_identity_or_occlusion': True,
              'splits': {}, 'validation': {}, 'new_perception_calls': 0}
    q = quality(x)
    np.testing.assert_array_equal(q['roles'][:n], q['roles'][n:])
    for split, ids in [('train',cache['inner_train']),('check',cache['inner_check'])]:
        result['splits'][split] = {
            'clips': len(ids), 'sources': len(set(cache['sources'][ids])),
            'role_proxy_counts': {k:int(v.sum()) for k,v in groups(x[ids]).items()},
            'missing_all_joints_per_role': (~q['visible'][ids].any(-1)).sum(0).tolist(),
            'joints_removed_by_confidence_per_role': (q['visible'][ids]&~q['strong'][ids]).sum((0,2)).tolist(),
            'observation_older_than_0_1s_per_role': (x[ids,184:186] > .1).sum(0).tolist(),
            'role_observation_times_differ': int((abs(x[ids,184]-x[ids,185]) > 1e-6).sum()),
            'baseline_geometry_fallback': int((~cache['original_valid'][ids]).sum()),
            'effective_changed_rows': {arm:int((np.any(transform(x[ids],arm)!=x[ids],axis=1)&cache['original_valid'][ids]).sum()) for arm in ARMS},
        }
    check = cache['inner_check']; truth = cache['labels'][check,:2]
    masks = {'all': np.ones(len(check), bool), **groups(x[check])}
    for group, mask in masks.items():
        trials = [metrics(truth[mask], np.asarray([r['prediction'][:2] for r in refs[str(seed)]])[mask]) for seed in SEEDS]
        result['validation'][group] = {'clips': int(mask.sum()), 'sources': len(set(cache['sources'][check][mask])),
            'geometry_valid_clips': int((mask & cache['original_valid'][check]).sum()),
            'kicker_passes_proxy': int((mask & q['roles'][check,0]).sum()),
            'keeper_passes_proxy': int((mask & q['roles'][check,1]).sum()),
            'seed_mean': {k:float(np.mean([t[k] for t in trials])) if mask.any() else None for k in ('kick_accuracy','keeper_accuracy','score_v2')}}
    return result


def gate(trials, count, plan):
    g = plan['gate']; eligible = []
    if count < g['high_confidence_group_minimum_clips']: return None
    for arm in ARMS:
        t = [r for r in trials if r['arm'] == arm]
        if (len(t) == len(SEEDS) and np.mean([r['delta'] for r in t]) >= g['mean_gain_minimum']-1e-12
                and all(r['delta'] > 0 for r in t)
                and np.mean([r['high_confidence_delta'] for r in t]) >= -g['high_confidence_group_maximum_drop']-1e-12):
            eligible.append((np.mean([r['score_v2'] for r in t]), -ARMS.index(arm), arm))
    return max(eligible)[2] if eligible else None


def run(inputs, out, budget=900):
    import torch
    import catboost
    import pose_tree_controls as controls
    import rgb_gru_data as data
    from sparse_role_comparison import source_interval
    assert torch.cuda.is_available(), 'GPU quota required; do not train on local CPU'
    assert catboost.__version__ == '1.2.10'
    inputs, out = Path(inputs), Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); manifest = json.loads((inputs/'input_manifest.json').read_text())
    for name,h in manifest['files'].items(): assert sha(inputs/name)==h
    plan = manifest['plan']; assert plan['metric_id']=='direction_accuracy_70_30_v2'
    assert plan['training']['seeds']==list(SEEDS)
    data.immutable_json(out/'protocol.json', {'source_sha256':sha(__file__), 'manifest':manifest})
    cache, refs = load(inputs); x = cache['original_full']; check = cache['inner_check']
    truth = cache['labels'][check,:2]; valid = cache['original_valid'][check]
    masks = groups(x[check]); high = masks['2_roles_pass_proxy']
    dump(out/'audit.json', audit(inputs))
    dump(out/'hardware.json', {'gpu':torch.cuda.get_device_name(0), 'training':'CatBoost GPU',
                             'replay':'CPU multiclass CatBoost', 'catboost':catboost.__version__})
    trials=[]; deltas={arm:[] for arm in ARMS}
    for seed in SEEDS:
        old = [np.asarray([r['probabilities'][k] for r in refs[str(seed)]]) for k in range(2)]
        baseline = np.column_stack([p.argmax(1) for p in old])
        # Verify frozen control inference, rather than retraining completed baselines.
        net=catboost.CatBoostClassifier(); net.load_model(str(inputs/f'control_{seed}.cbm'))
        replay=controls.softmax(net.predict(x[check],prediction_type='RawFormulaVal',thread_count=4)+cache['inner_margin'][check,:3])
        np.testing.assert_allclose(replay[valid],old[0][valid],rtol=1e-6,atol=1e-7)
        dump(out/f'control_replay_{seed}.json',{'max_difference':float(abs(replay[valid]-old[0][valid]).max()),'model_sha256':sha(inputs/f'control_{seed}.cbm')})
        for arm in ARMS:
            if time.monotonic()-started > budget-plan['limits']['save_reserve_seconds']:
                dump(out/'progress.json',{'state':'budget_boundary','completed':trials})
                return {'state':'budget_boundary','trials':trials}
            full=transform(x,arm); directory=out/arm/str(seed)
            probabilities,info=controls.fit_catboost(full,cache['inner_y'][:,0],cache['inner_margin'][:,:3],
                cache['inner_fit'],cache['inner_weights'],directory,seed,rounds=450)
            kick=probabilities[check].copy(); kick[~valid]=old[0][~valid]
            prediction=np.column_stack([kick.argmax(1),old[1].argmax(1)])
            row={'arm':arm,'seed':seed,**metrics(truth,prediction),
                'delta':metrics(truth,prediction)['score_v2']-metrics(truth,baseline)['score_v2'],
                'high_confidence_delta':metrics(truth[high],prediction[high])['score_v2']-metrics(truth[high],baseline[high])['score_v2'] if high.any() else None,
                'groups':{k:metrics(truth[m],prediction[m]) for k,m in masks.items()},'fit':info}
            trials.append(row)
            delta=((prediction==truth).astype(float)-(baseline==truth).astype(float))@np.array([.7,.3])
            deltas[arm].append(delta)
            np.save(directory/'raw_probabilities.npy',probabilities,allow_pickle=False)
            dump(directory/'result_v2.json',row)
            dump(directory/'predictions.json',[{'clip':str(cache['clips'][i]),'truth':truth[j].tolist(),
                'prediction':prediction[j].tolist(),'probabilities':[kick[j].tolist(),old[1][j].tolist()],
                'geometry_fallback':bool(not valid[j])} for j,i in enumerate(check)])
            dump(out/'progress.json',{'state':'running','completed':trials})
            print(json.dumps({'arm':arm,'seed':seed,'score_v2':row['score_v2'],'delta':row['delta']}),flush=True)
    summary={'state':'complete','metric_id':plan['metric_id'],'trials':trials,
             'candidate':gate(trials,int(high.sum()),plan),
             'paired_source_intervals':{arm:source_interval(np.mean(deltas[arm],axis=0),cache['sources'][check]) for arm in ARMS},
             'seconds':time.monotonic()-started,'competition_submissions':0}
    dump(out/'summary.json',summary)
    return summary
