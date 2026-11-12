"""R77 fixed per-target geometry subtraction and label-free reliability controls.

Only existing numerical caches are read. Quality proxies are not calibrated
correctness probabilities. No arm is selected from validation to build another.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pose_quality_ablation as q
from sparse_role_comparison import source_interval

digest = lambda a: hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
ARMS = {0: ('both', 'self', 'pose2d', 'pose2d_nofoot', 'body3d', 'field3d',
            'angles', 'ground', 'coordinates'),
        1: ('both', 'self', 'pose2d_nofoot', 'angles', 'ground', 'coordinates')}
MIXES = ('fixed_mix', 'calibration_gate', 'pose_calibration_gate')


def columns(target, arm):
    own = list(range(92*target, 92*(target+1))) + [184+target]
    if arm != 'pose2d_nofoot': own += [186, 187, 188]
    if arm == 'both': return list(range(435 if target == 0 else 189))
    if target == 0:
        if arm in ('self', 'body3d', 'angles', 'ground', 'coordinates'):
            own += list(range(189, 312))
        if arm in ('self', 'field3d', 'angles', 'ground', 'coordinates'):
            own += list(range(312, 435))
    if arm == 'coordinates':
        # Preserve binary missingness, remove confidence, uncertainty and age.
        remove = {184+target} | set(range(92*target+2, 92*(target+1), 4))
        if target == 0:
            for start in (189, 312):
                remove |= set(range(start+3, start+120, 5)) | {start+121, start+122}
        own = [i for i in own if i not in remove]
    return own


def design(cache, view, ground, target, arm):
    assert arm in ARMS[target]
    x = cache['original_full'][:, columns(target, arm)]
    if arm in ('angles', 'ground'): x = np.c_[x, view]
    elif arm == 'coordinates': x = np.c_[x, view[:, :4], view[:, 8]]
    if arm == 'ground': x = np.c_[x, ground]
    return np.asarray(x, np.float32)


def reliability(base, view, target):
    """Frozen heuristic weights; never inferred from validation performance."""
    pose = base[:, 92*target:92*(target+1)].reshape(-1, 23, 4)
    critical = [5, 6, 11, 12, 13, 14, 15, 16]
    confidence = np.clip(pose[:, critical, 2], 0, 1) * (pose[:, critical, 3] > 0)
    pose_weight = confidence.mean(1)
    # Existing core/leg gate is a structural proxy, not identity verification.
    pose_weight *= q.quality(base)['roles'][:, target]
    available = view[:, 8] > 0
    if target == 0: available &= base[:, 432] > 0
    # Residual / image diagonal: fixed 1% reference scale, no threshold search.
    calibration = 1. / (1. + np.maximum(view[:, 4], 0)/.01 + np.maximum(view[:, 5], 0)/.01)
    if target == 0:
        calibration *= np.clip(base[:, 434], 0, 1) / (1. + np.maximum(base[:, 433], 0)/.12)
    return pose_weight, np.where(available, calibration, 0), available


def mix(low, high, prior, base, view, target, arm):
    assert arm in MIXES
    wp, wc, available = reliability(base, view, target)
    weight = .5 * available if arm == 'fixed_mix' else wc
    result = low*(1-weight[:, None]) + high*weight[:, None]
    if arm == 'pose_calibration_gate': result = result*wp[:, None] + prior*(1-wp[:, None])
    return result


def load(inputs):
    inputs = Path(inputs); cache, refs = q.load(inputs)
    data = dict(np.load(inputs/'geometry.npz', allow_pickle=False))
    np.testing.assert_array_equal(data['clips'], cache['clips'])
    assert data['view'].shape == (2598, 9) and data['ground'].shape == (2598, 15)
    assert np.isfinite(data['view']).all() and np.isfinite(data['ground']).all()
    return cache, refs, data['view'], data['ground']


def summarize(predictions, cache, refs, view):
    check = cache['inner_check']; truth = cache['labels'][check, :2]; sources = cache['sources'][check]
    rows = []; candidates = []
    def compare(a, b, mask):
        d = (a.astype(float)-b.astype(float))[:, mask]
        return {'delta': float(d.mean()), 'seed_deltas': d.mean(1).tolist(),
                'interval': source_interval(d.mean(0), sources[mask])} if mask.any() else None
    groups = {'all': np.ones(len(check), bool),
              'calibration_available': view[check, 8] > 0,
              'calibration_missing': view[check, 8] == 0,
              **q.groups(cache['original_full'][check])}
    for target in (0, 1):
        old = np.array([[r['prediction'][target] == r['truth'][target] for r in refs[str(s)]] for s in q.SEEDS])
        names = ARMS[target] + ('prior',) + MIXES
        correct = {a: np.array([predictions[(target,a,s)]['probabilities'].argmax(1) == truth[:,target] for s in q.SEEDS]) for a in names}
        for arm in names:
            c = correct[arm]; raw = np.array([predictions[(target,arm,s)]['raw'].argmax(1) == truth[:,target] for s in q.SEEDS])
            group_rows = {name: {'clips':int(mask.sum()), 'sources':len(set(sources[mask])),
                         'accuracy':float(c[:,mask].mean()) if mask.any() else None,
                         'raw_accuracy':float(raw[:,mask].mean()) if mask.any() else None} for name,mask in groups.items()}
            row = {'target':target,'arm':arm,'accuracy':float(c.mean()),'raw_accuracy':float(raw.mean()),
                   'seed_accuracies':c.mean(1).tolist(),'vs_matched_both':compare(c,correct['both'],groups['all']),
                   'vs_retained':compare(c,old,groups['all']), 'groups':group_rows,
                   'confusion_sum_seeds':np.zeros((3,3),int).tolist()}
            matrix=np.zeros((3,3),int)
            for s in q.SEEDS:
                p=predictions[(target,arm,s)]['probabilities'].argmax(1)
                np.add.at(matrix,(truth[:,target],p),1)
            row['confusion_sum_seeds']=matrix.tolist()
            if arm in MIXES:
                row['vs_fixed_mix']=compare(c,correct['fixed_mix'],groups['all'])
                row['vs_quality_concat']=compare(c,correct['angles'],groups['all'])
            if arm == 'coordinates': row['vs_quality_concat']=compare(c,correct['angles'],groups['all'])
            d=row['vs_retained']; matched=row['vs_matched_both']
            candidate=(arm not in ('both','self','prior','pose2d_nofoot') and d['delta']>=.02
                       and min(d['seed_deltas'])>0 and d['interval']['source_cluster_95pct'][0]>0
                       and matched['delta']>0)
            if arm in MIXES: candidate &= row['vs_fixed_mix']['delta']>0 and row['vs_quality_concat']['delta']>0
            row['candidate_for_confirmation']=bool(candidate)
            if candidate:candidates.append({'target':target,'arm':arm})
            rows.append(row)
    return {'state':'complete','rows':rows,'candidates_for_confirmation':candidates,
            'new_perception_calls':0,'competition_submissions':0,
            'limitations':['Repeated exploratory source fold; candidates need separately frozen confirmation.',
              'Kick matched scores preserve R28 fallback on99 clips; raw scores use no fallback.',
              'Confidence and reprojection quality are not measured correctness probabilities.',
              'Ground cache was jointly gated in R42 and retains its original missingness.',
              'Keeper has no cached self3D; no claim that a kicker3D ablation tests keeper3D.',
              'Quality concat versus mixtures changes architecture; gate versus fixed mix isolates weighting.']}


def run(inputs, out, budget=1800):
    import torch
    import catboost
    from pose_tree_controls import fit_catboost, softmax
    assert torch.cuda.is_available() and catboost.__version__=='1.2.10'
    inputs,out=Path(inputs),Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'input_manifest.json').read_text())
    for name,h in manifest['files'].items(): assert q.sha(inputs/name)==h,name
    cache,refs,view,ground=load(inputs); check=cache['inner_check']; n=len(cache['clips'])
    q.dump(out/'protocol.json',{'source_sha256':q.sha(__file__),'manifest':manifest,
           'gpu':torch.cuda.get_device_name(0),'training':'CatBoost GPU','replay':'CPU multiclass',
           'train':779,'validation':262,'new_perception_calls':0})
    started=time.monotonic();predictions={};new=0;reused=0
    for target in (0,1):
        margin=cache['inner_margin'][:,target*3:target*3+3]; y=cache['inner_y'][:,target]
        for seed in q.SEEDS:
            raw_arms={}
            for arm in ARMS[target]:
                if time.monotonic()-started>budget-90: raise TimeoutError('Saved model boundary; resume exact models')
                folder=out/'fits'/str(target)/arm/str(seed);x=design(cache,view,ground,target,arm)
                offset=np.zeros_like(margin) if arm=='pose2d_nofoot' else margin
                if arm in ('both','self'):
                    source=inputs/'reuse'/str(target)/arm/str(seed)
                    shutil.copytree(source,folder,dirs_exist_ok=True)
                    p=json.loads((folder/'protocol.json').read_text())
                    assert p['x_sha256']==digest(x) and p['margin_sha256']==digest(offset) and p['y_sha256']==digest(y)
                    assert p['fit_indices']==cache['inner_fit'].tolist() and p['weights']==cache['inner_weights'].tolist()
                    net=catboost.CatBoostClassifier();net.load_model(str(folder/'model.cbm'))
                    raw=softmax(net.predict(x,prediction_type='RawFormulaVal',thread_count=4)+offset)
                    saved=np.load(folder/'raw_probabilities.npy',allow_pickle=False)
                    np.testing.assert_allclose(raw,saved,rtol=1e-6,atol=1e-7)
                    raw=saved;reused+=1
                    q.dump(folder/'r77_replay.json',{'passed':True,'source':'R70 original_static','new_training':False})
                else:
                    raw,_=fit_catboost(x,y,offset,cache['inner_fit'],cache['inner_weights'],folder,seed,rounds=450)
                    np.save(folder/'raw_probabilities.npy',raw,allow_pickle=False);new+=1
                raw_arms[arm]=raw[check]
                print(json.dumps({'stage':'fit','target':target,'arm':arm,'seed':seed,'new_fits':new,'reused':reused}),flush=True)
            prior=softmax(margin[check]);raw_arms['prior']=prior
            low=raw_arms['pose2d' if target==0 else 'self'];high=raw_arms['angles']
            for arm in MIXES:raw_arms[arm]=mix(low,high,prior,cache['original_full'][check],view[check],target,arm)
            old=np.array([r['probabilities'][target] for r in refs[str(seed)]])
            eligible=cache['original_valid'][check] if target==0 else np.ones(len(check),bool)
            for arm,raw in raw_arms.items():
                p=raw.copy();p[~eligible]=old[~eligible]
                predictions[(target,arm,seed)]={'probabilities':p,'raw':raw}
                q.dump(out/'predictions'/str(target)/arm/f'{seed}.json',{
                    'clips':cache['clips'][check].tolist(),'truth':cache['labels'][check,target].tolist(),
                    'probabilities':p.tolist(),'raw':raw.tolist(),'eligible':eligible.tolist()})
            q.dump(out/'progress.json',{'new_fits':new,'reused':reused,'last_target':target,'last_seed':seed})
    assert new==33 and reused==12
    summary=summarize(predictions,cache,refs,view)
    summary.update(new_gpu_fits=new,reused_models=reused,seconds=time.monotonic()-started)
    q.dump(out/'summary.json',summary);return summary
