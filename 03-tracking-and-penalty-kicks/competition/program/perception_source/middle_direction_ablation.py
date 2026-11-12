"""R78 distinguish minority ranking, loss weighting and decision effects.

No validation threshold search. Reuse fixed features, mirrored rows and model
controls; balanced losses and posterior restoration use training counts only.
"""
import json
import shutil
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pose_quality_ablation as q
from skeleton_geometry_ablation import digest
from sparse_role_comparison import source_interval

TRAIN_ARMS=('control','no_prior','balanced','balanced_no_prior')
ARMS=TRAIN_ARMS+('balanced_restored','balanced_no_prior_restored','prior_shift')


def class_weights(y, fit, weights):
    """Equal total fitting mass per class, counting mirrors with weight0.5."""
    counts=np.bincount(y[fit],weights=weights,minlength=3)
    assert (counts>0).all()
    return counts.sum()/(3*counts),counts


def reweight(probability, factors):
    p=np.asarray(probability)*np.asarray(factors)[None]
    return p/p.sum(1,keepdims=True)


def profile(y, probabilities):
    y=np.asarray(y);p=np.asarray(probabilities);pred=p.argmax(1);m=y==1;side=~m
    cm=np.zeros((3,3),int);np.add.at(cm,(y,pred),1)
    def ratio(a,b):return float(a/b) if b else None
    return {'clips':len(y),'accuracy':float((y==pred).mean()),'confusion':cm.tolist(),
        'middle_prevalence':float(m.mean()),'middle_predictions':int((pred==1).sum()),
        'middle_correct':int(cm[1,1]),'middle_false_positives':int(cm[0,1]+cm[2,1]),
        'middle_precision':ratio(cm[1,1],cm[:,1].sum()),'middle_recall':ratio(cm[1,1],cm[1].sum()),
        'middle_auc':float(roc_auc_score(m,p[:,1])) if m.any() and side.any() else None,
        'middle_ap':float(average_precision_score(m,p[:,1])) if m.any() else None,
        'middle_brier':float(np.mean((p[:,1]-m)**2)),
        'middle_probability_max':float(p[:,1].max()),
        'true_middle_mean_probability':float(p[m,1].mean()) if m.any() else None,
        'middle_rank_first':int((m&(pred==1)).sum()),
        'middle_rank_second':int((m&(np.argsort(p,axis=1)[:,1]==1)).sum()),
        'side_accuracy':float((pred[side]==y[side]).mean()) if side.any() else None,
        'left_right_swaps':int(cm[0,2]+cm[2,0]),
        'side_auc':float(roc_auc_score(y[side]==2,p[side,2]/p[side][:,[0,2]].sum(1))) if set(y[side])=={0,2} else None}


def design(cache,target,arm):
    assert arm in TRAIN_ARMS
    x=cache['original_full'] if target==0 else cache['original_full'][:,:189]
    y=cache['inner_y'][:,target];fit=cache['inner_fit'];weights=cache['inner_weights'].copy()
    factors,counts=class_weights(y,fit,weights)
    if arm.startswith('balanced'):weights=weights*factors[y[fit]]
    margin=cache['inner_margin'][:,3*target:3*(target+1)].copy()
    if arm in ('no_prior','balanced_no_prior'):margin[:]=0
    return x,y,margin,fit,weights,factors,counts


def derived(raw,factors):
    result=dict(raw)
    result['balanced_restored']=reweight(raw['balanced'],1/factors)
    result['balanced_no_prior_restored']=reweight(raw['balanced_no_prior'],1/factors)
    result['prior_shift']=reweight(raw['control'],factors)
    return result


def audit(inputs):
    """Read-only preflight on existing probabilities, with disjoint error groups."""
    inputs=Path(inputs);cache,refs=q.load(inputs);train,check=cache['inner_train'],cache['inner_check']
    base=cache['original_full'];quality=q.quality(base);out={'state':'audit_complete','targets':{},'cases':[]}
    for target in (0,1):
        y=cache['labels'][:,target];rows=[]
        for seed in q.SEEDS:
            raw=np.load(inputs/f'reuse/{target}/{seed}/raw_probabilities.npy',allow_pickle=False)
            rows.append({'seed':seed,'train_in_sample':profile(y[train],raw[train]),
                         'validation_raw':profile(y[check],raw[check])})
        mean=np.mean([np.load(inputs/f'reuse/{target}/{s}/raw_probabilities.npy')[check] for s in q.SEEDS],axis=0)
        pred=mean.argmax(1);truth=y[check]
        masks={'true_middle':truth==1,'left_right_wrong':(truth!=1)&(pred!=1)&(pred!=truth),
               'sides_correct':(truth!=1)&(pred==truth),'side_false_middle':(truth!=1)&(pred==1)}
        details={}
        for name,mask in masks.items():
            ids=check[mask];pose=base[ids,92*target:92*(target+1)].reshape(-1,23,4)
            details[name]={'clips':len(ids),'sources':len(set(cache['sources'][ids])),
                'role_proxy_pass':float(quality['roles'][ids,target].mean()) if len(ids) else None,
                'calibration_valid':float(cache['original_valid'][ids].mean()) if len(ids) else None,
                'mean_foot_confidence':float((pose[:,17:23,2]*(pose[:,17:23,3]>0)).mean()) if len(ids) else None,
                'source_counts':{str(s):int((cache['sources'][ids]==s).sum()) for s in sorted(set(cache['sources'][ids]))}}
            for j in np.flatnonzero(mask):
                i=check[j];out['cases'].append({'target':target,'group':name,'clip':str(cache['clips'][i]),
                    'source':str(cache['sources'][i]),'truth':int(truth[j]),'prediction':int(pred[j]),
                    'probabilities':mean[j].tolist(),'role_proxy_pass':bool(quality['roles'][i,target]),
                    'calibration_valid':bool(cache['original_valid'][i]),'pose2d':base[i,92*target:92*(target+1)].reshape(23,4).tolist()})
        factors,counts=class_weights(cache['inner_y'][:,target],cache['inner_fit'],cache['inner_weights'])
        out['targets'][str(target)]={'train_counts':np.bincount(y[train],minlength=3).tolist(),
            'validation_counts':np.bincount(y[check],minlength=3).tolist(),
            'middle_training_sources':len(set(cache['sources'][train[y[train]==1]])),
            'augmented_fit_class_mass':counts.tolist(),'balanced_factors':factors.tolist(),
            'profiles':rows,'error_groups':details}
    return out


def summarize(predictions,cache,refs):
    rows=[];candidates=[];check=cache['inner_check'];sources=cache['sources'][check]
    for target in (0,1):
        y=cache['labels'][check,target]
        old=np.array([[r['prediction'][target]==r['truth'][target] for r in refs[str(s)]] for s in q.SEEDS])
        control=np.array([predictions[target,'control',s]['probabilities'].argmax(1)==y for s in q.SEEDS])
        for arm in ARMS:
            trials=[profile(y,predictions[target,arm,s]['probabilities']) for s in q.SEEDS]
            raw_trials=[profile(y,predictions[target,arm,s]['raw']) for s in q.SEEDS]
            c=np.array([predictions[target,arm,s]['probabilities'].argmax(1)==y for s in q.SEEDS])
            d=c.astype(float)-control;dr=c.astype(float)-old
            ci=source_interval(d.mean(0),sources);ri=source_interval(dr.mean(0),sources)
            candidate=(arm not in ('control','prior_shift') and dr.mean()>=.02 and min(dr.mean(1))>0
                       and ri['source_cluster_95pct'][0]>0 and d.mean()>0)
            if candidate:candidates.append({'target':target,'arm':arm})
            rows.append({'target':target,'arm':arm,'accuracy':float(c.mean()),
                         'raw_accuracy':float(np.mean([a['accuracy'] for a in raw_trials])),
                         'trials':trials,'raw_trials':raw_trials,
                         'delta_vs_control':float(d.mean()),'seed_delta':d.mean(1).tolist(),'paired':ci,
                         'delta_vs_retained':float(dr.mean()),'paired_retained':ri,
                         'candidate_for_confirmation':bool(candidate)})
    return {'state':'complete','rows':rows,'candidates_for_confirmation':candidates,
            'new_perception_calls':0,'competition_submissions':0,
            'limits':['Raw train metrics are in-sample, not independent generalization estimates.',
              'Imbalance weights and decision shifts use only mirrored training class mass; no validation thresholds.',
              'Poor cross-source ranking does not prove the raw video lacks information.',
              'Kick matched scores preserve99 original fallbacks; raw full-cohort results are separate.',
              'Middle recall alone is not improvement; false positives and left/right swaps count too.']}


def run(inputs,out,budget=1200):
    import torch
    import catboost
    from pose_tree_controls import fit_catboost,softmax
    assert torch.cuda.is_available() and catboost.__version__=='1.2.10'
    inputs,out=Path(inputs),Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'input_manifest.json').read_text())
    for name,h in manifest['files'].items():assert q.sha(inputs/name)==h,name
    cache,refs=q.load(inputs);check=cache['inner_check'];train=cache['inner_train']
    q.dump(out/'protocol.json',{'source_sha256':q.sha(__file__),'manifest':manifest,
        'gpu':torch.cuda.get_device_name(0),'train':779,'validation':262,'new_perception_calls':0})
    started=time.monotonic();predictions={};new=reused=0
    for target in (0,1):
        for seed in q.SEEDS:
            raw={};train_profiles={}
            for arm in TRAIN_ARMS:
                if time.monotonic()-started>budget-90:raise TimeoutError('Saved fit boundary; preserve finished models')
                x,y,margin,fit,weights,factors,counts=design(cache,target,arm)
                folder=out/'fits'/str(target)/arm/str(seed)
                if arm=='control':
                    shutil.copytree(inputs/f'reuse/{target}/{seed}',folder,dirs_exist_ok=True)
                    p=json.loads((folder/'protocol.json').read_text())
                    assert p['x_sha256']==digest(x) and p['y_sha256']==digest(y) and p['margin_sha256']==digest(margin)
                    assert p['fit_indices']==fit.tolist() and p['weights']==weights.tolist()
                    net=catboost.CatBoostClassifier();net.load_model(str(folder/'model.cbm'))
                    replay=softmax(net.predict(x,prediction_type='RawFormulaVal',thread_count=4)+margin)
                    probability=np.load(folder/'raw_probabilities.npy',allow_pickle=False)
                    np.testing.assert_allclose(replay,probability,rtol=1e-6,atol=1e-7);reused+=1
                    q.dump(folder/'r78_replay.json',{'passed':True,'source':'R77 verified R70 both control'})
                else:
                    probability,_=fit_catboost(x,y,margin,fit,weights,folder,seed,rounds=450)
                    np.save(folder/'raw_probabilities.npy',probability,allow_pickle=False);new+=1
                raw[arm]=probability
                print(json.dumps({'stage':'fit','target':target,'arm':arm,'seed':seed,'new_fits':new,'reused':reused}),flush=True)
            raw=derived(raw,factors)
            old=np.array([r['probabilities'][target] for r in refs[str(seed)]])
            eligible=cache['original_valid'][check] if target==0 else np.ones(len(check),bool)
            for arm,p in raw.items():
                r=p[check];matched=r.copy();matched[~eligible]=old[~eligible]
                predictions[target,arm,seed]={'raw':r,'probabilities':matched}
                q.dump(out/f'predictions/{target}/{arm}/{seed}.json',{
                    'clips':cache['clips'][check].tolist(),'truth':cache['labels'][check,target].tolist(),
                    'raw':r.tolist(),'probabilities':matched.tolist(),'eligible':eligible.tolist(),
                    'train_in_sample':profile(cache['labels'][train,target],p[train])})
            q.dump(out/'progress.json',{'new_fits':new,'reused':reused})
    assert new==18 and reused==6
    summary=summarize(predictions,cache,refs);summary.update(new_gpu_fits=new,reused_models=reused,seconds=time.monotonic()-started)
    q.dump(out/'summary.json',summary);return summary
