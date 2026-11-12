"""R79 observed-window sampling on frozen R69 pose tracks. No perception calls."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pose_quality_ablation as q
from pose_mirror import JOINT_SWAP
from middle_direction_ablation import profile
from skeleton_geometry_ablation import digest
from sparse_role_comparison import source_interval

COMMON=('last','early3','uniform3','late3','peak3','five_pose','five_motion')
ARMS={0:COMMON+('last_geometry','five_geometry'),1:COMMON}
ANGLES=((11,13,15),(12,14,16),(5,11,13),(6,12,14))


def reflected(sequence):
    result=sequence.copy();p=result[...,:92].reshape(*result.shape[:-1],23,4)
    p[:]=p[...,JOINT_SWAP,:].copy();p[...,0]*=-1;result[...,92]*=-1
    return result


def peak_scores(sequence,times,target):
    s=np.asarray(sequence);p=s[:,:92].reshape(5,23,4);dt=np.diff(times)
    ids=[15,16,17,18,19,20,21,22] if target==0 else [5,6,7,8,9,10,11,12]
    scores=np.full(4,-1.,float)
    for k in range(4):
        good=(p[k,ids,3]>0)&(p[k+1,ids,3]>0)
        if not(s[k,94]>0 and s[k+1,94]>0 and good.sum()>=4):continue
        velocity=np.linalg.norm(p[k+1,ids,:2]-p[k,ids,:2],axis=1)/dt[k]
        scores[k]=float(np.median(velocity[good]))
        if target==1 and s[k,95]>0 and s[k+1,95]>0:
            scores[k]+=float(np.linalg.norm(s[k+1,92:94]-s[k,92:94])/dt[k])
    return scores


def select(sequence,times,target,arm):
    assert arm in ARMS[target]
    if arm in ('last','last_geometry'):return np.array([4])
    if arm=='early3':return np.array([0,1,2])
    if arm=='uniform3':return np.array([0,2,4])
    if arm=='late3':return np.array([2,3,4])
    if arm=='peak3':
        scores=peak_scores(sequence,times,target)
        if (scores<0).all():return np.array([2,3,4])
        start=int(np.clip(np.argmax(scores)-1,0,2));return np.arange(start,start+3)
    return np.arange(5)


def encode(sequence,times,target,arm):
    """Fixed slots, actual offsets, observed-pair velocity and angular rates.

    Cumulative root coordinates are excluded from static input: last-only must
    not secretly receive displacement from the beginning of the window.
    A selected pair never spans an unavailable core/joint observation.
    """
    s=np.asarray(sequence,np.float64);times=np.asarray(times,np.float64)
    assert s.shape==(5,96) and np.isfinite(s).all() and np.all(np.diff(times)>0)
    selected=select(s,times,target,arm);p=s[:,:92].reshape(5,23,4)
    static=np.zeros((5,96));velocity=np.zeros((5,23,3));angular=np.zeros((5,4,2));root=np.zeros((5,3));pair_time=np.zeros(5)
    for k in selected:
        static[k,:92]=s[k,:92];static[k,92:]=[s[k,94],s[k,95],times[k]-times[-1],1]
    if arm!='five_pose':
        for a,b in zip(selected[:-1],selected[1:]):
            if not (s[a:b+1,94]>0).all():continue
            dt=times[b]-times[a];pair_time[b]=dt
            good=(p[a:b+1,:,3]>0).all(0)
            velocity[b,good,:2]=(p[b,good,:2]-p[a,good,:2])/dt;velocity[b,good,2]=1
            for j,(u,v,w) in enumerate(ANGLES):
                if not good[[u,v,w]].all():continue
                vectors=p[[a,b]][:,[u,w],:2]-p[[a,b]][:,v,None,:2]
                norm=np.linalg.norm(vectors[:,0],axis=1)*np.linalg.norm(vectors[:,1],axis=1)
                if (norm<=1e-6).any():continue
                theta=np.arccos(np.clip((vectors[:,0]*vectors[:,1]).sum(1)/norm,-1,1))
                angular[b,j]=[(theta[1]-theta[0])/dt,1]
            if (s[a:b+1,95]>0).all():root[b]=[*(s[b,92:94]-s[a,92:94])/dt,1]
    return np.r_[static.ravel(),velocity.ravel(),angular.ravel(),root.ravel(),pair_time].astype(np.float32)


def assemble(cache,sequence,times,compatible,target,arm):
    n=len(cache['clips']);seq=np.concatenate([sequence,reflected(sequence)])
    stamps=np.concatenate([times,times]);rows=[]
    for i in range(2*n):rows.append(encode(seq[i,:,target],stamps[i],target,arm))
    x=np.c_[np.asarray(rows),cache['original_full'][:,186:189]]
    if arm.endswith('_geometry'):
        geom=cache['original_full'][:,189:435].copy();geom[~np.r_[compatible,compatible]]=0
        x=np.c_[x,geom]
    return x.astype(np.float32)


def load(inputs):
    inputs=Path(inputs);cache,refs=q.load(inputs)
    data=dict(np.load(inputs/'observations.npz',allow_pickle=False))
    np.testing.assert_array_equal(data['clips'],cache['clips'])
    encoded=dict(np.load(inputs/'encoded.npz',allow_pickle=False))
    return cache,refs,data,encoded


def audit(cache,data):
    result={'window_limit':'Existing last0.30-0.37s only; no true contact timestamp or outside-window poses.',
            'groups':{},'cases':[]};seq=data['sequence'];times=data['times']
    for split,ids in [('train',cache['inner_train']),('validation',cache['inner_check'])]:
        duration=times[ids,-1]-times[ids,0]
        result[split]={'clips':len(ids),'duration_quantiles':np.quantile(duration,[0,.25,.5,.75,1]).tolist(),
                       'interval_quantiles':np.quantile(np.diff(times[ids],axis=1),[0,.5,1]).tolist()}
        for target in (0,1):
            for cls in ('middle','sides'):
                use=ids[(cache['labels'][ids,target]==1)==(cls=='middle')]
                s=seq[use,:,target];p=s[:,:,:92].reshape(-1,5,23,4);obs=s[:,:,94]>0
                peaks=np.array([peak_scores(seq[i,:,target],times[i],target) for i in use])
                valid_peak=(peaks>=0).any(1);peak_index=peaks.argmax(1)
                result['groups'][f'{split}/{target}/{cls}']={
                    'clips':len(use),'sources':len(set(cache['sources'][use])),
                    'observed_per_slot':obs.sum(0).tolist(),'all_five':int(obs.all(1).sum()),
                    'foot_visible_fraction_per_slot':(p[:,:,17:,3]>0).mean((0,2)).tolist(),
                    'peak_interval_counts':np.bincount(peak_index[valid_peak],minlength=4).tolist(),
                    'no_usable_peak':int((~valid_peak).sum()),
                    'endpoint_pose_change_median':float(np.median(np.linalg.norm(p[:,-1,:,:2]-p[:,0,:,:2],axis=-1)[(p[:,-1,:,3]>0)&(p[:,0,:,3]>0)&obs[:,0,None]&obs[:,-1,None]]))}
    for i in cache['inner_check']:
        for target in (0,1):
            result['cases'].append({'clip':str(cache['clips'][i]),'target':target,'truth':int(cache['labels'][i,target]),
                'source':str(cache['sources'][i]),'frame_indices':data['frame_indices'][i].tolist(),
                'relative_ms':((times[i]-times[i,-1])*1000).tolist(),
                'observed':(seq[i,:,target,94]>0).tolist(),
                'peak_scores':peak_scores(seq[i,:,target],times[i],target).tolist(),
                'selected':{arm:select(seq[i,:,target],times[i],target,arm).tolist() for arm in COMMON},
                'pose2d':seq[i,:,target,:92].reshape(5,23,4).tolist()})
    return result


def summarize(predictions,cache,refs,data):
    check=cache['inner_check'];sources=cache['sources'][check];rows=[];candidates=[]
    for target in (0,1):
        y=cache['labels'][check,target];old=np.array([[r['prediction'][target]==r['truth'][target] for r in refs[str(s)]] for s in q.SEEDS])
        complete=(data['sequence'][check,:,target,94]>0).all(1)
        for arm in ('control',)+ARMS[target]:
            ts=[profile(y,predictions[target,arm,s]['probabilities']) for s in q.SEEDS]
            rt=[profile(y,predictions[target,arm,s]['raw']) for s in q.SEEDS]
            c=np.array([predictions[target,arm,s]['probabilities'].argmax(1)==y for s in q.SEEDS])
            reference='last_geometry' if arm=='five_geometry' else 'last'
            b=np.array([predictions[target,reference,s]['probabilities'].argmax(1)==y for s in q.SEEDS])
            d=c.astype(float)-b;dr=c.astype(float)-old;ci=source_interval(dr.mean(0),sources)
            raw_complete=[profile(y[complete],predictions[target,arm,s]['raw'][complete]) for s in q.SEEDS]
            candidate=(arm not in ('control','last','last_geometry') and dr.mean()>=.02 and min(dr.mean(1))>0
                       and ci['source_cluster_95pct'][0]>0 and d.mean()>0)
            row={'target':target,'arm':arm,'accuracy':float(c.mean()),'raw_accuracy':float(np.mean([r['accuracy'] for r in rt])),
                 'trials':ts,'raw_trials':rt,'complete_five_clips':int(complete.sum()),'complete_five_raw':raw_complete,
                 'reference':reference,'delta_vs_reference':float(d.mean()),'paired_reference':source_interval(d.mean(0),sources),
                 'delta_vs_retained':float(dr.mean()),'seed_delta_retained':dr.mean(1).tolist(),'paired_retained':ci,
                 'candidate_for_confirmation':bool(candidate)}
            if candidate:candidates.append({'target':target,'arm':arm})
            rows.append(row)
    return {'state':'complete','rows':rows,'candidates_for_confirmation':candidates,
            'new_perception_calls':0,'competition_submissions':0,
            'limits':['No observation outside the existing last0.30-0.37s window; no known contact instant.',
             'Peak motion is an automatic proxy, not detected ball contact; jitter can produce peaks.',
             'All arms reuse whole-window tracking/normalization; early-window scores are not online early prediction.',
             'Cumulative root displacement is excluded from last-only; actual selected-pair velocity requires all intermediate cached observations.',
             'Quality masks do not prove identity or joint correctness. Reused exploratory source fold.',
             'Kick geometry arms retain old3D only under the original R69 identity-compatibility proxy.']}


def run(inputs,out,budget=1800):
    import torch,catboost
    from pose_tree_controls import fit_catboost,softmax
    assert torch.cuda.is_available() and catboost.__version__=='1.2.10'
    inputs,out=Path(inputs),Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'input_manifest.json').read_text())
    for name,h in manifest['files'].items():assert q.sha(inputs/name)==h,name
    cache,refs,data,encoded=load(inputs);check=cache['inner_check'];train=cache['inner_train']
    q.dump(out/'protocol.json',{'source_sha256':q.sha(__file__),'manifest':manifest,'gpu':torch.cuda.get_device_name(0),
           'train':779,'validation':262,'new_perception_calls':0})
    started=time.monotonic();predictions={};new=reused=0
    for target in (0,1):
        margin=cache['inner_margin'][:,target*3:target*3+3];y=cache['inner_y'][:,target]
        eligible=data['eligible'][check,target]&(cache['original_valid'][check] if target==0 else True)
        for seed in q.SEEDS:
            old=np.array([r['probabilities'][target] for r in refs[str(seed)]])
            for arm in ('control',)+ARMS[target]:
                if time.monotonic()-started>budget-90:raise TimeoutError('Saved model boundary')
                folder=out/'fits'/str(target)/arm/str(seed)
                if arm=='control':
                    x=cache['original_full'] if target==0 else cache['original_full'][:,:189]
                    shutil.copytree(inputs/f'reuse/{target}/{seed}',folder,dirs_exist_ok=True)
                    net=catboost.CatBoostClassifier();net.load_model(str(folder/'model.cbm'))
                    raw=np.load(folder/'raw_probabilities.npy',allow_pickle=False)
                    replay=softmax(net.predict(x,prediction_type='RawFormulaVal',thread_count=4)+margin)
                    np.testing.assert_allclose(raw,replay,rtol=1e-6,atol=1e-7);reused+=1
                    q.dump(folder/'r79_replay.json',{'passed':True,'source':'verified R78 control'})
                    valid=cache['original_valid'][check] if target==0 else np.ones(len(check),bool)
                else:
                    x=encoded[f't{target}_{arm}'];assert np.isfinite(x).all()
                    raw,_=fit_catboost(x,y,margin,cache['inner_fit'],cache['inner_weights'],folder,seed,rounds=450)
                    np.save(folder/'raw_probabilities.npy',raw,allow_pickle=False);new+=1;valid=eligible
                prob=raw[check].copy();prob[~valid]=old[~valid]
                predictions[target,arm,seed]={'raw':raw[check],'probabilities':prob}
                q.dump(out/f'predictions/{target}/{arm}/{seed}.json',{'clips':cache['clips'][check].tolist(),
                    'truth':cache['labels'][check,target].tolist(),'probabilities':prob.tolist(),'raw':raw[check].tolist(),
                    'eligible':valid.tolist(),'train_in_sample':profile(cache['labels'][train,target],raw[train])})
                q.dump(out/'progress.json',{'new_fits':new,'reused':reused})
                print(json.dumps({'stage':'fit','target':target,'arm':arm,'seed':seed,'new_fits':new,'reused':reused}),flush=True)
    assert new==48 and reused==6
    summary=summarize(predictions,cache,refs,data);summary.update(new_gpu_fits=new,reused_models=reused,seconds=time.monotonic()-started)
    q.dump(out/'summary.json',summary);return summary
