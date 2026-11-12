"""R9: supervised role ranking from cached boxes; clicks never enter features."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import rgb_gru_data as data


def inside(box, point):
    return bool(box is not None and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def frame_features(frame):
    """Accept only detector output. No metadata, clicks, or outcome labels."""
    width, height = frame['width'], frame['height']
    objects = {}
    for kind in ['goal', 'ball']:
        candidates = [o for o in frame['objects'] if o['kind'] == kind]
        objects[kind] = max(candidates, key=lambda o: o['score']) if candidates else None
    features = []
    for person in frame['persons']:
        x1, y1, x2, y2 = person['bbox']; w, h = x2-x1, y2-y1
        cx, cy, foot = (x1+x2)/2, (y1+y2)/2, np.array([(x1+x2)/2, y2])
        body = [x1/width, y1/height, x2/width, y2/height, cx/width, cy/height,
                w/width, h/height, w/max(h, 1), w*h/(width*height)]
        context = []
        for kind in ['goal', 'ball']:
            obj = objects[kind]
            if obj is None:
                context.extend([0.]*10); continue
            a,b,c,d = obj['bbox']; ow,oh = max(c-a,1),max(d-b,1)
            center = np.array([(a+c)/2,(b+d)/2]); delta = (foot-center)/max(h,1)
            context.extend([1., obj['score'], center[0]/width, center[1]/height,
                            ow/width, oh/height, delta[0], delta[1], h/oh, w/ow])
        features.append(body+context)
    return np.clip(np.asarray(features, np.float32).reshape(-1,30), -20,20)


def choose_pair(probabilities):
    if len(probabilities) < 2:
        return [-1,-1]
    p = np.clip(probabilities, 1e-12, 1)
    scores = np.log(p[:,1,None]) + np.log(p[None,:,2])
    np.fill_diagonal(scores, -np.inf)
    return list(map(int, np.unravel_index(scores.argmax(), scores.shape)))


def coverage(probabilities, frames):
    correct = []; selected = []
    for frame in frames:
        pair = choose_pair(probabilities[frame['start']:frame['end']]); selected.append(pair)
        correct.append([index >= 0 and index in matches for index,matches in zip(pair,frame['matches'])])
    correct = np.asarray(correct, bool).reshape(-1,2)
    return {'kicker':float(correct[:,0].mean()),'keeper':float(correct[:,1].mean()),
            'mean':float(correct.mean()),'both':float(correct.all(1).mean()),'clips':len(frames)}, selected, correct


def pack(cache, folds, clicks, out):
    rows = pd.read_csv(folds); annotations = pd.read_csv(clicks).set_index('clip')
    assert len(rows)==1299 and rows.groupby('source_id').fold.nunique().max()==1
    original = json.loads((cache/'artifact_manifest.json').read_text())
    arrays, labels, frames, offset = [], [], [], 0
    for row in rows.itertuples():
        name = 'detections/'+row.clip+'.json'; assert data.sha(cache/name)==original[name]
        record = json.loads((cache/name).read_text())
        ann = annotations.loc[row.clip] if row.clip in annotations.index else None
        nearest = None if ann is None else min(range(len(record['anchors'])),key=lambda i:abs(record['anchors'][i]['timestamp']-ann.kicker_click_time_seconds))
        for i, frame in enumerate(record['anchors']):
            x = frame_features(frame); y = np.full(len(x), -1, np.int64); matches = [[],[]]
            annotated = i==nearest
            if annotated:
                points = [[ann[role+'_x_norm']*frame['width'],ann[role+'_y_norm']*frame['height']] for role in ['kicker','keeper']]
                matches = [[j for j,p in enumerate(frame['persons']) if inside(p['bbox'],point)] for point in points]
            eligible = annotated and all(len(a)==1 for a in matches) and matches[0]!=matches[1]
            if eligible:
                y[:]=0
                for role,match in enumerate(matches,1): y[match[0]]=role
            old = record['selected'][i]['roles']
            baseline = None if not annotated else [inside(old[role],point) for role,point in zip(['kicker','keeper'],points)]
            frames.append({'clip':row.clip,'source_id':row.source_id,'fold':int(row.fold),'anchor':i,
                           'timestamp':frame['timestamp'],'start':offset,'end':offset+len(x),
                           'annotated':annotated,'eligible_supervision':eligible,'matches':matches,
                           'old_point_coverage':baseline,'boxes':[p['bbox'] for p in frame['persons']],
                           'annotation_time_offset':None if not annotated else frame['timestamp']-ann.kicker_click_time_seconds})
            arrays.append(x);labels.append(y);offset+=len(x)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'candidates.npz',features=np.concatenate(arrays),labels=np.concatenate(labels))
    data.dump(out/'frames.json',frames)
    rows.to_csv(out/'public_folds.csv',index=False)
    data.dump(out/'manifest.json',{'source_sha256':data.sha(__file__),'folds_sha256':data.sha(folds),
              'annotation_sha256':data.sha(clicks),'cached_detection_manifest_sha256':data.sha(cache/'artifact_manifest.json'),
              'files':{name:data.sha(out/name) for name in ['candidates.npz','frames.json','public_folds.csv']},
              'feature_columns':'10 person-box geometry, then 10 goal-context and 10 ball-context; no clicks/times/source IDs/labels',
              'evaluation':'Nearest cached frame point coverage; temporal offset recorded; unmatched candidates count as failure.'})
    return {'frames':len(frames),'candidates':offset,'annotated_clips':sum(f['annotated'] for f in frames),
            'eligible_supervision_clips':sum(f['eligible_supervision'] for f in frames)}


def execute(bundle, out, budget=180):
    import torch
    import xgboost as xgb
    import xgb_probe
    assert torch.cuda.is_available()
    manifest = json.loads((bundle/'manifest.json').read_text())
    assert manifest['source_sha256']==data.sha(__file__)
    for name,digest in manifest['files'].items(): assert data.sha(bundle/name)==digest
    with np.load(bundle/'candidates.npz') as a: x,y=a['features'],a['labels']
    frames=json.loads((bundle/'frames.json').read_text());out.mkdir(parents=True,exist_ok=True)
    def indices(folds):
        return np.array([i for f in frames if f['fold'] in folds and f['eligible_supervision'] for i in range(f['start'],f['end'])],dtype=int)
    inner,train=indices([2,3,4]),indices([1,2,3,4])
    checks=[f for f in frames if f['fold']==1 and f['annotated']]
    outer=[f for f in frames if f['fold']==0 and f['annotated']]
    assert set(f['source_id'] for f in frames if f['fold'] in [2,3,4]).isdisjoint(f['source_id'] for f in checks)
    assert set(f['source_id'] for f in frames if f['fold']!=0).isdisjoint(f['source_id'] for f in outer)
    protocol={'source_sha256':data.sha(__file__),'bundle_sha256':data.sha(bundle/'manifest.json'),
              'gpu':torch.cuda.get_device_name(0),'xgboost':xgb.__version__,'depths':[3,5],
              'round_grid':[1]+list(range(10,201,10)),'seed':6643,'variants':['body','context'],
              'selection':'Inner fold 1 mean role point coverage; earliest rounds then shallower tree win ties.',
              'fit_filter':'Training clicks must uniquely match distinct cached candidate boxes. Evaluation includes unmatched cases.',
              'no_clicks_in_inference':True}
    data.immutable_json(out/'protocol.json',protocol)
    started=time.monotonic();summary={}
    baseline=lambda fs:{'kicker':float(np.mean([f['old_point_coverage'][0] for f in fs])),
                       'keeper':float(np.mean([f['old_point_coverage'][1] for f in fs]))}
    def fit(matrix,fit_indices,depth,rounds,path):
        weights=np.sqrt(len(fit_indices)/(3*np.bincount(y[fit_indices],minlength=3)))
        training=matrix.slice(fit_indices);training.set_weight(weights[y[fit_indices]])
        history={};tick=time.monotonic()
        model=xgb.train({'objective':'multi:softprob','num_class':3,'device':'cuda','tree_method':'hist',
                        'max_depth':depth,'eta':.05,'min_child_weight':3,'lambda':10,'subsample':1.,
                        'colsample_bytree':1.,'seed':6643,'nthread':4,'eval_metric':'mlogloss'},
                       training,num_boost_round=rounds,evals=[(training,'fit')],evals_result=history,verbose_eval=False)
        runtime=xgb_probe.gpu_config(model);path.mkdir(parents=True,exist_ok=True)
        model.save_model(path/'model.ubj');data.dump(path/'history.json',history);data.dump(path/'runtime.json',runtime)
        info={'seconds':time.monotonic()-tick,'model_sha256':data.sha(path/'model.ubj'),'fit_indices':fit_indices.tolist(),
              'depth':depth,'rounds':rounds,'class_weights':weights.tolist()};data.dump(path/'fit.json',info)
        return model,info
    # A bounded cost fit is discarded, never used as a selected model.
    matrix=xgb.DMatrix(x,label=np.maximum(y,0))
    _,cost=fit(matrix,inner,3,20,out/'cost_pilot')
    estimate=cost['seconds']*60
    data.dump(out/'cost_profile.json',{'seconds_20_rounds':cost['seconds'],'conservative_full_estimate_seconds':estimate})
    if estimate>budget-30: raise RuntimeError('Cost gate failed; no full role screen started')
    for variant,features in [('body',x[:,:10]),('context',x)]:
        matrix=xgb.DMatrix(features,label=np.maximum(y,0));candidates=[]
        for depth in protocol['depths']:
            if time.monotonic()-started>budget-30: raise TimeoutError('Role screen time budget reached')
            model,info=fit(matrix,inner,depth,200,out/variant/f'inner_depth{depth}')
            for rounds in protocol['round_grid']:
                metrics,_,_=coverage(model.predict(matrix,iteration_range=(0,rounds)),checks)
                candidates.append({'depth':depth,'rounds':rounds,'metrics':metrics})
        chosen=max(candidates,key=lambda a:(a['metrics']['mean'],-a['rounds'],-a['depth']))
        model,info=fit(matrix,train,chosen['depth'],chosen['rounds'],out/variant/'outer')
        probabilities=model.predict(matrix);reloaded=xgb.Booster(model_file=out/variant/'outer/model.ubj')
        reloaded.set_param({'device':'cuda'});xgb_probe.gpu_config(reloaded)
        np.testing.assert_allclose(probabilities,reloaded.predict(matrix),atol=1e-7,rtol=1e-6)
        metrics,pairs,_=coverage(probabilities,outer)
        predictions=[{'clip':f['clip'],'anchor':f['anchor'],'pair':choose_pair(probabilities[f['start']:f['end']])} for f in frames]
        data.dump(out/variant/'selections.json',predictions)
        np.savez_compressed(out/variant/'probabilities.npz',probabilities=probabilities)
        summary[variant]={'chosen':chosen,'candidates':candidates,'outer':metrics,'fit':info}
        data.dump(out/'partial_summary.json',summary)
        print(json.dumps({'variant':variant,'inner':chosen['metrics'],'outer':metrics}),flush=True)
    report={'variants':summary,'baseline_inner':baseline(checks),'baseline_outer':baseline(outer),
            'seconds':time.monotonic()-started,'cost_estimate_seconds':estimate,
            'limitations':'Approximate-frame point coverage, not identity accuracy or direction prediction score.'}
    data.dump(out/'summary.json',report)
    data.dump(out/'artifact_manifest.json',{str(p.relative_to(out)):data.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    return report
