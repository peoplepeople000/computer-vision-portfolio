"""R28 matched calibrated-body/field additions to the frozen312D baseline."""
import hashlib
import json
from pathlib import Path
import numpy as np
import calibrated_nlf as camera
import nlf_pose_features as nlf
import nlf_pose_pilot as pilot
import pose_3d_classifier as exact


def input_identity(sample):
    """All neural input arguments, independent of labels, source or fold."""
    info=sample.get('camera')
    if info is None:return None,None
    box=pilot.observed_box(sample['points'],sample['scores'])
    if box is None:return None,None
    desc={'version':'r28_calibrated_v1','image_sha256':sample['image_sha256'],
          'model_sha256':pilot.MODEL_SHA,'intrinsic':np.asarray(info['intrinsic'],np.float32).tolist(),
          'camera_up':np.asarray(info['camera_up'],np.float32).tolist(),'box':box.tolist(),
          'rtmw_sha256':hashlib.sha256(np.asarray(sample['points'],np.float32).tobytes()+np.asarray(sample['scores'],np.float32).tobytes()).hexdigest(),
          'calibration_sha256':sample['calibration_sha256'],'num_aug':1,'internal_batch_size':4,
          'identity_extrinsic':True,'zero_distortion':True}
    digest=hashlib.sha256(json.dumps(desc,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return desc,digest


def representations(points, uncertainty, agreement, rotation, side):
    body,bvalid=nlf.geometry(points,uncertainty,agreement,'body')
    mirrored_body,bmvalid=nlf.geometry(*nlf.reflect(points,uncertainty),agreement,'body')
    field,fvalid=camera.field_features(points,uncertainty,agreement,rotation,side)
    mirrored_field,fmvalid=camera.field_features(points,uncertainty,agreement,rotation,side,mirrored=True)
    valid=bool(bvalid and bmvalid and fvalid and fmvalid)
    arrays={'calibrated_body':np.stack([body,mirrored_body]),'calibrated_field':np.stack([field,mirrored_field])}
    if not valid:arrays={k:np.zeros_like(v) for k,v in arrays.items()}
    return arrays,valid


def choose(scores, control, denominator):
    if exact.passes_gate(scores['calibrated_field'],[control,scores['calibrated_body']],denominator):return 'calibrated_field'
    if exact.passes_gate(scores['calibrated_body'],[control],denominator):return 'calibrated_body'
    return None


def execute(inputs, payload, previous, assets, out, budget=900):
    import gc
    import shutil
    import time
    import torch
    import torchvision
    import catboost
    from PIL import Image
    import rgb_gru_data as data
    import rgb_gru as metrics
    import pose_tree_controls as controls
    import sparse_role_comparison as sparse
    assert torch.cuda.is_available() and catboost.__version__=='1.2.10'
    inputs,payload,previous,assets,out=map(Path,(inputs,payload,previous,assets,out))
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(6643)
    manifest=json.loads((inputs/'r28-inputs.json').read_text())
    for name,h in manifest['files'].items():assert data.sha(inputs/name)==h
    spec=json.loads((payload/'samples.json').read_text());assert spec['stage']=='inner' and len(spec['samples'])==1041
    for name,h in spec['files'].items():assert data.sha(payload/name)==h
    assert spec['camera_helper_sha256']==data.sha(camera.__file__) and spec['crop_helper_sha256']==data.sha(data.__file__)
    with np.load(inputs/'baseline.npz') as a:cache={k:a[k] for k in a.files}
    refs=json.loads((inputs/'references.json').read_text());samples=spec['samples']
    assert all(s['fold']!=0 and cache['clips'][s['index']]==s['clip'] for s in samples)
    weights=assets/'nlf_l_multi_0.3.2.patch6.torchscript';assert data.sha(weights)==pilot.MODEL_SHA
    protocol={'source_sha256':data.sha(__file__),'input_manifest_sha256':data.sha(inputs/'r28-inputs.json'),
        'camera_helper_sha256':data.sha(camera.__file__),'nlf_helper_sha256':data.sha(nlf.__file__),
        'classifier_helper_sha256':data.sha(controls.__file__),'model_sha256':pilot.MODEL_SHA,
        'inference':'Exact input_identity arguments; same official24joints. Reuse10pilot tensors without reexecution.',
        'arms':['calibrated_body','calibrated_field'],'feature_size':435,'seeds':[6643,6644,6645],
        'recipe':'Same R19 CatBoost450/depth4/.035/L2=15, train-onlyprior, mirror weights, all original fitting rows.',
        'fallback':'Saved matched R19 probabilities when calibrated extra invalid; keeper/outcome unchanged.',
        'gate':manifest['plan']['promotion'],'budget':budget}
    data.immutable_json(out/'protocol.json',protocol)
    (out/'poses').mkdir(exist_ok=True);(out/'records').mkdir(exist_ok=True)
    previous_checks=json.loads((previous/'artifact_manifest.json').read_text())
    for name,h in previous_checks.items():assert data.sha(previous/name)==h
    old_protocol=json.loads((previous/'protocol.json').read_text())
    assert old_protocol['camera_helper_sha256']==data.sha(camera.__file__)
    assert old_protocol['nlf_helper_sha256']==data.sha(nlf.__file__) and old_protocol['model_sha256']==pilot.MODEL_SHA
    old_samples={s['clip']:s for s in json.loads((previous/'input_manifest.json').read_text())['samples']}
    reused={};new_calls=0
    shutil.copy2(previous/'protocol.json',out/'pilot_protocol.json')
    for sample in samples:
        if sample['clip'] not in old_samples:continue
        source=previous/'records'/(sample['clip']+'.json');record=json.loads(source.read_text())
        target=out/'pilot_records'/source.name;target.parent.mkdir(exist_ok=True)
        if target.exists():assert data.sha(target)==data.sha(source)
        else:shutil.copy2(source,target)
        reused[str(target.relative_to(out))]=data.sha(source)
        if not sample['input_key']:continue
        assert record['cuda_verified'] and record['calibration_sha256']==sample['calibration_sha256']
        desc,key=input_identity(sample);old_desc,_=input_identity(old_samples[sample['clip']]);assert desc==old_desc and key==sample['input_key']
        for name in ['intrinsic','camera_up','box']:np.testing.assert_array_equal(desc[name],record[name])
        source=previous/'poses'/(sample['clip']+'.npz');assert data.sha(source)==record['output_sha256']
        target=out/'poses'/(key+'.npz')
        if target.exists():assert data.sha(target)==data.sha(source)
        else:shutil.copy2(source,target)
        reused[str(target.relative_to(out))]=data.sha(source)
        wrapper={'identity':desc,'input_key':key,'origin':'pilot','pose_sha256':data.sha(source),
            'pilot_record_sha256':data.sha(previous/'records'/(sample['clip']+'.json')),
            'cuda_verified':True,'inference_valid':True,'agreement':record['agreement'],
            'self_reprojection_max_pixels':record['self_reprojection_max_pixels'],'seconds':record['seconds']}
        data.immutable_json(out/'records'/(key+'.json'),wrapper)
    assert sum(k.startswith('poses/') for k in reused)==10
    data.dump(out/'reused_pilot_payloads.json',reused)
    net=None;canonical=None;completed=0
    def boundary():
        if time.monotonic()-started>budget-120:raise TimeoutError('Safe item boundary; completed camera-keyed poses/checkpoints retained')
    for sample in samples:
        desc,key=input_identity(sample);assert desc==sample['identity'] and key==sample['input_key']
        if desc is None:continue
        boundary();path=out/'records'/(key+'.json');destination=out/'poses'/(key+'.npz')
        if path.exists():
            record=json.loads(path.read_text());assert record['identity']==desc and data.sha(destination)==record['pose_sha256']
            completed+=1;continue
        info=camera.camera_inputs(sample['calibration'],sample['box'],sample['frame_index'])
        for name in info:np.testing.assert_allclose(info[name],sample['camera'][name],atol=1e-7,rtol=1e-7)
        assert sample['anchor']==2 and desc['identity_extrinsic'] and desc['zero_distortion']
        if net is None:
            net=torch.jit.load(str(weights),map_location='cuda').eval()
            with torch.inference_mode(),torch.device('cuda'):
                canonical=net.get_weights_for_canonical_points(net.cano_all['smpl'][-24:].cuda().float())
        image_path=payload/'images'/(desc['image_sha256']+'.jpg');assert data.sha(image_path)==desc['image_sha256']
        pixels=np.asarray(Image.open(image_path).convert('RGB')).copy();assert pixels.shape==(224,224,3)
        intrinsic=np.asarray(desc['intrinsic'],np.float32);up=np.asarray(desc['camera_up'],np.float32);box=np.asarray(desc['box'],np.float32)
        image=torch.from_numpy(pixels).permute(2,0,1)[None].cuda();torch.cuda.synchronize();tick=time.monotonic()
        with torch.inference_mode(),torch.device('cuda'):
            pred=net.estimate_poses_batched(image,[torch.from_numpy(box)[None].cuda()],canonical,
                intrinsic_matrix=torch.from_numpy(intrinsic)[None].cuda(),world_up_vector=torch.from_numpy(up).cuda(),
                extrinsic_matrix=torch.eye(4,device='cuda')[None],distortion_coeffs=torch.zeros((1,5),device='cuda'),
                num_aug=desc['num_aug'],internal_batch_size=desc['internal_batch_size'])
        torch.cuda.synchronize();seconds=time.monotonic()-tick
        tensors=[pred[k][0][0] for k in ['poses3d','poses2d','uncertainties']];assert all(t.is_cuda for t in tensors)
        points,projected,u=[t.cpu().numpy() for t in tensors]
        assert points.shape==(24,3) and projected.shape==(24,2) and u.shape==(24,)
        expected,visible=camera.project_camera(points,intrinsic)
        finite=np.isfinite(points).all() and np.isfinite(projected).all() and np.isfinite(u).all()
        error=float(np.linalg.norm(expected-projected,axis=1).max()) if finite and visible.all() else None
        valid=bool(error is not None and error<=.05)
        agreement=pilot.projection_error(projected,sample['points'],sample['scores'],box[3]) if valid else None
        if agreement is not None and not np.isfinite(agreement):agreement=None
        with destination.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,points3d=points,points2d=projected,uncertainty=u,intrinsic=intrinsic,camera_up=up,image_sha256=desc['image_sha256'])
        destination.with_suffix('.tmp').replace(destination)
        data.dump(path,{'identity':desc,'input_key':key,'origin':'new','pose_sha256':data.sha(destination),
            'cuda_verified':True,'inference_valid':valid,'agreement':agreement,'self_reprojection_max_pixels':error,'seconds':seconds})
        new_calls+=1;completed+=1
        if completed%100==0:print(json.dumps({'stage':'calibrated_nlf','completed':completed,'new':new_calls,'pilot_reused':10}),flush=True)
    peak=torch.cuda.max_memory_allocated()/1e9;del net;canonical=None;gc.collect();torch.cuda.empty_cache()
    n=len(cache['clips']);assert n==1299
    extras={name:np.zeros((2*n,123),np.float32) for name in protocol['arms']};valid=np.zeros(n,bool)
    for sample in samples:
        key=sample['input_key']
        if not key:continue
        record=json.loads((out/'records'/(key+'.json')).read_text());i=sample['index']
        with np.load(out/'poses'/(key+'.npz')) as a:
            values,ok=representations(a['points3d'],a['uncertainty'],record['agreement'],sample['camera']['rotation'],sample['camera']['goal_side'])
        valid[i]=ok
        for name in extras:extras[name][[i,i+n]]=values[name]
    np.savez_compressed(out/'calibrated_features.npz',**extras,valid=valid,clips=cache['clips'])
    for name,h in reused.items():assert data.sha(out/name)==h
    data.dump(out/'extraction_summary.json',{'samples':len(samples),'eligible':completed,'new_inferences':new_calls,
        'pilot_reused':10,'valid_by_fold':{str(f):int(valid[cache['folds']==f].sum()) for f in (1,2,3,4)},
        'peak_cuda_gb':peak,'seconds':time.monotonic()-started,'new_detector_rtmw_calibration_calls':0})
    data.dump(out/'hardware.json',{'gpu':torch.cuda.get_device_name(0),'nlf_cuda_outputs_verified':True,
        'catboost_training':'GPU','catboost_multiclass_prediction':'Kaggle CPU'})
    check,fit,train=cache['inner_check'],cache['inner_fit'],cache['inner_train']
    assert len(check)==262 and set(cache['sources'][train]).isdisjoint(cache['sources'][check])
    np.testing.assert_array_equal(np.sort(fit),np.sort(np.r_[train,train+n]))
    original={};control_num=0
    for seed in protocol['seeds']:
        saved=refs[str(seed)]['predictions'];assert [r['clip'] for r in saved]==cache['clips'][check].tolist()
        np.testing.assert_array_equal([r['truth'] for r in saved],cache['labels'][check])
        p=[np.asarray([r['probabilities'][k] for r in saved]) for k in range(3)]
        original[seed]=p;control_num+=exact.integer_score(cache['labels'][check],np.column_stack([a.argmax(1) for a in p]))[0]
    denominator=100*len(check)*len(protocol['seeds']);scores={};report={'state':'running','arms':{},'control_numerator':control_num,'denominator':denominator}
    for name,extra in extras.items():
        full=np.c_[cache['inner_base'],extra].astype(np.float32);assert full.shape==(2598,435) and np.isfinite(full).all()
        data.dump(out/(name+'_input.json'),{'base_sha256':hashlib.sha256(cache['inner_base'].tobytes()).hexdigest(),
            'extra_sha256':hashlib.sha256(extra.tobytes()).hexdigest(),'full_sha256':hashlib.sha256(full.tobytes()).hexdigest(),
            'valid_validation':int(valid[check].sum()),'invalid_validation':int((~valid[check]).sum())})
        trials=[];numerator=0;deltas=[]
        for seed in protocol['seeds']:
            boundary();directory=out/name/str(seed)
            p,info=controls.fit_catboost(full,cache['inner_y'][:,0],cache['inner_margin'][:,:3],fit,cache['inner_weights'],directory/'target0',seed)
            raw=p[check];kick=raw.copy();kick[~valid[check]]=original[seed][0][~valid[check]]
            probabilities=[kick,*original[seed][1:]];pred=np.column_stack([a.argmax(1) for a in probabilities])
            control=np.column_stack([a.argmax(1) for a in original[seed]])
            numerator+=exact.integer_score(cache['labels'][check],pred)[0]
            deltas.append(((pred==cache['labels'][check]).astype(float)-(control==cache['labels'][check]).astype(float))@np.asarray(data.WEIGHTS))
            result={'seed':seed,**metrics.metric_report(cache['labels'][check],pred),'fit':info};trials.append(result)
            data.dump(directory/'result_summary.json',result);np.save(directory/'raw_kick.npy',raw,allow_pickle=False)
            data.dump(directory/'predictions.json',[{'clip':str(cache['clips'][i]),'truth':cache['labels'][i].tolist(),
                'prediction':pred[j].tolist(),'probabilities':[a[j].tolist() for a in probabilities],'extra_valid':bool(valid[i])} for j,i in enumerate(check)])
            print(json.dumps({'stage':'classifier','arm':name,'seed':seed,'score':result['weighted_score']}),flush=True)
        scores[name]=numerator;report['arms'][name]={'trials':trials,'numerator':numerator,'mean':numerator/denominator,
            'paired_vs_control':sparse.source_interval(np.mean(deltas,0),cache['sources'][check])}
        data.dump(out/'summary.json',report)
    report.update(state='complete',promoted=choose(scores,control_num,denominator),control_mean=control_num/denominator,
        seconds=time.monotonic()-started,competition_score=None,goal_target=.8)
    data.dump(out/'summary.json',report);return report
