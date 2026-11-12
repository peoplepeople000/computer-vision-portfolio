"""R29 frozen calibrated-field inference over the completed R25 private cache."""
import hashlib
import json
from pathlib import Path
import numpy as np
import calibrated_nlf as camera
import calibrated_nlf_experiment as representation
import field_calibration_pilot as field
import nlf_pose_pilot as pilot
import rgb_gru_data as data


def ordered_clips(clips):
    return sorted(clips,key=lambda c:hashlib.sha256(('r29-calibrated-private|'+c).encode()).hexdigest())


def clean_dump(path,value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temporary.replace(path)


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


def camera_sample(clip,role,calibration,calibration_sha):
    """Only the matching last-anchor crop can use the last-frame calibration."""
    info=camera.camera_inputs(calibration,role['box'],clip['frame_index']) if role['anchor']==2 and role['image_sha256'] else None
    result={'clip':clip['clip'],'frame_index':clip['frame_index'],'anchor':role['anchor'],
        'image_sha256':role['image_sha256'],'points':role['points'],'scores':role['scores'],
        'box':role['box'],'calibration_sha256':calibration_sha,
        'camera':None if info is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()}}
    desc,key=representation.input_identity(result);result.update(identity=desc,input_key=key)
    return result


def replace_kick(new,original,valid):
    """Exact per-seed fallback; the two unchanged heads are never recomputed."""
    new=np.asarray(new);valid=np.asarray(valid,bool)
    assert len(original)==3 and new.shape==original[0].shape==(len(valid),3)
    result=[np.asarray(a).copy() for a in original]
    result[0][valid]=new[valid]
    for p,classes in zip(result,(3,3,2)):
        assert p.shape==(len(valid),classes) and np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-6)
    return result


class FieldCalibrator:
    """Same pretrained R26 inference, calibration and quality recipe."""
    def __init__(self,upstream,weights,weight_hashes,out,protocol_sha):
        import sys
        import torch
        import yaml
        sys.path.insert(0,str(upstream))
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_line_net
        from utils.utils_calib import FramebyFrameCalib,keypoint_world_coords_2D,line_world_coords_3D
        from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool as get_kp
        from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool_l as get_lines
        from utils.utils_heatmap import coords_to_dict,complete_keypoints
        self.camera=FramebyFrameCalib;self.get_kp=get_kp;self.get_lines=get_lines
        self.coords=coords_to_dict;self.complete=complete_keypoints
        self.out=Path(out);self.protocol_sha=protocol_sha;self.models=[]
        self.world={i+1:[*xy,-2.44 if i+1 in (12,15,16,19) else 0.] for i,xy in enumerate(keypoint_world_coords_2D)}
        self.line_world={i+1:v for i,v in enumerate(line_world_coords_3D)}
        for maker,cfg,name in ((get_cls_net,'hrnetv2_w48.yaml','SV_kp'),(get_line_net,'hrnetv2_w48_l.yaml','SV_lines')):
            assert data.sha(Path(weights)/name)==weight_hashes[name]
            net=maker(yaml.safe_load((Path(upstream)/'config'/cfg).read_text()))
            net.load_state_dict(torch.load(Path(weights)/name,map_location='cpu',weights_only=True))
            net=net.cuda().eval();assert next(net.parameters()).is_cuda;self.models.append(net)
        clean_dump(self.out/'world_coordinates.json',{'keypoints':self.world,'lines':self.line_world})

    def infer(self,sample,frame):
        import copy
        import signal
        import time
        import cv2
        import torch
        import torchvision.transforms.functional as tf
        from PIL import Image
        clip=sample['clip'];height,width=frame.shape[:2];assert (width,height)==(sample['width'],sample['height'])
        pixel_sha=hashlib.sha256(frame.tobytes()).hexdigest();tick=time.monotonic()
        record={'clip':clip,'frame_index':sample['frame_index'],'timestamp':sample['timestamp'],
            'video_sha256':sample['video_sha256'],'pixel_sha256':pixel_sha,'width':width,'height':height,
            'protocol_sha256':self.protocol_sha,'quality':{'passed':False}}
        path=self.out/'detections'/(clip+'.json')
        if path.exists():
            raw=json.loads(path.read_text());assert raw['pixel_sha256']==pixel_sha and raw['protocol_sha256']==self.protocol_sha
            kp={int(k):v for k,v in raw['keypoints'].items()};lines={int(k):v for k,v in raw['lines'].items()}
            record['network_reused']=True
        else:
            tensor=tf.to_tensor(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))).unsqueeze(0)
            tensor=tf.resize(tensor,[540,960],antialias=True).cuda();torch.cuda.synchronize();inference_tick=time.monotonic()
            with torch.inference_mode():
                heatmaps=self.models[0](tensor);heatmaps_l=self.models[1](tensor);assert heatmaps.is_cuda and heatmaps_l.is_cuda
                kp=self.coords(self.get_kp(heatmaps[:,:-1]),threshold=.3434)[0]
                lines=self.coords(self.get_lines(heatmaps_l[:,:-1]),threshold=.7867)[0]
            torch.cuda.synchronize();record.update(network_seconds=time.monotonic()-inference_tick,network_reused=False,cuda_outputs=True)
            clean_dump(path,{'pixel_sha256':pixel_sha,'protocol_sha256':self.protocol_sha,'keypoints':kp,'lines':lines,'network_seconds':record['network_seconds']})
        raw_kp={k:{**v,'x':v['x']*width/960,'y':v['y']*height/540} for k,v in kp.items()}
        raw_lines={k:{**v,'x_1':v['x_1']*width/960,'y_1':v['y_1']*height/540,'x_2':v['x_2']*width/960,'y_2':v['y_2']*height/540} for k,v in lines.items()}
        def timeout(*args):raise TimeoutError('Calibration exceeded45seconds; saved raw detections retained')
        previous=signal.signal(signal.SIGALRM,timeout)
        try:
            signal.alarm(45)
            complete,completed_lines=self.complete(copy.deepcopy(kp),copy.deepcopy(lines),w=960,h=540,normalize=True)
            solver=self.camera(iwidth=width,iheight=height,denormalize=True);solver.update(complete,completed_lines)
            params=solver.heuristic_voting(refine_lines=True);record['parameters']=params
            if params is not None:
                p=field.projection(params);record['projection']=p.tolist()
                record['quality']=field.quality(p,width,height,raw_kp,raw_lines,self.world,self.line_world)
        except Exception as e:record['error']=type(e).__name__+': '+str(e)[:400]
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        record['seconds']=time.monotonic()-tick;clean_dump(self.out/'records'/(clip+'.json'),record)
        return record


class CalibratedPose:
    def __init__(self,weights,images,out):
        self.weights=Path(weights);self.images=Path(images);self.out=Path(out);self.net=None;self.canonical=None
        assert data.sha(self.weights)==pilot.MODEL_SHA

    def infer(self,sample):
        import time
        import torch
        from PIL import Image
        desc,key=representation.input_identity(sample);assert desc==sample['identity'] and key==sample['input_key']
        if key is None:return np.zeros(123,np.float32),False,False
        path=self.out/'records'/(key+'.json');destination=self.out/'poses'/(key+'.npz');new=False
        if path.exists():
            record=json.loads(path.read_text());assert record['identity']==desc and record['pose_sha256']==data.sha(destination)
        else:
            if self.net is None:
                self.net=torch.jit.load(str(self.weights),map_location='cuda').eval()
                with torch.inference_mode(),torch.device('cuda'):
                    self.canonical=self.net.get_weights_for_canonical_points(self.net.cano_all['smpl'][-24:].cuda().float())
            image_path=self.images/(desc['image_sha256']+'.jpg');assert data.sha(image_path)==desc['image_sha256']
            pixels=np.asarray(Image.open(image_path).convert('RGB')).copy();assert pixels.shape==(224,224,3)
            intrinsic=np.asarray(desc['intrinsic'],np.float32);up=np.asarray(desc['camera_up'],np.float32);box=np.asarray(desc['box'],np.float32)
            image=torch.from_numpy(pixels).permute(2,0,1)[None].cuda();torch.cuda.synchronize();tick=time.monotonic()
            with torch.inference_mode(),torch.device('cuda'):
                pred=self.net.estimate_poses_batched(image,[torch.from_numpy(box)[None].cuda()],self.canonical,
                    intrinsic_matrix=torch.from_numpy(intrinsic)[None].cuda(),world_up_vector=torch.from_numpy(up).cuda(),
                    extrinsic_matrix=torch.eye(4,device='cuda')[None],distortion_coeffs=torch.zeros((1,5),device='cuda'),
                    num_aug=desc['num_aug'],internal_batch_size=desc['internal_batch_size'])
            torch.cuda.synchronize();seconds=time.monotonic()-tick
            tensors=[pred[k][0][0] for k in ('poses3d','poses2d','uncertainties')];assert all(t.is_cuda for t in tensors)
            p,uv,u=[t.cpu().numpy() for t in tensors];assert p.shape==(24,3) and uv.shape==(24,2) and u.shape==(24,)
            expected,positive=camera.project_camera(p,intrinsic);finite=all(np.isfinite(v).all() for v in (p,uv,u))
            error=float(np.linalg.norm(expected-uv,axis=1).max()) if finite and positive.all() else None
            inference_valid=bool(error is not None and error<=.05)
            agreement=pilot.projection_error(uv,sample['points'],sample['scores'],box[3]) if inference_valid else None
            if agreement is not None and not np.isfinite(agreement):agreement=None
            atomic_npz(destination,points3d=p,points2d=uv,uncertainty=u,intrinsic=intrinsic,camera_up=up,image_sha256=desc['image_sha256'])
            record={'identity':desc,'input_key':key,'pose_sha256':data.sha(destination),'cuda_verified':True,
                'inference_valid':inference_valid,'agreement':agreement,'self_reprojection_max_pixels':error,'seconds':seconds}
            clean_dump(path,record);new=True
        with np.load(destination) as a:
            features,valid=representation.representations(a['points3d'],a['uncertainty'],record['agreement'],sample['camera']['rotation'],sample['camera']['goal_side'])
        return features['calibrated_field'][0],valid,new


def draw_example(sample,frame,calibration,roles,valid,out):
    import cv2
    canvas=frame.copy();h,w=canvas.shape[:2]
    if calibration['quality']['passed']:
        corners,_=field.project(np.asarray(calibration['projection']),field.goal_corners(calibration['quality']['unique_goal_side']))
        cv2.polylines(canvas,[np.round(corners).astype(np.int32)],True,(80,230,80),3)
    for stage,color in (('fold0',(0,210,255)),('fold2',(255,160,70))):
        box=roles[stage]['box']
        if box is not None and roles[stage]['anchor']==2:
            a,b,c,d=np.round(box).astype(int);cv2.rectangle(canvas,(a,b),(c,d),color,2)
            cv2.putText(canvas,stage,(a,max(14,b-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    scale=1000/w;canvas=cv2.resize(canvas,(1000,round(h*scale)))
    heading=np.zeros((48,1000,3),np.uint8)
    title=f"{sample['clip']} | field quality={calibration['quality']['passed']} | NLF usable f0={valid['fold0']} f2={valid['fold2']}"
    cv2.putText(heading,title,(10,29),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
    path=Path(out)/(sample['clip']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(path),np.vstack([heading,canvas]),[cv2.IMWRITE_JPEG_QUALITY,90])


def execute(frozen,field_reference,upstream,nlf_weights,videos,out,mode='pilot',budget=1800):
    import gc
    import time
    import cv2
    import torch
    import catboost
    import pandas as pd
    import pose_tree_controls as controls
    from catboost import CatBoostClassifier
    assert mode in ('pilot','full') and torch.cuda.is_available() and catboost.__version__=='1.2.10'
    frozen,field_reference,upstream,nlf_weights,out=map(Path,(frozen,field_reference,upstream,nlf_weights,out))
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(6643);cv2.setRNGSeed(6643)
    started=time.monotonic();spec=json.loads((frozen/'frozen-manifest.json').read_text())
    for name,h in spec['files'].items():assert data.sha(frozen/name)==h
    assert spec['crop_helper_sha256']==data.sha(data.__file__)
    old_checks=json.loads((field_reference/'artifact_manifest.json').read_text())
    for name,h in old_checks.items():assert data.sha(field_reference/name)==h
    assert old_checks['field_calibration_pilot.py']==data.sha(field.__file__)
    old=json.loads((field_reference/'protocol.json').read_text())
    assert old['kp_threshold']==.3434 and old['line_threshold']==.7867 and old['alpha']==.7 and old['refine_lines'] is True
    assert data.sha(nlf_weights)==pilot.MODEL_SHA
    samples={s['clip']:s for s in spec['samples']};assert len(samples)==900 and set(samples)==set(videos)
    order=ordered_clips(samples);assert order[:12]==spec['pilot_clips']
    assert all(c.endswith('_private.mp4') for c in order)
    protocol={'source_sha256':data.sha(__file__),'frozen_manifest_sha256':data.sha(frozen/'frozen-manifest.json'),
        'field_helper_sha256':data.sha(field.__file__),'camera_helper_sha256':data.sha(camera.__file__),
        'representation_helper_sha256':data.sha(representation.__file__),'field_reference_protocol_sha256':data.sha(field_reference/'protocol.json'),
        'field_weights':old['weights'],'field_upstream_commit':old['upstream_commit'],
        'upstream_files':{str(p.relative_to(upstream)):data.sha(p) for p in upstream.rglob('*') if p.is_file() and p.suffix!='.pyc'},
        'field_recipe':{k:old[k] for k in ('input_size','kp_threshold','line_threshold','refine_lines','alpha','quality_gate')},
        'nlf_model_sha256':pilot.MODEL_SHA,'models':spec['new_models'],'pilot_clips':spec['pilot_clips'],
        'fallback':'Exact R25 per-fold/seed kicker probabilities; keeper/outcome unchanged.',
        'ensemble':'Equal probabilities across folds0/2 andseeds6643/6644/6645; no fitting or private tuning.'}
    data.immutable_json(out/'protocol.json',protocol);protocol_sha=data.sha(out/'protocol.json')
    clean_dump(out/'input_manifest.json',spec)
    field_net=None;nlf_net=CalibratedPose(nlf_weights,frozen/'images',out/'calibrated');records=[];new_fields=new_poses=0
    limit=12 if mode=='pilot' else 900
    for clip in order[:limit]:
        sample=samples[clip];destination=out/'clips'/(clip+'.json')
        if destination.exists():
            record=json.loads(destination.read_text());assert record['protocol_sha256']==protocol_sha and record['video_sha256']==sample['video_sha256']
            for name,h in record['files'].items():assert data.sha(out/name)==h
            records.append(record)
        else:
            if time.monotonic()-started>budget-180:break
            tick=time.monotonic();assert data.sha(videos[clip])==sample['video_sha256']
            field_path=out/'field/records'/(clip+'.json');frame=None
            if not field_path.exists() or clip in spec['pilot_clips']:
                cap=cv2.VideoCapture(str(videos[clip]))
                for index in range(sample['frame_index']+1):
                    ok,frame=cap.read();assert ok,(clip,index)
                cap.release();assert frame.shape[:2]==(sample['height'],sample['width'])
            if field_path.exists():
                cal=json.loads(field_path.read_text());assert cal['frame_index']==sample['frame_index'] and cal['video_sha256']==sample['video_sha256'] and cal['protocol_sha256']==protocol_sha
            else:
                if field_net is None:field_net=FieldCalibrator(upstream,field_reference/'weights',old['weights'],out/'field',protocol_sha)
                cal=field_net.infer(sample,frame);new_fields+=1
            role_samples={};valid={};features={};files={str(field_path.relative_to(out)):data.sha(field_path)}
            raw=out/'field/detections'/(clip+'.json');files[str(raw.relative_to(out))]=data.sha(raw)
            for stage,role in sample['roles'].items():
                entry=camera_sample(sample,role,cal,data.sha(field_path));extra,usable,new=nlf_net.infer(entry);new_poses+=new
                role_samples[stage]=entry;valid[stage]=bool(usable);features[stage]=extra;features[stage+'_valid']=np.array(usable)
                if entry['input_key']:
                    for name in ('records/'+entry['input_key']+'.json','poses/'+entry['input_key']+'.npz'):
                        p=out/'calibrated'/name;files[str(p.relative_to(out))]=data.sha(p)
            path=out/'features'/(clip+'.npz');atomic_npz(path,**features);files[str(path.relative_to(out))]=data.sha(path)
            if clip in spec['pilot_clips']:
                draw_example(sample,frame,cal,sample['roles'],valid,out/'examples')
                path=out/'examples'/(clip+'.jpg');files[str(path.relative_to(out))]=data.sha(path)
            record={'clip':clip,'index':sample['index'],'protocol_sha256':protocol_sha,'video_sha256':sample['video_sha256'],
                'files':files,'samples':role_samples,'valid':valid,'field_quality_passed':cal['quality']['passed'],'seconds':time.monotonic()-tick}
            clean_dump(destination,record);records.append(record)
        if len(records)==12:
            keys={s['input_key'] for r in records for s in r['samples'].values() if s['input_key']}
            inference=[json.loads((out/'calibrated/records'/(key+'.json')).read_text()) for key in sorted(keys)]
            estimate=float(np.median([r['seconds'] for r in records[2:]])*900)
            quality=sum(r['field_quality_passed'] for r in records)
            projection_ok=bool(inference) and all(r['inference_valid'] and r['cuda_verified'] for r in inference)
            report={'clips':[r['clip'] for r in records],'field_quality_passed':quality,'steady_estimate_seconds':estimate,
                'unique_calibrated_inputs':len(keys),'all_camera_selfprojections_passed':projection_ok,
                'usable_by_fold':{stage:sum(r['valid'][stage] for r in records) for stage in ('fold0','fold2')},
                'passed':bool(quality>=6 and estimate<=2400 and projection_ok)}
            data.immutable_json(out/'cost_pilot.json',report)
            if not report['passed']:break
        if len(records)%25==0 or len(records)==12:
            progress={'stage':'private_calibrated_features','completed':len(records),'target':limit,'new_fields':new_fields,'new_calibrated_poses':new_poses,'seconds':time.monotonic()-started}
            clean_dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    peak=torch.cuda.max_memory_allocated()/1e9;del field_net,nlf_net;gc.collect();torch.cuda.empty_cache()
    cost=json.loads((out/'cost_pilot.json').read_text()) if (out/'cost_pilot.json').exists() else None
    complete=len(records)==900 and cost and cost['passed'];prediction_ready=False
    if complete:
        with np.load(frozen/'private_features.npz') as a:base={k:a[k] for k in a.files}
        template=pd.read_csv(frozen/'submission_template.csv');np.testing.assert_array_equal(base['clips'],template['clip'])
        priors=np.load(frozen/'models/priors.npy',allow_pickle=False);all_probabilities=[];raw_kicks={}
        for f,stage in enumerate(('fold0','fold2')):
            extra=[];valid=[]
            for clip in base['clips']:
                with np.load(out/'features'/(str(clip)+'.npz')) as a:extra.append(a[stage]);valid.append(bool(a[stage+'_valid']))
            extra=np.asarray(extra,np.float32);valid=np.asarray(valid,bool);full=np.c_[base[stage+'_base'],extra].astype(np.float32)
            assert full.shape==(900,435) and np.isfinite(full).all()
            margin=priors[f][base[stage+'_base'][:,186:189].argmax(1),:3];group=[];saved={}
            for seed in (6643,6644,6645):
                path=frozen/'models'/stage/(str(seed)+'.cbm');assert data.sha(path)==spec['new_models'][str(path.relative_to(frozen))]
                net=CatBoostClassifier();net.load_model(str(path));assert net.tree_count_==450 and net.get_all_params()['task_type']=='GPU'
                raw=net.predict(full,prediction_type='RawFormulaVal',task_type='CPU',thread_count=4);p=controls.softmax(raw+margin)
                original=[base[f'{stage}_seed{seed}_target{k}'] for k in range(3)];ps=replace_kick(p,original,valid)
                raw_kicks[stage+'_'+str(seed)]=p;group.append(ps)
                for k,q in enumerate(ps):saved[f'seed{seed}_target{k}']=q
            atomic_npz(out/(stage+'_probabilities.npz'),**saved)
            atomic_npz(out/(stage+'_inputs.npz'),features=full,valid=valid,clips=base['clips'],margin=margin)
            all_probabilities.append(group)
        probabilities=[np.mean([ps[k] for group in all_probabilities for ps in group],0) for k in range(3)]
        columns={name:[classes[int(i)] for i in p.argmax(1)] for (name,classes),p in zip(data.TARGETS.items(),probabilities)}
        submission=pd.DataFrame({'clip':template['clip'],**columns});assert submission.columns.tolist()==template.columns.tolist() and not submission.isna().any().any()
        submission.to_csv(out/'submission.csv',index=False)
        atomic_npz(out/'probabilities.npz',**{name:p for name,p in zip(data.TARGETS,probabilities)})
        atomic_npz(out/'raw_kicks.npz',**raw_kicks);prediction_ready=True
    state='complete' if complete else 'pilot_complete' if mode=='pilot' and len(records)==12 else 'checkpoint'
    if cost and not cost['passed']:state='pilot_gate_failed'
    summary={'state':state,'mode':mode,'completed':len(records),'total':900,'pilot':cost,'new_fields':new_fields,'new_calibrated_poses':new_poses,
        'field_quality_passed':sum(r['field_quality_passed'] for r in records),'usable_by_fold':{s:sum(r['valid'][s] for r in records) for s in ('fold0','fold2')},
        'seconds':time.monotonic()-started,'peak_cuda_gb':peak,'new_detector_crop_rtmw_oldfov_calls':0,'new_model_fits':0,
        'gpu':torch.cuda.get_device_name(0),'submission_ready':prediction_ready,'competition_score':None}
    if prediction_ready:summary['submission_sha256']=data.sha(out/'submission.csv')
    clean_dump(out/'summary.json',summary);return summary
