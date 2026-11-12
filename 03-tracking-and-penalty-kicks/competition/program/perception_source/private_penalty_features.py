"""One-time sparse private-clip inference with per-item, hash-verified resume."""
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import rgb_gru_data as data


def pilot_order(clips):
    return sorted(clips,key=lambda c:hashlib.sha256(('r25-private|'+c).encode()).hexdigest())


def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


class Models:
    def __init__(self, assets, nlf_weights, models, out):
        import torch
        import onnxruntime as ort
        import xgboost as xgb
        from rtmlib import YOLOX,RTMPose
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        import sparse_role_gpu as sparse
        import nlf_pose_pilot as nlf
        import xgb_probe
        assert torch.cuda.is_available();ort.preload_dlls()
        assert 'CUDAExecutionProvider' in ort.get_available_providers()
        self.out=Path(out);self.nlf_weights=Path(nlf_weights);self.nlf=None;self.field=None
        self.detect_impl=sparse.Models.detect
        checkpoints=Path(assets)/'models/rtmlib/hub/checkpoints'
        self.people=YOLOX(str(checkpoints/'yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),
                          model_input_size=(640,640),backend='onnxruntime',device='cuda')
        self.pose=RTMPose(str(checkpoints/'rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx'),
                          model_input_size=(192,256),backend='onnxruntime',device='cuda')
        for net,name,weight in [(self.people,'person','yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),
            (self.pose,'pose','rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx')]:
            options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(self.out/(name+'_cuda'))
            net.session=ort.InferenceSession(str(checkpoints/weight),sess_options=options,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            assert net.session.get_providers()[0]=='CUDAExecutionProvider'
        path=Path(assets)/'models/grounding-dino-tiny'
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        self.grounder=AutoModelForZeroShotObjectDetection.from_pretrained(path,local_files_only=True).eval().cuda()
        self.role_models={f:xgb.Booster(model_file=Path(models)/f'roles/fold{f}.ubj') for f in (0,2)}
        for net in self.role_models.values():net.set_param({'device':'cuda'});xgb_probe.gpu_config(net)
        assert data.sha(self.nlf_weights)==nlf.MODEL_SHA
        self.profiled=set()
        data.dump(self.out/'runtime.json',{'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
            'onnxruntime':ort.__version__,'grounder_device':str(next(self.grounder.parameters()).device),
            'uses_tracking':False,'rgb_encoder':False})

    def verify_cuda(self,net,name):
        if name in self.profiled:return
        events=json.loads(Path(net.session.end_profiling()).read_text())
        nodes=sum(e.get('args',{}).get('provider')=='CUDAExecutionProvider' for e in events)
        assert nodes>0;self.profiled.add(name)
        data.dump(self.out/(name+'_cuda_verified.json'),{'cuda_nodes':nodes,'providers':net.session.get_providers()})

    def detect(self,frame):
        result,timing=self.detect_impl(self,frame);self.verify_cuda(self.people,'person')
        return result,timing

    def pose_for(self,path):
        import pose_probe
        h=data.sha(path);target=self.out/'poses'/(h+'.npz');record=target.with_suffix('.json')
        if record.exists():
            saved=json.loads(record.read_text());assert saved['input_sha256']==h and saved['weights_sha256']==pose_probe.WEIGHT_SHA
            assert data.sha(target)==saved['sha256']
        else:
            image=cv2.imread(str(path));assert image.shape==(224,224,3)
            p,s=self.pose(image);self.verify_cuda(self.pose,'pose')
            assert p.shape==(1,133,2) and s.shape==(1,133) and np.isfinite(p).all() and np.isfinite(s).all()
            atomic_npz(target,points=p[0,:23],scores=s[0,:23])
            data.dump(record,{'input_sha256':h,'weights_sha256':pose_probe.WEIGHT_SHA,'sha256':data.sha(target)})
        with np.load(target) as z:return h,(z['points'],z['scores'])

    def body_for(self,path,pose):
        import torch
        from PIL import Image
        import nlf_pose_pilot as nlf
        import nlf_pose_features as features
        h=data.sha(path);target=self.out/'nlf'/(h+'.npz');record=target.with_suffix('.json')
        if record.exists():
            r=json.loads(record.read_text());assert r['input_sha256']==h and r['model_sha256']==nlf.MODEL_SHA
            assert data.sha(target)==r['sha256']
        else:
            box=nlf.observed_box(*pose);agreement=None
            p=np.zeros((24,3),np.float32);projected=np.zeros((24,2),np.float32);u=np.ones(24,np.float32)
            if box is not None:
                if self.nlf is None:
                    self.nlf=torch.jit.load(str(self.nlf_weights),map_location='cuda').eval()
                    with torch.inference_mode(),torch.device('cuda'):
                        self.field=self.nlf.get_weights_for_canonical_points(self.nlf.cano_all['smpl'][-24:].cuda().float())
                pixels=np.asarray(Image.open(path).convert('RGB')).copy()
                image=torch.from_numpy(pixels).permute(2,0,1)[None].cuda()
                with torch.inference_mode(),torch.device('cuda'):
                    pred=self.nlf.estimate_poses_batched(image,[torch.from_numpy(box)[None].cuda()],self.field,
                        default_fov_degrees=55.,num_aug=1,internal_batch_size=4)
                assert pred['poses3d'][0].is_cuda
                p,projected,u=[pred[k][0][0].cpu().numpy() for k in ('poses3d','poses2d','uncertainties')]
                agreement=nlf.projection_error(projected,*pose,box[3])
                if agreement is not None and not np.isfinite(agreement):agreement=None
                data.dump(self.out/'nlf_cuda_verified.json',{'tensor_device':str(pred['poses3d'][0].device),'model_sha256':nlf.MODEL_SHA})
            atomic_npz(target,points3d=p,points2d=projected,uncertainty=u)
            r={'input_sha256':h,'model_sha256':nlf.MODEL_SHA,'agreement':agreement,'sha256':data.sha(target)};data.dump(record,r)
        with np.load(target) as z:return features.geometry(z['points3d'],z['uncertainty'],r['agreement'],'body')


def process(row,video,models,out,protocol_sha):
    clip=row['clip'];out=Path(out);record=out/'records'/(clip+'.json')
    provenance={'video_sha256':data.sha(video),'protocol_sha256':protocol_sha,'preferred_foot':str(row['preferred_foot'])}
    if record.exists():
        saved=json.loads(record.read_text());assert saved['input']==provenance
        for name,h in saved['files'].items():assert data.sha(out/name)==h
        return saved
    import xgboost as xgb
    import role_selector as roles
    import sparse_role_gpu as sparse
    import pose_rgb_fusion as aligned
    started=time.monotonic();frames=sparse.choose_anchors(sparse.sparse.baseline.read_tail(video))
    assert len(frames)==3
    detections=[];files={};cost={'person_seconds':0.,'grounding_seconds':0.}
    for t,(frame_index,timestamp,bgr) in enumerate(frames):
        path=out/'detections'/clip/(str(t)+'.json')
        item_input={'video_sha256':provenance['video_sha256'],'protocol_sha256':protocol_sha,'frame_index':frame_index,'timestamp':timestamp}
        if path.exists():
            d=json.loads(path.read_text());assert d['input']==item_input
        else:
            det,timing=models.detect(bgr);d={'input':item_input,'detection':det,'timing':timing};data.dump(path,d)
        detections.append(d['detection']);files[str(path.relative_to(out))]=data.sha(path)
        for k in cost:cost[k]+=d['timing'][k]
    role_records={f:{'clip':clip,'keys':[],'mask':[],'timestamps':[]} for f in (0,2)}
    image_hashes={};pose_cache={};paths={};selections={f:[] for f in (0,2)}
    for t,((_,timestamp,bgr),det) in enumerate(zip(frames,detections)):
        candidates=roles.frame_features(det)
        for fold in (0,2):
            pair=roles.choose_pair(models.role_models[fold].predict(xgb.DMatrix(candidates))) if len(candidates)>=2 else [-1,-1]
            keys=[];selections[fold].append(pair)
            for j in pair:
                key=None if j<0 else f'{clip}/{t}_{j}.jpg';keys.append(key)
                if key is None or key in image_hashes:continue
                path=out/'images'/key
                if not path.exists():
                    pixels=data.crop_view(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),det['persons'][j]['bbox'])
                    assert pixels.shape==(224,224,3)
                    ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded.tobytes())
                h,pose=models.pose_for(path);image_hashes[key]=h;pose_cache[h]=pose;paths[h]=path
                files[str(path.relative_to(out))]=h
                for p in [out/'poses'/(h+'.npz'),out/'poses'/(h+'.json')]:files[str(p.relative_to(out))]=data.sha(p)
            r=role_records[fold];r['keys'].append(keys);r['mask'].append([j>=0 for j in pair]);r['timestamps'].append(timestamp)
    features={};selected={}
    foot=np.eye(3,dtype=np.float32)[{'left':0,'right':1}.get(row['preferred_foot'],2)]
    for fold in (0,2):
        pose,hashes,anchors=aligned.latest_selection(role_records[fold],image_hashes,pose_cache)
        body,valid=np.zeros(123,np.float32),False
        if hashes[0]:
            h=hashes[0];body,valid=models.body_for(paths[h],pose_cache[h])
            for p in [out/'nlf'/(h+'.npz'),out/'nlf'/(h+'.json')]:files[str(p.relative_to(out))]=data.sha(p)
        features.update({f'pose{fold}':np.r_[pose,foot].astype(np.float32),f'body{fold}':body,f'valid{fold}':np.array(valid)})
        selected[str(fold)]={'pairs':selections[fold],'record':role_records[fold],'hashes':list(hashes),'anchors':list(map(int,anchors))}
    path=out/'features'/(clip+'.npz');atomic_npz(path,**features);files[str(path.relative_to(out))]=data.sha(path)
    saved={'clip':clip,'input':provenance,'files':files,'selections':selected,'image_hashes':image_hashes,
        'seconds':time.monotonic()-started,'detector_timing':cost,'body_valid':{str(f):bool(features[f'valid{f}']) for f in (0,2)}}
    data.dump(record,saved);return saved


def run(metadata,template,videos,assets,nlf_weights,prepared,out,budget=1800):
    import torch
    import penalty_inference as inference
    import sparse_role_gpu as sparse
    import pose_rgb_fusion as aligned
    import nlf_pose_features as nlf
    assert torch.cuda.is_available();torch.set_num_threads(4);cv2.setNumThreads(1)
    out,prepared=Path(out),Path(prepared);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    inference.verify_archive(prepared);assert json.loads((prepared/'summary.json').read_text())['ready_for_private_inference']
    modelroot=prepared/'models'
    for name,h in json.loads((modelroot/'model_manifest.json').read_text()).items():assert data.sha(modelroot/name)==h
    rows=pd.read_csv(metadata);format_=pd.read_csv(template)
    assert len(rows)==len(format_)==900 and rows['clip'].is_unique and set(rows['clip'])==set(format_['clip'])
    assert all(c.endswith('_private.mp4') for c in rows['clip']) and set(rows['clip'])==set(videos)
    assert not any(k in rows for k in data.TARGETS)
    used_assets={str(p.relative_to(assets)):data.sha(p) for p in Path(assets).rglob('*') if p.is_file()}
    protocol={'sources':{Path(p).name:data.sha(p) for p in (__file__,inference.__file__,sparse.__file__,aligned.__file__,nlf.__file__,data.__file__)},
        'asset_hashes':used_assets,'nlf_model_sha256':data.sha(nlf_weights),
        'models_manifest_sha256':data.sha(modelroot/'model_manifest.json'),'metadata_sha256':data.sha(metadata),'template_sha256':data.sha(template),
        'anchors':[-.35,-.16,0.],'perception':'Frozen public sparse YOLOX/DINO + JPEG95/224 RTMW + latest kicker NLF; no dense tracking.',
        'scope':'Private competition inference only. Same role models/priors/classifiers as replayed validation. No label or public-score tuning.',
        'cost_gate':'First 12 deterministic private clips; steady median seconds * 900 < 2400; otherwise save pilot and stop.'}
    data.immutable_json(out/'protocol.json',protocol);protocol_sha=data.sha(out/'protocol.json')
    models=None;ordered=pilot_order(rows['clip']);indexed=rows.set_index('clip',drop=False);records=[]
    for clip in ordered:
        if time.monotonic()-started>budget-120:break
        path=out/'records'/(clip+'.json')
        if not path.exists() and models is None:models=Models(assets,nlf_weights,modelroot,out)
        record=process(indexed.loc[clip],videos[clip],models,out,protocol_sha);records.append(record)
        if len(records)==12:
            estimate=float(np.median([r['seconds'] for r in records[2:]])*900)
            report={'pilot_clips':ordered[:12],'steady_estimate_seconds':estimate,'passed':estimate<2400}
            data.dump(out/'cost_pilot.json',report)
            if not report['passed']:break
        if len(records)%25==0 or len(records)==12:
            print(json.dumps({'stage':'private_features','completed':len(records),'total':900,'seconds':time.monotonic()-started}),flush=True)
            data.dump(out/'progress.json',{'completed':len(records),'total':900,'last_clip':clip})
    complete=len(records)==900
    if complete:
        inputs={f:{'pose':[],'body':[],'valid':[]} for f in (0,2)}
        for clip in format_['clip']:
            with np.load(out/'features'/(clip+'.npz')) as z:
                for fold in (0,2):
                    for k in ('pose','body','valid'):inputs[fold][k].append(z[k+str(fold)])
        predicted=[]
        for fold in (0,2):
            group=inference.predict(**inputs[fold],fold=fold,models=modelroot);predicted.append(group)
            atomic_npz(out/f'fold{fold}_probabilities.npz',**{f'seed{seed}_target{k}':p for seed,ps in zip(inference.SEEDS,group) for k,p in enumerate(ps)})
        probabilities=inference.ensemble(predicted);columns={}
        for k,(name,classes) in enumerate(data.TARGETS.items()):
            columns[name]=[classes[int(i)] for i in probabilities[k].argmax(1)]
        submission=pd.DataFrame({'clip':format_['clip'],**columns});assert submission.columns.tolist()==format_.columns.tolist()
        assert not submission.isna().any().any();submission.to_csv(out/'submission.csv',index=False)
        atomic_npz(out/'probabilities.npz',**{name:p for name,p in zip(data.TARGETS,probabilities)})
    report={'state':'complete' if complete else 'checkpointed','completed':len(records),'total':900,
        'seconds':time.monotonic()-started,'new_public_preprocessing':0,'submission_ready':complete,
        'competition_score':None,'next':'Verify saved features/predictions and submit once' if complete else 'Resume saved clip/hash cache; never restart completed clips.'}
    if complete:report['submission_sha256']=data.sha(out/'submission.csv')
    data.dump(out/'summary.json',report);return report
