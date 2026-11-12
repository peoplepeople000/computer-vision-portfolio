"""R26 calibrated view and support-foot proxies; no cropped-NLF rotation."""
import numpy as np
import field_calibration_pilot as field

DIMENSION = 18
MIRROR_SIGN_COLUMNS = (0, 5, 8)


def crop_to_image(points, box, width, height, size=224, margin=.20):
    """Invert the exact crop/rounded letterbox with OpenCV pixel-center scaling."""
    x1,y1,x2,y2=np.asarray(box,float)
    if not np.isfinite([x1,y1,x2,y2]).all() or x2-x1<2 or y2-y1<2:
        return np.full_like(np.asarray(points,float),np.nan)
    dx,dy=(x2-x1)*margin,(y2-y1)*margin
    x1,x2=np.clip([np.floor(x1-dx),np.ceil(x2+dx)],0,width).astype(int)
    y1,y2=np.clip([np.floor(y1-dy),np.ceil(y2+dy)],0,height).astype(int)
    w,h=x2-x1,y2-y1
    if min(w,h)<=0:return np.full_like(np.asarray(points,float),np.nan)
    scale=size/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    offset=np.array([(size-nw)//2,(size-nh)//2])
    return (np.asarray(points,float)-offset+.5)*[w/nw,h/nh]-.5+[x1,y1]


def representation(record, points, scores, box, preferred_foot, anchor):
    """One last-frame 18D context. Feet are ground proxies, not measured contacts."""
    result=np.zeros(DIMENSION,np.float32)
    if anchor!=2 or not record['quality']['passed'] or preferred_foot not in ('left','right'):
        return result,False
    side=record['quality']['unique_goal_side'];p=np.asarray(record['projection'],float)
    c=record['parameters']['cam_params'];w,h=record['width'],record['height']
    center=np.asarray(c['position_meters'],float)
    view=field.goal_relative(center[:2],side)
    result[:5]=[view[0]/100,view[1]/100,-center[2]/100,c['x_focal_length']/w,c['y_focal_length']/h]
    # preferred_foot denotes the kicking foot; the opposite side is a support-foot proxy.
    indices=[17,18,19] if preferred_foot=='right' else [20,21,22]
    points,scores=np.asarray(points,float),np.asarray(scores,float)
    selected=points[indices];confidence=scores[indices]
    valid=np.isfinite(selected).all() and np.isfinite(confidence).all() and (confidence>.3).all()
    valid=valid and (selected>=0).all() and (selected<224).all()
    if valid:
        image=crop_to_image(selected,box,w,h)
        valid=np.isfinite(image).all() and (image>=0).all() and (image<[w,h]).all()
        if valid:
            ground,visible=field.ground_points(p,image)
            relative=field.goal_relative(ground,side)
            valid=visible.all() and np.isfinite(relative).all()
            valid=valid and (abs(relative[:,0])<=20).all() and (relative[:,1]>=-1).all() and (relative[:,1]<=30).all()
            if valid:
                toe=relative[:2].mean(0);heel=relative[2];middle=(toe+heel)/2
                result[5:8]=[middle[0]/11,middle[1]/11,1]
                delta=toe-heel;length=np.linalg.norm(delta)
                # A broad geometric plausibility gate; it cannot establish foot contact.
                if .05<=length<=1.:
                    result[8:13]=[delta[0]/length,-delta[1]/length,length,1,float(confidence.min())]
    q=record['quality']
    result[13:]=[q['keypoint_median_diag'],q['line_median_diag'],q['raw_keypoints_used']/57,q['raw_lines_used']/23,1]
    assert np.isfinite(result).all()
    return np.clip(result,-5,5),True


def reflected(features):
    result=np.asarray(features).copy();assert result.shape[-1]==DIMENSION
    result[...,list(MIRROR_SIGN_COLUMNS)]*=-1
    return result


def extract(samples_file, pilot, upstream, videos, out, budget=1800):
    """Add only missing last-anchor calibrations; reuse all12 verified pilot records."""
    import copy
    import hashlib
    import json
    import shutil
    import signal
    import sys
    import time
    from pathlib import Path
    import cv2
    import torch
    import yaml
    import torchvision.transforms.functional as tf
    from PIL import Image
    samples_file,pilot,upstream,out=map(Path,(samples_file,pilot,upstream,out))
    sys.path.insert(0,str(upstream))
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_line_net
    from utils.utils_calib import FramebyFrameCalib,keypoint_world_coords_2D,line_world_coords_3D
    from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool as get_kp
    from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool_l as get_lines
    from utils.utils_heatmap import coords_to_dict,complete_keypoints
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        return h.hexdigest()
    def dump(path,value):
        def clean(x):
            if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
            if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
            if isinstance(x,np.generic):return clean(x.item())
            if isinstance(x,float) and not np.isfinite(x):return None
            return x
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(6643);cv2.setRNGSeed(6643)
    samples=json.loads(samples_file.read_text());assert len(samples)==1299 and len({s['clip'] for s in samples})==1299
    assert all(s['clip'].endswith('_public.mp4') and s['clip']!='0526_public.mp4' for s in samples)
    old_checks=json.loads((pilot/'artifact_manifest.json').read_text())
    for name,h in old_checks.items():assert sha(pilot/name)==h
    assert old_checks['field_calibration_pilot.py']==sha(field.__file__)
    old=json.loads((pilot/'protocol.json').read_text())
    assert old['refine_lines'] is True and old['alpha']==.7 and old['kp_threshold']==.3434 and old['line_threshold']==.7867
    out.mkdir(parents=True,exist_ok=True)
    protocol={'source_sha256':sha(__file__),'pilot_helper_sha256':sha(field.__file__),
        'samples_sha256':sha(samples_file),'pilot_manifest_sha256':sha(pilot/'artifact_manifest.json'),
        'pilot_protocol_sha256':sha(pilot/'protocol.json'),'weights':old['weights'],'upstream_commit':old['upstream_commit'],
        'inference_recipe':{k:old[k] for k in ('input_size','kp_threshold','line_threshold','refine_lines','alpha','quality_gate')},
        'scope':'New field calibration only, exactly last cached sparse frame per1299 public clips; immutable reuse of12 pilot records.',
        'gpu':torch.cuda.get_device_name(0),'budget_seconds':budget,'calibration_timeout_seconds':45}
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:dump(path,protocol)
    shutil.copy2(pilot/'protocol.json',out/'pilot_protocol.json')
    dump(out/'input_samples.json',samples)
    pilot_samples={s['clip']:s for s in json.loads((pilot/'pilot_input.json').read_text())['samples']}
    expected={s['clip']:s for s in samples}
    for clip,s in pilot_samples.items():
        e=expected[clip]
        assert s['video_sha256']==e['video_sha256'] and s['frame_index']==e['frame_index']
        assert sha(videos[clip])==e['video_sha256']
        for name in ['records/'+clip+'.json','detections/'+clip+'.json']:
            target=out/name;target.parent.mkdir(exist_ok=True)
            if target.exists():assert sha(target)==old_checks[name]
            else:shutil.copy2(pilot/name,target)
    model_pair=None
    def models():
        nonlocal model_pair
        if model_pair is None:
            model_pair=[]
            for maker,cfg,name in [(get_cls_net,'hrnetv2_w48.yaml','SV_kp'),(get_line_net,'hrnetv2_w48_l.yaml','SV_lines')]:
                assert sha(pilot/'weights'/name)==old['weights'][name]
                net=maker(yaml.safe_load((upstream/'config'/cfg).read_text()))
                net.load_state_dict(torch.load(pilot/'weights'/name,map_location='cpu',weights_only=True))
                net=net.to('cuda').eval();assert next(net.parameters()).is_cuda;model_pair.append(net)
        return model_pair
    world={i+1:[*xy,-2.44 if i+1 in (12,15,16,19) else 0.] for i,xy in enumerate(keypoint_world_coords_2D)}
    line_world={i+1:v for i,v in enumerate(line_world_coords_3D)}
    def timeout(*args):raise TimeoutError('Calibration exceeded45seconds; detection retained')
    signal.signal(signal.SIGALRM,timeout);started=time.monotonic();new=0
    for sample in samples:
        clip=sample['clip'];destination=out/'records'/(clip+'.json')
        if destination.exists():
            r=json.loads(destination.read_text());assert r['frame_index']==sample['frame_index']
            assert r['protocol_sha256'] in (sha(path),protocol['pilot_protocol_sha256'])
            continue
        if time.monotonic()-started>budget-120:break
        assert sha(videos[clip])==sample['video_sha256']
        cap=cv2.VideoCapture(str(videos[clip]))
        for index in range(sample['frame_index']+1):
            ok,frame=cap.read();assert ok,(clip,index)
        cap.release();height,width=frame.shape[:2]
        assert (width,height)==(sample['width'],sample['height'])
        pixel_sha=hashlib.sha256(frame.tobytes()).hexdigest()
        r={'clip':clip,'frame_index':sample['frame_index'],'timestamp':sample['timestamp'],
           'video_sha256':sample['video_sha256'],'pixel_sha256':pixel_sha,'width':width,'height':height,
           'protocol_sha256':sha(path),'quality':{'passed':False}}
        tick=time.monotonic();raw_path=out/'detections'/(clip+'.json')
        if raw_path.exists():
            raw=json.loads(raw_path.read_text());assert raw['pixel_sha256']==pixel_sha and raw['protocol_sha256']==sha(path)
            kp={int(k):v for k,v in raw['keypoints'].items()};lines={int(k):v for k,v in raw['lines'].items()}
            r['network_reused']=True
        else:
            model,model_l=models()
            tensor=tf.to_tensor(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))).unsqueeze(0)
            tensor=tf.resize(tensor,[540,960],antialias=True).to('cuda');torch.cuda.synchronize();inference_tick=time.monotonic()
            with torch.inference_mode():
                heatmaps=model(tensor);heatmaps_l=model_l(tensor);assert heatmaps.is_cuda and heatmaps_l.is_cuda
                kp=coords_to_dict(get_kp(heatmaps[:,:-1]),threshold=.3434)[0]
                lines=coords_to_dict(get_lines(heatmaps_l[:,:-1]),threshold=.7867)[0]
            torch.cuda.synchronize();r['network_seconds']=time.monotonic()-inference_tick
            r['network_reused']=False;r['cuda_outputs']=True
            raw={'pixel_sha256':pixel_sha,'protocol_sha256':sha(path),'keypoints':kp,'lines':lines,'network_seconds':r['network_seconds']}
            dump(raw_path,raw)
        raw_kp={k:{**v,'x':v['x']*width/960,'y':v['y']*height/540} for k,v in kp.items()}
        raw_lines={k:{**v,'x_1':v['x_1']*width/960,'y_1':v['y_1']*height/540,
                      'x_2':v['x_2']*width/960,'y_2':v['y_2']*height/540} for k,v in lines.items()}
        try:
            signal.alarm(45)
            complete,completed_lines=complete_keypoints(copy.deepcopy(kp),copy.deepcopy(lines),w=960,h=540,normalize=True)
            camera=FramebyFrameCalib(iwidth=width,iheight=height,denormalize=True);camera.update(complete,completed_lines)
            params=camera.heuristic_voting(refine_lines=True);r['parameters']=params
            if params is not None:
                p=field.projection(params);r['projection']=p.tolist();r['quality']=field.quality(p,width,height,raw_kp,raw_lines,world,line_world)
        except Exception as e:r['error']=type(e).__name__+': '+str(e)[:400]
        finally:signal.alarm(0)
        r['seconds']=time.monotonic()-tick;dump(destination,r);new+=1
        if new%50==0:print(json.dumps({'stage':'field_extract','completed':len(list((out/'records').glob('*.json'))),'new_this_run':new,'seconds':time.monotonic()-started}),flush=True)
    records=[json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
    summary={'state':'complete' if len(records)==1299 else 'checkpoint','completed':len(records),'total':1299,
             'new_this_run':new,'pilot_reused':12,'quality_passed':sum(r['quality']['passed'] for r in records),
             'seconds':time.monotonic()-started,'new_role_pose_nlf_inferences':0}
    dump(out/'summary.json',summary);return summary
