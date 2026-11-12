"""Sparse PnLCalib pilot; field geometry only, never a cropped-NLF transform."""
import numpy as np


def projection(parameters):
    c = parameters['cam_params']
    k = np.array([[c['x_focal_length'], 0, c['principal_point'][0]],
                  [0, c['y_focal_length'], c['principal_point'][1]], [0, 0, 1.]])
    r = np.asarray(c['rotation_matrix'], float)
    center = np.asarray(c['position_meters'], float)
    assert np.isfinite(k).all() and np.isfinite(r).all() and np.isfinite(center).all()
    assert k[0, 0] > 0 and k[1, 1] > 0
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-3)
    assert abs(np.linalg.det(r)-1) < 1e-3
    return k @ r @ np.c_[np.eye(3), -center]


def project(p, world):
    xyz = np.asarray(world, float).reshape(-1, 3)
    q = np.c_[xyz, np.ones(len(xyz))] @ np.asarray(p).T
    valid = np.isfinite(q).all(1) & (q[:, 2] > 1e-7)
    image = np.full((len(xyz), 2), np.nan)
    image[valid] = q[valid, :2] / q[valid, 2:]
    return image, valid


def ground_points(p, image):
    """Ray/ground-plane intersections. Only use on plausible ground proxies."""
    xy = np.asarray(image, float).reshape(-1, 2)
    h = np.asarray(p, float)[:, [0, 1, 3]]
    out = np.full((len(xy), 2), np.nan)
    if not np.isfinite(h).all() or np.linalg.cond(h) > 1e12:
        return out, np.zeros(len(xy), bool)
    q = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(h).T
    valid = np.isfinite(q).all(1) & (abs(q[:, 2]) > 1e-9)
    out[valid] = q[valid, :2] / q[valid, 2:]
    _, visible = project(p, np.c_[out, np.zeros(len(out))])
    valid &= visible
    out[~valid] = np.nan
    return out, valid


def goal_relative(xy, side):
    """Positive left from kicker view; positive depth into pitch from goal line."""
    assert side in (-1, 1)
    xy = np.asarray(xy, float)
    return np.stack([-side*xy[..., 1], 52.5-side*xy[..., 0]], -1)


def goal_corners(side):
    assert side in (-1, 1)
    return np.array([[side*52.5, -3.66, 0], [side*52.5, 3.66, 0],
                     [side*52.5, 3.66, -2.44], [side*52.5, -3.66, -2.44]])


def line_distances(p, world_endpoints, observed_endpoints):
    predicted, visible = project(p, world_endpoints)
    if not visible.all():
        return None
    line = np.cross(np.r_[predicted[0], 1], np.r_[predicted[1], 1])
    scale = np.linalg.norm(line[:2])
    if scale < 1e-8:
        return None
    observed = np.asarray(observed_endpoints, float)
    return abs(np.c_[observed, np.ones(len(observed))] @ line) / scale


def quality(p, width, height, raw_keypoints, raw_lines, keypoint_world, line_world):
    """Same-detector consistency proxies; not independent calibration accuracy."""
    diag = np.hypot(width, height)
    errors = []
    for key, point in raw_keypoints.items():
        key = int(key)
        if key not in keypoint_world:
            continue
        predicted, valid = project(p, [keypoint_world[key]])
        if valid[0]:
            errors.append(float(np.linalg.norm(predicted[0]-[point['x'], point['y']])/diag))
    line_errors = []
    for key, observed in raw_lines.items():
        key = int(key)
        if key not in line_world:
            continue
        distances = line_distances(p, line_world[key],
            [[observed['x_1'], observed['y_1']], [observed['x_2'], observed['y_2']]])
        if distances is not None:
            line_errors.append(float(np.mean(distances)/diag))
    goals = []
    for side, ids in [(-1, {7, 8, 9}), (1, {10, 11, 12})]:
        corners, valid = project(p, goal_corners(side))
        within = (corners >= [-.1*width, -.1*height]).all() and (corners <= [1.1*width, 1.1*height]).all()
        support = len(ids & {int(k) for k in raw_lines})
        plausible = bool(valid.all() and within and support >= 2
                         and np.linalg.norm(corners[0]-corners[1])/diag >= .025)
        goals.append({'side':side, 'corners':corners.tolist(), 'line_support':support, 'plausible':plausible})
    median = float(np.median(errors)) if errors else None
    p90 = float(np.quantile(errors, .9)) if errors else None
    line_median = float(np.median(line_errors)) if line_errors else None
    unique = [g for g in goals if g['plausible']]
    passed = bool(len(errors) >= 6 and median <= .01 and p90 <= .025
                  and len(line_errors) >= 2 and line_median <= .01 and len(unique) == 1)
    return {'passed':passed, 'raw_keypoints_used':len(errors), 'keypoint_median_diag':median,
            'keypoint_p90_diag':p90, 'raw_lines_used':len(line_errors), 'line_median_diag':line_median,
            'goals':goals, 'unique_goal_side':unique[0]['side'] if len(unique)==1 else None}


def execute(inputs, upstream, weights, out):
    """Twelve public training sources, one sparse frame each. No classifier fit."""
    import copy
    import hashlib
    import json
    import signal
    import sys
    import time
    from pathlib import Path
    import cv2
    import torch
    import yaml
    import torchvision.transforms.functional as tf
    from PIL import Image

    inputs, upstream, weights, out = map(Path, (inputs, upstream, weights, out))
    sys.path.insert(0, str(upstream))
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_line_net
    from utils.utils_calib import FramebyFrameCalib, keypoint_world_coords_2D, line_world_coords_3D
    from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool as get_kp
    from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool_l as get_lines
    from utils.utils_heatmap import coords_to_dict, complete_keypoints

    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def dump(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        def clean(x):
            if isinstance(x, dict):return {str(k):clean(v) for k,v in x.items()}
            if isinstance(x, (list,tuple,np.ndarray)):return [clean(v) for v in x]
            if isinstance(x, np.generic):return clean(x.item())
            if isinstance(x, float) and not np.isfinite(x):return None
            return x
        path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    manifest = json.loads((inputs/'pilot.json').read_text())
    for name,h in manifest['files'].items():assert sha(inputs/name)==h
    assert len(manifest['samples'])==12 and len({s['source_id'] for s in manifest['samples']})==12
    assert all(s['fold'] in (2,3,4) for s in manifest['samples'])
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.manual_seed(6643);cv2.setRNGSeed(6643)
    out.mkdir(parents=True,exist_ok=True)
    protocol={'source_sha256':sha(__file__), 'input_sha256':sha(inputs/'pilot.json'),
        'upstream_commit':manifest['upstream_commit'], 'weights':{n:sha(weights/n) for n in ['SV_kp','SV_lines']},
        'gpu':torch.cuda.get_device_name(0), 'input_size':[540,960],
        'kp_threshold':.3434,'line_threshold':.7867,'refine_lines':True,'alpha':.7,
        'quality_gate':'At least6 raw keypoints; median <=.01 and p90 <=.025 diagonal; at least2 raw lines, median <=.01 diagonal; exactly1 supported visible goal.',
        'expansion_gate':'At least6/12 pass quality and median total frame time excluding first2 <=10s; visual review still required.',
        'limits':'Same-model consistency, not geometric ground truth. No cropped NLF transform, pose rerun or classifier tuning.',
        'per_frame_calibration_timeout_seconds':45}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:dump(protocol_path,protocol)
    models=[]
    for maker,cfg,name in [(get_cls_net,'hrnetv2_w48.yaml','SV_kp'),(get_line_net,'hrnetv2_w48_l.yaml','SV_lines')]:
        net=maker(yaml.safe_load((upstream/'config'/cfg).read_text()))
        net.load_state_dict(torch.load(weights/name,map_location='cpu',weights_only=True))
        net=net.to('cuda').eval();assert next(net.parameters()).is_cuda;models.append(net)
    world={i+1:[*xy,-2.44 if i+1 in (12,15,16,19) else 0.] for i,xy in enumerate(keypoint_world_coords_2D)}
    line_world={i+1:v for i,v in enumerate(line_world_coords_3D)}
    records=[]
    def timeout(*args):raise TimeoutError('Calibration exceeded45seconds; saved detections retained')
    signal.signal(signal.SIGALRM,timeout)
    for sample in manifest['samples']:
        record_path=out/'records'/(sample['clip']+'.json')
        if record_path.exists():
            record=json.loads(record_path.read_text())
            assert record['image_sha256']==sample['image_sha256'] and record['protocol_sha256']==sha(protocol_path)
            records.append(record);continue
        frame=cv2.imread(str(inputs/sample['image']));assert frame is not None
        height,width=frame.shape[:2]
        record={'clip':sample['clip'],'source_id':sample['source_id'],'fold':sample['fold'],
            'frame_index':sample['frame_index'],'image_sha256':sample['image_sha256'],
            'protocol_sha256':sha(protocol_path),'width':width,'height':height}
        started=time.monotonic();raw_path=out/'detections'/(sample['clip']+'.json')
        if raw_path.exists():
            raw=json.loads(raw_path.read_text());assert raw['image_sha256']==sample['image_sha256'] and raw['protocol_sha256']==sha(protocol_path)
            kp={int(k):v for k,v in raw['keypoints'].items()};lines={int(k):v for k,v in raw['lines'].items()}
            record['network_reused']=True
        else:
            tensor=tf.to_tensor(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))).unsqueeze(0)
            tensor=tf.resize(tensor,[540,960],antialias=True).to('cuda');torch.cuda.synchronize();tick=time.monotonic()
            with torch.inference_mode():
                heatmaps=models[0](tensor);heatmaps_l=models[1](tensor)
                assert heatmaps.is_cuda and heatmaps_l.is_cuda
                kp=coords_to_dict(get_kp(heatmaps[:,:-1]),threshold=.3434)[0]
                lines=coords_to_dict(get_lines(heatmaps_l[:,:-1]),threshold=.7867)[0]
            torch.cuda.synchronize();record['network_seconds']=time.monotonic()-tick
            record['network_reused']=False;record['cuda_outputs']=True
            raw={'image_sha256':sample['image_sha256'],'protocol_sha256':sha(protocol_path),
                 'keypoints':kp,'lines':lines,'network_seconds':record['network_seconds']}
            dump(raw_path,raw)
        raw_kp={k:{**v,'x':v['x']*width/960,'y':v['y']*height/540} for k,v in kp.items()}
        raw_lines={k:{**v,'x_1':v['x_1']*width/960,'y_1':v['y_1']*height/540,
                    'x_2':v['x_2']*width/960,'y_2':v['y_2']*height/540} for k,v in lines.items()}
        overlay=frame.copy();record['quality']={'passed':False}
        try:
            signal.alarm(45)
            complete,completed_lines=complete_keypoints(copy.deepcopy(kp),copy.deepcopy(lines),w=960,h=540,normalize=True)
            camera=FramebyFrameCalib(iwidth=width,iheight=height,denormalize=True)
            camera.update(complete,completed_lines)
            params=camera.heuristic_voting(refine_lines=True)
            record['parameters']=params
            if params is not None:
                p=projection(params);record['projection']=p.tolist()
                record['quality']=quality(p,width,height,raw_kp,raw_lines,world,line_world)
                for ends in line_world.values():
                    uv,valid=project(p,ends)
                    if valid.all() and abs(uv).max()<1e6:
                        cv2.line(overlay,tuple(uv[0].astype(int)),tuple(uv[1].astype(int)),(255,100,0),2)
                spots,valid=project(p,[[-41.5,0,0],[41.5,0,0]])
                for uv,ok in zip(spots,valid):
                    if ok and (uv>=0).all() and (uv<[width,height]).all():cv2.circle(overlay,tuple(uv.astype(int)),6,(255,0,255),-1)
        except Exception as e:
            record['error']=type(e).__name__+': '+str(e)[:400]
        finally:signal.alarm(0)
        for point in raw_kp.values():cv2.circle(overlay,(round(point['x']),round(point['y'])),3,(0,255,0),-1)
        record['seconds']=time.monotonic()-started
        record['peak_cuda_gb']=torch.cuda.max_memory_allocated()/1e9
        caption=sample['clip']+'  '+('PASS' if record['quality']['passed'] else 'REJECT')
        cv2.putText(overlay,caption,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),5)
        cv2.putText(overlay,caption,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
        (out/'examples').mkdir(exist_ok=True)
        assert cv2.imwrite(str(out/'examples'/(sample['clip']+'.jpg')),overlay)
        dump(record_path,record);records.append(record)
        print(json.dumps({'stage':'field_pilot','completed':len(records),'passed':record['quality']['passed'],'seconds':record['seconds']}),flush=True)
    times=[r['seconds'] for r in records[2:]]
    passed=sum(r['quality']['passed'] for r in records)
    summary={'state':'complete','clips':len(records),'quality_passed':passed,'median_seconds_excluding_first2':float(np.median(times)),
        'expansion_gate_passed':bool(passed>=6 and np.median(times)<=10),'visual_review_required':True,
        'model_gpu':protocol['gpu'],'new_role_pose_nlf_inferences':0,'competition_score':None}
    dump(out/'summary.json',summary)
    dump(out/'artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    return summary
