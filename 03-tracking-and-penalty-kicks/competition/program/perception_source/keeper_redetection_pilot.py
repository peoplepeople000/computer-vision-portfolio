"""R66 native five-frame keeper redetection/appearance association pilot, frozen backbones."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import field_calibration_pilot as field
from pose_quality_ablation import sha, dump

WEIGHTS={'rtmw':'9a9bc17c13ff0a1e37507f45a037a21bc410cd08374b0e664c57d0080843aa58',
         'yolox':'3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23'}
EDGES=[(5,6),(5,11),(6,12),(11,12),(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]


def clamp(box,w,h):
    b=np.asarray(box,float).copy();b[[0,2]]=np.clip(b[[0,2]],0,w-1);b[[1,3]]=np.clip(b[[1,3]],0,h-1)
    return b if np.isfinite(b).all() and min(b[2:]-b[:2])>=4 else None


def iou(a,b):
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));over=np.prod(size)
    return float(over/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-over+1e-9))


def goal_roi(cal,w,h):
    if cal is None or not cal['quality']['passed']:return None
    pts,ok=field.project(cal['projection'],field.goal_corners(cal['quality']['unique_goal_side']))
    if not ok.all():return None
    lo,hi=pts.min(0),pts.max(0);pad=(hi-lo)*[.15,.35]
    return clamp(np.r_[lo-pad,hi+pad],w,h)


def select(candidates,cal):
    """No clicks/labels; box feet are approximate ground proxies only."""
    unique=[]
    for box in candidates:
        box=np.asarray(box,float)
        if not any(iou(box,b)>.5 for b in unique):unique.append(box)
    scores=[]
    for i,b in enumerate(unique):
        xy,valid=field.ground_points(cal['projection'],[[(b[0]+b[2])/2,b[3]]])
        if not valid[0]:continue
        left,depth=field.goal_relative(xy,cal['quality']['unique_goal_side'])[0]
        if abs(left)>5.5 or not -1.5<=depth<=2.5:continue
        cost=abs(depth)/2+abs(left)/7.32
        scores.append({'candidate':i,'box':b.tolist(),'left':float(left),'depth':float(depth),'cost':float(cost)})
    scores.sort(key=lambda s:s['cost'])
    if not scores:return None,{'reason':'no_ground_plausible_candidate','candidates':scores}
    if len(scores)>1 and scores[1]['cost']-scores[0]['cost']<.15:
        return None,{'reason':'ambiguous_candidates','candidates':scores}
    return np.asarray(scores[0]['box']),{'reason':'selected','candidates':scores}


def flow_affine(first,second,mask,min_points):
    points=cv2.goodFeaturesToTrack(first,maxCorners=250,qualityLevel=.01,minDistance=3,mask=mask)
    if points is None or len(points)<min_points:return None,{'reason':'too_few_corners','corners':0 if points is None else len(points)}
    opts=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(first,second,points,None,**opts)
    back,bok,_=cv2.calcOpticalFlowPyrLK(second,first,nxt,None,**opts)
    good=(ok.ravel()>0)&(bok.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
    if good.sum()<min_points:return None,{'reason':'insufficient_fb_tracks','good':int(good.sum())}
    matrix,inliers=cv2.estimateAffinePartial2D(points[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2)
    if matrix is None or inliers.sum()<min_points or inliers.mean()<.6:return None,{'reason':'unstable_affine'}
    scale=float(np.linalg.norm(matrix[:,0]))
    if not .8<=scale<=1.25:return None,{'reason':'scale_jump','scale':scale}
    return np.vstack([matrix,[0,0,1]]),{'reason':'accepted','inliers':int(inliers.sum()),'scale':scale,'median_fb':float(np.median(np.linalg.norm(back[good]-points[good],axis=2)))}


def move_box(box,matrix,w,h):
    x1,y1,x2,y2=box;p=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]])@matrix.T
    return clamp(np.r_[p[:,:2].min(0),p[:,:2].max(0)],w,h)


def appearance(frame,box):
    h,w=frame.shape[:2];b=clamp(box,w,h)
    if b is None:return None
    lo,hi=b[:2],b[2:];size=hi-lo
    x1,y1,x2,y2=np.rint(np.r_[lo+size*[.2,.15],lo+size*[.8,.65]]).astype(int)
    crop=frame[y1:y2,x1:x2]
    if crop.size==0:return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[16,8],[0,180,0,256]);hist/=max(float(hist.sum()),1e-8)
    return hist


def associate(previous,candidates,colors):
    scored=[];hp=previous[3]-previous[1]
    for i,(box,color) in enumerate(zip(candidates,colors)):
        h=box[3]-box[1];ratio=h/hp
        distance=np.linalg.norm((box[:2]+box[2:]-previous[:2]-previous[2:])/2)/max(h,hp)
        if not .5<=ratio<=2 or distance>1.5 or color>.65:continue
        cost=.55*color+.35*distance+.1*abs(np.log(ratio))
        scored.append({'candidate':i,'cost':float(cost),'color':float(color),'distance':float(distance),'height_ratio':float(ratio)})
    scored.sort(key=lambda x:x['cost'])
    if not scored:return None,{'reason':'no_appearance_motion_candidate','candidates':scored}
    if len(scored)>1 and scored[1]['cost']-scored[0]['cost']<.12:return None,{'reason':'ambiguous_appearance_motion','candidates':scored}
    return candidates[scored[0]['candidate']],{'reason':'accepted','candidates':scored}


def track(frames,seed,detector,roi,calls):
    boxes=[seed];audit=[];reference=appearance(frames[0],seed);previous=reference
    if reference is None:return None,[{'reason':'missing_reference_appearance'}],[]
    x1,y1,x2,y2=np.rint(roi).astype(int)
    for frame in frames[1:]:
        raw=detector(frame[y1:y2,x1:x2]);calls['detector']+=1
        candidates=[b+np.array([x1,y1,x1,y1]) for b in raw];colors=[]
        for b in candidates:
            hist=appearance(frame,b)
            color=1. if hist is None else .5*cv2.compareHist(reference,hist,cv2.HISTCMP_BHATTACHARYYA)+.5*cv2.compareHist(previous,hist,cv2.HISTCMP_BHATTACHARYYA)
            colors.append(color)
        box,info=associate(boxes[-1],candidates,colors);audit.append(info)
        if box is None:return None,audit,[]
        boxes.append(box);previous=appearance(frame,box)
    return boxes,audit,[]


def interpolate(sample,indices):
    anchors=sample['anchors'];times=np.array([a['frame_index'] for a in anchors])
    out=np.full((len(indices),2,4),np.nan)
    for role in range(2):
        if any(a['people'][role] is None for a in anchors):continue
        boxes=np.array([a['people'][role]['box'] for a in anchors])
        for k in range(4):out[:,role,k]=np.interp(indices,times,boxes[:,k])
    return out


def draw_pose(img,points,scores,offset,scale,color):
    p=np.rint((points-offset)*scale).astype(int);v=scores>.3
    for a,b in EDGES:
        if v[a] and v[b]:cv2.line(img,tuple(p[a]),tuple(p[b]),color,1,cv2.LINE_AA)
    for xy,valid in zip(p,v):
        if valid:cv2.circle(img,tuple(xy),2,color,-1)


def run(inputs,out,weights,videos,budget=900):
    import torch
    import onnxruntime as ort
    from rtmlib import YOLOX,RTMPose
    inputs,out,weights=map(Path,(inputs,out,weights));out.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((inputs/'input_manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(inputs/name)==h
    for name,h in WEIGHTS.items():assert sha(weights/(name+'.onnx'))==h
    assert torch.cuda.is_available();torch.set_num_threads(4);cv2.setNumThreads(1)
    if hasattr(ort,'preload_dlls'):ort.preload_dlls()
    assert 'CUDAExecutionProvider' in ort.get_available_providers()
    detector=YOLOX(str(weights/'yolox.onnx'),model_input_size=(640,640),backend='onnxruntime',device='cuda')
    pose=RTMPose(str(weights/'rtmw.onnx'),model_input_size=(192,256),backend='onnxruntime',device='cuda')
    profiles={}
    for name,model in [('detector',detector),('pose',pose)]:
        options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(out/name)
        model.session=ort.InferenceSession(str(weights/('yolox.onnx' if name=='detector' else 'rtmw.onnx')),sess_options=options,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        assert model.session.get_providers()[0]=='CUDAExecutionProvider'
    protocol={'manifest':manifest,'source_sha256':sha(__file__),'weights':WEIGHTS}
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:dump(path,protocol)
    started=time.monotonic();reports=[];calls={'detector':0,'pose':0};samples=json.loads((inputs/'samples.json').read_text())
    (out/'records').mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'features').mkdir(exist_ok=True)
    for sample in samples:
        name=sample['clip'];record=out/'records'/(name+'.json')
        if record.exists():
            old=json.loads(record.read_text());assert old['input']==sample and sha(out/'features'/(name+'.npz'))==old['feature_sha256'];reports.append(old);continue
        if time.monotonic()-started>budget-120:break
        video=Path(videos[name]);assert sha(video)==sample['anchors'][0]['video_sha256']
        start,end=[sample['anchors'][i]['frame_index'] for i in (0,2)]
        indices=np.unique(np.rint(np.linspace(start,end,5)).astype(int));assert len(indices)==5
        cap=cv2.VideoCapture(str(video));frames=[];times=[];index=0
        while index<=end:
            ok,frame=cap.read()
            if not ok:break
            if index in indices:frames.append(frame);times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            index+=1
        cap.release();assert len(frames)==5 and np.all(np.diff(times)>0)
        for ai in [0,2]:assert abs(times[0 if ai==0 else -1]-sample['anchors'][ai]['timestamp'])<.003
        h,w=frames[0].shape[:2];old_boxes=interpolate(sample,indices)
        cal=sample['anchors'][0]['calibration'];roi=goal_roi(cal,w,h)
        selected=None;selection={'reason':'no_valid_early_calibration'};flow=[];background=[]
        if roi is not None:
            x1,y1,x2,y2=np.rint(roi).astype(int);candidate=detector(frames[0][y1:y2,x1:x2]);calls['detector']+=1
            candidates=[b+np.array([x1,y1,x1,y1]) for b in candidate]
            if np.isfinite(old_boxes[0,1]).all():candidates.append(old_boxes[0,1])
            selected,selection=select(candidates,cal)
        tracked=None
        if selected is not None:tracked,flow,background=track(frames,selected,detector,roi,calls)
        new_boxes=old_boxes.copy()
        if tracked is not None:new_boxes[:,1]=tracked
        arrays={};reused={}
        prior_path=inputs/'reuse/features'/(name+'.npz')
        if prior_path.exists():
            prior_record=json.loads((inputs/'reuse/records'/(name+'.json')).read_text())
            assert prior_record['input']==sample and sha(prior_path)==prior_record['feature_sha256']
            prior=np.load(prior_path,allow_pickle=False)
            np.testing.assert_array_equal(prior['frame_indices'],indices);np.testing.assert_allclose(prior['timestamps'],times,atol=1e-9)
            np.testing.assert_array_equal(prior['old_boxes'],old_boxes)
            for old_arm,bkey in [('native_control','old_boxes'),('early_track','new_boxes')]:
                for t in range(5):
                    for role in range(2):
                        if not prior[old_arm+'_mask'][t,role]:continue
                        b=clamp(prior[bkey][t,role],w,h)
                        center=(b[:2]+b[2:])/2;side=max(b[2:]-b[:2])*1.15
                        expanded=clamp(np.r_[center-side/2,center+side/2],w,h)
                        reused[(t,tuple(expanded))]=(prior[old_arm+'_points'][t,role],prior[old_arm+'_scores'][t,role])
        for arm,boxes in [('native_control',old_boxes),('early_track',new_boxes)]:
            pts=np.zeros((5,2,23,2),np.float32);scores=np.zeros((5,2,23),np.float32);mask=np.zeros((5,2),bool)
            for t in range(5):
                for role in range(2):
                    b=clamp(boxes[t,role],w,h)
                    if b is None:continue
                    center=(b[:2]+b[2:])/2;side=max(b[2:]-b[:2])*1.15
                    expanded=clamp(np.r_[center-side/2,center+side/2],w,h)
                    key=(t,tuple(expanded))
                    if key not in reused:
                        p,s=pose(frames[t],bboxes=[expanded]);calls['pose']+=1
                        assert p.shape==(1,133,2) and np.isfinite(p).all() and np.isfinite(s).all()
                        reused[key]=(p[0,:23].copy(),s[0,:23].copy())
                    pts[t,role],scores[t,role]=reused[key];mask[t,role]=True
            arrays[arm+'_points']=pts;arrays[arm+'_scores']=scores;arrays[arm+'_mask']=mask
        feature=out/'features'/(name+'.npz')
        np.savez_compressed(feature,**arrays,old_boxes=old_boxes,new_boxes=new_boxes,frame_indices=indices,timestamps=times)
        # Fixed scene crops preserve identity context; do not draw skeleton across absent observations.
        canvas=np.full((740,1600,3),24,np.uint8)
        for t,frame in enumerate(frames):
            region=roi if roi is not None else np.array([0,0,w-1,h-1])
            if tracked is not None:region=np.r_[np.minimum(region[:2],new_boxes[t,1,:2]-30),np.maximum(region[2:],new_boxes[t,1,2:]+30)]
            region=clamp(region,w,h);a,b,c,d=np.rint(region).astype(int)
            patch=frame[b:d,a:c];scale=min(320/patch.shape[1],220/patch.shape[0]);res=cv2.resize(patch,None,fx=scale,fy=scale)
            for row in range(3):
                tile=np.full((220,320,3),24,np.uint8);tile[:res.shape[0],:res.shape[1]]=res
                boxes=old_boxes if row<2 else new_boxes
                if np.isfinite(boxes[t,1]).all():
                    rect=np.rint((boxes[t,1].reshape(2,2)-[a,b])*scale).astype(int)
                    cv2.rectangle(tile,tuple(rect[0]),tuple(rect[1]),(255,180,0) if row<2 else (0,230,80),1)
                if row:
                    arm='native_control' if row==1 else 'early_track'
                    draw_pose(tile,arrays[arm+'_points'][t,1],arrays[arm+'_scores'][t,1],np.array([a,b]),scale,(0,180,255) if row==1 else (0,230,80))
                canvas[60+row*225:280+row*225,t*320:(t+1)*320]=tile
                cv2.putText(canvas,f'{indices[t]} / {times[t]:.3f}s',(t*320+5,50),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1)
        title=f'{name} | top: original box | middle: native pose | bottom: early track | accepted={tracked is not None}'
        cv2.putText(canvas,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        assert cv2.imwrite(str(out/'images'/(name+'.jpg')),canvas)
        result={'input':sample,'feature_sha256':sha(feature),'selection':selection,'tracking':flow,'tracking_accepted':tracked is not None,
                'background_transforms':[None if x is None else x.tolist() for x in background],
                'native_pixels':True,'identity_review_pending':True,'reused_r63':prior_path.exists()}
        dump(record,result);reports.append(result)
        print(json.dumps({'clip':name,'selection':selection['reason'],'tracked':tracked is not None}),flush=True)
    for name,model in [('detector',detector),('pose',pose)]:
        events=json.loads(Path(model.session.end_profiling()).read_text())
        profiles[name]=sum(e.get('args',{}).get('provider')=='CUDAExecutionProvider' for e in events)
        assert profiles[name]>0
    dump(out/'hardware.json',{'gpu':torch.cuda.get_device_name(0),'cuda_node_events':profiles,'providers':pose.session.get_providers()})
    summary={'state':'complete' if len(reports)==len(samples) else 'budget_boundary','clips':len(reports),
        'tracked':sum(r['tracking_accepted'] for r in reports),'calls':calls,'seconds':time.monotonic()-started,
        'visual_review_pending':True,'competition_submissions':0}
    dump(out/'summary.json',summary);return summary
