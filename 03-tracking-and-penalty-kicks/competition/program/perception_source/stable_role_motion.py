"""R67 partial role tracks and explicit motion; frozen R66 remains replayable.

Coordinates are observed image measurements, never metric field positions.
Missing observations never acquire interpolated poses or cross-gap velocity.
"""
import numpy as np
import cv2
import json
import time
from pathlib import Path
import keeper_redetection_pilot as old
from pose_quality_ablation import sha,dump
from keeper_redetection_pilot import appearance, clamp, iou, flow_affine
from pose_mirror import JOINT_SWAP


def associate(previous, candidates, colors):
    """R66 gates, with every rejection preserved for diagnosis."""
    rows=[]; hp=previous[3]-previous[1]
    for i,(b,color) in enumerate(zip(candidates,colors)):
        h=b[3]-b[1]; ratio=h/hp
        distance=np.linalg.norm((b[:2]+b[2:]-previous[:2]-previous[2:])/2)/max(h,hp)
        reasons=[]
        if not .5<=ratio<=2: reasons.append('height')
        if distance>1.5: reasons.append('distance')
        if color>.65: reasons.append('appearance')
        rows.append(dict(candidate=i,box=b.tolist(),color=float(color),distance=float(distance),
                         ratio=float(ratio),rejected=reasons,
                         cost=float(.55*color+.35*distance+.1*abs(np.log(ratio)))))
    eligible=sorted([r for r in rows if not r['rejected']],key=lambda r:r['cost'])
    reason='accepted'
    if not eligible: reason='no_candidate'
    elif len(eligible)>1 and eligible[1]['cost']-eligible[0]['cost']<.12: reason='ambiguous'
    index=eligible[0]['candidate'] if reason=='accepted' else None
    return index,dict(reason=reason,candidates=rows)


def track_observations(frames,candidates,seed_index,seed_box):
    """Track both directions from an observed seed, retaining partial segments.

    Search at most two steps beyond the last observation; use a fixed appearance
    reference. Never replace a failed track with old independently selected boxes.
    """
    n=len(frames); boxes=np.full((n,4),np.nan); states=['unconfirmed']*n; logs=[None]*n
    reference=appearance(frames[seed_index],seed_box)
    if reference is None:return boxes,states,logs
    boxes[seed_index]=seed_box;states[seed_index]='seed';logs[seed_index]={'reason':'seed'}
    for direction in (1,-1):
        last=seed_index; previous=reference
        for t in range(seed_index+direction,n if direction==1 else -1,direction):
            gap=abs(t-last)
            if gap>2:
                states[t]='lost';logs[t]={'reason':'gap_limit','candidates':[]};continue
            colors=[]
            for b in candidates[t]:
                h=appearance(frames[t],b)
                colors.append(1. if h is None else .5*cv2.compareHist(reference,h,cv2.HISTCMP_BHATTACHARYYA)
                              +.5*cv2.compareHist(previous,h,cv2.HISTCMP_BHATTACHARYYA))
            j,logs[t]=associate(boxes[last],candidates[t],colors)
            if j is None:states[t]='lost';continue
            boxes[t]=candidates[t][j];states[t]='reacquired' if gap>1 else 'observed'
            last=t;previous=appearance(frames[t],boxes[t])
    return boxes,states,logs


def background_reference(frames,candidates):
    """Current-to-first background affine; fail closed for root displacement."""
    transforms=np.tile(np.eye(3),(len(frames),1,1));valid=np.zeros(len(frames),bool);valid[0]=True
    logs=[{'reason':'reference'}];h,w=frames[0].shape[:2]
    first=cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
    for t in range(1,len(frames)):
        current=cv2.cvtColor(frames[t],cv2.COLOR_BGR2GRAY);mask=np.full((h,w),255,np.uint8)
        # Exclude people at both ends, including their immediate surrounding pixels.
        for boxes in (candidates[0],candidates[t]):
            for b in boxes:
                size=b[2:]-b[:2];b=clamp(np.r_[b[:2]-.25*size,b[2:]+.25*size],w,h)
                if b is None:continue
                x1,y1,x2,y2=np.rint(b).astype(int);mask[y1:y2,x1:x2]=0
        matrix,info=flow_affine(current,first,mask,16)
        logs.append(info)
        if matrix is not None:transforms[t]=matrix;valid[t]=True
    return transforms,valid,logs


def stable_coordinates(points,scores,observed,transforms,bg_valid):
    """One torso scale per role/window; no hip/shoulder center substitution.

    Local joints are hip-relative. Root trajectories require background alignment
    at EVERY observed frame. Coordinates without alignment remain local only.
    """
    points=np.asarray(points,float);scores=np.asarray(scores,float)
    n=len(points);sequence=np.zeros((n,2,96),np.float32)
    for role in range(2):
        raw=points[:,role];v=(scores[:,role]>.3)&observed[:,role,None]&np.isfinite(raw).all(-1)
        core=v[:,11]&v[:,12]&v[:,5]&v[:,6]
        if not core.any():continue
        camera_ok=bool(np.asarray(bg_valid)[core].all())
        p=raw.copy()
        if camera_ok:
            p=np.einsum('tij,tkj->tki',transforms,np.concatenate([raw,np.ones((*raw.shape[:2],1))],-1))[...,:2]
        hip=p[:,[11,12]].mean(1);shoulder=p[:,[5,6]].mean(1)
        torso=np.linalg.norm(shoulder-hip,axis=1);scale=float(np.median(torso[core]))
        if scale<5:continue
        # A gross pose-scale discontinuity is unavailable, not smoothed into motion.
        core&=(torso/scale>=.5)&(torso/scale<=2.)
        if not core.any():continue
        origin=hip[np.flatnonzero(core)[0]]
        for t in np.flatnonzero(core):
            joint=sequence[t,role,:92].reshape(23,4)
            joint[v[t],:2]=np.clip((p[t,v[t]]-hip[t])/scale,-8,8)
            joint[v[t],2]=scores[t,role,v[t]];joint[v[t],3]=1
            if camera_ok:sequence[t,role,92:94]=(hip[t]-origin)/scale
            sequence[t,role,94:]=[1,float(camera_ok)]
    return sequence


def mirror_sequence(sequence):
    result=sequence.copy();j=result[...,:92].reshape(*result.shape[:-1],23,4)
    j[:]=j[...,JOINT_SWAP,:].copy();j[...,0]*=-1;result[...,92]*=-1
    return result


ANGLE_TRIPLES=((11,13,15),(12,14,16),(5,11,13),(6,12,14))


def features(sequence,times):
    """Static 194D; dynamic 270D: observed local velocity, angles, root velocity.

    Velocity is torso-lengths/second; joint angle rates are radians/second.
    Both use actual presentation timestamps and adjacent observed pairs only.
    """
    seq=np.asarray(sequence);times=np.asarray(times,float);dt=np.diff(times)
    assert seq.shape==(5,2,96) and np.isfinite(seq).all() and (dt>0).all()
    static=[];dynamic=[];eligible=[]
    for r in range(2):
        s=seq[:,r];obs=s[:,94]>0;ids=np.flatnonzero(obs)
        static.extend(s[ids[-1]] if len(ids) else np.zeros(96))
        static.append(float(times[-1]-times[ids[-1]]) if len(ids) else 0.)
        p=s[:,:92].reshape(5,23,4);valid=p[...,3]>0
        pair=valid[1:]&valid[:-1]&obs[1:,None]&obs[:-1,None]
        velocity=np.diff(p[...,:2],axis=0)/dt[:,None,None]
        mean=(velocity*pair[...,None]).sum(0)/np.maximum(pair.sum(0)[:,None],1)
        last=np.zeros((23,2));last_valid=np.zeros(23)
        for j in range(23):
            use=np.flatnonzero(pair[:,j])
            if len(use):last[j]=velocity[use[-1],j];last_valid[j]=1
        angle_parts=[]
        for a,b,c in ANGLE_TRIPLES:
            u=p[:,a,:2]-p[:,b,:2];v=p[:,c,:2]-p[:,b,:2]
            norm=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
            good=valid[:,[a,b,c]].all(1)&obs&(norm>1e-6)
            angle=np.arccos(np.clip((u*v).sum(-1)/np.maximum(norm,1e-6),-1,1))
            ok=good[1:]&good[:-1];use=np.flatnonzero(ok)
            angle_parts.extend([float(np.diff(angle)[ok].sum()/dt[ok].sum()) if ok.any() else 0.,
                                float(np.diff(angle)[use[-1]]/dt[use[-1]]) if len(use) else 0.,float(ok.mean())])
        root_pair=obs[1:]&obs[:-1]&(s[1:,95]>0)&(s[:-1,95]>0)
        rv=np.diff(s[:,92:94],axis=0)/dt[:,None];use=np.flatnonzero(root_pair)
        root=np.r_[rv[root_pair].mean(0) if len(use) else [0,0],rv[use[-1]] if len(use) else [0,0],float(root_pair.mean())]
        dynamic.extend(np.r_[mean.ravel(),last.ravel(),pair.mean(0),angle_parts,root,
                             obs.mean(),(obs[1:]&obs[:-1]).mean(),times[-1]-times[0]])
        eligible.append(bool(obs.sum()>=3 and (obs[1:]&obs[:-1]).sum()>=2))
    return np.asarray(static,np.float32),np.asarray(dynamic,np.float32),np.array(eligible)


def unique_boxes(boxes):
    result=[]
    for b in boxes:
        if not any(iou(b,a)>.5 for a in result):result.append(np.asarray(b,float))
    return result


def seed_roles(sample,frames,candidates,indices):
    seeds=[None,None];audit=[{'reason':'no_spot_or_ball_kicker'},{'reason':'no_goal_keeper'}]
    # Use ONLY exact observed anchor frames; never apply another frame's camera.
    for a in sample['anchors']:
        if a['frame_index'] not in indices:continue
        t=indices.index(a['frame_index'])
        cal=a['calibration']
        if seeds[0] is None:
            ranked=[];method=None
            if cal is not None and cal['quality']['passed']:
                feet=[[(b[0]+b[2])/2,b[3]] for b in candidates[t]]
                xy,valid=old.field.ground_points(cal['projection'],feet)
                relative=old.field.goal_relative(xy,cal['quality']['unique_goal_side'])
                for j,((left,depth),ok) in enumerate(zip(relative,valid)):
                    if ok and abs(left)<=5 and 7<=depth<=16:
                        ranked.append((abs(depth-11)/5+abs(left)/5,j))
                method='calibrated_penalty_region'
            else:
                balls=[o for o in a.get('objects',[]) if o['kind']=='ball' and o['score']>=.4]
                if balls:
                    ball=np.asarray(max(balls,key=lambda o:o['score'])['bbox']);center=(ball[:2]+ball[2:])/2
                    for j,b in enumerate(candidates[t]):
                        distance=np.linalg.norm(np.array([(b[0]+b[2])/2,b[3]])-center)/(b[3]-b[1])
                        if distance<=1.75:ranked.append((distance,j))
                    method='observed_ball_proximity'
            ranked.sort()
            if ranked and (len(ranked)<2 or ranked[1][0]-ranked[0][0]>=.15):
                seeds[0]=(t,candidates[t][ranked[0][1]])
                audit[0]={'reason':method,'frame':t,'ranked':ranked}
        if seeds[1] is None and cal is not None and cal['quality']['passed']:
            b,info=old.select(candidates[t],cal)
            if b is not None:
                seeds[1]=(t,b);audit[1]={**info,'frame':t}
        elif seeds[1] is None:
            goals=[o for o in a.get('objects',[]) if o['kind']=='goal' and o['score']>=.4]
            if goals:
                g=np.asarray(max(goals,key=lambda o:o['score'])['bbox']);gw,gh=g[2:]-g[:2];ranked=[]
                for j,b in enumerate(candidates[t]):
                    x=(b[0]+b[2])/2;y=b[3];height=b[3]-b[1]
                    if not (g[0]<=x<=g[2] and g[1]+.45*gh<=y<=g[3]+.12*gh and .15*gh<=height<=1.2*gh):continue
                    ranked.append((abs(x-(g[0]+g[2])/2)/gw+abs(y-g[3])/gh,j))
                ranked.sort()
                if ranked and (len(ranked)<2 or ranked[1][0]-ranked[0][0]>=.15):
                    seeds[1]=(t,candidates[t][ranked[0][1]])
                    audit[1]={'reason':'observed_goal_box_proxy','frame':t,'ranked':ranked}
    return seeds,audit


def render(path,frames,boxes,points,scores,states,clip):
    """Full-frame and role closeups share the same actual five observations."""
    canvas=np.full((810,1600,3),24,np.uint8)
    for t,frame in enumerate(frames):
        h,w=frame.shape[:2];scale=min(320/w,210/h);scene=cv2.resize(frame,None,fx=scale,fy=scale)
        canvas[35:35+scene.shape[0],t*320:t*320+scene.shape[1]]=scene
        for r in range(2):
            color=(0,220,100) if r==0 else (255,180,0)
            b=clamp(boxes[t,r],w,h)
            if b is not None:
                p=np.rint(b.reshape(2,2)*scale).astype(int);p+=np.array([t*320,35])
                cv2.rectangle(canvas,tuple(p[0]),tuple(p[1]),color,2)
            # A fixed union crop per role preserves the context of any jumps.
            good=boxes[:,r][np.isfinite(boxes[:,r]).all(1)]
            if not len(good):continue
            lo=good[:,:2].min(0);hi=good[:,2:].max(0);size=hi-lo
            region=clamp(np.r_[lo-.3*size,hi+.3*size],w,h)
            a,c,d,e=np.rint(region).astype(int);patch=frame[c:e,a:d]
            z=min(315/patch.shape[1],245/patch.shape[0]);tile=cv2.resize(patch,None,fx=z,fy=z)
            if b is not None:
                rect=np.rint((b.reshape(2,2)-[a,c])*z).astype(int)
                cv2.rectangle(tile,tuple(rect[0]),tuple(rect[1]),color,1)
                old.draw_pose(tile,points[t,r],scores[t,r],np.array([a,c]),z,color)
            y=290+r*255;canvas[y:y+tile.shape[0],t*320:t*320+tile.shape[1]]=tile
            cv2.putText(canvas,f'{"kick" if r==0 else "keeper"}: {states[r][t]}',(t*320+3,y-7),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
    cv2.putText(canvas,clip+' | observed boxes only; missing = no pose',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    assert cv2.imwrite(str(path),canvas)


def run(inputs,out,weights,videos,budget=900):
    import torch
    import onnxruntime as ort
    from rtmlib import YOLOX,RTMPose
    inputs,out,weights=map(Path,(inputs,out,weights));out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'input_manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(inputs/name)==h
    for name,h in old.WEIGHTS.items():assert sha(weights/(name+'.onnx'))==h
    assert torch.cuda.is_available();torch.set_num_threads(4);cv2.setNumThreads(1)
    if hasattr(ort,'preload_dlls'):ort.preload_dlls()
    detector=YOLOX(str(weights/'yolox.onnx'),model_input_size=(640,640),backend='onnxruntime',device='cuda')
    pose=RTMPose(str(weights/'rtmw.onnx'),model_input_size=(192,256),backend='onnxruntime',device='cuda')
    for name,model in [('detector',detector),('pose',pose)]:
        options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(out/name)
        model.session=ort.InferenceSession(str(weights/('yolox.onnx' if name=='detector' else 'rtmw.onnx')),sess_options=options,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    protocol={'manifest':manifest,'source_sha256':sha(__file__),'weights':old.WEIGHTS}
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:dump(out/'protocol.json',protocol)
    for folder in ('features','records','images'): (out/folder).mkdir(exist_ok=True)
    started=time.monotonic();calls={'detector':0,'pose':0,'reused_pose':0};reports=[]
    samples=json.loads((inputs/'samples.json').read_text())
    for sample in samples:
        clip=sample['clip'];record=out/'records'/(clip+'.json');feature=out/'features'/(clip+'.npz')
        if record.exists():
            r=json.loads(record.read_text());assert r['input']==sample and sha(feature)==r['feature_sha256'];reports.append(r);continue
        if time.monotonic()-started>budget-120:break
        video=Path(videos[clip]);assert sha(video)==sample['anchors'][0]['video_sha256']
        start,end=[sample['anchors'][i]['frame_index'] for i in (0,2)]
        indices=np.unique(np.rint(np.linspace(start,end,5)).astype(int)).tolist();assert len(indices)==5
        frames=[];times=[];cap=cv2.VideoCapture(str(video));i=0
        while i<=end:
            ok,frame=cap.read()
            if not ok:break
            if i in indices:frames.append(frame);times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            i+=1
        cap.release();assert len(frames)==5 and np.all(np.diff(times)>0)
        for a,t in ((sample['anchors'][0],0),(sample['anchors'][2],4)):assert abs(times[t]-a['timestamp'])<.003
        h,w=frames[0].shape[:2];rois=[]
        for a in (sample['anchors'][0],sample['anchors'][2]):
            roi=old.goal_roi(a['calibration'],w,h)
            if roi is not None:rois.append(roi);break
        kb=next((np.asarray(a['people'][0]['box']) for a in sample['anchors'] if a['people'][0] is not None),None)
        if kb is not None:
            size=kb[2:]-kb[:2];roi=clamp(np.r_[kb[:2]-size*[2,1],kb[2:]+size*[2,1]],w,h)
            if roi is not None:rois.append(roi)
        candidate_cache=inputs/'candidate_reuse'/(clip+'.json')
        candidates=[]
        if candidate_cache.exists():
            cached=json.loads(candidate_cache.read_text())
            assert cached['video_sha256']==sample['anchors'][0]['video_sha256'] and cached['frame_indices']==indices
            candidates=[[np.asarray(b,float) for b in row] for row in cached['candidate_boxes']]
        for frame in ([] if candidates else frames):
            boxes=[]
            for roi in rois:
                x1,y1,x2,y2=np.rint(roi).astype(int)
                boxes.extend(np.asarray(detector(frame[y1:y2,x1:x2]))+np.array([x1,y1,x1,y1]));calls['detector']+=1
            boxes.extend(detector(frame));calls['detector']+=1
            candidates.append(unique_boxes([b for b in boxes if clamp(b,w,h) is not None]))
        seeds,selection=seed_roles(sample,frames,candidates,indices)
        boxes=np.full((5,2,4),np.nan);states=[];logs=[]
        for r,seed in enumerate(seeds):
            if seed is None:states.append(['unconfirmed']*5);logs.append([None]*5);continue
            b,s,l=track_observations(frames,candidates,*seed);boxes[:,r]=b;states.append(s);logs.append(l)
        for t in range(5):
            if np.isfinite(boxes[t]).all() and iou(*boxes[t])>.5:
                boxes[t]=np.nan
                for r in range(2):states[r][t]='role_collision'
        bg,bg_valid,bg_audit=background_reference(frames,candidates)
        points=np.zeros((5,2,23,2),np.float32);scores=np.zeros((5,2,23),np.float32);observed=np.zeros((5,2),bool)
        reuse={};prior=inputs/'reuse/features'/(clip+'.npz')
        if prior.exists():
            with np.load(prior,allow_pickle=False) as a:
                np.testing.assert_array_equal(a['frame_indices'],indices);np.testing.assert_allclose(a['timestamps'],times,atol=1e-9)
                arms=[('native_control','old_boxes'),('early_track','new_boxes')] if 'old_boxes' in a else [('', 'boxes')]
                for arm,bkey in arms:
                    for t in range(5):
                        for r in range(2):
                            if not a[arm+'_mask' if arm else 'observed'][t,r]:continue
                            b=clamp(a[bkey][t,r],w,h)
                            if b is None:continue
                            center=(b[:2]+b[2:])/2;side=max(b[2:]-b[:2])*1.15
                            crop=clamp(np.r_[center-side/2,center+side/2],w,h)
                            reuse[(t,tuple(crop))]=(a[arm+'_points' if arm else 'points'][t,r].copy(),a[arm+'_scores' if arm else 'scores'][t,r].copy())
        for t in range(5):
            for r in range(2):
                b=clamp(boxes[t,r],w,h)
                if b is None:continue
                center=(b[:2]+b[2:])/2;side=max(b[2:]-b[:2])*1.15;crop=clamp(np.r_[center-side/2,center+side/2],w,h)
                key=(t,tuple(crop))
                if key in reuse:points[t,r],scores[t,r]=reuse[key];calls['reused_pose']+=1
                else:
                    p,s=pose(frames[t],bboxes=[crop]);calls['pose']+=1
                    points[t,r],scores[t,r]=p[0,:23],s[0,:23]
                observed[t,r]=True
        seq=stable_coordinates(points,scores,observed,bg,bg_valid);stat,dyn,eligible=features(seq,times)
        np.savez_compressed(feature,points=points,scores=scores,observed=observed,boxes=boxes,sequence=seq,
                            transforms=bg,bg_valid=bg_valid,static=stat,dynamic=dyn,eligible=eligible,
                            frame_indices=indices,timestamps=times)
        render(out/'images'/(clip+'.jpg'),frames,boxes,points,scores,states,clip)
        result={'input':sample,'feature_sha256':sha(feature),'selection':selection,'states':states,'association':logs,
                'candidate_boxes':[[b.tolist() for b in f] for f in candidates],'background':bg_audit,
                'eligible':eligible.tolist(),'observed':observed.sum(0).tolist(),'identity_review_pending':True}
        dump(record,result);reports.append(result)
        print(json.dumps({'clip':clip,'observed':result['observed'],'eligible':result['eligible'],'completed':len(reports)}),flush=True)
    profiles={}
    for name,model in [('detector',detector),('pose',pose)]:
        events=json.loads(Path(model.session.end_profiling()).read_text())
        profiles[name]=sum(e.get('args',{}).get('provider')=='CUDAExecutionProvider' for e in events)
        if calls[name]:assert profiles[name]>0
    dump(out/'hardware.json',{'gpu':torch.cuda.get_device_name(0),'cuda_node_events':profiles})
    summary={'state':'complete' if len(reports)==len(samples) else 'budget_boundary','clips':len(reports),
             'eligible_roles':np.sum([r['eligible'] for r in reports],axis=0).tolist(),'calls':calls,
             'seconds':time.monotonic()-started,'competition_submissions':0,'identity_review_pending':True}
    dump(out/'summary.json',summary);return summary
