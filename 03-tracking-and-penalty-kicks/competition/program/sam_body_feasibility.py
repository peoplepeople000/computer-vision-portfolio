"""R92 official body-only model feasibility, on unchanged authorized pilot input."""
import hashlib,json,time,gc,os
from pathlib import Path
import numpy as np

DINO_REVISION='6876159a11b4df116f30f667f8c9888617df0751'
INITIALIZED_ONLY=frozenset(('backbone.encoder.mask_token','head_pose.hand_pose_comps_ori','head_pose_hand.hand_pose_comps_ori'))


def audit_keys(keys,provided):
    missing=sorted(set(keys)-set(provided));unexpected=sorted(set(provided)-set(keys))
    external=lambda k:k.startswith(('head_pose.mhr.','head_pose_hand.mhr.'))
    unsafe=[k for k in missing if not external(k) and k not in INITIALIZED_ONLY and not k.endswith('num_batches_tracked')]
    return {'missing':missing,'unexpected':unexpected,'unsafe_missing':unsafe,
            'initialized_only':sorted(set(missing)&INITIALIZED_ONLY),'externalMHR_separately_hashed':True}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def project(points,translation,intrinsic):
    camera=np.asarray(points)+np.asarray(translation).reshape(1,3)
    assert np.isfinite(camera).all() and (camera[:,2]>0).all()
    homogeneous=camera@np.asarray(intrinsic).T
    return homogeneous[:,:2]/homogeneous[:,2:3]


def run(inputs,out,assets):
    import torch
    from huggingface_hub import hf_hub_download
    from kaggle_secrets import UserSecretsClient
    assert torch.cuda.is_available(),'CUDA required'
    inputs,out,assets=map(Path,(inputs,out,assets));out.mkdir(parents=True,exist_ok=True);assets.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'input_manifest.json').read_text());plan=manifest['plan']
    for name,h in manifest['files'].items():assert sha(inputs/name)==h
    dump(out/'protocol.json',{'manifest':manifest,'worker_sha256':sha(__file__)})
    torch.set_num_threads(4);os.environ['MOMENTUM_ENABLED']='0';os.environ['WANDB_MODE']='disabled'
    # Import official modules before downloading GBs, to expose setup failures early.
    from sam_3d_body import SAM3DBodyEstimator
    import sam_3d_body.build_models as builder
    hub_load=torch.hub.load
    def pinned_hub(repo,*args,**kwargs):
        if repo=='facebookresearch/dinov3':repo+=':'+DINO_REVISION
        return hub_load(repo,*args,**kwargs)
    torch.hub.load=pinned_hub
    model_info=plan['official_model'];token=UserSecretsClient().get_secret('HF_TOKEN').strip()
    expected={'model_config.yaml':plan['config_sha256'],'model.ckpt':model_info['checkpoint']['sha256'],'assets/mhr_model.pt':model_info['mhr']['sha256']}
    for filename,h in expected.items():
        target=assets/filename
        if not target.exists():
            hf_hub_download(model_info['huggingface_repository'],filename,revision=model_info['huggingface_revision'],token=token,local_dir=str(assets))
        assert sha(target)==h,filename
        print(json.dumps({'stage':'asset_verified','file':filename,'bytes':target.stat().st_size}),flush=True)
    del token
    dump(out/'asset_receipt.json',{'files':expected,'storage':'Private notebook model_cache output; never republish publicly'})
    original_load=builder.load_state_dict;load_audits=[]
    def audited_load(model,state_dict,strict=False,**kwargs):
        values=model.state_dict();record=audit_keys(values,state_dict)
        # These exact defaults are created by the reviewed official constructors.
        for key in record['initialized_only']:
            value=values[key]
            expected=torch.zeros_like(value) if key.endswith('mask_token') else torch.eye(54,device=value.device,dtype=value.dtype)
            assert torch.equal(value,expected),key
        record['initialized_defaults_verified']=True
        record['dinov3_revision']=DINO_REVISION
        load_audits.append(record);dump(out/'checkpoint_load.json',load_audits)
        assert not record['unsafe_missing'] and not record['unexpected'],'Checkpoint mismatch: inspect saved key lists'
        return original_load(model,state_dict,strict=False,**kwargs)
    builder.load_state_dict=audited_load
    def load():
        model,cfg=builder.load_sam_3d_body(str(assets/'model.ckpt'),device='cuda',mhr_path=str(assets/'assets/mhr_model.pt'))
        assert cfg.MODEL.DECODER.USE_INTRIN_CENTER is True
        assert next(model.parameters()).is_cuda
        return model,SAM3DBodyEstimator(model,cfg)
    model,est=load();records=json.loads((inputs/'records.json').read_text());assert len(records)==12
    seen=[]
    def hook(module,args):
        tensors=[v for v in args if isinstance(v,torch.Tensor)]
        assert tensors and tensors[0].is_cuda
        seen.append({'device':str(tensors[0].device),'shape':list(tensors[0].shape)})
    handle=model.backbone.register_forward_pre_hook(hook)
    def infer(record):
        K=torch.tensor(record['intrinsic'],dtype=torch.float32,device='cuda')[None]
        bbox=np.asarray(record['bbox_xyxy'],np.float32)[None]
        with torch.inference_mode():
            result=est.process_one_image(str(inputs/record['image']),bboxes=bbox,cam_int=K,inference_type='body')
        assert len(result)==1
        return {k:np.asarray(v) for k,v in result[0].items() if v is not None}
    infer(records[0]);torch.cuda.synchronize();started=time.monotonic();timings=[];first=None
    for i,record in enumerate(records):
        assert time.monotonic()-started<plan['bounds']['soft_seconds']
        torch.cuda.reset_peak_memory_stats();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();t=time.monotonic();start.record();r=infer(record);end.record();torch.cuda.synchronize()
        seconds=time.monotonic()-t;timings.append(seconds)
        for value in r.values():assert np.isfinite(value).all()
        projected=project(r['pred_keypoints_3d'],r['pred_cam_t'],record['intrinsic'])
        err=float(np.abs(projected-r['pred_keypoints_2d']).max());assert err<.02,err
        peak=torch.cuda.max_memory_allocated()/2**30
        item={'clip':record['clip'],'seconds':seconds,'cuda_ms':start.elapsed_time(end),'peak_gib':peak,'projection_max_pixels':err,'known_wrong_role':record['clip'] in plan['known_role_failures']}
        assert item['cuda_ms']>0 and seen and peak<plan['bounds']['gpu_peak_gib']
        np.savez_compressed(out/(record['clip']+'.npz'),**r)
        dump(out/(record['clip']+'.json'),item)
        print(json.dumps({'stage':'image_complete',**item}),flush=True)
        if first is None:first=r
        if i==1:assert np.median(timings)<=plan['bounds']['median_seconds_per_image'],'Cost gate'
    handle.remove();del est,model;gc.collect();torch.cuda.empty_cache()
    model,est=load();replay=infer(records[0])
    for key in first:np.testing.assert_allclose(replay[key],first[key],rtol=1e-4,atol=1e-4,err_msg=key)
    dump(out/'reload.json',{'passed':True,'keys':list(first),'independent_model_loads':len(load_audits)})
    dump(out/'hardware.json',{'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'torch':str(torch.__version__),'backbone_inputs':seen})
    summary={'state':'technical_complete_visual_review_pending','images':12,'median_seconds':float(np.median(timings)),'known_wrong_roles':plan['known_role_failures'],'new_control_perception_calls':0,'classifier_fits':0,'competition_submissions':0,'claim':'Internal camera/reload consistency is not measured3D accuracy or R89 repair-gate success.'}
    dump(out/'summary.json',summary);return summary
