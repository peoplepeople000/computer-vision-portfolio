"""Train an IV-conditioned EX instance predictor with an excluded-mouse protocol."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from .learning import SmallUNet,loss_function
from .guided_ex_hard_negative_data import training_records,make_record,crop,sample
from .data import read_rows,split_id,image_path
from .stage3 import FOLDS
from .rle import encode


class GuidedNet(SmallUNet):
    def __init__(self,width,initial=None):
        super().__init__(width)
        if initial is not None:self.load_state_dict(initial)
        old=self.a[0];new=nn.Conv2d(9,width,3,padding=1)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:3].copy_(old.weight);new.bias.copy_(old.bias)
        self.a[0]=new;self.presence=nn.Linear(width*4,1)

    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
        d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(e),self.presence(c.mean((2,3))).ravel()


def objective(logits,presence,target,exists):
    return loss_function(logits,target)+.5*F.binary_cross_entropy_with_logits(presence,exists)


def proposal(prob,presence,x,cell,origin,shape):
    labels,count=ndi.label(prob[0]>.5);options=[]
    for label in range(1,count+1):
        mask=ndi.binary_fill_holes(labels==label);area=int(mask.sum())
        if area<6 or area>500:continue
        yy,xx=np.where(mask);distance=float(np.linalg.norm(np.array([xx.mean(),yy.mean()])-24))
        if distance>=8:continue
        score=float(prob[0][mask].mean())*np.exp(-.5*(distance/5)**2)
        options.append((score,mask,distance))
    if not options:return None
    _,mask,distance=max(options,key=lambda v:v[0]);yy,xx=np.where(mask);x0,y0=origin
    gx,gy=xx+x0,yy+y0;valid=(gx>=0)&(gy>=0)&(gx<shape[1])&(gy<shape[0])
    if valid.sum()<6:return None
    canvas=np.zeros(shape,bool);canvas[gy[valid],gx[valid]]=True
    # Features describe image evidence and the projected cell, never annotations.
    area=float(valid.sum());height=float(yy.max()-yy.min()+1);width=float(xx.max()-xx.min()+1)
    boundary=mask&~ndi.binary_erosion(mask);prior=x[3]>.5;union=(mask|prior).sum()
    features=[np.log1p(area),area/(width*height),np.log(width/height),float(prob[0][mask].mean()),float(prob[0][mask].max()),float(prob[0][mask].std()),float(prob[0][boundary].mean()),float(prob[1][mask].max()),float(presence),distance,np.log1p(cell['patch'].sum()),np.log(max(area,1)/max(cell['patch'].sum(),1)),float((mask&prior).sum()/max(union,1))]
    return dict(iv=cell['key'],rle=encode(canvas),features=[float(v) for v in features],presence=float(presence))


@torch.inference_mode()
def infer(model,record,device):
    results=[]
    for start in range(0,len(record['cells']),32):
        cells=record['cells'][start:start+32];crops=[crop(record,c) for c in cells]
        x=np.stack([c[0] for c in crops]);logits,presence=model(torch.from_numpy(x).to(device))
        probabilities=logits.sigmoid().cpu().numpy();presence=presence.sigmoid().cpu().numpy()
        for cell,values,pr,xx,(_,_,origin) in zip(cells,probabilities,presence,x,crops):
            value=proposal(values,pr,xx,cell,origin,record['shape'])
            if value is not None:results.append(value)
    return dict(sample_id=record['sample_id'],proposals=results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/stage54_guided_ex'));p.add_argument('--steps',type=int,default=1500)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(2078);rng=np.random.default_rng(2078)
    device=torch.device('mps')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold]
    ids=sorted(s for s in truth if split_id(s)[0]!=subject);initializer=Path(f'outputs/stage6_ex/unet_holdout_{a.fold}/model.pt')
    initial=torch.load(initializer,map_location='cpu',weights_only=False)
    if initial['held_out']!=subject or sorted(initial['train_ids'])!=ids:raise ValueError('Invalid initializer partition')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    config=dict(held_out=subject,train_ids=ids,steps=a.steps,seed=2078,width=initial['width'],batch=16,patch=48,channels=9,positive_sampling=.7,translation_jitter=4.,initializer=str(initializer),initializer_sha256=sha(initializer),source_sha256=sha(__file__),data_source_sha256=sha(Path(__file__).with_name('guided_ex_hard_negative_data.py')),base_data_source_sha256=sha(Path(__file__).with_name('guided_ex_data.py')),hard_negative_probability=.2,annotation_sha256=sha(root/'training/train_ground_truth.csv'),policy='Paired-EX target conditioned on a predicted IV cell. Development GT affines with 20% 40-100 pixel shifted verified-cell negatives; held-out image-only geometry. Unverified IV proposals are competition-credit negatives, not claimed biological negatives.')
    inputs=[Path(f'outputs/stage8_iv/{a.fold}/development.csv'),Path(f'outputs/stage8_iv/{a.fold}/training_config.json'),Path(f'outputs/stage11_native_ex/{a.fold}/training_config.json'),Path(f'outputs/stage8_iv_quality_capacity/{a.fold}.pt'),Path('outputs/stage28_cpdino_iv_full/predictions.csv')]
    for sid in ids:
        inputs.extend([Path(f'outputs/stage8_iv/{a.fold}/maps/{sid}_invivo.npz'),Path(f'outputs/stage11_native_ex/{a.fold}/maps/{sid}_exvivo.npz')])
        inputs.extend(image_path(root,'training',sid,mode) for mode in ['invivo','exvivo'])
    config['development_inputs_sha256']={str(p):sha(p) for p in inputs}
    out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True);cfg=out/'configuration.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Changed guided recipe')
    cfg.write_text(json.dumps(config,indent=2)+'\n')
    model=GuidedNet(initial['width'],initial['state_dict']).to(device);final=out/'model.pt';resume=out/'resume.pt'
    records,loaded_ids=training_records(root,truth,a.fold)
    if loaded_ids!=ids:raise ValueError('Training data differs')
    if final.exists():
        saved=torch.load(final,map_location='cpu',weights_only=False)
        if saved['config']!=config or saved['step']!=a.steps:raise ValueError('Final checkpoint mismatch')
        model.load_state_dict(saved['state_dict'])
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=3e-5);start=0;history=[]
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            if saved['config']!=config:raise ValueError('Resume mismatch')
            model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            start=saved['step'];history=saved['history'];rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng']);torch.mps.set_rng_state(saved['mps_rng'])
        model.train();tick=time.perf_counter()
        for step in range(start+1,a.steps+1):
            x,y,exists=(t.to(device) for t in sample(records,rng));optimizer.zero_grad(set_to_none=True)
            logits,presence=model(x);loss=objective(logits,presence,y,exists)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite guided loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                item=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.perf_counter()-tick);history.append(item);print('Guided EX',a.fold,item,flush=True)
            if step%500==0 or step==a.steps:
                torch.save(dict(config=config,step=step,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),out/'resume.tmp');(out/'resume.tmp').replace(resume)
        torch.save(dict(config=config,step=a.steps,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),final)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n');del optimizer,scheduler,x,y,exists,logits,presence,loss
    model.eval();proposal_folder=out/'proposals';proposal_folder.mkdir(exist_ok=True)
    development_ids=[]
    for record in records:
        sid=record['sample_id'];development_ids.append(sid);path=proposal_folder/f'{sid}.json'
        if not path.exists():path.write_text(json.dumps(infer(model,record,device))+'\n')
        print('Guided EX development inference',sid,flush=True)
    del records
    baseline=read_rows('outputs/stage32_contour_consensus/predicted_all.csv')
    geometry=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    matrices={r['sample_id']:np.array(r['matrix']) for r in json.loads(geometry.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    for sid in sorted(set(truth)-set(ids)):
        path=proposal_folder/f'{sid}.json'
        if path.exists():continue
        if sid not in matrices:value=dict(sample_id=sid,proposals=[])
        else:
            maps=np.load(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')['maps'].astype(np.float32)
            record=make_record(root,sid,baseline[sid],matrices[sid],maps)
            value=infer(model,record,device)
        path.write_text(json.dumps(value)+'\n');print('Guided EX held-out inference',sid,len(value['proposals']),flush=True)
    (out/'completion.json').write_text(json.dumps(dict(model_sha256=sha(final),development_ids=development_ids,held_out=subject,train_ids=ids,geometry_sha256=sha(geometry),baseline_sha256=sha('outputs/stage32_contour_consensus/predicted_all.csv'),heldout_maps_sha256={s:sha(f'outputs/stage7_tiles/maps/{s}_exvivo.npz') for s in sorted(set(truth)-set(ids))},proposals={p.stem:sha(p) for p in proposal_folder.glob('*.json')}),indent=2)+'\n')


if __name__=='__main__':main()
