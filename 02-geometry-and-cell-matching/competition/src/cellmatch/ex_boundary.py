"""Local mask-conditioned EX boundary refinement with fixed detections."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy import ndimage as ndi
from .learning import SmallUNet,image_features,loss_function,choose_device
from .data import read_rows,write_rows,parse_instances,read_image,image_path,split_id
from .rle import Mask
from .stage3 import FOLDS,development_signature
from .learning import instances_from_maps
from .rle import labels_to_instances
from .cli import evaluate


def origin(mask,size=32):
    x,y=np.round(mask.coordinates().mean(0)).astype(int)
    return int(np.clip(x-size//2,0,max(0,mask.shape[1]-size))),int(np.clip(y-size//2,0,max(0,mask.shape[0]-size)))


def local_mask(mask,x,y,size=32):
    patch=np.zeros((size,size),np.float32)
    if mask is not None:
        xy=mask.coordinates()-[x,y];valid=((xy>=0)&(xy<size)).all(1);xy=xy[valid]
        patch[xy[:,1],xy[:,0]]=1
    return patch


def example(features,proposal,target):
    x,y=origin(proposal);prior=local_mask(proposal,x,y);fg=local_mask(target,x,y)
    distance=ndi.distance_transform_edt(np.pad(fg,1))[1:-1,1:-1].astype(np.float32)
    distance/=max(float(distance.max()),1)
    return np.concatenate([features[:,y:y+32,x:x+32],prior[None]]).astype(np.float16),np.stack([fg,distance]).astype(np.float16)


def patch_rle(patch,x,y,shape):
    runs=[]
    for yy,line in enumerate(patch):
        edges=np.flatnonzero(np.diff(np.r_[False,line,False])).reshape(-1,2)
        for a,b in edges:runs.extend(((yy+y)*shape[1]+a+x,b-a))
    return ' '.join(map(str,runs))


def prepare(root,tag,truth):
    heldout=FOLDS[tag];folder=Path(f'outputs/stage3/{tag}')
    if json.loads((folder/'development_manifest.json').read_text())!=development_signature(f'outputs/stage2/unet_holdout_{tag}/model.pt'):
        raise ValueError('Stale development predictions')
    dev=read_rows(folder/'development.csv');expected={s for s in truth if split_id(s)[0]!=heldout}
    if set(dev)!=expected:raise ValueError('Development includes held-out mouse')
    calibration=json.loads(Path('outputs/stage4_calibrated/calibration.json').read_text())[heldout]
    if set(calibration['train_ids'])!=expected:raise ValueError('Wrong threshold calibration split')
    records=[];rng=np.random.default_rng(2029)
    for sid in sorted(dev):
        image=read_image(image_path(root,'training',sid,'exvivo'));features=image_features(image)
        actual=parse_instances(truth[sid],'exvivo',image.shape)
        maps=np.load(folder/'development_maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
        instances=labels_to_instances(instances_from_maps(maps,calibration['thresholds']['exvivo'],.35),'EXP')
        masks=parse_instances({'exvivo_instances':instances},'exvivo',image.shape)
        for proposal in masks.values():
            overlaps=[(proposal.iou(gt),key) for key,gt in actual.items()]
            score,key=max(overlaps,default=(0,None));target=actual[key] if score>.2 else None
            x,y=example(features,proposal,target);records.append((split_id(sid)[0],target is not None,x,y))
        # Synthetic imperfect contours cover annotated cells missed by the detector.
        for gt in actual.values():
            if not gt.area:continue
            x0,y0=origin(gt);patch=local_mask(gt,x0,y0).astype(bool)
            mode=int(rng.integers(3))
            if mode==0:patch=ndi.binary_dilation(patch)
            elif mode==1:
                eroded=ndi.binary_erosion(patch)
                if eroded.any():patch=eroded
            patch=ndi.shift(patch.astype(float),rng.integers(-1,2,2),order=0,mode='constant')>0
            if not patch.any():continue
            proposal=Mask.parse(patch_rle(patch,x0,y0,image.shape),image.shape)
            x,y=example(features,proposal,gt);records.append((split_id(sid)[0],True,x,y))
        print('Boundary training patches',tag,sid,len(records),flush=True)
    return records,sorted(expected)


def sample(records,rng,batch=32):
    groups=sorted({r[0] for r in records});xs=[];ys=[]
    for _ in range(batch):
        subject=groups[int(rng.integers(len(groups)))];positive=rng.random()<.8
        choices=records[(subject,positive)] if isinstance(records,dict) else [r for r in records if r[0]==subject and r[1]==positive]
        if not choices:raise ValueError('Empty positive/negative group')
        _,_,a,b=choices[int(rng.integers(len(choices)))];x=a.astype(np.float32);y=b.astype(np.float32)
        k=int(rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k,(1,2))
        if rng.random()<.5:x=x[:,:,::-1];y=y[:,:,::-1]
        x=x.copy();x[:2]=np.clip(x[:2]**rng.uniform(.8,1.3)*rng.uniform(.85,1.15),0,1)
        xs.append(x);ys.append(y)
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys))


def refine(masks,features,model,device):
    keys=list(masks);inputs=[];origins=[]
    for key in keys:
        m=masks[key];x,y=origin(m);origins.append((x,y));prior=local_mask(m,x,y)
        inputs.append(np.concatenate([features[:,y:y+32,x:x+32],prior[None]]))
    probabilities=[]
    with torch.inference_mode():
        for i in range(0,len(inputs),64):
            probabilities.extend(model(torch.from_numpy(np.stack(inputs[i:i+64])).to(device))[:,:1].sigmoid().cpu().numpy()[:,0])
    occupied=np.zeros(features.shape[1:],np.int32)
    for n,m in enumerate(masks.values(),1):
        for a,b in zip(m.starts,m.ends):occupied.ravel()[a:b]=n
    result={};accepted=0
    for n,(key,prob,(x,y)) in enumerate(zip(keys,probabilities,origins),1):
        old=masks[key];labels,count=ndi.label(prob>.5);prior=local_mask(old,x,y).astype(bool)
        overlaps=np.bincount(labels[prior],minlength=count+1);overlaps[0]=0
        candidate=labels==int(np.argmax(overlaps)) if overlaps.max()>0 else np.zeros_like(prior)
        candidate &= (occupied[y:y+32,x:x+32]==0)|(occupied[y:y+32,x:x+32]==n)
        encoded=patch_rle(candidate,x,y,old.shape);new=Mask.parse(encoded,old.shape)
        if new.area and .5<=new.area/max(old.area,1)<=1.7 and new.iou(old)>.4:
            result[key]=encoded;occupied[y:y+32,x:x+32][candidate]=n;accepted+=1
        else:result[key]=' '.join(f'{a} {b-a}' for a,b in zip(old.starts,old.ends))
    return result,accepted


def main():
    torch.set_num_threads(2);device=choose_device('mps');root=Path('data/raw/Project_2_Dataset')
    truth=read_rows(root/'training/train_ground_truth.csv');baseline=read_rows('outputs/stage5_hybrid_geometry/predicted_all.csv')
    out=Path('outputs/stage6_boundary');out.mkdir(exist_ok=True);allrows=[];diagnostics=[]
    for tag,subject in FOLDS.items():
        folder=out/tag;folder.mkdir(exist_ok=True)
        if (folder/'model.pt').exists():raise FileExistsError('Preserving previous boundary model')
        records,ids=prepare(root,tag,truth)
        records={key:[r for r in records if (r[0],r[1])==key] for key in sorted({(r[0],r[1]) for r in records})}
        torch.manual_seed(2029);rng=np.random.default_rng(2029)
        model=SmallUNet(8).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1200,eta_min=.0001);history=[];start=time.perf_counter()
        model.train()
        for step in range(1,1201):
            x,y=sample(records,rng);optimizer.zero_grad(set_to_none=True);loss=loss_function(model(x.to(device)),y.to(device))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
            value=float(loss.detach().cpu())
            if not np.isfinite(value):raise RuntimeError('Nonfinite boundary loss')
            if step==1 or step%200==0:
                history.append({'step':step,'loss':value,'seconds':time.perf_counter()-start});print('Boundary',tag,history[-1],flush=True)
        state={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'width':8,'held_out':subject,'train_ids':ids,
               'steps':1200,'seed':2029,'patch':32,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        torch.save(state,folder/'model.pt');(folder/'history.json').write_text(json.dumps(history,indent=2)+'\n');del records;model.eval()
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            image=read_image(image_path(root,'training',sid,'exvivo'));masks=parse_instances(row,'exvivo',image.shape)
            instances,accepted=refine(masks,image_features(image),model,device)
            allrows.append({'sample_id':sid,'invivo_instances':row['invivo_instances'],'exvivo_instances':instances,'match_pairs':[]})
            diagnostics.append({'sample_id':sid,'accepted_boundary_updates':accepted,'instances':len(masks)})
        del model;torch.mps.empty_cache()
    path=out/'segmentation_oof.csv';write_rows(allrows,path)
    (out/'manifest.json').write_text(json.dumps({'policy':'Fixed detections; local boundary model; fall back to original on invalid/large changes',
        'regions':diagnostics,'training':'80% positive and 20% negative patches, original development proposals plus synthetic training-GT contour perturbations',
        'folds':FOLDS},indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
