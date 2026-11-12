"""Boundary-aware fine-tuning with training-only hard-background sampling."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .learning import SmallUNet,load_training,choose_device,predict
from .data import read_rows,parse_instances,write_rows
from .stage3 import FOLDS,development_signature
from .cli import evaluate


def boundary_loss(logits,target):
    foreground=target[:,:1];probability=logits.sigmoid()
    local=F.avg_pool2d(foreground,3,stride=1,padding=1)
    boundary=((local>0)&(local<1)).float()
    weight=1+3*boundary
    bce=F.binary_cross_entropy_with_logits(logits[:,:1],foreground,reduction='none',
        pos_weight=torch.tensor(1.5,device=logits.device))
    bce=(bce*weight).sum()/weight.sum()
    dims=(1,2,3)
    dice=1-((2*(probability[:,:1]*foreground).sum(dims)+1)/
        (probability[:,:1].sum(dims)+foreground.sum(dims)+1)).mean()
    distance=(F.smooth_l1_loss(probability[:,1:],target[:,1:],reduction='none')*(1+4*foreground)).mean()
    return bce+dice+2*distance


def attach_hard_centers(items,development):
    for item in items:
        mode='invivo' if item['mode']==0 else 'exvivo';sid=item['sample_id']
        shape=item['y'].shape[1:]
        masks=parse_instances(development[sid],mode,shape)
        centers=[]
        for mask in masks.values():
            if not mask.area:continue
            x,y=mask.coordinates().mean(0)
            if item['y'][0,int(round(y)),int(round(x))]==0:centers.append([y,x])
        item['hard_centers']=np.asarray(centers).reshape(-1,2)


def sample_refinement(items,batch,patch,rng):
    groups=sorted({(r['subject'],r['mode']) for r in items});xs=[];ys=[]
    for _ in range(batch):
        subject,mode=groups[int(rng.integers(len(groups)))]
        candidates=[r for r in items if (r['subject'],r['mode'])==(subject,mode)]
        item=candidates[int(rng.integers(len(candidates)))];height,width=item['x'].shape[1:]
        scale=rng.uniform(.85,1.4);size=int(round(patch/scale));choice=rng.random()
        candidates=item['hard_centers'] if choice<.3 else item['centers']
        if choice<.85 and len(candidates):
            cy,cx=candidates[int(rng.integers(len(candidates)))]+rng.uniform(-size/3,size/3,2)
        else:cy,cx=rng.uniform(0,height),rng.uniform(0,width)
        y=int(np.clip(cy-size/2,0,max(0,height-size)));x=int(np.clip(cx-size/2,0,max(0,width-size)))
        feature=item['x'][:,y:y+size,x:x+size].astype(np.float32)
        target=item['y'][:,y:y+size,x:x+size].astype(np.float32)
        pad=((0,0),(0,size-feature.shape[1]),(0,size-feature.shape[2]))
        feature=np.pad(feature,pad,mode='reflect');target=np.pad(target,pad)
        feature=np.stack([cv2.resize(f,(patch,patch),interpolation=cv2.INTER_LINEAR) for f in feature])
        target=np.stack([cv2.resize(target[0],(patch,patch),interpolation=cv2.INTER_NEAREST),
                         cv2.resize(target[1],(patch,patch),interpolation=cv2.INTER_LINEAR)])
        k=int(rng.integers(4));feature=np.rot90(feature,k,(1,2));target=np.rot90(target,k,(1,2))
        if rng.random()<.5:feature=feature[:,:,::-1];target=target[:,:,::-1]
        feature=np.clip(feature**rng.uniform(.7,1.4)*rng.uniform(.8,1.2)+rng.normal(0,.01,feature.shape),0,1).astype(np.float32)
        xs.append(np.concatenate([feature,np.full((1,patch,patch),mode,np.float32)]));ys.append(target)
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys))


def train_fold(args,tag):
    heldout=FOLDS[tag];device=choose_device(args.device)
    output=Path(args.output)/f'unet_holdout_{tag}';output.mkdir(parents=True,exist_ok=True)
    if (output/'model.pt').exists():raise FileExistsError('Choose a new output; existing checkpoints are preserved')
    checkpoint_path=Path(f'outputs/stage2/unet_holdout_{tag}/model.pt')
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if state['held_out']!=heldout:raise ValueError('Wrong initializer fold')
    items,train_ids=load_training(args.data,heldout)
    if set(train_ids)!=set(state['train_ids']):raise ValueError('Initializer training IDs differ')
    folder=Path(f'outputs/stage3/{tag}')
    if json.loads((folder/'development_manifest.json').read_text())!=development_signature(checkpoint_path):raise ValueError('Wrong hard-negative cache provenance')
    development=read_rows(folder/'development.csv')
    if set(development)!=set(train_ids):raise ValueError('Hard-negative cache contains wrong subjects')
    attach_hard_centers(items,development)
    torch.manual_seed(2027);rng=np.random.default_rng(2027)
    model=SmallUNet(state['width']).to(device);model.load_state_dict(state['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=.00003)
    config={'held_out':heldout,'train_ids':train_ids,'steps':args.steps,'seed':2027,'device':device,
            'initializer_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'loss':'boundary-weighted BCE (positive weight 1.5), Dice, normalized distance',
            'sampling':'balanced subject/modality; 30% hard-center, 55% positive-center, 15% random; fallback when centers absent',
            'scale_range':[.85,1.4],'hard_centers':sum(len(r['hard_centers']) for r in items)}
    (output/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
    history=[];start=time.perf_counter();model.train()
    for step in range(1,args.steps+1):
        x,y=sample_refinement(items,8,128,rng);optimizer.zero_grad(set_to_none=True)
        loss=boundary_loss(model(x.to(device)),y.to(device));loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        value=float(loss.detach().cpu())
        if not np.isfinite(value):raise RuntimeError('Nonfinite loss')
        history.append({'step':step,'loss':value,'seconds':time.perf_counter()-start})
        if step==1 or step%100==0:print(tag,history[-1],flush=True)
    checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'width':state['width'],
                'step':state['step']+args.steps,'held_out':heldout,'train_ids':train_ids,'config':config}
    torch.save(checkpoint,output/'model.pt');(output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    del model,items
    if device=='mps':torch.mps.empty_cache()
    path=Path(args.output)/f'unet_holdout_{tag}.csv'
    predict(SimpleNamespace(data=args.data,checkpoint=str(output/'model.pt'),split='training',subject=heldout,
        allow_training_subjects=False,output=str(path),threshold=.5,seed_threshold=.35,device=device,action='predict'))
    evaluate(SimpleNamespace(data=args.data,predictions=path,subject=heldout,output=Path(args.output)/f'unet_holdout_{tag}_metrics.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/raw/Project_2_Dataset');parser.add_argument('--output',default='outputs/stage5_refinement')
    parser.add_argument('--device',default='auto');parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--fold',choices=['all',*FOLDS],default='all');args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    for tag in (FOLDS if args.fold=='all' else [args.fold]):train_fold(args,tag)
    if args.fold=='all':
        rows=[]
        for tag in FOLDS:rows.extend(read_rows(Path(args.output)/f'unet_holdout_{tag}.csv').values())
        path=Path(args.output)/'segmentation_oof.csv';write_rows(rows,path)
        evaluate(SimpleNamespace(data=args.data,predictions=path,subject=None,output=Path(args.output)/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
