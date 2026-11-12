"""Small image-patch correspondence model with subject-held-out retrieval tests."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from .data import read_rows,read_image,image_path,parse_instances,split_id
from .learning import image_features,choose_device
from .stage3 import FOLDS


class PatchDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        def tower():
            layers=[]
            for incoming,outgoing in [(3,16),(16,32),(32,64)]:
                layers.extend([nn.Conv2d(incoming,outgoing,3,stride=2,padding=1),nn.GroupNorm(4,outgoing),nn.SiLU()])
            return nn.Sequential(*layers,nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(64*16,64))
        self.towers=nn.ModuleList([tower(),tower()])
    def forward(self,x,mode):
        return F.normalize(self.towers[mode](x),dim=-1)


def patches(image,masks,size=64):
    features=image_features(image)
    keys=[k for k,m in masks.items() if m.area]
    result=[]
    for key in keys:
        xy=masks[key].coordinates();cx,cy=xy.mean(0)
        yy,xx=np.mgrid[:size,:size].astype(np.float32)
        yy+=cy-size//2;xx+=cx-size//2
        crop=np.stack([ndi.map_coordinates(f,[yy,xx],order=1,mode='reflect') for f in features])
        low=np.quantile(crop,.05,axis=(1,2),keepdims=True)
        high=np.quantile(crop,.95,axis=(1,2),keepdims=True)
        crop=np.clip((crop-low)/np.maximum(high-low,.03),0,1)
        mask=np.zeros((size,size),np.float32)
        local=np.rint(xy-[cx-size//2,cy-size//2]).astype(int)
        keep=(local>=0).all(1)&(local<size).all(1)
        mask[local[keep,1],local[keep,0]]=1
        result.append(np.concatenate([crop,mask[None]]))
    return keys,np.asarray(result,np.float16).reshape(-1,3,size,size)


def encode(model,x,mode,device):
    output=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            output.append(model(torch.from_numpy(x[start:start+128].astype(np.float32)).to(device),mode).cpu().numpy())
    return np.concatenate(output) if output else np.empty((0,64))


def augment(x,rng,k,flip):
    x=np.rot90(x.astype(np.float32),k,axes=(2,3)).copy()
    if flip:x=x[:,:,:,::-1].copy()
    x[:,:2]=np.clip(x[:,:2]**rng.uniform(.75,1.3,(len(x),1,1,1))+
                         rng.normal(0,.02,x[:,:2].shape),0,1)
    return torch.from_numpy(x)


def train_fold(records,heldout,steps,device,output):
    train=[r for r in records if r['subject']!=heldout and len(r['pairs'])>=2]
    if not train:raise ValueError('No training pairs')
    torch.manual_seed(2026);rng=np.random.default_rng(2026)
    model=PatchDescriptor().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.00007)
    groups=[]
    for record in train:
        groups.append([record['patches'][mode][[record['keys'][mode].index(p[mode]) for p in record['pairs']]] for mode in range(2)])
    history=[];start=time.perf_counter();model.train()
    for step in range(steps):
        a,b=groups[int(rng.integers(len(groups)))];ids=rng.choice(len(a),min(len(a),32),replace=False)
        k=int(rng.integers(4));flip=bool(rng.integers(2))
        # Shared spatial transform, independent noise/intensity perturbations.
        x=augment(a[ids],rng,k,flip);y=augment(b[ids],rng,k,flip)
        logits=model(x.to(device),0)@model(y.to(device),1).T/.12
        target=torch.arange(len(ids),device=device)
        loss=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        if step%100==0 or step==steps-1:
            history.append({'step':step+1,'loss':float(loss.detach().cpu()),'seconds':time.perf_counter()-start})
            print(heldout,history[-1],flush=True)
    output.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'held_out':heldout,
                'train_ids':[r['sample_id'] for r in train],'steps':steps},output/'model.pt')
    model.eval();reports={}
    for group in ['training','held_out']:
        ranks=[];chance=[]
        for record in records:
            if (record['subject']==heldout)!=(group=='held_out'):continue
            z=[encode(model,record['patches'][mode],mode,device) for mode in range(2)]
            similarity=z[0]@z[1].T
            for iv,ex in record['pairs']:
                i=record['keys'][0].index(iv);j=record['keys'][1].index(ex)
                ranks.append(1+int((similarity[:,j]>similarity[i,j]).sum()));chance.append(5/len(z[0]))
        ranks=np.asarray(ranks)
        reports[group]={'pairs':len(ranks),'top1':float((ranks<=1).mean()),'top5':float((ranks<=5).mean()),
                        'top10':float((ranks<=10).mean()),'median_rank':float(np.median(ranks)),
                        'random_top5':float(np.mean(chance))}
    reports.update(held_out_subject=heldout,history=history,steps=steps,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   note='Oracle-mask retrieval diagnostic only; held-out masks do not enter fitting')
    (output/'retrieval.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps({k:reports[k] for k in ['held_out_subject','training','held_out']}),flush=True)
    del model
    if device=='mps':torch.mps.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto');p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--output',default='outputs/stage4_patches');args=p.parse_args()
    device=choose_device(args.device);torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    records=[]
    for sid,row in truth.items():
        record={'sample_id':sid,'subject':split_id(sid)[0],'keys':[],'patches':[],'pairs':row['match_pairs']}
        for modality in ['invivo','exvivo']:
            image=read_image(image_path(root,'training',sid,modality));masks=parse_instances(row,modality,image.shape)
            keys,x=patches(image,masks);record['keys'].append(keys);record['patches'].append(x)
        records.append(record)
    for tag,heldout in FOLDS.items():
        output=Path(args.output)/tag
        if (output/'model.pt').exists():raise FileExistsError('Use a new experiment directory')
        train_fold(records,heldout,args.steps,device,output)


if __name__=='__main__':main()
