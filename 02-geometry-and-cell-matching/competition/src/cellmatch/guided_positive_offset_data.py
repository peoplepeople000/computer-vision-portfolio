"""Complete-target positive displacement sampling; inference is unchanged."""
import numpy as np
import torch
from .guided_ex_data import crop

STATS={'attempted':0,'accepted':0,'fallback':0}

def sample(records,rng,batch=16,wide_fraction=.5):
    groups={}
    for i,r in enumerate(records):
        for j,c in enumerate(r['cells']):groups.setdefault((r['subject'],bool(c['target'])),[]).append((i,j))
    subjects=sorted({r['subject'] for r in records});xs=[];ys=[];present=[]
    for _ in range(batch):
        subject=subjects[int(rng.integers(len(subjects)))];positive=bool(rng.random()<.7)
        candidates=groups.get((subject,positive)) or groups[(subject,not positive)]
        i,j=candidates[int(rng.integers(len(candidates)))];r=records[i];cell=r['cells'][j];chosen=None
        if cell['target'] and wide_fraction>0 and rng.random()<wide_fraction:
            STATS['attempted']+=1
            if '_target_areas' not in r:r['_target_areas']=np.bincount(r['labels'].ravel())
            full_area=r['_target_areas'][cell['target']]
            for _ in range(16):
                x,y,_=crop(r,cell,rng.uniform(-12,12,2),with_target=True)
                if y[0].sum()!=full_area:continue
                yy,xx=np.where(y[0]>0);radius=float(np.hypot(xx.mean()-24,yy.mean()-24))
                if 8<=radius<12:chosen=(x,y);STATS['accepted']+=1;break
            if chosen is None:STATS['fallback']+=1
        if chosen is None:x,y,_=crop(r,cell,rng.uniform(-4,4,2),with_target=True)
        else:x,y=chosen
        k=int(rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k,(1,2))
        if rng.random()<.5:x=x[:,:,::-1];y=y[:,:,::-1]
        x=x.copy();y=y.copy()
        for channel in [0,1,7,8]:x[channel]=np.clip(np.maximum(x[channel],0)**rng.uniform(.7,1.4)*rng.uniform(.8,1.2),0,1)
        xs.append(x);ys.append(y);present.append(float(y[0].any()))
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys)),torch.tensor(present,dtype=torch.float32)
