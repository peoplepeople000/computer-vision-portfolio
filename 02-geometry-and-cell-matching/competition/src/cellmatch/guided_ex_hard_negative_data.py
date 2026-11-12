"""Guided EX crops with development-only deliberately misregistered negatives.

For 20% of sampled verified cells, translate the projected IV prompt by 40--100
pixels. The original EX target remains the only eligible target; no unrelated
EX cell is relabeled as the match. Reject shifts where that target enters crop.
"""
import numpy as np,torch
from .guided_ex_data import training_records,make_record,crop


def shifted_example(record,cell,rng,hard=False):
    if hard:
        for _ in range(12):
            angle=rng.uniform(0,2*np.pi);distance=rng.uniform(40,100);shift=np.array([np.cos(angle),np.sin(angle)])*distance
            x,y,origin=crop(record,cell,shift,with_target=True)
            # Keep an informative in-image crop and verify the true instance is absent.
            cx,cy=np.array(origin)+24
            if 24<=cx<record['shape'][1]-24 and 24<=cy<record['shape'][0]-24 and not y[0].any():return x,y
    x,y,_=crop(record,cell,rng.uniform(-4,4,2),with_target=True)
    return x,y


def sample(records,rng,batch=16):
    groups={}
    for i,r in enumerate(records):
        for j,c in enumerate(r['cells']):groups.setdefault((r['subject'],bool(c['target'])),[]).append((i,j))
    subjects=sorted({r['subject'] for r in records});xs=[];ys=[];present=[]
    for _ in range(batch):
        subject=subjects[int(rng.integers(len(subjects)))];positive=bool(rng.random()<.7)
        candidates=groups.get((subject,positive)) or groups[(subject,not positive)]
        i,j=candidates[int(rng.integers(len(candidates)))];r=records[i];cell=r['cells'][j]
        x,y=shifted_example(r,cell,rng,hard=bool(cell['target'] and rng.random()<.2))
        k=int(rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k,(1,2))
        if rng.random()<.5:x=x[:,:,::-1];y=y[:,:,::-1]
        x=x.copy();y=y.copy()
        for channel in [0,1,7,8]:x[channel]=np.clip(np.maximum(x[channel],0)**rng.uniform(.7,1.4)*rng.uniform(.8,1.2),0,1)
        xs.append(x);ys.append(y);present.append(float(y[0].any()))
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys)),torch.tensor(present,dtype=torch.float32)
