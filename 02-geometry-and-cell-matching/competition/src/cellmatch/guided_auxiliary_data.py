"""Original guided sampling plus an all-instance EX auxiliary target."""
import numpy as np
import torch
from scipy import ndimage as ndi
from .guided_ex_data import crop as target_crop,cut

def crop(record,cell,shift=(0.,0.),size=48,with_target=False):
 x,y,origin=target_crop(record,cell,shift,size,with_target)
 if not with_target:return x,y,origin
 labels=cut(record['labels'],*origin,size)[0];fg=labels>0;distance=np.zeros(labels.shape,np.float32)
 for value in np.unique(labels):
  if value==0:continue
  mask=labels==value;d=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1].astype(np.float32);d/=max(float(d.max()),1.);distance[mask]=d[mask]
 assert np.all(y[0]<=fg)
 return x,np.concatenate([y,np.stack([fg,distance]).astype(np.float32)]),origin

def sample(records,rng,batch=16):
 groups={}
 for i,r in enumerate(records):
  for j,c in enumerate(r['cells']):groups.setdefault((r['subject'],bool(c['target'])),[]).append((i,j))
 subjects=sorted({r['subject'] for r in records});xs=[];ys=[];present=[]
 for _ in range(batch):
  subject=subjects[int(rng.integers(len(subjects)))];positive=bool(rng.random()<.7)
  candidates=groups.get((subject,positive)) or groups[(subject,not positive)]
  i,j=candidates[int(rng.integers(len(candidates)))];r=records[i];cell=r['cells'][j]
  x,y,_=crop(r,cell,rng.uniform(-4,4,2),with_target=True)
  k=int(rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k,(1,2))
  if rng.random()<.5:x=x[:,:,::-1];y=y[:,:,::-1]
  x=x.copy();y=y.copy()
  for channel in [0,1,7,8]:x[channel]=np.clip(np.maximum(x[channel],0)**rng.uniform(.7,1.4)*rng.uniform(.8,1.2),0,1)
  xs.append(x);ys.append(y);present.append(float(y[0].any()))
 return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys)),torch.tensor(present,dtype=torch.float32)
