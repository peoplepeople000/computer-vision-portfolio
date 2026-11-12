"""Paired geometric scaling with sufficient native context and fixed anchor.
A72px native crop covers the48px output at every scale in[.7,1.4]. Image, IV
prior and EX target share a transform; the search anchor keeps its48px meaning.
"""
import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from .guided_ex_data import crop


def resize_crop(x,y,scale):
    assert x.shape==(9,72,72) and y.shape==(2,72,72) and .7<=scale<=1.4
    transform=np.array([[scale,0,24-scale*36],[0,scale,24-scale*36]],np.float64)
    output=np.stack([cv2.warpAffine(channel,transform,(48,48),flags=cv2.INTER_NEAREST if i==3 else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for i,channel in enumerate(x)])
    output[2]=1.
    yy,xx=np.mgrid[:48,:48];output[4]=np.exp(-((xx-24)**2+(yy-24)**2)/18.)
    fg=cv2.warpAffine(y[0],transform,(48,48),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>.5
    distance=ndi.distance_transform_edt(np.pad(fg,1))[1:-1,1:-1].astype(np.float32);distance/=max(float(distance.max()),1.)
    return output.astype(np.float32),np.stack([fg,distance]).astype(np.float32)


def sample(records,rng,scale_rng,batch=16):
    groups={}
    for i,r in enumerate(records):
        for j,c in enumerate(r['cells']):groups.setdefault((r['subject'],bool(c['target'])),[]).append((i,j))
    subjects=sorted({r['subject'] for r in records});xs=[];ys=[];present=[]
    for _ in range(batch):
        subject=subjects[int(rng.integers(len(subjects)))];positive=bool(rng.random()<.7)
        candidates=groups.get((subject,positive)) or groups[(subject,not positive)]
        i,j=candidates[int(rng.integers(len(candidates)))];r=records[i];cell=r['cells'][j]
        x,y,_=crop(r,cell,rng.uniform(-4,4,2),size=72,with_target=True)
        scale=float(np.exp(scale_rng.uniform(np.log(.7),np.log(1.4))))
        x,y=resize_crop(x,y,scale)
        k=int(rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k,(1,2))
        if rng.random()<.5:x=x[:,:,::-1];y=y[:,:,::-1]
        x=x.copy();y=y.copy()
        for channel in [0,1,7,8]:x[channel]=np.clip(np.maximum(x[channel],0)**rng.uniform(.7,1.4)*rng.uniform(.8,1.2),0,1)
        xs.append(x);ys.append(y);present.append(float(y[0].any()))
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys)),torch.tensor(present,dtype=torch.float32)
