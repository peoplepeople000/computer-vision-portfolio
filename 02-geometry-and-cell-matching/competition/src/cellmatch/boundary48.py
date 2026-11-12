"""48-pixel local instance contour helpers; preserve IDs and occupied neighbors."""
import numpy as np
import torch
from scipy import ndimage as ndi
from .rle import Mask

def origin(mask,size=48):
    x,y=np.round(mask.coordinates().mean(0)).astype(int)
    return int(np.clip(x-size//2,0,max(0,mask.shape[1]-size))),int(np.clip(y-size//2,0,max(0,mask.shape[0]-size)))


def local_mask(mask,x,y,size=48):
    patch=np.zeros((size,size),np.float32)
    if mask is not None:
        xy=mask.coordinates()-[x,y];valid=((xy>=0)&(xy<size)).all(1);xy=xy[valid]
        patch[xy[:,1],xy[:,0]]=1
    return patch


def example(features,proposal,target):
    x,y=origin(proposal);prior=local_mask(proposal,x,y);fg=local_mask(target,x,y)
    distance=ndi.distance_transform_edt(np.pad(fg,1))[1:-1,1:-1].astype(np.float32)
    distance/=max(float(distance.max()),1)
    return np.concatenate([features[:,y:y+48,x:x+48],prior[None]]).astype(np.float16),np.stack([fg,distance]).astype(np.float16)


def patch_rle(patch,x,y,shape):
    runs=[]
    for yy,line in enumerate(patch):
        edges=np.flatnonzero(np.diff(np.r_[False,line,False])).reshape(-1,2)
        for a,b in edges:runs.extend(((yy+y)*shape[1]+a+x,b-a))
    return ' '.join(map(str,runs))


def refine(masks,features,model,device):
    keys=list(masks);inputs=[];origins=[]
    for key in keys:
        m=masks[key];x,y=origin(m);origins.append((x,y));prior=local_mask(m,x,y)
        inputs.append(np.concatenate([features[:,y:y+48,x:x+48],prior[None]]))
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
        candidate &= (occupied[y:y+48,x:x+48]==0)|(occupied[y:y+48,x:x+48]==n)
        encoded=patch_rle(candidate,x,y,old.shape);new=Mask.parse(encoded,old.shape)
        if new.area and .5<=new.area/max(old.area,1)<=1.7 and new.iou(old)>.4:
            result[key]=encoded;occupied[y:y+48,x:x+48][candidate]=n;accepted+=1
        else:result[key]=' '.join(f'{a} {b-a}' for a,b in zip(old.starts,old.ends))
    return result,accepted

