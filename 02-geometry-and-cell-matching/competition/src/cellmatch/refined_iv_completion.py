"""Frozen Stage328 added-IV-only contour operator."""
import copy
import numpy as np
import torch
from scipy import ndimage as ndi
from .data import parse_instances
from .learning import image_features
from .ex_boundary import origin,local_mask,patch_rle
from .rle import Mask

def refine_added(oldrow, keys, image, model, device):
    features=image_features(image);masks=parse_instances(oldrow,'invivo',image.shape);row=copy.deepcopy(oldrow);owner=np.zeros(image.shape,np.int32);numbers={k:n for n,k in enumerate(masks,1)}
    for k,m in masks.items():
        xy=m.coordinates();owner[xy[:,1],xy[:,0]]=numbers[k]
    patches=[];origins=[]
    for k in keys:
     x,y=origin(masks[k]);origins.append((x,y));patches.append(np.concatenate([features[:,y:y+32,x:x+32],local_mask(masks[k],x,y)[None]]))
    probabilities=[]
    with torch.inference_mode():
     for n in range(0,len(keys),64):probabilities.extend(model(torch.from_numpy(np.stack(patches[n:n+64])).to(device))[:,:1].sigmoid().cpu().numpy()[:,0])
    accepted=[];changed=[]
    for k,prob,(x,y) in zip(keys,probabilities,origins):
     old=masks[k];prior=local_mask(old,x,y).astype(bool);labels,count=ndi.label(prob>.5);overlap=np.bincount(labels[prior],minlength=count+1);overlap[0]=0
     candidate=labels==int(overlap.argmax()) if overlap.max()>0 else np.zeros_like(prior);block=owner[y:y+32,x:x+32];candidate&=(block==0)|(block==numbers[k]);encoded=patch_rle(candidate,x,y,image.shape);new=Mask.parse(encoded,image.shape)
     if new.area and .5<=new.area/old.area<=1.7 and new.iou(old)>.4:
      row['invivo_instances'][k]=encoded;block[candidate]=numbers[k];accepted.append(k)
      if encoded!=oldrow['invivo_instances'][k]:changed.append(k)
    return row, dict(sample_id=oldrow["sample_id"], accepted=accepted, changed=changed), np.array(probabilities,dtype=np.float32).reshape(-1,32,32), np.array(origins,dtype=np.int32).reshape(-1,2)
