"""Frozen Stage231 native EX contour operator, extracted without policy changes."""
import copy
import numpy as np
import torch
from scipy import ndimage as ndi
from .learning import image_features
from .data import parse_instances
from .ex_boundary import origin,local_mask,patch_rle
from .rle import Mask


def refine_row(base_row,source_row,image,model,device):
    features=image_features(image);row=copy.deepcopy(base_row);masks=parse_instances(row,'exvivo',image.shape)
    paired={e for i,e in row['match_pairs']};rles=set(source_row['exvivo_instances'].values())
    keys=[k for k in masks if k not in paired and row['exvivo_instances'][k] in rles]
    occupied=np.zeros(image.shape,np.int32);numbers={k:n for n,k in enumerate(masks,1)}
    for k,m in masks.items():
        for start,end in zip(m.starts,m.ends):occupied.ravel()[start:end]=numbers[k]
    accepted=[]
    for offset in range(0,len(keys),64):
        batchkeys=keys[offset:offset+64];patches=[];origins=[]
        for key in batchkeys:
            x0,y0=origin(masks[key]);origins.append((x0,y0));prior=local_mask(masks[key],x0,y0)
            patches.append(np.concatenate([features[:,y0:y0+32,x0:x0+32],prior[None]]))
        with torch.inference_mode():probs=model(torch.from_numpy(np.stack(patches)).to(device))[:,0].sigmoid().cpu().numpy()
        for key,(x0,y0),prob in zip(batchkeys,origins,probs):
            old=masks[key];prior=local_mask(old,x0,y0).astype(bool);labels,count=ndi.label(prob>.5)
            overlap=np.bincount(labels[prior],minlength=count+1);overlap[0]=0
            candidate=labels==int(overlap.argmax()) if overlap.max() else np.zeros_like(prior)
            block=occupied[y0:y0+32,x0:x0+32];candidate&=(block==0)|(block==numbers[key])
            encoded=patch_rle(candidate,x0,y0,image.shape);new=Mask.parse(encoded,image.shape)
            if new.area and .5<=new.area/old.area<=1.7 and new.iou(old)>.4:
                row['exvivo_instances'][key]=encoded;block[candidate]=numbers[key];accepted.append(key)
    assert row['match_pairs']==base_row['match_pairs'] and row['invivo_instances']==base_row['invivo_instances']
    assert set(row['exvivo_instances'])==set(base_row['exvivo_instances']) and not set(accepted)&paired
    taken=np.zeros(image.shape,bool)
    for mask in parse_instances(row,'exvivo',image.shape).values():
        xy=mask.coordinates();assert not taken[xy[:,1],xy[:,0]].any();taken[xy[:,1],xy[:,0]]=True
    return row,dict(sample_id=base_row['sample_id'],eligible=keys,accepted=accepted,changed=sum(row['exvivo_instances'][k]!=base_row['exvivo_instances'][k] for k in keys))
