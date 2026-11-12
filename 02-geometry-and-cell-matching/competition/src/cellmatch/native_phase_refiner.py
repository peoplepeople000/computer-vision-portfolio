"""Frozen16phase crop averaging for native EX; retain original ownership/component policy."""
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
        # Cover all four pooling phases in each axis, keeping global pixels aligned.
        sums=np.zeros((len(batchkeys),32,32),np.float64)
        weights=np.zeros_like(sums)
        for dy in [-1,0,1,2]:
            for dx in [-1,0,1,2]:
                shifted=[]; offsets=[]
                for key,(x0,y0) in zip(batchkeys,origins):
                    x1=int(np.clip(x0+dx,0,image.shape[1]-32));y1=int(np.clip(y0+dy,0,image.shape[0]-32))
                    shifted.append(np.concatenate([features[:,y1:y1+32,x1:x1+32],local_mask(masks[key],x1,y1)[None]]))
                    offsets.append((x1-x0,y1-y0))
                with torch.inference_mode():values=model(torch.from_numpy(np.stack(shifted)).to(device))[:,0].sigmoid().cpu().numpy()
                for n,((sx,sy),value) in enumerate(zip(offsets,values)):
                    left,right=max(0,sx),min(32,32+sx);top,bottom=max(0,sy),min(32,32+sy)
                    sums[n,top:bottom,left:right]+=value[top-sy:bottom-sy,left-sx:right-sx]
                    weights[n,top:bottom,left:right]+=1
        assert (weights>0).all()
        probs=sums/weights
        for key,(x0,y0),prob in zip(batchkeys,origins,probs):
            old=masks[key];prior=local_mask(old,x0,y0).astype(bool);labels,count=ndi.label(prob>.5)
            overlap=np.bincount(labels[prior],minlength=count+1);overlap[0]=0
            candidate=labels==int(overlap.argmax()) if overlap.max() else np.zeros_like(prior)
            block=occupied[y0:y0+32,x0:x0+32];candidate&=(block==0)|(block==numbers[key])
            encoded=patch_rle(candidate,x0,y0,image.shape);new=Mask.parse(encoded,image.shape)
            if new.area:
                row['exvivo_instances'][key]=encoded;block[candidate]=numbers[key];accepted.append(key)
    assert row['match_pairs']==base_row['match_pairs'] and row['invivo_instances']==base_row['invivo_instances']
    assert set(row['exvivo_instances'])==set(base_row['exvivo_instances']) and not set(accepted)&paired
    taken=np.zeros(image.shape,bool)
    for mask in parse_instances(row,'exvivo',image.shape).values():
        xy=mask.coordinates();assert not taken[xy[:,1],xy[:,0]].any();taken[xy[:,1],xy[:,0]]=True
    return row,dict(sample_id=base_row['sample_id'],eligible=keys,accepted=accepted,changed=sum(row['exvivo_instances'][k]!=base_row['exvivo_instances'][k] for k in keys))
