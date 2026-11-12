"""Frozen Stage325 public/LoRA IV completion rule, without annotations."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from .data import parse_instances
from .cellpose_contour_consensus import blend

def complete(old, public, lora, shape):
    assert old['sample_id']==public['sample_id']==lora['sample_id']
    a=parse_instances(public,'invivo',shape);b=parse_instances(lora,'invivo',shape)
    ak,bk=list(a),list(b)
    ious=np.array([[a[i].iou(b[j]) for j in bk] for i in ak]).reshape(len(ak),len(bk))
    ii,jj=linear_sum_assignment(ious,maximize=True)
    mixed,_=blend(public,lora,shape);candidates=parse_instances(mixed,'invivo',shape)
    row=copy.deepcopy(old);owner=np.zeros(shape,bool)
    for m in parse_instances(old,'invivo',shape).values():
        xy=m.coordinates();owner[xy[:,1],xy[:,0]]=True
    added=[];screen=[]
    for i,j in sorted(zip(ii,jj),key=lambda ij:(-ious[ij],ij[0])):
        if ious[i,j]<=.75:continue
        proposal=candidates[ak[i]];xy=proposal.coordinates();blocked=bool(owner[xy[:,1],xy[:,0]].any())
        screen.append(dict(public_key=ak[i],lora_key=bk[j],agreement=float(ious[i,j]),blocked=blocked))
        if blocked:continue
        key=f'CONSENSUS_ADD_{i:06d}';assert key not in row['invivo_instances']
        row['invivo_instances'][key]=mixed['invivo_instances'][ak[i]];owner[xy[:,1],xy[:,0]]=True
        added.append(dict(key=key,public_key=ak[i],lora_key=bk[j],agreement=float(ious[i,j])))
    return row,dict(sample_id=old['sample_id'],added=added,screen=screen)
