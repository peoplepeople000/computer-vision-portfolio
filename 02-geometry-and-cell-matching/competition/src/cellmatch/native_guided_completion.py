"""Fixed EX completion from native/IV-guided mask agreement, without labels."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from .data import parse_instances
from .rle import Mask


def complete(row,native,proposals,shape):
    native_masks={k:Mask.parse(v,shape) for k,v in native.items()}
    native_keys=list(native_masks)
    guided=[Mask.parse(q['rle'],shape) for q in proposals]
    overlaps=np.array([[a.iou(native_masks[k]) for k in native_keys] for a in guided]).reshape(len(guided),len(native_keys))
    ii,jj=linear_sum_assignment(overlaps,maximize=True)
    occupied=np.zeros(shape,bool)
    for mask in parse_instances(row,'exvivo',shape).values():
        xy=mask.coordinates();occupied[xy[:,1],xy[:,0]]=True
    result=copy.deepcopy(row);decisions=[]
    for i,j in sorted(zip(ii,jj),key=lambda ij:(-overlaps[ij],ij[0],ij[1])):
        if overlaps[i,j]<=.75:continue
        prior=native_masks[native_keys[j]];xy=prior.coordinates()
        blocked=bool(occupied[xy[:,1],xy[:,0]].any())
        key=f'EX_CONSENSUS_ADD_{j:06d}'
        accepted=not blocked
        if accepted:
            assert key not in result['exvivo_instances']
            result['exvivo_instances'][key]=native[native_keys[j]]
            occupied[xy[:,1],xy[:,0]]=True
        decisions.append(dict(key=key,native_key=native_keys[j],proposal=int(i),agreement=float(overlaps[i,j]),blocked=blocked,accepted=accepted))
    return result,decisions
