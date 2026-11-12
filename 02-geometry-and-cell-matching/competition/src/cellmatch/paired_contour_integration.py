"""Integrate fixed paired contour proposals without changing any identity/link."""
import copy
import numpy as np
from .data import parse_instances
from .rle import Mask


def integrate(row, proposal, original_paired_ids, shape):
    result=copy.deepcopy(row)
    masks=parse_instances(row,'exvivo',shape)
    numbers={k:n for n,k in enumerate(masks,1)}
    owners=np.zeros(shape,np.int32)
    for key,mask in masks.items():
        xy=mask.coordinates()
        if owners[xy[:,1],xy[:,0]].any():
            raise ValueError('Input masks overlap')
        owners[xy[:,1],xy[:,0]]=numbers[key]
    current_paired={key for _,key in row['match_pairs']}
    eligible=sorted(current_paired & set(original_paired_ids))
    decisions=[]
    for key in eligible:
        encoded=proposal['exvivo_instances'][key]
        mask=Mask.parse(encoded,shape);xy=mask.coordinates()
        blocked=bool(((owners[xy[:,1],xy[:,0]]!=0)&(owners[xy[:,1],xy[:,0]]!=numbers[key])).any())
        accepted=bool(mask.area and not blocked)
        if accepted:
            result['exvivo_instances'][key]=encoded
            # Keep original occupied support as a conservative sequential guard.
            owners[xy[:,1],xy[:,0]]=numbers[key]
        decisions.append(dict(key=key,blocked=blocked,accepted=accepted,
                              changed=result['exvivo_instances'][key]!=row['exvivo_instances'][key]))
    return result,decisions
