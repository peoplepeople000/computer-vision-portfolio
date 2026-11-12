"""EX-conditioned IV crops for missing-IV recovery (research preparation).

The legacy crop API calls its source image `iv_image` and destination `ex_image`.
Here those storage slots deliberately contain EX and IV respectively. The affine
is inverted once, and supervision maps a predicted EX cell to a verified IV cell.
Held-out calls pass no truth and cannot construct labels or target identities.
"""
import numpy as np
from .data import parse_instances,split_id
from .learning import image_features
from .guided_ex_data import cell_record
from .metrics import segmentation_score
from .cellpose_finetune import instance_labels

def invert_affine(matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.shape!=(2,3) or not np.isfinite(matrix).all():raise ValueError('Expected finite 2x3 IV-to-EX affine')
    return np.linalg.inv(np.vstack([matrix,[0.,0.,1.]]))[:2]

def reverse_record(sid,iv_image,ex_image,ex_instances,iv_to_ex,truth=None):
    bank=parse_instances({'exvivo_instances':ex_instances},'exvivo',ex_image.shape)
    numbers={k:n for n,k in enumerate(truth['invivo_instances'],1)} if truth is not None else {}
    reverse_pairs={e:i for i,e in truth['match_pairs']} if truth is not None else {}
    mapping=segmentation_score(bank,parse_instances(truth,'exvivo',ex_image.shape))['pred_to_gt'] if truth is not None else {}
    cells=[cell_record(k,m,numbers.get(reverse_pairs.get(mapping.get(k)),0)) for k,m in bank.items()]
    labels=instance_labels(truth['invivo_instances'],iv_image.shape) if truth is not None else None
    return dict(sample_id=sid,subject=split_id(sid)[0],matrix=invert_affine(iv_to_ex),iv_image=image_features(ex_image),ex_image=image_features(iv_image),ex_maps=np.zeros((2,*iv_image.shape),np.float32),labels=labels,cells=cells,shape=iv_image.shape)
