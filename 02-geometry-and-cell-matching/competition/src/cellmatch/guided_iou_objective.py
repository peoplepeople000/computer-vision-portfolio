"""Replace guided per-instance Dice with unit Lovasz hinge; other terms unchanged."""
import torch
from torch.nn import functional as F
from .iou_flow_loss import lovasz_hinge

def objective(logits,presence,target,exists):
 fg,center=target[:,:1],target[:,1:]
 probability=logits.sigmoid()
 bce=F.binary_cross_entropy_with_logits(logits[:,:1],fg,pos_weight=torch.tensor(2.,device=logits.device))
 overlap=lovasz_hinge(logits[:,0],fg[:,0]>.5)
 distance=(F.smooth_l1_loss(probability[:,1:],center,reduction='none')*(1+4*fg)).mean()
 return bce+overlap+2*distance+.5*F.binary_cross_entropy_with_logits(presence,exists)
