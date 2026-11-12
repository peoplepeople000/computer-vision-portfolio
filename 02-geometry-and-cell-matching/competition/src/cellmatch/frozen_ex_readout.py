"""Small native-pixel EX foreground/distance decoder over frozen public tokens."""
import torch
from torch import nn
from torch.nn import functional as F

class EXReadout(nn.Linear):
    def __init__(self,public_state):
        super().__init__(1024,128)
        with torch.no_grad():
            self.weight.zero_();self.bias.fill_(-2.)
            self.weight[:64].copy_(public_state['weight'][128:192])
            self.bias[:64].copy_(public_state['bias'][128:192])


def decode_maps(values,height,width,ps=8):
    batch,count,channels=values.shape
    assert height%ps==width%ps==0 and count==(height//ps)*(width//ps) and channels==2*ps*ps
    return values.reshape(batch,height//ps,width//ps,2,ps,ps).permute(0,3,1,4,2,5).reshape(batch,2,height,width)


def pixel_loss(logits,target,valid):
    """Global masked Dice, weighted BCE and normalized-distance regression."""
    fg,center=target[:,0],target[:,1];p=logits.sigmoid();den=valid.sum().clamp_min(1.)
    bce=(F.binary_cross_entropy_with_logits(logits[:,0],fg,reduction='none',pos_weight=logits.new_tensor(2.))*valid).sum()/den
    dice=1-(2*(p[:,0]*fg*valid).sum()+1)/(((p[:,0]+fg)*valid).sum()+1)
    distance=(F.smooth_l1_loss(p[:,1],center,reduction='none')*(1+4*fg)*valid).sum()/den
    return bce+dice+2*distance


def token_loss(values,target,valid):
    return pixel_loss(values.reshape(-1,2,8,8),target.reshape(-1,2,8,8),valid.reshape(-1,8,8))
