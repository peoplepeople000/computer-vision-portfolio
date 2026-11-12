"""One supervised spatial correction stage over frozen foreground posteriors."""
import torch
from torch import nn
from torch.nn import functional as F
from .learning import SmallUNet


class PosteriorSpatialCorrection(SmallUNet):
    """Raw/local/prior/probability -> foreground logit residual at32px."""
    def __init__(self, base_state=None):
        super().__init__(16)
        self.a[0]=nn.Conv2d(4,16,3,padding=1)
        self.out=nn.Conv2d(16,1,1)
        if base_state is not None:
            state={k:v.detach().clone() for k,v in base_state.items()}
            expanded=torch.zeros((16,4,3,3),dtype=state['a.0.weight'].dtype)
            expanded[:,:3]=state['a.0.weight']
            state['a.0.weight']=expanded
            state['out.weight']=torch.zeros((1,16,1,1),dtype=state['out.weight'].dtype)
            state['out.bias']=torch.zeros((1,),dtype=state['out.bias'].dtype)
            self.load_state_dict(state,strict=True)
        else:
            nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)


def foreground_loss(logits,target):
    probability=logits.sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=logits.new_tensor(2.))
    overlap=(probability*target).sum((-2,-1))
    dice=1-((2*overlap+1)/(probability.sum((-2,-1))+target.sum((-2,-1))+1)).mean()
    return bce+dice


def training_logits(model,x):
    baseline=torch.logit(x[:,3:4].clamp(1e-6,1-1e-6))
    return baseline+model(x)


class PosteriorCorrectedPredictor(nn.Module):
    """OriginalD4mean + eight-view logit-residual mean; center unchanged."""
    def __init__(self,original,correction):
        super().__init__();self.original=original;self.correction=correction
        self.original.requires_grad_(False)

    def forward(self,x):
        z=self.original(x)
        features=torch.cat((x,z[:,0:1].sigmoid()),1)
        residuals=[]
        for flip in (False,True):
            base=features.flip(-1) if flip else features
            for k in range(4):
                residual=self.correction(torch.rot90(base,k,(-2,-1)))
                residual=torch.rot90(residual,-k,(-2,-1))
                if flip:residual=residual.flip(-1)
                residuals.append(residual)
        correction=torch.stack(residuals).mean(0)
        return torch.cat((z[:,0:1]+correction,z[:,1:2]),1)
