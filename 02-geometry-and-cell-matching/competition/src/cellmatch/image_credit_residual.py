"""Mask-conditioned image residuals for frozen endpoint/joint quality logits."""
import numpy as np
import torch
from torch import nn
from .guided_ex_data import cut
from .rle import Mask


def candidate_crops(record, images, size=48):
    banks=[]; truncations=[]
    for key, image in zip(['iv_rle','ex_rle'],images):
        cache={}; clipped={}
        for c in record['candidates']:
            rle=c[key]
            if rle in cache:continue
            xy=Mask.parse(rle,image.shape[1:]).coordinates()
            x0,y0=np.rint(xy.mean(0)).astype(int)-size//2
            local=xy-np.array([x0,y0]);valid=((local>=0)&(local<size)).all(1)
            mask=np.zeros((1,size,size),np.float32)
            mask[0,local[valid,1],local[valid,0]]=1
            cache[rle]=np.concatenate([cut(image,int(x0),int(y0),size),mask])
            clipped[rle]=int((~valid).sum())
        banks.append(np.stack([cache[c[key]] for c in record['candidates']]))
        truncations.append(sum(clipped[c[key]]>0 for c in record['candidates']))
    return np.concatenate(banks,axis=1).astype(np.float16),truncations


class ImageCreditResidual(nn.Module):
    def __init__(self):
        super().__init__()
        def tower():
            return nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
                                 nn.AvgPool2d(2),nn.Conv2d(16,32,3,stride=2,padding=1),
                                 nn.GroupNorm(4,32),nn.SiLU(),nn.AdaptiveAvgPool2d(4),nn.Flatten())
        self.iv=tower();self.ex=tower()
        self.iv_head=nn.Linear(512,1);self.ex_head=nn.Linear(512,1)
        self.joint=nn.Sequential(nn.Linear(1024,64),nn.SiLU(),nn.Linear(64,1))
        for layer in [self.iv_head,self.ex_head,self.joint[-1]]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,x):
        iv=self.iv(x[:,:3]);ex=self.ex(x[:,3:])
        return torch.cat([self.iv_head(iv),self.ex_head(ex),self.joint(torch.cat([iv,ex],1))],1)


def base_logits(heads,x):
    output=[]
    for h,cols in [(heads[0],slice(13,43)),(heads[1],slice(47,73)),(heads[3],slice(None))]:
        model,mean,sd=h
        with torch.inference_mode():output.append(model(torch.from_numpy(((x[:,cols]-mean)/sd).clip(-8,8))).numpy().ravel())
    return np.stack(output,axis=1)


def joint_score(logits):
    p=torch.from_numpy(np.asarray(logits)).sigmoid().numpy()
    return np.sqrt(p[:,0]*p[:,1])*p[:,2]
