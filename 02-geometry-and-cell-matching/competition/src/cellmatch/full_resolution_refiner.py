"""Same parameter shapes as SmallUNet, with dilations instead of resampling."""
import torch
from .learning import SmallUNet

class FullResolutionRefiner(SmallUNet):
    def __init__(self,width=16):
        super().__init__(width)
        for block,dilation in [(self.b,2),(self.c,4),(self.d,2)]:
            for layer in block:
                if isinstance(layer,torch.nn.Conv2d):
                    layer.dilation=(dilation,dilation);layer.padding=(dilation,dilation)

    def forward(self,x):
        a=self.a(x);b=self.b(a);c=self.c(b)
        d=self.d(torch.cat([c,b],1))
        return self.out(self.e(torch.cat([d,a],1)))
