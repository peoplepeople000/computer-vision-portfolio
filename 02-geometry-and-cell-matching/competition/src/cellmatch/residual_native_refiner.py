"""Native contour correction initialized to preserve the supplied binary prior."""
import torch
from .learning import SmallUNet

class ResidualNativeRefiner(SmallUNet):
    def __init__(self, width=16):
        super().__init__(width)
        with torch.no_grad():
            self.out.weight[0].zero_()
            self.out.bias[0].zero_()

    def forward(self, x):
        logits = super().forward(x)
        foreground = logits[:, :1] + 2.0 * (2.0 * x[:, 2:3] - 1.0)
        return torch.cat((foreground, logits[:, 1:]), dim=1)
