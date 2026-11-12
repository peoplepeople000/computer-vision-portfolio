"""Per-channel spatial normalization ablation; not an IBN-Net reproduction."""
from torch import nn
from .learning import SmallUNet

class InstanceNormalizedUNet(SmallUNet):
    def __init__(self,width=16):
        super().__init__(width)
        # One channel per group computes instance normalization without running stats.
        # Preserve every parameter tensor and its initialization sequence.
        for layer in self.modules():
            if isinstance(layer,nn.GroupNorm):
                layer.num_groups=layer.num_channels
