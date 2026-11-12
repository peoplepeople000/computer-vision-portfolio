"""Activation checkpointing without changing microscopy decoder parameters."""
from functools import wraps
import torch
from torch.utils.checkpoint import checkpoint


BLOCKS=('base','deconv1','deconv2','deconv3','deconv4','decoder','deconv_out','decoder_head')


def enable_decoder_checkpointing(decoder):
    for name in BLOCKS:
        module=getattr(decoder,name);assert not getattr(module,'_q5_checkpointed',False)
        original=module.forward
        def wrap(function,block):
            @wraps(function)
            def forward(*args,**kwargs):
                if block.training and torch.is_grad_enabled():
                    return checkpoint(function,*args,**kwargs,use_reentrant=False,preserve_rng_state=True)
                return function(*args,**kwargs)
            return forward
        module.forward=wrap(original,module);module._q5_checkpointed=True
    return decoder
