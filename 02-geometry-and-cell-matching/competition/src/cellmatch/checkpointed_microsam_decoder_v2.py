"""Checkpointed full decoder with one-update BatchNorm buffer semantics."""
from contextlib import contextmanager,nullcontext
from functools import wraps
import torch
from torch.utils.checkpoint import checkpoint

BLOCKS=('base','deconv1','deconv2','deconv3','deconv4','decoder','deconv_out','decoder_head')


@contextmanager
def isolated_running_statistics(block):
    """Recomputation uses temporary BN buffers; original updates occur once."""
    saved=[]
    for module in block.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
            originals={key:module._buffers[key] for key in ('running_mean','running_var','num_batches_tracked')}
            saved.append((module,originals))
            for key,value in originals.items():module._buffers[key]=value.clone() if value is not None else None
    try:yield
    finally:
        for module,originals in saved:
            for key,value in originals.items():module._buffers[key]=value


def enable_decoder_checkpointing(decoder):
    for name in BLOCKS:
        module=getattr(decoder,name);assert not getattr(module,'_q5_checkpointed',False)
        original=module.forward
        def wrap(function,block):
            @wraps(function)
            def forward(*args,**kwargs):
                if block.training and torch.is_grad_enabled():
                    return checkpoint(function,*args,**kwargs,use_reentrant=False,preserve_rng_state=True,
                                      context_fn=lambda:(nullcontext(),isolated_running_statistics(block)))
                return function(*args,**kwargs)
            return forward
        module.forward=wrap(original,module);module._q5_checkpointed=True
    return decoder
