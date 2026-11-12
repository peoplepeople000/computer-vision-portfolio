"""Rank-four SAM attention adapters inspired by CellSeg1; task recipe separate."""
import math
import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, original, rank=4):
        super().__init__()
        self.original=original
        self.down=nn.Linear(original.in_features,rank,bias=False)
        self.up=nn.Linear(rank,original.out_features,bias=False)
        nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self,value):
        return self.original(value)+self.up(self.down(value))


class LowRankQKV(nn.Module):
    def __init__(self,original,rank=4):
        super().__init__();self.original=original
        dimension=original.in_features
        assert original.out_features==3*dimension
        self.query_down=nn.Linear(dimension,rank,bias=False)
        self.query_up=nn.Linear(rank,dimension,bias=False)
        self.value_down=nn.Linear(dimension,rank,bias=False)
        self.value_up=nn.Linear(rank,dimension,bias=False)
        for layer in [self.query_down,self.value_down]:nn.init.kaiming_uniform_(layer.weight,a=math.sqrt(5))
        for layer in [self.query_up,self.value_up]:nn.init.zeros_(layer.weight)

    def forward(self,value):
        query,key,values=self.original(value).chunk(3,dim=-1)
        return torch.cat([query+self.query_up(self.query_down(value)),key,
                          values+self.value_up(self.value_down(value))],dim=-1)


def insert_adapters(sam,rank=4):
    """Freeze all pretrained weights; fit only image/decoder attention adapters."""
    sam.requires_grad_(False);names=[]
    for index,block in enumerate(sam.image_encoder.blocks):
        block.attn.qkv=LowRankQKV(block.attn.qkv,rank)
        names.append(f'image_encoder.blocks.{index}.attn.qkv')
    transformer=sam.mask_decoder.transformer
    attention=[]
    for index,block in enumerate(transformer.layers):
        for name in ['self_attn','cross_attn_token_to_image','cross_attn_image_to_token']:
            attention.append((f'mask_decoder.transformer.layers.{index}.{name}',getattr(block,name)))
    attention.append(('mask_decoder.transformer.final_attn_token_to_image',transformer.final_attn_token_to_image))
    for name,block in attention:
        for projection in ['q_proj','v_proj']:
            setattr(block,projection,LowRankLinear(getattr(block,projection),rank));names.append(name+'.'+projection)
    return names


def adapter_state(sam):
    return {name:parameter.detach().cpu().clone() for name,parameter in sam.named_parameters() if parameter.requires_grad}
