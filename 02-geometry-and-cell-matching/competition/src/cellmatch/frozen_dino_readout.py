"""Pure layout and loss helpers for cached CellposeDINO linear readout fitting.
The backbone is external and frozen. Helpers never choose subjects or labels.
"""
import numpy as np
import torch
from torch.nn import functional as F


def decode_tokens(values, height, width, ps=8):
    batch, count, channels = values.shape
    assert height % ps == width % ps == 0
    assert count == (height//ps)*(width//ps) and channels == 3*ps*ps
    return values.reshape(batch,height//ps,width//ps,3,ps,ps).permute(0,3,1,4,2,5).reshape(batch,3,height,width)


def pack_pixels(values, ps=8):
    batch, channels, height, width = values.shape
    assert height % ps == width % ps == 0
    return values.reshape(batch,channels,height//ps,ps,width//ps,ps).permute(0,2,4,1,3,5).reshape(batch,-1,channels*ps*ps)


def token_loss(prediction, target, valid, ps=8):
    prediction=prediction.reshape(-1,3,ps*ps)
    target=target.reshape(-1,3,ps*ps)
    valid=valid.reshape(-1,ps*ps)
    denominator=valid.sum().clamp_min(1.)
    flow=((prediction[:,:2]-5.*target[:,1:])**2*valid[:,None]).sum()/(4.*denominator)
    foreground=(F.binary_cross_entropy_with_logits(prediction[:,2],(target[:,0]>.5).to(prediction.dtype),reduction='none')*valid).sum()/denominator
    return flow+foreground


def tile_layout(shape, bsize=384):
    from cellpose import transforms
    y1,y2,x1,x2=map(int,transforms.get_pad_yx(*shape,min_size=(bsize,bsize)))
    canvas=np.zeros((1,shape[0]+y1+y2,shape[1]+x1+x2),np.float32)
    tiles,ys,xs,height,width=transforms.make_tiles(canvas,bsize=bsize,augment=False,tile_overlap=.1)
    return dict(shape=list(shape),bsize=bsize,padding=[y1,y2,x1,x2],ys=[list(map(int,v)) for v in ys],xs=[list(map(int,v)) for v in xs],height=int(height),width=int(width),tiles=len(ys))


def tile_pixels(values, layout):
    y1,y2,x1,x2=layout['padding']
    padded=np.pad(values,((0,0),(y1,y2),(x1,x2)),mode='constant')
    return np.stack([padded[:,a:b,c:d] for (a,b),(c,d) in zip(layout['ys'],layout['xs'])])


def reassemble(values, layout):
    from cellpose import transforms
    result=transforms.average_tiles(values,layout['ys'],layout['xs'],layout['height'],layout['width'])
    y1,y2,x1,x2=layout['padding'];height,width=layout['shape']
    return result[:,y1:y1+height,x1:x1+width]
