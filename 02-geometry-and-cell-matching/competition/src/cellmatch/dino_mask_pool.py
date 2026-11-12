"""Image-only candidate-mask pooling of cached public ViT-L tokens.
Each8x8 output block inherits its token embedding. Overlapping tiles use the
same taper weights as Cellpose average_tiles, normalized per image pixel.
"""
import numpy as np
from scipy.sparse import csr_matrix
from .rle import Mask


def taper(size):
    bsize=max(224,size);x=np.abs(np.arange(bsize)-(bsize-1)/2)
    values=1/(1+np.exp((x-(bsize/2-20))/7.5))
    full=values[:,None]*values[None,:];start=bsize//2-size//2
    return full[start:start+size,start:start+size]


def pool_masks(rles,features,layout):
    size=layout['bsize'];side=size//8;per_tile=side*side
    assert features.shape[:2]==(layout['tiles'],per_tile)
    weight=taper(size);den=np.zeros((layout['height'],layout['width']),np.float64)
    for (a,b),(c,d) in zip(layout['ys'],layout['xs']):den[a:b,c:d]+=weight
    unique=list(dict.fromkeys(rles));rows=[];cols=[];data=[]
    for n,rle in enumerate(unique):
        mask=Mask.parse(rle,tuple(layout['shape']));xy=mask.coordinates();assert len(xy)>0
        x=xy[:,0]+layout['padding'][2];y=xy[:,1]+layout['padding'][0]
        for tile,((a,b),(c,d)) in enumerate(zip(layout['ys'],layout['xs'])):
            valid=(y>=a)&(y<b)&(x>=c)&(x<d)
            if not valid.any():continue
            yy,xx=y[valid]-a,x[valid]-c
            token=(yy//8)*side+xx//8
            coefficients=weight[yy,xx]/den[y[valid],x[valid]]/len(xy)
            rows.extend([n]*len(token));cols.extend((tile*per_tile+token).tolist());data.extend(coefficients.tolist())
    matrix=csr_matrix((data,(rows,cols)),shape=(len(unique),layout['tiles']*per_tile),dtype=np.float32)
    np.testing.assert_allclose(np.asarray(matrix.sum(1)).ravel(),1.,rtol=1e-5,atol=1e-6)
    pooled=matrix@features.reshape(-1,features.shape[-1]);index={r:i for i,r in enumerate(unique)}
    return np.asarray([pooled[index[r]] for r in rles],np.float32).reshape(len(rles),features.shape[-1])
