"""Label-free empirical percentile appearance features from unique mask identities."""
import numpy as np
from scipy.stats import rankdata


def augment(record):
    x=record['x']; blocks=[]
    for key,columns in [('iv_rle',slice(18,39)),('ex_rle',slice(52,73))]:
        unique={}
        for c,v in zip(record['candidates'],x[:,columns]):
            rle=c[key]
            if rle in unique:
                assert np.array_equal(unique[rle],v), 'One mask has inconsistent appearance features'
            else:unique[rle]=v
        keys=sorted(unique)
        if not keys:
            blocks.append(np.empty((0,21),np.float32));continue
        values=np.stack([unique[k] for k in keys]);rank=(rankdata(values,axis=0,method='average')-.5)/len(keys)
        lookup={k:v for k,v in zip(keys,rank)}
        blocks.append(np.array([lookup[c[key]] for c in record['candidates']],np.float32))
    return dict(record,x=np.c_[x,*blocks].astype(np.float32))
