"""Training-only strict-pair utility targets for ranking eligible candidates."""
import numpy as np
from .data import parse_instances
from .rle import Mask


def targets(record, truth, shapes):
    gt=[parse_instances(truth,m,s) for m,s in zip(['invivo','exvivo'],shapes)]
    pairs=truth['match_pairs']
    output=np.zeros(len(record['candidates']),np.float32)
    cache=[{},{}]
    for n,c in enumerate(record['candidates']):
        if not record['y'][n]:
            continue
        for k,key in enumerate(['iv_rle','ex_rle']):
            rle=c[key]
            if rle not in cache[k]:
                mask=Mask.parse(rle,shapes[k])
                cache[k][rle]={g:mask.iou(m) for g,m in gt[k].items()}
        value=max((min(cache[0][c['iv_rle']][i],cache[1][c['ex_rle']][e]) for i,e in pairs),default=0.)
        assert value>.75, (record['sample_id'],n,value)
        output[n]=value
    return output
