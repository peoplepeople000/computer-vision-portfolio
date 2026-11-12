"""Image-only continuous boundary context for IV/EX candidate masks."""
import numpy as np
from .rle import Mask

def features(record,shapes):
 cache=[{},{}];rows=[]
 for c in record['candidates']:
  row=[]
  for j,(key,shape) in enumerate(zip(['iv_rle','ex_rle'],shapes)):
   rr=c[key]
   if rr not in cache[j]:
    m=Mask.parse(rr,shape);xy=m.coordinates();x,y=xy.mean(0);h,w=shape;l,t,r,b=m.bbox
    margin=min(x,y,w-1-x,h-1-y);assert margin>=0
    cache[j][rr]=[np.log1p(margin/np.sqrt(max(m.area,1)/np.pi)),float(min(l,t,w-1-r,h-1-b)==0)]
   row.extend(cache[j][rr])
  rows.append(row)
 result=np.asarray(rows,np.float32).reshape(-1,4);assert np.isfinite(result).all();return result
