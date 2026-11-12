"""Frozen guided prediction with explicitly calibrated foreground decoding."""
import numpy as np
import torch
from scipy import ndimage as ndi
from .rle import encode
from .guided_ex_data import crop

THRESHOLDS=(.35,.45,.55,.65)

def proposal(prob,presence,x,cell,origin,shape,foreground=.5):
    labels,count=ndi.label(prob[0]>foreground);options=[]
    for label in range(1,count+1):
        mask=ndi.binary_fill_holes(labels==label);area=int(mask.sum())
        if area<6 or area>500:continue
        yy,xx=np.where(mask);distance=float(np.linalg.norm(np.array([xx.mean(),yy.mean()])-24))
        if distance>=8:continue
        score=float(prob[0][mask].mean())*np.exp(-.5*(distance/5)**2)
        options.append((score,mask,distance))
    if not options:return None
    _,mask,distance=max(options,key=lambda v:v[0]);yy,xx=np.where(mask);x0,y0=origin
    gx,gy=xx+x0,yy+y0;valid=(gx>=0)&(gy>=0)&(gx<shape[1])&(gy<shape[0])
    if valid.sum()<6:return None
    canvas=np.zeros(shape,bool);canvas[gy[valid],gx[valid]]=True
    # Features describe image evidence and the projected cell, never annotations.
    area=float(valid.sum());height=float(yy.max()-yy.min()+1);width=float(xx.max()-xx.min()+1)
    boundary=mask&~ndi.binary_erosion(mask);prior=x[3]>.5;union=(mask|prior).sum()
    features=[np.log1p(area),area/(width*height),np.log(width/height),float(prob[0][mask].mean()),float(prob[0][mask].max()),float(prob[0][mask].std()),float(prob[0][boundary].mean()),float(prob[1][mask].max()),float(presence),distance,np.log1p(cell['patch'].sum()),np.log(max(area,1)/max(cell['patch'].sum(),1)),float((mask&prior).sum()/max(union,1))]
    return dict(iv=cell['key'],rle=encode(canvas),features=[float(v) for v in features],presence=float(presence))


@torch.inference_mode()
def infer_thresholds(model,record,device):
    results={str(t):dict(sample_id=record['sample_id'],proposals=[]) for t in THRESHOLDS}
    for start in range(0,len(record['cells']),32):
        cells=record['cells'][start:start+32];crops=[crop(record,c) for c in cells]
        x=np.stack([c[0] for c in crops]);logits,presence=model(torch.from_numpy(x).to(device));probabilities=logits.sigmoid().cpu().numpy();presence=presence.sigmoid().cpu().numpy()
        for cell,values,pr,xx,(_,_,origin) in zip(cells,probabilities,presence,x,crops):
            for threshold in THRESHOLDS:
                value=proposal(values,pr,xx,cell,origin,record['shape'],threshold)
                if value is not None:results[str(threshold)]['proposals'].append(value)
    return results
