"""Decode original guided EX component plus up to two alternatives within 12px.
The original component within 8px is always retained when present. Fixed .5
foreground and 6..500 area constraints; no labels enter this decoder.
"""
import numpy as np
import torch
from scipy import ndimage as ndi
from .rle import encode
from .guided_ex_data import crop

def proposals(prob,presence,x,cell,origin,shape):
    labels,count=ndi.label(prob[0]>.5);options=[]
    for label in range(1,count+1):
        mask=ndi.binary_fill_holes(labels==label);area=int(mask.sum())
        if area<6 or area>500:continue
        yy,xx=np.where(mask);distance=float(np.linalg.norm(np.array([xx.mean(),yy.mean()])-24))
        if distance>=12:continue
        score=float(prob[0][mask].mean())*np.exp(-.5*(distance/5)**2)
        options.append((score,mask,distance))
    if not options:return []
    ranked=sorted(options,key=lambda v:v[0],reverse=True)
    original=next((v for v in ranked if v[2]<8),None)
    selected=([original] if original is not None else [])+[v for v in ranked if v is not original]
    results=[]
    for _,mask,distance in selected[:3]:
        yy,xx=np.where(mask);x0,y0=origin
        gx,gy=xx+x0,yy+y0;valid=(gx>=0)&(gy>=0)&(gx<shape[1])&(gy<shape[0])
        if valid.sum()<6:continue
        canvas=np.zeros(shape,bool);canvas[gy[valid],gx[valid]]=True
        # Features describe image evidence and the projected cell, never annotations.
        area=float(valid.sum());height=float(yy.max()-yy.min()+1);width=float(xx.max()-xx.min()+1)
        boundary=mask&~ndi.binary_erosion(mask);prior=x[3]>.5;union=(mask|prior).sum()
        features=[np.log1p(area),area/(width*height),np.log(width/height),float(prob[0][mask].mean()),float(prob[0][mask].max()),float(prob[0][mask].std()),float(prob[0][boundary].mean()),float(prob[1][mask].max()),float(presence),distance,np.log1p(cell['patch'].sum()),np.log(max(area,1)/max(cell['patch'].sum(),1)),float((mask&prior).sum()/max(union,1))]
        results.append(dict(iv=cell['key'],rle=encode(canvas),features=[float(v) for v in features],presence=float(presence)))
    return results

@torch.inference_mode()
def infer(model,record,device):
    results=[]
    for start in range(0,len(record['cells']),32):
        cells=record['cells'][start:start+32];crops=[crop(record,c) for c in cells]
        x=np.stack([c[0] for c in crops]);logits,presence=model(torch.from_numpy(x).to(device))
        probabilities=logits.sigmoid().cpu().numpy();presence=presence.sigmoid().cpu().numpy()
        for cell,values,pr,xx,(_,_,origin) in zip(cells,probabilities,presence,x,crops):
            values_out=proposals(values,pr,xx,cell,origin,record['shape'])
            results.extend(values_out)
    return dict(sample_id=record['sample_id'],proposals=results)

