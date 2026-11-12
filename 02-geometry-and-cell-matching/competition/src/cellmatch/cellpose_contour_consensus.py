"""Average agreeing baseline/Cellpose IV contours with conservative pixel ties.

Match masks at IoU > 0.5, average their signed distance fields, retain baseline
ownership when distances tie, and never invade another baseline cell's pixels.
No model fitting or annotations enter this inference-only ensemble.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from .data import read_rows,write_rows,parse_instances,image_shape,image_path,validate_submission
from .rle import encode
from .cli import evaluate


def blend(baseline,candidate,shape):
    if baseline['sample_id']!=candidate['sample_id']:raise ValueError('Different samples')
    old=parse_instances(baseline,'invivo',shape);new=parse_instances(candidate,'invivo',shape)
    keys=list(old);other=list(new);ious=np.array([[old[i].iou(new[j]) for j in other] for i in keys]).reshape(len(keys),len(other))
    ii,jj=linear_sum_assignment(ious,maximize=True)
    owner=np.zeros(shape,np.int32)
    for index,key in enumerate(keys,1):
        xy=old[key].coordinates();owner[xy[:,1],xy[:,0]]=index
    result=copy.deepcopy(baseline);changed=0
    for i,j in zip(ii,jj):
        if ious[i,j]<=.5:continue
        aa,bb=old[keys[i]],new[other[j]]
        x0=max(0,min(aa.bbox[0],bb.bbox[0])-2);y0=max(0,min(aa.bbox[1],bb.bbox[1])-2)
        x1=min(shape[1],max(aa.bbox[2],bb.bbox[2])+3);y1=min(shape[0],max(aa.bbox[3],bb.bbox[3])+3)
        arrays=[]
        for mask in [aa,bb]:
            patch=np.zeros((y1-y0,x1-x0),bool);xy=mask.coordinates();patch[xy[:,1]-y0,xy[:,0]-x0]=True;arrays.append(patch)
        # Explicit false padding makes masks at image edges behave consistently.
        signed=[]
        for patch in arrays:
            padded=np.pad(patch,1);signed.append((ndi.distance_transform_edt(padded)-ndi.distance_transform_edt(~padded))[1:-1,1:-1])
        summed=signed[0]+signed[1]
        merged=((summed>0)|((summed==0)&arrays[0]))&(arrays[0]|arrays[1])
        local_owner=owner[y0:y1,x0:x1];merged&=(local_owner==0)|(local_owner==i+1)
        if not merged.any():continue
        canvas=np.zeros(shape,bool);canvas[y0:y1,x0:x1]=merged
        value=encode(canvas)
        if value!=result['invivo_instances'][keys[i]]:changed+=1
        result['invivo_instances'][keys[i]]=value
    return result,changed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['training','hidden_test'],default='training');a=p.parse_args()
    root=Path('data/raw/Project_2_Dataset');test=a.split=='hidden_test'
    baseline_path=Path('outputs/submission_stage8/submission.csv' if test else 'outputs/stage13_native_relative/preserved/predicted_all.csv');source_path=a.source/'predictions.csv'
    baseline=read_rows(baseline_path);source=read_rows(source_path)
    if set(baseline)!=set(source) or len(source)!=(29 if test else 47):raise ValueError('Incomplete split')
    rows=[];counts={}
    for sid,row in baseline.items():
        value,counts[sid]=blend(row,source[sid],image_shape(image_path(root,a.split,sid,'invivo')));rows.append(value)
        print('Contour consensus',sid,counts[sid],flush=True)
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/('submission.csv' if test else 'predicted_all.csv');write_rows(rows,path)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps({'source_sha256':digest(__file__),'input_sha256':{str(p):digest(p) for p in [baseline_path,source_path,a.source/'configuration.json']},'changed_masks':counts,'policy':__doc__,'fitting':'None','split':a.split},indent=2)+'\n')
    if test:(a.output/'validation.json').write_text(json.dumps({'status':'passed','counts':validate_submission(root,path),'csv_sha256':digest(path),'uploaded':False,'hidden_score':None},indent=2)+'\n')
    else:evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=a.output/'metrics.json'))

if __name__=='__main__':main()
