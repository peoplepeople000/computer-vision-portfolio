"""Bounded IV contour union for agreeing independent segmentation masks.

A fixed inference rule, with no annotation or fitted threshold input. Retain
baseline cell IDs/pairs and prohibit invading another baseline cell's pixels.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment
from .data import read_rows,write_rows,parse_instances,image_path,image_shape,split_id
from .rle import encode
from .stage3 import FOLDS
from .cli import evaluate


def bounded_union(base,candidate,shape):
    if base['sample_id']!=candidate['sample_id']:raise ValueError('Different samples')
    old=parse_instances(base,'invivo',shape);new=parse_instances(candidate,'invivo',shape)
    keys,others=list(old),list(new)
    overlaps=np.array([[old[k].iou(new[j]) for j in others] for k in keys]).reshape(len(keys),len(others))
    ii,jj=linear_sum_assignment(overlaps,maximize=True)
    owner=np.zeros(shape,np.int32)
    for index,key in enumerate(keys,1):
        xy=old[key].coordinates();owner[xy[:,1],xy[:,0]]=index
    row=copy.deepcopy(base);changed=0
    for i,j in zip(ii,jj):
        if overlaps[i,j]<=.5:continue
        mask=np.zeros(shape,bool)
        for value in [old[keys[i]],new[others[j]]]:
            xy=value.coordinates();mask[xy[:,1],xy[:,0]]=True
        mask&=(owner==0)|(owner==i+1)
        # Never expand a correct-looking baseline contour by more than 30%.
        if mask.sum()>1.3*old[keys[i]].area:continue
        rle=encode(mask)
        if rle!=row['invivo_instances'][keys[i]]:
            row['invivo_instances'][keys[i]]=rle;changed+=1
            # Reserve new pixels to preserve nonoverlap between successive unions.
            owner[mask]=i+1
    return row,changed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/stage41_iv_union'));a=p.parse_args()
    basepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');newpath=Path(f'outputs/stage35_cpdino_iv_adapted/{a.fold}/predictions.csv')
    base,new=read_rows(basepath),read_rows(newpath);subject=FOLDS[a.fold]
    if set(new)!={s for s in base if split_id(s)[0]==subject}:raise ValueError('Incomplete fold')
    cfgpath=newpath.with_name('configuration.json');cfg=json.loads(cfgpath.read_text())
    if cfg['held_out']!=subject or set(cfg['train_ids'])!=set(base)-set(new):raise ValueError('Invalid excluded mouse')
    rows=[];counts={};root=Path('data/raw/Project_2_Dataset')
    for sid in sorted(new):
        row,counts[sid]=bounded_union(base[sid],new[sid],image_shape(image_path(root,'training',sid,'invivo')));rows.append(row)
    out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True);path=out/'predictions.csv';write_rows(rows,path)
    (out/'manifest.json').write_text(json.dumps(dict(changed=counts,held_out=subject,train_ids=cfg['train_ids'],
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [basepath,newpath,cfgpath]},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy='IoU > 0.5, growth <= 30%, fixed baseline IDs/pairs, exclusive pixel ownership'),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=subject,output=out/'metrics.json'))

if __name__=='__main__':main()
