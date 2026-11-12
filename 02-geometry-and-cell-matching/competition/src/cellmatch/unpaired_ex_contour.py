"""Stage84: development-only intensity boundary calibration for unpaired EX.
Keep every IV mask, pair and paired EX mask unchanged. Tune a one-pixel-band
contour rule on synthetic perturbations of development nonverified EX masks.
No held-out annotation enters fitting or transformation.
"""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import ndimage as ndi
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .rle import Mask
from .cli import evaluate

LEVELS=[None,.1,.2,.3,.4,.5,.6,.7]

def local_patch(mask,image):
 x0,y0,x1,y1=mask.bbox;x0=max(0,x0-5);y0=max(0,y0-5);x1=min(image.shape[1],x1+6);y1=min(image.shape[0],y1+6)
 prior=np.zeros((y1-y0,x1-x0),bool);xy=mask.coordinates();prior[xy[:,1]-y0,xy[:,0]-x0]=True
 return image[y0:y1,x0:x1],prior,x0,y0


def contour(image,prior,level,blocked=None):
 if level is None:return prior.copy()
 core=ndi.binary_erosion(prior);outer=ndi.binary_dilation(prior);ring=ndi.binary_dilation(prior,iterations=4)&~outer
 bg=float(np.median(image[ring])) if ring.any() else 0.;peak=float(np.percentile(image[prior],95))
 if peak<=bg:return prior.copy()
 candidate=core|(outer&((image-bg)/(peak-bg)>=level))
 if blocked is not None:candidate&=~blocked
 labels,_=ndi.label(candidate);overlap=np.bincount(labels[prior].ravel());overlap[0]=0
 if not overlap.any():return prior.copy()
 candidate=labels==int(overlap.argmax());ratio=candidate.sum()/prior.sum();iou=(candidate&prior).sum()/max(1,(candidate|prior).sum())
 if not .5<=ratio<=1.8 or iou<.5:return prior.copy()
 return candidate


def fit(root,truth,ids):
 subjects=sorted({split_id(s)[0] for s in ids});sums={s:np.zeros(len(LEVELS)) for s in subjects};counts={s:0 for s in subjects}
 for sid in ids:
  subject=split_id(sid)[0];im=read_image(image_path(root,'training',sid,'exvivo')).astype(float);verified={b for a,b in truth[sid]['match_pairs']}
  for key,mask in parse_instances(truth[sid],'exvivo',im.shape).items():
   if key in verified:continue
   image,target,_,_=local_patch(mask,im)
   for prior in [target,ndi.binary_erosion(target),ndi.binary_dilation(target)]:
    if prior.sum()<6:continue
    for i,level in enumerate(LEVELS):
     candidate=contour(image,prior,level);sums[subject][i]+=(candidate&target).sum()/max(1,(candidate|target).sum())
    counts[subject]+=1
 assert all(counts.values());scores=np.mean([sums[s]/counts[s] for s in subjects],axis=0);best=int(scores.argmax())
 return dict(level=LEVELS[best],scores=[dict(level=l,mean_iou=float(v)) for l,v in zip(LEVELS,scores)],examples=counts,train_ids=ids)


def patch_rle(mask,x0,y0,width):
 runs=[]
 for yy,line in enumerate(mask):
  edges=np.flatnonzero(np.diff(np.r_[False,line,False])).reshape(-1,2)
  for a,b in edges:runs.extend(((yy+y0)*width+int(a)+x0,int(b-a)))
 return ' '.join(map(str,runs))


def transform(row,image,level):
 result=copy.deepcopy(row);masks=parse_instances(row,'exvivo',image.shape);owner=np.zeros(image.shape,np.int32);current=np.zeros_like(owner)
 for i,m in enumerate(masks.values(),1):
  xy=m.coordinates();assert not owner[xy[:,1],xy[:,0]].any();owner[xy[:,1],xy[:,0]]=i
 current[:]=owner;paired={b for a,b in row['match_pairs']}
 for i,(key,m) in enumerate(masks.items(),1):
  if key in paired:continue
  patch,prior,x,y=local_patch(m,image);h,w=prior.shape;old=owner[y:y+h,x:x+w];now=current[y:y+h,x:x+w];blocked=((old>0)&(old!=i))|((now>0)&(now!=i))
  candidate=contour(patch,prior,level,blocked);now[now==i]=0;now[candidate]=i
  result['exvivo_instances'][key]=patch_rle(candidate,x,y,image.shape[1])
 assert result['match_pairs']==row['match_pairs'] and result['invivo_instances']==row['invivo_instances']
 assert all(result['exvivo_instances'][k]==row['exvivo_instances'][k] for k in paired)
 return result


def main():
 root=Path('data/raw/Project_2_Dataset');bp=Path('outputs/stage77_multi_ex/full/gated_joint/predictions.csv');base=read_rows(bp);truth=read_rows(root/'training/train_ground_truth.csv');out=Path('outputs/stage84_unpaired_contour');out.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();config=dict(policy=__doc__,levels=LEVELS,source_sha256=sha(__file__),input_sha256={str(p):sha(p) for p in [bp,root/'training/train_ground_truth.csv',*[image_path(root,'training',sid,'exvivo') for sid in sorted(truth)]]})
 if (out/'configuration.json').exists():assert json.loads((out/'configuration.json').read_text())==config
 (out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n');rows=[]
 for subject in sorted({split_id(s)[0] for s in truth}):
  ids=sorted(s for s in truth if split_id(s)[0]!=subject);cal=fit(root,truth,ids);(out/f'{subject}_calibration.json').write_text(json.dumps(cal,indent=2)+'\n');print('Selected',subject,cal['level'],cal['examples'],flush=True)
  for sid in sorted(set(truth)-set(ids)):
   rows.append(transform(base[sid],read_image(image_path(root,'training',sid,'exvivo')).astype(float),cal['level']))
 write_rows(rows,out/'predictions.csv');evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=None,output=out/'metrics.json'));(out/'manifest.json').write_text(json.dumps(dict(csv_sha256=sha(out/'predictions.csv'),configuration_sha256=sha(out/'configuration.json')),indent=2)+'\n')
if __name__=='__main__':main()
