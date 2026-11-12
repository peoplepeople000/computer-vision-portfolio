"""Leakage-aware data and native-coordinate crops for IV-guided EX prediction."""
from pathlib import Path
import copy,json
import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from .data import read_rows,read_image,image_path,parse_instances,split_id
from .learning import image_features
from .cellpose_finetune import instance_labels
from .cellpose_contour_consensus import blend
from .ex_joint_calibration import morphology_features
from .stage3 import FOLDS,quality_probabilities
from .metrics import segmentation_score


def cut(array,x0,y0,size=48):
    """Zero-pad a channel-first image crop; retain exact integer pixel coordinates."""
    if array.ndim==2:array=array[None]
    out=np.zeros((array.shape[0],size,size),array.dtype);h,w=array.shape[-2:]
    left,right=max(x0,0),min(x0+size,w);top,bottom=max(y0,0),min(y0+size,h)
    if right>left and bottom>top:out[:,top-y0:bottom-y0,left-x0:right-x0]=array[:,top:bottom,left:right]
    return out


def cell_record(key,mask,target=0):
    xy=mask.coordinates();x0,y0,x1,y1=mask.bbox
    patch=np.zeros((y1-y0+1,x1-x0+1),np.float32);patch[xy[:,1]-y0,xy[:,0]-x0]=1
    return dict(key=key,center=xy.mean(0),origin=np.array([x0,y0]),patch=patch,target=int(target))


def crop(record,cell,shift=(0.,0.),size=48,with_target=False):
    """Project one IV cell into EX; optional labels are only a training output."""
    matrix=np.asarray(record['matrix'],np.float64).copy();matrix[:,2]+=shift
    point=matrix[:,:2]@cell['center']+matrix[:,2]
    x0,y0=np.rint(point).astype(int)-size//2
    local=matrix.copy();local[:,2]-=np.array([x0,y0])
    mask_matrix=local.copy();mask_matrix[:,2]+=matrix[:,:2]@cell['origin']
    mask=cv2.warpAffine(cell['patch'],mask_matrix,(size,size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    iv=np.stack([cv2.warpAffine(v,local,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for v in record['iv_image']])
    yy,xx=np.mgrid[:size,:size];anchor=np.exp(-((xx-size//2)**2+(yy-size//2)**2)/18.).astype(np.float32)
    x=np.concatenate([cut(record['ex_image'],x0,y0,size),np.ones((1,size,size),np.float32),mask[None],anchor[None],cut(record['ex_maps'],x0,y0,size),iv]).astype(np.float32)
    if x.shape!=(9,size,size) or not np.isfinite(x).all():raise ValueError('Invalid guided crop')
    y=None
    if with_target:
        if record.get('labels') is None:raise ValueError('Training labels required for a target')
        fg=(cut(record['labels'],x0,y0,size)[0]==cell['target']) if cell['target'] else np.zeros((size,size),bool)
        distance=ndi.distance_transform_edt(np.pad(fg,1))[1:-1,1:-1].astype(np.float32)
        distance/=max(float(distance.max()),1.)
        y=np.stack([fg,distance]).astype(np.float32)
    return x,y,(int(x0),int(y0))


def make_record(root,sid,row,matrix,exmaps,truth=None):
    iv=read_image(image_path(root,'training',sid,'invivo'));ex=read_image(image_path(root,'training',sid,'exvivo'))
    masks=parse_instances(row,'invivo',iv.shape)
    labels=None;mapping={};matched={};ex_numbers={}
    if truth is not None:
        labels=instance_labels(truth['exvivo_instances'],ex.shape)
        ex_numbers={key:i for i,key in enumerate(truth['exvivo_instances'],1)}
        actual=parse_instances(truth,'invivo',iv.shape)
        mapping=segmentation_score(masks,actual)['pred_to_gt'];matched=dict(truth['match_pairs'])
    cells=[]
    for key,mask in masks.items():
        target=ex_numbers.get(matched.get(mapping.get(key)),0)
        cells.append(cell_record(key,mask,target))
    return dict(sample_id=sid,subject=split_id(sid)[0],matrix=np.asarray(matrix),
        iv_image=image_features(iv),ex_image=image_features(ex),ex_maps=exmaps,labels=labels,cells=cells,shape=ex.shape)


def training_records(root,truth,tag):
    """Use only same-outer-fold IV/EX predictions and development annotations."""
    subject=FOLDS[tag];ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    folder=Path('outputs/stage8_iv')/tag;exfolder=Path('outputs/stage11_native_ex')/tag
    for path in [folder/'training_config.json',exfolder/'training_config.json']:
        cfg=json.loads(path.read_text())
        if cfg['held_out']!=subject or sorted(cfg['train_ids'])!=ids:raise ValueError('Invalid development predictor partition')
    rows=read_rows(folder/'development.csv')
    if set(rows)!=set(ids):raise ValueError('Unexpected development images')
    qpath=Path(f'outputs/stage8_iv_quality_capacity/{tag}.pt');state=torch.load(qpath,map_location='cpu',weights_only=False)
    if state['held_out']!=subject or sorted(state['train_ids'])!=ids:raise ValueError('Invalid IV quality partition')
    model=torch.nn.Linear(30,1);model.load_state_dict(state['quality']);qfit=model.eval(),state['mean'],state['sd']
    public=read_rows('outputs/stage28_cpdino_iv_full/predictions.csv');records=[]
    for sid in ids:
        if len(truth[sid]['match_pairs'])<5:continue
        im=np.load(folder/'maps'/f'{sid}_invivo.npz')['maps'].astype(np.float32)
        em=np.load(exfolder/'maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
        row=copy.deepcopy(rows[sid]);masks=parse_instances(row,'invivo',im.shape[1:]);keys,x=morphology_features(masks,im)
        keep={key for key,p in zip(keys,quality_probabilities(qfit,x)) if p>=state['threshold']}
        row['invivo_instances']={k:v for k,v in row['invivo_instances'].items() if k in keep};row['match_pairs']=[]
        row,_=blend(row,public[sid],im.shape[1:])
        actual=[parse_instances(truth[sid],m,maps.shape[1:]) for m,maps in [('invivo',im),('exvivo',em)]]
        a=np.array([actual[0][i].coordinates().mean(0) for i,j in truth[sid]['match_pairs']]);b=np.array([actual[1][j].coordinates().mean(0) for i,j in truth[sid]['match_pairs']])
        matrix=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0].T
        records.append(make_record(root,sid,row,matrix,em,truth[sid]))
        print('Guided EX development',sid,len(records[-1]['cells']),sum(c['target']>0 for c in records[-1]['cells']),flush=True)
    if any(r['subject']==subject for r in records):raise ValueError('Outer annotations entered training')
    return records,ids


def sample(records,rng,batch=16):
    groups={}
    for i,r in enumerate(records):
        for j,c in enumerate(r['cells']):groups.setdefault((r['subject'],bool(c['target'])),[]).append((i,j))
    subjects=sorted({r['subject'] for r in records});xs=[];ys=[];present=[]
    for _ in range(batch):
        subject=subjects[int(rng.integers(len(subjects)))];positive=bool(rng.random()<.7)
        candidates=groups.get((subject,positive)) or groups[(subject,not positive)]
        i,j=candidates[int(rng.integers(len(candidates)))];r=records[i];cell=r['cells'][j]
        x,y,_=crop(r,cell,rng.uniform(-4,4,2),with_target=True)
        k=int(rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k,(1,2))
        if rng.random()<.5:x=x[:,:,::-1];y=y[:,:,::-1]
        x=x.copy();y=y.copy()
        for channel in [0,1,7,8]:x[channel]=np.clip(np.maximum(x[channel],0)**rng.uniform(.7,1.4)*rng.uniform(.8,1.2),0,1)
        xs.append(x);ys.append(y);present.append(float(y[0].any()))
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys)),torch.tensor(present,dtype=torch.float32)
