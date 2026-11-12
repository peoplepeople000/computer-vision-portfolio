"""Subject-held-out cell descriptors, spatial consistency, and EX quality filtering.

Run with PYTHONPATH=src python -m cellmatch.stage3 from Q5. This is a compact
research baseline, not an implementation of SuperGlue or PointDSC.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from .data import read_rows, read_image, image_path, parse_instances, split_id, write_rows
from .rle import labels_to_instances
from .learning import SmallUNet, probability_maps, instances_from_maps, choose_device
from .cell_features import extract_features, fit_feature_geometry
from .metrics import segmentation_score
from .registration import assign_pairs
from .cli import evaluate

FOLDS={'5d':'subject_5d294c','b2':'subject_b2ba5e','db':'subject_db6b8b'}


def development_signature(checkpoint_path):
    return {'checkpoint_sha256':hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
            'learning_source_sha256':hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest(),
            'normalization_source_sha256':hashlib.sha256(Path(__file__).with_name('segmentation.py').read_bytes()).hexdigest(),
            'foreground_threshold':.5,'seed_threshold':.35,'format':1}


class Descriptor(nn.Module):
    def __init__(self, dimensions=26):
        super().__init__()
        self.towers=nn.ModuleList([nn.Sequential(nn.Linear(dimensions,64),nn.SiLU(),nn.Linear(64,32)) for _ in range(2)])
    def forward(self,x,mode):
        return F.normalize(self.towers[mode](x),dim=-1)


def fit_descriptor(records, heldout, steps=800):
    selected=[r for r in records if r['subject']!=heldout and len(r['pairs'])>=2]
    if not selected: raise ValueError('No descriptor training pairs')
    normalization=[]
    for mode in range(2):
        x=np.concatenate([r['features'][mode][:,:26] for r in selected])
        normalization.append((x.mean(0),np.maximum(x.std(0),.03)))
    groups=[]
    for r in selected:
        xs=[]
        for mode in range(2):
            indices=[r['keys'][mode].index(pair[mode]) for pair in r['pairs']]
            mean,sd=normalization[mode]
            xs.append(torch.from_numpy(((r['features'][mode][indices,:26]-mean)/sd).clip(-8,8)))
        groups.append(xs)
    torch.manual_seed(2026)
    rng=np.random.default_rng(2026)
    model=Descriptor()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    history=[]
    for step in range(steps):
        x,y=groups[int(rng.integers(len(groups)))]
        # Only different verified identities serve as negatives. Unverified
        # cells are not implicitly labeled biologically nonmatching.
        za=model(x+torch.randn_like(x)*.04,0)
        zb=model(y+torch.randn_like(y)*.04,1)
        logits=za@zb.T/.15
        labels=torch.arange(len(x))
        loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%100==0: history.append(float(loss.detach()))
    model.eval()
    return model,normalization,history


def similarities(model,normalization,features):
    with torch.inference_mode():
        z=[]
        for mode in range(2):
            mean,sd=normalization[mode]
            x=torch.from_numpy(((features[mode][:,:26]-mean)/sd).clip(-8,8))
            z.append(model(x,mode).numpy())
    return z[0]@z[1].T


def fit_quality(records):
    x=np.concatenate([r['features'][1] for r in records])
    y=np.concatenate([r['quality'] for r in records]).astype(np.float32)
    mean,sd=x.mean(0),np.maximum(x.std(0),.03)
    xt=torch.from_numpy(((x-mean)/sd).clip(-8,8));yt=torch.from_numpy(y)
    torch.manual_seed(2026)
    model=nn.Linear(30,1)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=.5,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=F.binary_cross_entropy_with_logits(model(xt).ravel(),yt)+.02*model.weight.square().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    return model.eval(),mean,sd


def quality_probabilities(fitted,features):
    model,mean,sd=fitted
    with torch.inference_mode():
        return model(torch.from_numpy(((features-mean)/sd).clip(-8,8))).sigmoid().numpy().ravel()


def calibrate_quality(records):
    """Threshold selection uses only the two outer-training mice."""
    subjects=sorted({r['subject'] for r in records})
    if len(subjects)!=2: raise ValueError('Expected exactly two development mice')
    scored=[]
    for subject in subjects:
        fitted=fit_quality([r for r in records if r['subject']!=subject])
        for record in records:
            if record['subject']==subject:
                scored.append((record,quality_probabilities(fitted,record['features'][1])))
    measurements=[]
    for threshold in (0.,.15,.25,.35,.45,.55,.65):
        pqs=[]
        for record,probability in scored:
            keep=probability>=threshold
            denominator=.5*(record['gt_count']+int(keep.sum()))
            pqs.append(float(record['ious'][keep].sum()/denominator) if denominator else 0.)
        measurements.append({'threshold':threshold,'mean_pq':float(np.mean(pqs))})
    threshold=max(measurements,key=lambda r:r['mean_pq'])['threshold']
    return fit_quality(records),threshold,measurements


def build_record(root,sid,row,maps=None,truth=None):
    record={'sample_id':sid,'subject':split_id(sid)[0],'keys':[],'centers':[],'features':[],'masks':[],'shapes':[]}
    for modality in ('invivo','exvivo'):
        image=read_image(image_path(root,'training',sid,modality))
        masks=parse_instances(row,modality,image.shape)
        keys,centers,features=extract_features(image,masks,None if maps is None else maps[modality])
        record['keys'].append(keys);record['centers'].append(centers);record['features'].append(features)
        record['masks'].append(masks);record['shapes'].append(image.shape)
    if truth is not None:
        gt=parse_instances(truth,'exvivo',record['shapes'][1])
        score=segmentation_score(record['masks'][1],gt)
        mapping=score['pred_to_gt']
        record['quality']=np.array([k in mapping for k in record['keys'][1]])
        record['ious']=np.array([record['masks'][1][k].iou(gt[mapping[k]]) if k in mapping else 0 for k in record['keys'][1]])
        record['gt_count']=len(gt)
    return record


def development_predictions(args,tag,truth,checkpoint,device):
    heldout=FOLDS[tag]
    train_ids=set(checkpoint['train_ids'])
    expected={sid for sid in truth if split_id(sid)[0]!=heldout}
    if train_ids!=expected: raise ValueError('Outer-fold training partition mismatch')
    output=Path(args.output)/tag
    path=output/'development.csv'
    cache=output/'development_maps'
    signature=development_signature(Path(args.stage2)/f'unet_holdout_{tag}/model.pt')
    manifest_path=output/'development_manifest.json'
    output.mkdir(parents=True,exist_ok=True);cache.mkdir(exist_ok=True)
    if path.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text())!=signature:
            raise ValueError('Development cache provenance differs; choose a new output directory')
        rows=read_rows(path)
        if set(rows)!=expected: raise ValueError('Cached development IDs differ')
        return rows,cache
    model=SmallUNet(checkpoint['width']).to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    rows=[]
    for n,sid in enumerate(sorted(expected),1):
        row={'sample_id':sid,'match_pairs':[]}
        for mode,modality in enumerate(('invivo','exvivo')):
            image=read_image(image_path(args.data,'training',sid,modality))
            maps=probability_maps(model,image,mode,device)
            np.savez_compressed(cache/f'{sid}_{modality}.npz',maps=maps.astype(np.float16))
            row[f'{modality}_instances']=labels_to_instances(instances_from_maps(maps),'IVP' if mode==0 else 'EXP')
        rows.append(row)
        print(f'{tag}: development inference {n}/{len(expected)}',flush=True)
    write_rows(rows,path)
    manifest_path.write_text(json.dumps(signature,indent=2)+'\n')
    del model
    if device=='mps': torch.mps.empty_cache()
    return read_rows(path),cache


def load_maps(cache,sid):
    return {m:np.load(cache/f'{sid}_{m}.npz')['maps'].astype(np.float32) for m in ('invivo','exvivo')}


def run_fold(args,tag,truth,ground_truth_records,device):
    heldout=FOLDS[tag]
    output=Path(args.output)/tag
    checkpoint_path=Path(args.stage2)/f'unet_holdout_{tag}/model.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint['held_out']!=heldout: raise ValueError('Checkpoint held-out subject mismatch')
    development,cache=development_predictions(args,tag,truth,checkpoint,device)
    dev_records=[build_record(args.data,sid,row,load_maps(cache,sid),truth[sid]) for sid,row in development.items()]
    fitted,threshold,calibration=calibrate_quality(dev_records)
    descriptor,normalization,losses=fit_descriptor(ground_truth_records,heldout,args.steps)
    torch.save({'descriptor':descriptor.state_dict(),'normalization':normalization,
                'quality':fitted[0].state_dict(),'quality_mean':fitted[1],'quality_sd':fitted[2],
                'threshold':threshold,'held_out':heldout,'train_ids':checkpoint['train_ids']},output/'models.pt')
    # Outer held-out labels are not used until evaluation below.
    predictions=read_rows(Path(args.stage2)/f'unet_holdout_{tag}.csv')
    expected={sid for sid in truth if split_id(sid)[0]==heldout}
    if set(predictions)!=expected: raise ValueError('Held-out predictions do not exactly cover subject')
    map_cache=Path(args.stage2)/f'unet_holdout_{tag}_maps'
    variants={name:[] for name in ('quality','matching','combined','geometry_filter')}
    diagnostics=[]
    for n,(sid,row) in enumerate(predictions.items(),1):
        record=build_record(args.data,sid,row,load_maps(map_cache,sid))
        probability=quality_probabilities(fitted,record['features'][1])
        similarity=similarities(descriptor,normalization,record['features'])
        matrix,diagnostic=fit_feature_geometry(*record['centers'],similarity)
        pairs=assign_pairs(*record['masks'],matrix,radius=5.,ambiguity_ratio=.75)
        keep={k for k,p in zip(record['keys'][1],probability) if p>=threshold}
        quality=copy.deepcopy(row)
        quality['exvivo_instances']={k:v for k,v in row['exvivo_instances'].items() if k in keep}
        matching=copy.deepcopy(row);matching['match_pairs']=pairs
        combined=copy.deepcopy(quality)
        combined['match_pairs']=[p for p in pairs if p[1] in keep]
        geometric=copy.deepcopy(row)
        removed=set()
        if matrix is not None and diagnostic['inliers']>=12:
            inverse=np.linalg.inv(matrix[:,:2])
            projected=(record['centers'][1]-matrix[:,2])@inverse.T
            height,width=record['shapes'][0]
            inside=(projected[:,0]>8)&(projected[:,0]<width-8)&(projected[:,1]>8)&(projected[:,1]<height-8)
            distance=cKDTree(record['centers'][0]).query(projected)[0]
            # Experimental geometry-only ablation: reject weak candidates only
            # inside the inferred overlapping field, never all unmatched cells.
            removed={k for k,p,ok,d in zip(record['keys'][1],probability,inside,distance)
                     if ok and d>12 and p<max(threshold,.35)}
            geometric['exvivo_instances']={k:v for k,v in row['exvivo_instances'].items() if k not in removed}
        for name,value in zip(variants,(quality,matching,combined,geometric)): variants[name].append(value)
        diagnostic.update(sample_id=sid,pairs=len(pairs),quality_removed=len(row['exvivo_instances'])-len(keep),geometry_removed=len(removed))
        diagnostics.append(diagnostic)
        print(f"{tag}: held-out {n}/{len(predictions)}; pairs={len(pairs)}, EX removed={diagnostic['quality_removed']}",flush=True)
    for name,rows in variants.items():
        path=output/f'{name}.csv';write_rows(rows,path)
        evaluate(SimpleNamespace(data=args.data,predictions=path,subject=heldout,output=output/f'{name}_metrics.json'))
    report={'held_out':heldout,'train_ids':checkpoint['train_ids'],'quality_threshold':threshold,
            'quality_calibration':calibration,'descriptor_losses':losses,'descriptor_steps':args.steps,
            'segmentation_checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'feature_source_sha256':hashlib.sha256(Path(__file__).with_name('cell_features.py').read_bytes()).hexdigest(),
            'regions':diagnostics}
    (output/'experiment.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/raw/Project_2_Dataset')
    parser.add_argument('--stage2',default='outputs/stage2')
    parser.add_argument('--output',default='outputs/stage3')
    parser.add_argument('--fold',choices=['all',*FOLDS],default='all')
    parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=choose_device(args.device)
    truth=read_rows(Path(args.data)/'training/train_ground_truth.csv')
    selected=list(FOLDS) if args.fold=='all' else [args.fold]
    # Ground-truth masks are used only for descriptor fitting on outer-training
    # mice, not to extract held-out prediction features or initialize transforms.
    subjects={FOLDS[tag] for tag in selected}
    records=[]
    for sid,row in truth.items():
        if len(selected)==1 and split_id(sid)[0] in subjects: continue
        record=build_record(args.data,sid,row)
        record['pairs']=row['match_pairs'];records.append(record)
    for tag in selected: run_fold(args,tag,truth,records,device)
    if args.fold=='all':
        for name in ('quality','matching','combined','geometry_filter'):
            rows=[]
            for tag in FOLDS: rows.extend(read_rows(Path(args.output)/tag/f'{name}.csv').values())
            path=Path(args.output)/f'{name}_oof.csv';write_rows(rows,path)
            evaluate(SimpleNamespace(data=args.data,predictions=path,subject=None,output=Path(args.output)/f'{name}_oof_metrics.json'))


if __name__=='__main__': main()
