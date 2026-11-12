"""Training-only calibration of conservative EX probability contour changes."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy import ndimage as ndi
from .data import read_rows, write_rows, parse_instances, read_image, image_path, split_id
from .rle import Mask, labels_to_instances
from .ex_boundary import patch_rle
from .learning import instances_from_maps
from .cell_features import extract_features
from .stage3 import FOLDS, quality_probabilities
from .metrics import segmentation_score
from .cli import evaluate

POLICIES = [(0., 0.)] + [(delta, sigma) for sigma in (0., .6) for delta in (-.1, -.05, 0., .05, .1) if (delta, sigma) != (0., 0.)]


def adjust(masks, maps, threshold, delta, sigma):
    """Keep identities and constrain updates to two-pixel contour neighborhoods."""
    if delta == 0 and sigma == 0:
        return {k: ' '.join(f'{a} {b-a}' for a,b in zip(m.starts,m.ends)) for k,m in masks.items()}
    shape = maps.shape[1:]
    occupied = np.zeros(shape, np.int32)
    for index, m in enumerate(masks.values(), 1):
        for a,b in zip(m.starts,m.ends): occupied.ravel()[a:b] = index
    probability = ndi.gaussian_filter(maps[0], sigma) if sigma else maps[0]
    result = {}
    for index,(key,m) in enumerate(masks.items(),1):
        xmin,ymin,xmax,ymax = m.bbox
        x0,y0=max(0,xmin-3),max(0,ymin-3)
        x1,y1=min(shape[1],xmax+4),min(shape[0],ymax+4)
        prior=occupied[y0:y1,x0:x1]==index
        candidate=(probability[y0:y1,x0:x1] > threshold+delta)
        candidate &= ndi.binary_dilation(prior,iterations=2)
        candidate &= (occupied[y0:y1,x0:x1]==0)|prior
        components,count=ndi.label(candidate)
        overlap=np.bincount(components[prior],minlength=count+1);overlap[0]=0
        candidate=components==int(np.argmax(overlap)) if overlap.max() else prior.copy()
        new=Mask.parse(patch_rle(candidate,x0,y0,shape),shape)
        if not (.65 <= new.area/m.area <= 1.4 and new.iou(m)>.6):
            candidate=prior;new=m
        encoded=patch_rle(candidate,x0,y0,shape)
        result[key]=encoded
        # Reserve accepted growth so output instances cannot overlap.
        occupied[y0:y1,x0:x1][candidate]=index
    return result


def main():
    torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    baseline=read_rows('outputs/stage7_tiles/native/predicted_all.csv')
    calibration=json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    out=Path('outputs/stage8_ex_contour');out.mkdir(exist_ok=True);allrows=[];reports={}
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject}
        config=calibration[subject]
        state=torch.load(f'outputs/stage6_specialist_quality/{tag}/model.pt',map_location='cpu',weights_only=False)
        manifest=json.loads(Path(f'outputs/stage6_ex/development_{tag}/manifest.json').read_text())
        checkpoint=Path(f'outputs/stage6_ex/unet_holdout_{tag}/model.pt')
        if set(config['train_ids'])!=ids or set(state['train_ids'])!=ids or set(manifest['train_ids'])!=ids or manifest['held_out']!=subject:
            raise ValueError('Held-out data in calibration')
        if manifest['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():raise ValueError('Stale EX checkpoint cache')
        model=nn.Linear(30,1);model.load_state_dict(state['quality']);fitted=(model.eval(),state['mean'],state['sd'])
        scores=[[] for _ in POLICIES]
        for sid in sorted(ids):
            maps=np.load(f'outputs/stage6_ex/development_{tag}/{sid}_exvivo.npz')['maps'].astype(np.float32)
            image=read_image(image_path(root,'training',sid,'exvivo'))
            instances=labels_to_instances(instances_from_maps(maps,config['threshold'],.35),'EXP')
            masks=parse_instances({'exvivo_instances':instances},'exvivo',image.shape)
            keys,_,features=extract_features(image,masks,maps);features[:,5:26]=0
            p=quality_probabilities(fitted,features);masks={k:masks[k] for k,v in zip(keys,p) if v>=state['threshold']}
            actual=parse_instances(truth[sid],'exvivo',image.shape)
            for i,(delta,sigma) in enumerate(POLICIES):
                proposed=adjust(masks,maps,config['threshold'],delta,sigma)
                parsed=parse_instances({'exvivo_instances':proposed},'exvivo',image.shape)
                scores[i].append(segmentation_score(parsed,actual)['pq'])
            print('Contour calibration',tag,sid,flush=True)
        means=np.mean(scores,axis=1);chosen=int(np.argmax(means));delta,sigma=POLICIES[chosen]
        reports[subject]={'train_ids':sorted(ids),'threshold':config['threshold'],'delta':delta,'sigma':sigma,
            'training_scores':[{'delta':d,'sigma':s,'pq':float(v)} for (d,s),v in zip(POLICIES,means)],
            'caveat':'Training image calibration uses 256px inference; held-out inference uses 128px tiles.'}
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            maps=np.load(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')['maps'].astype(np.float32)
            masks=parse_instances(row,'exvivo',maps.shape[1:]);value=copy.deepcopy(row)
            value['exvivo_instances']=adjust(masks,maps,config['threshold'],delta,sigma);value['match_pairs']=[];allrows.append(value)
        print('Selected contour',tag,delta,sigma,float(means[chosen]),flush=True)
        (out/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n')
    path=out/'segmentation_oof.csv';write_rows(allrows,path)
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))

if __name__=='__main__':main()
