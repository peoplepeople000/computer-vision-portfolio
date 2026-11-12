"""Training-selected local EX contour regularization with frozen detections."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image
from .data import read_rows, write_rows, parse_instances, split_id, image_path, image_shape
from .rle import Mask, labels_to_instances
from .learning import instances_from_maps
from .stage3 import FOLDS, development_signature
from .metrics import segmentation_score
from .cli import evaluate


def regularize(masks, method):
    if method not in ('none','closing','convex'): raise ValueError('Unknown regularizer')
    if not masks: return {}
    shape=next(iter(masks.values())).shape
    occupied=np.zeros(shape,np.int32)
    for n,mask in enumerate(masks.values(),1):
        for a,b in zip(mask.starts,mask.ends): occupied.ravel()[a:b]=n
    output={}
    for n,(key,mask) in enumerate(masks.items(),1):
        if method=='none' or not mask.area:
            output[key]=' '.join(f'{a} {b-a}' for a,b in zip(mask.starts,mask.ends));continue
        xmin,ymin,xmax,ymax=mask.bbox
        x0=max(0,xmin-2);y0=max(0,ymin-2);x1=min(shape[1],xmax+3);y1=min(shape[0],ymax+3)
        patch=np.zeros((y1-y0,x1-x0),bool);xy=mask.coordinates();patch[xy[:,1]-y0,xy[:,0]-x0]=True
        fixed=convex_hull_image(patch) if method=='convex' else ndi.binary_closing(patch,structure=ndi.generate_binary_structure(2,1))
        fixed |= patch
        fixed &= ((occupied[y0:y1,x0:x1]==0)|(occupied[y0:y1,x0:x1]==n))
        # Reserve additions to prevent overlap with later instances.
        occupied[y0:y1,x0:x1][fixed]=n
        runs=[]
        for y,line in enumerate(fixed):
            edges=np.flatnonzero(np.diff(np.r_[False,line,False])).reshape(-1,2)
            for a,b in edges:runs.extend(((y+y0)*shape[1]+a+x0,b-a))
        output[key]=' '.join(map(str,runs))
    return output


def main():
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    baseline=read_rows('outputs/stage5_hybrid_geometry/predicted_all.csv')
    calibration=json.loads(Path('outputs/stage4_calibrated/calibration.json').read_text())
    out=Path('outputs/stage6_shape');out.mkdir(exist_ok=True);rows=[];reports={}
    for tag,subject in FOLDS.items():
        folder=Path(f'outputs/stage3/{tag}');checkpoint=f'outputs/stage2/unet_holdout_{tag}/model.pt'
        if json.loads((folder/'development_manifest.json').read_text())!=development_signature(checkpoint):raise ValueError('Stale maps')
        development=read_rows(folder/'development.csv');expected={s for s in truth if split_id(s)[0]!=subject}
        if set(development)!=expected or set(calibration[subject]['train_ids'])!=expected:raise ValueError('Training partition mismatch')
        threshold=calibration[subject]['thresholds']['exvivo'];scores={m:[] for m in ['none','closing','convex']}
        for sid in development:
            maps=np.load(folder/'development_maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
            instances=(development[sid]['exvivo_instances'] if threshold==.5 else labels_to_instances(instances_from_maps(maps,threshold,.35),'EXP'))
            masks=parse_instances({'exvivo_instances':instances},'exvivo',maps.shape[1:]);actual=parse_instances(truth[sid],'exvivo',maps.shape[1:])
            for method in scores:
                transformed=regularize(masks,method)
                guess=parse_instances({'exvivo_instances':transformed},'exvivo',maps.shape[1:])
                scores[method].append(segmentation_score(guess,actual)['pq'])
        means={k:float(np.mean(v)) for k,v in scores.items()};chosen=max(means,key=means.get)
        reports[subject]={'train_ids':sorted(expected),'threshold':threshold,'training_mean_pq':means,'chosen':chosen}
        print(subject,reports[subject]['training_mean_pq'],chosen,flush=True)
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            shape=image_shape(image_path(root,'training',sid,'exvivo'))
            masks=parse_instances(row,'exvivo',shape)
            rows.append({'sample_id':sid,'invivo_instances':row['invivo_instances'],
                         'exvivo_instances':regularize(masks,chosen),'match_pairs':[]})
    path=out/'segmentation_oof.csv';write_rows(rows,path)
    (out/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
