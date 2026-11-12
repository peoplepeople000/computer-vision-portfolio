"""Training-only edge-credit learning followed by optional unmatched assignment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from .data import read_rows,write_rows,parse_instances,read_image,image_path,split_id
from .matching import centroids
from .cell_features import extract_features
from .pair_quality import pair_features,training_record
from .stage3 import FOLDS
from .metrics import segmentation_score
from .cli import evaluate


def candidate_features(masks,features,matrix):
    if matrix is None or any(not m for m in masks):return [],np.empty((0,33),np.float32)
    keys,points=zip(*(centroids(m) for m in masks))
    distances=cdist(points[0]@matrix[:,:2].T+matrix[:,2],points[1])
    ii,jj=np.where(distances<8)
    pairs=[[keys[0][i],keys[1][j]] for i,j in zip(ii,jj)]
    x=pair_features(masks,features,pairs,matrix)
    differences=np.abs(x[:,:9]-x[:,9:18])
    return pairs,np.c_[x,differences,x[:,18]**2,np.log1p(x[:,20])].astype(np.float32)


def assign_edges(pairs,probabilities,threshold,distances,radius):
    if not pairs:return []
    left=sorted({a for a,b in pairs});right=sorted({b for a,b in pairs})
    ai={k:i for i,k in enumerate(left)};bi={k:i for i,k in enumerate(right)}
    utility=np.zeros((len(left),len(right)+len(left)),np.float64)
    utility[:,:len(right)]=-1e6
    for n,((a,b),p) in enumerate(zip(pairs,probabilities)):
        if distances[n]<radius and p>threshold:utility[ai[a],bi[b]]=p-threshold
    rows,cols=linear_sum_assignment(-utility)
    return [[left[i],right[j]] for i,j in zip(rows,cols) if j<len(right) and utility[i,j]>0]


def fit(records):
    x=np.concatenate([r['x'] for r in records]);y=np.concatenate([r['y'] for r in records]).astype(np.float32)
    mean=x.mean(0);sd=np.maximum(x.std(0),.03);xt=torch.from_numpy(((x-mean)/sd).clip(-8,8));yt=torch.from_numpy(y)
    torch.manual_seed(2033);model=nn.Linear(x.shape[1],1);optimizer=torch.optim.LBFGS(model.parameters(),max_iter=150,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(xt).ravel(),yt)+.02*model.weight.square().mean();loss.backward();return loss
    optimizer.step(closure);return model.eval(),mean,sd


def probability(fitted,x):
    model,mean,sd=fitted
    with torch.inference_mode():return model(torch.from_numpy(((x-mean)/sd).clip(-8,8))).sigmoid().numpy().ravel()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--iv-capacity',action='store_true');parser.add_argument('--iv-quality',action='store_true');parser.add_argument('--rescue-ex',action='store_true')
    parser.add_argument('--relative-ex-quality',action='store_true')
    parser.add_argument('--ex-capacity-root')
    parser.add_argument('--geometry',default='outputs/stage7_consensus')
    parser.add_argument('--output');args=parser.parse_args()
    if args.relative_ex_quality and args.ex_capacity_root:raise ValueError('Choose one EX family')
    if (args.relative_ex_quality or args.ex_capacity_root) and not (args.iv_capacity and args.iv_quality and args.output):raise ValueError('New EX families require capacity IV, IV quality, and explicit output')
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    baseline_path=('outputs/stage8_iv_quality_capacity/segmentation_oof.csv' if args.iv_capacity else 'outputs/stage8_iv_quality/segmentation_oof.csv') if args.iv_quality else ('outputs/stage8_iv/segmentation_oof.csv' if args.iv_capacity else 'outputs/stage7_tiles/native/predicted_all.csv')
    if args.relative_ex_quality:baseline_path='outputs/stage10_relative_ex_quality/segmentation_oof.csv'
    if args.ex_capacity_root:baseline_path=str(Path(args.ex_capacity_root)/'segmentation_oof.csv')
    baseline=read_rows(baseline_path)
    manifest=json.loads((Path(args.geometry)/'predicted_all_manifest.json').read_text())
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in manifest['regions'] if r.get('matrix') and r.get('status')=='registered'}
    calibration=json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    out=Path(args.output or ('outputs/stage8_joint'+('_capacity' if args.iv_capacity else '')+('_iv_quality' if args.iv_quality else '')+('_rescue_ex' if args.rescue_ex else '')));out.mkdir(exist_ok=True);results=[];reports={}
    if any((out/f'{tag}.pt').exists() for tag in FOLDS):raise FileExistsError('Preserving existing pair checkpoints; choose a new output')
    (out/'pipeline.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject}
        if args.iv_capacity:
            iv_folder=Path(f'outputs/stage8_iv/{tag}');base=read_rows(iv_folder/'development.csv');iv_maps=iv_folder/'maps'
            config=json.loads((iv_folder/'training_config.json').read_text())
        else:
            iv_folder=Path(f'outputs/stage7_iv_development/{tag}');base=read_rows(iv_folder/'predictions.csv');iv_maps=iv_folder
            config=json.loads((iv_folder/'manifest.json').read_text())
        if set(base)!=ids or set(config['train_ids'])!=ids:raise ValueError('Invalid IV development split')
        quality_path=Path(f'outputs/stage6_specialist_quality/{tag}/model.pt')
        if args.relative_ex_quality:quality_path=Path(f'outputs/stage10_relative_ex_quality/{tag}.pt')
        if args.ex_capacity_root:quality_path=Path(args.ex_capacity_root)/tag/'quality.pt'
        state=torch.load(quality_path,map_location='cpu',weights_only=False)
        if state['held_out']!=subject or set(state['train_ids'])!=ids or set(calibration[subject]['train_ids'])!=ids:raise ValueError('Invalid EX split')
        ex_manifest=json.loads(Path(f'outputs/stage6_ex/development_{tag}/manifest.json').read_text())
        ex_cache=Path(f'outputs/stage6_ex/development_{tag}')
        ex_foreground=calibration[subject]['threshold']
        if args.ex_capacity_root:
            ex_cache=Path(args.ex_capacity_root)/tag/'maps'
            ex_manifest=json.loads((Path(args.ex_capacity_root)/tag/'training_config.json').read_text())
            ex_foreground=state['foreground']
        quality_transform=None
        if args.relative_ex_quality:
            from .relative_ex_quality import relative_features
            quality_transform=relative_features
        if set(ex_manifest['train_ids'])!=ids or ex_manifest['held_out']!=subject:raise ValueError('Invalid EX development maps')
        q=nn.Linear(30,1);q.load_state_dict(state['quality']);quality=(q.eval(),state['mean'],state['sd'],state['threshold'])
        iv_quality=None
        if args.iv_quality:
            from .ex_joint_calibration import morphology_features
            from .stage3 import quality_probabilities
            qfolder=Path('outputs/stage8_iv_quality_capacity' if args.iv_capacity else 'outputs/stage8_iv_quality')
            iv_state=torch.load(qfolder/f'{tag}.pt',map_location='cpu',weights_only=False)
            if set(iv_state['train_ids'])!=ids or iv_state['held_out']!=subject:raise ValueError('Wrong IV quality fold')
            iv_model=nn.Linear(30,1);iv_model.load_state_dict(iv_state['quality']);iv_quality=(iv_model.eval(),iv_state['mean'],iv_state['sd'])
        records=[]
        for sid,row in base.items():
            if iv_quality is not None:
                iv_probability=np.load(iv_maps/f'{sid}_invivo.npz')['maps'].astype(np.float32)
                iv_masks=parse_instances(row,'invivo',iv_probability.shape[1:]);iv_keys,iv_features=morphology_features(iv_masks,iv_probability)
                iv_prob=quality_probabilities(iv_quality,iv_features);iv_keep={k for k,v in zip(iv_keys,iv_prob) if v>=iv_state['threshold']}
                row=copy.deepcopy(row);row['invivo_instances']={k:v for k,v in row['invivo_instances'].items() if k in iv_keep}
            maps=np.load(ex_cache/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
            record=training_record(root,sid,truth[sid],row,maps,ex_foreground,quality[:3]+(0.,) if args.rescue_ex else quality,quality_transform)
            if record is None:continue
            _,masks,actual,matrix=record;features=[]
            for mode,modality in enumerate(['invivo','exvivo']):
                probability_maps=np.load(iv_maps/f'{sid}_invivo.npz')['maps'].astype(np.float32) if mode==0 else maps
                features.append(extract_features(read_image(image_path(root,'training',sid,modality)),masks[mode],probability_maps)[2])
            pairs,x=candidate_features(masks,features,matrix)
            mapping=[segmentation_score(m,g)['pred_to_gt'] for m,g in zip(masks,actual)];verified=set(map(tuple,truth[sid]['match_pairs']))
            y=np.array([(mapping[0].get(a),mapping[1].get(b)) in verified for a,b in pairs])
            records.append({'sample_id':sid,'subject':split_id(sid)[0],'pairs':pairs,'x':x,'y':y})
            print('Joint training edges',tag,sid,len(pairs),flush=True)
        inner=[]
        for mouse in sorted({r['subject'] for r in records}):
            fitted=fit([r for r in records if r['subject']!=mouse])
            inner.extend((r,probability(fitted,r['x'])) for r in records if r['subject']==mouse)
        measurements=[];verified=sum(len(truth[s]['match_pairs']) for s in ids)
        for radius in (4.,5.,6.,8.):
            for threshold in (.05,.1,.15,.2,.25,.3,.35,.4,.45,.5):
                correct=count=0
                for r,p in inner:
                    chosen=assign_edges(r['pairs'],p,threshold,r['x'][:,18],radius);positive={tuple(pair) for pair,y in zip(r['pairs'],r['y']) if y}
                    correct+=sum(tuple(pair) in positive for pair in chosen);count+=len(chosen)
                measurements.append({'radius':radius,'threshold':threshold,'f1':2*correct/(count+verified)})
        selected=max(measurements,key=lambda v:v['f1']);fitted=fit(records)
        torch.save({'state_dict':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],'held_out':subject,'train_ids':sorted(ids),'selected':selected,'feature_count':33},out/f'{tag}.pt')
        reports[subject]={'train_ids':sorted(ids),'selected':selected,'inner_scores':measurements,
            'rescue_ex':args.rescue_ex,'iv_quality':args.iv_quality,'iv_capacity':args.iv_capacity,'calibration_note':'Inner subject classifier holdout; segmentation models see development mice. Regions with fewer than five verified pairs contribute denominator only.'}
        for sid,row in baseline.items():
            if split_id(sid)[0]!=subject:continue
            original_ex_keep=None
            if args.rescue_ex:
                from .learning import instances_from_maps
                from .rle import labels_to_instances
                from .ex_joint_calibration import morphology_features
                from .stage3 import quality_probabilities
                row=copy.deepcopy(row)
                heldout_ex_cache=ex_cache if args.ex_capacity_root else Path('outputs/stage7_tiles/maps')
                raw_maps=np.load(heldout_ex_cache/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
                row['exvivo_instances']=labels_to_instances(instances_from_maps(raw_maps,ex_foreground,.35),'EXP')
                raw_masks=parse_instances(row,'exvivo',raw_maps.shape[1:]);raw_keys,raw_features=morphology_features(raw_masks,raw_maps)
                if quality_transform is not None:raw_features=quality_transform(raw_features)
                raw_prob=quality_probabilities(quality[:3],raw_features);original_ex_keep={k for k,v in zip(raw_keys,raw_prob) if v>=quality[3]}
            masks=[];features=[]
            for modality in ['invivo','exvivo']:
                image=read_image(image_path(root,'training',sid,modality));masks.append(parse_instances(row,modality,image.shape))
                cache=Path(f'outputs/stage8_iv/{tag}/maps') if args.iv_capacity and modality=='invivo' else Path('outputs/stage7_tiles/maps')
                if args.ex_capacity_root and modality=='exvivo':cache=ex_cache
                maps=np.load(cache/f'{sid}_{modality}.npz')['maps'].astype(np.float32)
                features.append(extract_features(image,masks[-1],maps)[2])
            pairs,x=candidate_features(masks,features,matrices.get(sid));p=probability(fitted,x)
            value=copy.deepcopy(row);value['match_pairs']=assign_edges(pairs,p,selected['threshold'],x[:,18],selected['radius'])
            if original_ex_keep is not None:
                keep=original_ex_keep|{b for a,b in value['match_pairs']}
                value['exvivo_instances']={k:v for k,v in value['exvivo_instances'].items() if k in keep}
            results.append(value)
        print('Joint selection',tag,selected,flush=True)
    write_rows(results,out/'predicted_all.csv');(out/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n')
    (out/'predicted_all_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predicted_all.csv',subject=None,output=out/'predicted_all_metrics.json'))

if __name__=='__main__':main()
