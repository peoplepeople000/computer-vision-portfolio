"""Apply a frozen outer-fold pair calibrator to an alternate registration."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .cell_features import extract_features
from .stage3 import FOLDS
from .pair_quality import pair_features,probabilities
from .cli import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--models',default='outputs/stage7_pair_quality');p.add_argument('--output',required=True);p.add_argument('--geometry');p.add_argument('--tile-variant',choices=['native','mean']);args=p.parse_args()
    root=Path('data/raw/Project_2_Dataset');folder=Path(args.input);rows=read_rows(folder/'predicted_all.csv')
    manifest=json.loads(((Path(args.geometry) if args.geometry else folder)/'predicted_all_manifest.json').read_text());matrices={r['sample_id']:np.asarray(r['matrix']) for r in manifest['regions'] if r.get('matrix') and r.get('status')=='registered'}
    result=[];reports=[]
    for tag,subject in FOLDS.items():
        state=torch.load(Path(args.models)/f'{tag}.pt',map_location='cpu',weights_only=False)
        if state['held_out']!=subject or any(split_id(s)[0]==subject for s in state['train_ids']):raise ValueError('Wrong pair model partition')
        model=nn.Linear(22,1);model.load_state_dict(state['state_dict']);fitted=(model.eval(),state['mean'],state['sd'])
        for sid,row in rows.items():
            if split_id(sid)[0]!=subject:continue
            value=copy.deepcopy(row)
            if row['match_pairs']:
                masks=[];features=[]
                for mode,modality in enumerate(['invivo','exvivo']):
                    image=read_image(image_path(root,'training',sid,modality));m=parse_instances(row,modality,image.shape);masks.append(m)
                    cache=f'outputs/stage5_refinement/unet_holdout_{tag}_maps' if mode==0 else f'outputs/stage6_ex/unet_holdout_{tag}_maps'
                    maps=np.load(f'{cache}/{sid}_{modality}.npz')['maps'].astype(np.float32)
                    if args.tile_variant:
                        native=np.load(f'outputs/stage7_tiles/maps/{sid}_{modality}.npz')['maps'].astype(np.float32)
                        maps=native if args.tile_variant=='native' else (maps+native)/2
                    features.append(extract_features(image,m,maps)[2])
                probability=probabilities(fitted,pair_features(masks,features,row['match_pairs'],matrices[sid]))
                value['match_pairs']=[pair for pair,v in zip(row['match_pairs'],probability) if v>=state['threshold']]
                reports.append({'sample_id':sid,'threshold':state['threshold'],'input_pairs':row['match_pairs'],'probabilities':probability.tolist()})
            result.append(value)
    out=Path(args.output);out.mkdir(exist_ok=True);path=out/'predicted_all.csv';write_rows(result,path)
    (out/'pair_filter.json').write_text(json.dumps({'models':args.models,'input':args.input,'regions':reports},indent=2)+'\n')
    (out/'predicted_all_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'predicted_all_metrics.json'))


if __name__=='__main__':main()
