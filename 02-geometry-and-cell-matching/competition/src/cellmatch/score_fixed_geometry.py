"""Score new masks with frozen training-derived transforms and pair models."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .data import read_rows,write_rows,parse_instances,read_image,image_path,split_id
from .stage3 import FOLDS
from .registration import assign_pairs
from .cell_features import extract_features
from .pair_quality import pair_features,probabilities
from .cli import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--iv-capacity',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');rows=read_rows(args.input)
    manifest=json.loads(Path('outputs/stage7_consensus/predicted_all_manifest.json').read_text())
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in manifest['regions'] if r.get('matrix')}
    result=[]
    for tag,subject in FOLDS.items():
        state=torch.load(f'outputs/stage7_pair_quality_refined/{tag}.pt',map_location='cpu',weights_only=False)
        if state['held_out']!=subject or set(state['train_ids'])!={s for s in truth if split_id(s)[0]!=subject}:raise ValueError('Invalid pair fold')
        model=nn.Linear(22,1);model.load_state_dict(state['state_dict']);fitted=(model.eval(),state['mean'],state['sd'])
        for sid,row in rows.items():
            if split_id(sid)[0]!=subject:continue
            masks=[];features=[]
            for modality in ['invivo','exvivo']:
                image=read_image(image_path(root,'training',sid,modality));masks.append(parse_instances(row,modality,image.shape))
                cache=Path(f'outputs/stage8_iv/{tag}/maps') if args.iv_capacity and modality=='invivo' else Path('outputs/stage7_tiles/maps')
                maps=np.load(cache/f'{sid}_{modality}.npz')['maps'].astype(np.float32)
                features.append(extract_features(image,masks[-1],maps)[2])
            matrix=matrices.get(sid);pairs=assign_pairs(*masks,matrix,radius=5,ambiguity_ratio=.8)
            x=pair_features(masks,features,pairs,matrix) if pairs else np.empty((0,22),np.float32)
            p=probabilities(fitted,x);value=copy.deepcopy(row);value['match_pairs']=[pair for pair,v in zip(pairs,p) if v>=state['threshold']];result.append(value)
            print('Frozen geometry scoring',sid,flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);write_rows(result,out/'predicted_all.csv')
    (out/'predicted_all_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'pipeline.json').write_text(json.dumps({'input':args.input,'geometry':'stage7_consensus','pair_models':'stage7_pair_quality_refined','iv_capacity':args.iv_capacity},indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predicted_all.csv',subject=None,output=out/'predicted_all_metrics.json'))

if __name__=='__main__':main()
