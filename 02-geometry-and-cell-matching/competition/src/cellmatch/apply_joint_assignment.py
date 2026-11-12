"""Apply saved joint edge-credit models to a complete new mask prediction set."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .data import read_rows,write_rows,parse_instances,split_id
from .ex_joint_calibration import morphology_features
from .joint_pair_assignment import candidate_features,probability,assign_edges
from .stage3 import FOLDS
from .cli import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--geometry',default='outputs/stage7_consensus');parser.add_argument('--models',default='outputs/stage8_joint');parser.add_argument('--iv-capacity',action='store_true')
    parser.add_argument('--ex-map-root',help='Optional EX experiment root with <fold>/model.pt and <fold>/maps/')
    args=parser.parse_args()
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');rows=read_rows(args.input)
    if set(rows)!=set(truth):raise ValueError('Expected complete 47-region OOF inputs')
    manifest=json.loads((Path(args.geometry)/'predicted_all_manifest.json').read_text());matrices={r['sample_id']:np.asarray(r['matrix']) for r in manifest['regions'] if r.get('matrix') and r.get('status')=='registered'}
    results=[]
    for tag,subject in FOLDS.items():
        state=torch.load(Path(args.models)/f'{tag}.pt',map_location='cpu',weights_only=False)
        if state['held_out']!=subject or set(state['train_ids'])!={s for s in truth if split_id(s)[0]!=subject}:raise ValueError('Invalid pair model fold')
        if args.ex_map_root:
            ex_state=torch.load(Path(args.ex_map_root)/tag/'model.pt',map_location='cpu',weights_only=False)
            if ex_state['held_out']!=subject or set(ex_state['train_ids'])!={s for s in truth if split_id(s)[0]!=subject}:raise ValueError('Invalid EX map model fold')
        model=nn.Linear(state['feature_count'],1);model.load_state_dict(state['state_dict']);fitted=(model.eval(),state['mean'],state['sd']);config=state['selected']
        for sid,row in rows.items():
            if split_id(sid)[0]!=subject:continue
            masks=[];features=[]
            for modality in ['invivo','exvivo']:
                cache=Path(f'outputs/stage8_iv/{tag}/maps') if args.iv_capacity and modality=='invivo' else Path('outputs/stage7_tiles/maps')
                if args.ex_map_root and modality=='exvivo':cache=Path(args.ex_map_root)/tag/'maps'
                maps=np.load(cache/f'{sid}_{modality}.npz')['maps'].astype(np.float32)
                masks.append(parse_instances(row,modality,maps.shape[1:]));features.append(morphology_features(masks[-1],maps)[1])
            pairs,x=candidate_features(masks,features,matrices.get(sid));p=probability(fitted,x)
            value=copy.deepcopy(row);value['match_pairs']=assign_edges(pairs,p,config['threshold'],x[:,18],config['radius']);results.append(value)
            print('Joint inference',sid,flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);write_rows(results,out/'predicted_all.csv')
    (out/'predicted_all_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'pipeline.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predicted_all.csv',subject=None,output=out/'predicted_all_metrics.json'))

if __name__=='__main__':main()
