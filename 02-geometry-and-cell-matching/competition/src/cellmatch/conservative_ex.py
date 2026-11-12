"""Exploratory conservative EX rejection using existing held-out quality models.

Masks below quality probability 0.15 are rejected, capped at 20% per region.
These are global research settings, not per-mouse post-hoc choices.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .stage3 import FOLDS,build_record,load_maps,quality_probabilities
from .stage3_refine import morphology_confidence
from .data import read_rows,write_rows,split_id
from .cli import evaluate


def rejected_keys(keys,probability,threshold=.15,max_fraction=.2):
    if len(keys)!=len(probability):raise ValueError('Key/probability length mismatch')
    count=int(np.floor(len(keys)*max_fraction))
    order=np.argsort(probability,kind='stable')[:count]
    return {keys[i] for i in order if probability[i]<threshold}


def main():
    torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');output=Path('outputs/stage4_conservative');output.mkdir(exist_ok=True)
    allrows=[];diagnostics=[]
    for tag,heldout in FOLDS.items():
        state=torch.load(f'outputs/stage3_refine/{tag}/quality_model.pt',map_location='cpu',weights_only=False)
        if state['held_out']!=heldout or any(split_id(sid)[0]==heldout for sid in state['train_ids']):raise ValueError('Training leakage')
        model=nn.Linear(30,1);model.load_state_dict(state['quality']);model.eval()
        fitted=(model,state['mean'],state['sd'])
        rows=read_rows(f'outputs/stage2/unet_holdout_{tag}.csv')
        for sid,row in rows.items():
            record=morphology_confidence(build_record(root,sid,row,load_maps(Path(f'outputs/stage2/unet_holdout_{tag}_maps'),sid)))
            probability=quality_probabilities(fitted,record['features'][1])
            removed=rejected_keys(record['keys'][1],probability)
            result=copy.deepcopy(row)
            result['exvivo_instances']={k:v for k,v in row['exvivo_instances'].items() if k not in removed}
            allrows.append(result);diagnostics.append({'sample_id':sid,'removed':len(removed),'input_cells':len(record['keys'][1])})
    path=output/'quality_oof.csv';write_rows(allrows,path)
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=output/'quality_oof_metrics.json'))
    (output/'manifest.json').write_text(json.dumps({'threshold':.15,'max_removed_fraction':.2,'exploratory':True,'regions':diagnostics},indent=2)+'\n')


if __name__=='__main__':main()
