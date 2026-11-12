"""Exploratory stage-three ablations after initial fold inspection.

Removes brightness/context inputs from EX quality classification. Separately,
uses relative descriptor affinities because contrastive cross-entropy learns
rankings, not a calibrated absolute cosine confidence. Outer labels remain
excluded from fitting, but these ablations are exploratory model selection.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .stage3 import FOLDS,Descriptor,build_record,load_maps,calibrate_quality,quality_probabilities,similarities
from .cell_features import fit_feature_geometry
from .registration import assign_pairs
from .data import read_rows,write_rows,split_id
from .cli import evaluate


def relative_affinity(scores):
    if not scores.size: return scores.copy()
    return 1+(.25/.15)*(scores-scores.max(axis=0,keepdims=True))


def morphology_confidence(record):
    result=dict(record)
    result['features']=[record['features'][0],record['features'][1].copy()]
    result['features'][1][:,5:26]=0
    return result


def main():
    torch.set_num_threads(2)
    data=Path('data/raw/Project_2_Dataset');previous=Path('outputs/stage3')
    output=Path('outputs/stage3_refine');output.mkdir(parents=True,exist_ok=True)
    truth=read_rows(data/'training/train_ground_truth.csv')
    all_quality=[];all_matching=[];reports=[]
    for tag,heldout in FOLDS.items():
        folder=output/tag;folder.mkdir(exist_ok=True)
        train=read_rows(previous/tag/'development.csv')
        checkpoint=torch.load(previous/tag/'models.pt',map_location='cpu',weights_only=False)
        if checkpoint['held_out']!=heldout or any(split_id(sid)[0]==heldout for sid in train):
            raise ValueError('Quality-training data includes the held-out subject')
        if set(train)!=set(checkpoint['train_ids']): raise ValueError('Development cache partition mismatch')
        records=[morphology_confidence(build_record(data,sid,row,load_maps(previous/tag/'development_maps',sid),truth[sid])) for sid,row in train.items()]
        fitted,threshold,calibration=calibrate_quality(records)
        torch.save({'quality':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],
                    'zero_columns':list(range(5,26)),'threshold':threshold,'held_out':heldout,
                    'train_ids':list(train)},folder/'quality_model.pt')
        model=Descriptor();model.load_state_dict(checkpoint['descriptor']);model.eval()
        prediction=read_rows(f'outputs/stage2/unet_holdout_{tag}.csv')
        if set(prediction)!={sid for sid in truth if split_id(sid)[0]==heldout}:
            raise ValueError('Prediction IDs must exactly match the held-out subject')
        filtered=[];matched=[];diagnostics=[]
        for sid,row in prediction.items():
            record=build_record(data,sid,row,load_maps(Path(f'outputs/stage2/unet_holdout_{tag}_maps'),sid))
            probabilities=quality_probabilities(fitted,morphology_confidence(record)['features'][1])
            keep={k for k,p in zip(record['keys'][1],probabilities) if p>=threshold}
            clean=copy.deepcopy(row)
            clean['exvivo_instances']={k:v for k,v in row['exvivo_instances'].items() if k in keep}
            filtered.append(clean)
            affinity=relative_affinity(similarities(model,checkpoint['normalization'],record['features']))
            matrix,diagnostic=fit_feature_geometry(*record['centers'],affinity)
            pairs=assign_pairs(*record['masks'],matrix,radius=5.,ambiguity_ratio=.75)
            pair_row=copy.deepcopy(row);pair_row['match_pairs']=pairs;matched.append(pair_row)
            diagnostic.update(sample_id=sid,pairs=len(pairs),quality_removed=len(row['exvivo_instances'])-len(keep))
            diagnostics.append(diagnostic)
            print(f'{tag} {sid}: EX removed {diagnostic["quality_removed"]}, candidate pairs {len(pairs)}',flush=True)
        for name,rows in [('quality',filtered),('matching',matched)]:
            path=folder/f'{name}.csv';write_rows(rows,path)
            evaluate(SimpleNamespace(data=data,predictions=path,subject=heldout,output=folder/f'{name}_metrics.json'))
        report={'held_out':heldout,'train_ids':list(train),'threshold':threshold,'calibration':calibration,
                'regions':diagnostics,'exploratory':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (folder/'experiment.json').write_text(json.dumps(report,indent=2)+'\n')
        all_quality.extend(filtered);all_matching.extend(matched);reports.append(report)
    for name,rows in [('quality',all_quality),('matching',all_matching)]:
        path=output/f'{name}_oof.csv';write_rows(rows,path)
        evaluate(SimpleNamespace(data=data,predictions=path,subject=None,output=output/f'{name}_oof_metrics.json'))


if __name__=='__main__': main()
