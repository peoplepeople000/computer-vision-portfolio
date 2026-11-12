"""Calibrate foreground thresholds using only each outer fold's training mice.

Keeps the U-Net checkpoints, seeds, and area constraints fixed. Cached maps are
float16; the original CSV is reused exactly when threshold 0.5 is selected.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .stage3 import FOLDS,development_signature
from .learning import instances_from_maps
from .data import read_rows,parse_instances,write_rows,split_id
from .rle import labels_to_instances
from .metrics import segmentation_score
from .cli import evaluate


def main():
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    output=Path('outputs/stage4_calibrated');output.mkdir(exist_ok=True)
    variants=[];reports={}
    for tag,heldout in FOLDS.items():
        folder=Path(f'outputs/stage3/{tag}')
        if json.loads((folder/'development_manifest.json').read_text())!=development_signature(f'outputs/stage2/unet_holdout_{tag}/model.pt'):
            raise ValueError('Stale development cache')
        development=read_rows(folder/'development.csv')
        expected={sid for sid in truth if split_id(sid)[0]!=heldout}
        if set(development)!=expected:raise ValueError('Development fold does not exclude the entire held-out mouse')
        thresholds=(.3,.4,.5,.6,.7)
        measurements={m:{t:[] for t in thresholds} for m in ('invivo','exvivo')}
        for n,(sid,row) in enumerate(development.items(),1):
            for mode,modality in enumerate(('invivo','exvivo')):
                maps=np.load(folder/'development_maps'/f'{sid}_{modality}.npz')['maps'].astype(np.float32)
                shape=maps.shape[1:];actual=parse_instances(truth[sid],modality,shape)
                for threshold in thresholds:
                    instances=row[f'{modality}_instances'] if threshold==.5 else labels_to_instances(instances_from_maps(maps,threshold=threshold),'IVP' if mode==0 else 'EXP')
                    guess=parse_instances({f'{modality}_instances':instances},modality,shape)
                    pq=segmentation_score(guess,actual)['pq']
                    measurements[modality][threshold].append({'sample_id':sid,'pq':pq})
            print(f'{tag}: threshold calibration {n}/{len(development)}',flush=True)
        chosen={}
        scores={}
        for modality in measurements:
            scores[modality]={t:float(np.mean([r['pq'] for r in values])) for t,values in measurements[modality].items()}
            # Prefer the original threshold if tied.
            chosen[modality]=max(thresholds,key=lambda t:(scores[modality][t],-abs(t-.5)))
        rows=read_rows(f'outputs/stage2/unet_holdout_{tag}.csv')
        for sid,row in rows.items():
            if split_id(sid)[0]!=heldout:raise ValueError('Wrong evaluation subject')
            new={'sample_id':sid,'match_pairs':[]}
            for mode,modality in enumerate(('invivo','exvivo')):
                if chosen[modality]==.5:new[f'{modality}_instances']=row[f'{modality}_instances']
                else:
                    maps=np.load(f'outputs/stage2/unet_holdout_{tag}_maps/{sid}_{modality}.npz')['maps'].astype(np.float32)
                    new[f'{modality}_instances']=labels_to_instances(instances_from_maps(maps,threshold=chosen[modality]),'IVP' if mode==0 else 'EXP')
            variants.append(new)
        reports[heldout]={'thresholds':chosen,'training_scores':scores,'train_ids':sorted(development)}
        print(heldout,chosen,flush=True)
    path=output/'segmentation_oof.csv';write_rows(variants,path)
    (output/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=output/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
