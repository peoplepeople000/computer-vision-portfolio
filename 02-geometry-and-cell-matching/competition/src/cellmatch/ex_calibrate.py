"""EX specialist development inference and training-only foreground calibration."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .learning import SmallUNet, choose_device, probability_maps, instances_from_maps
from .data import read_rows, write_rows, parse_instances, read_image, image_path, split_id
from .rle import labels_to_instances
from .stage3 import FOLDS, development_signature
from .metrics import segmentation_score
from .cli import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generate',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    variants={'specialist':[],'ensemble':[]};reports={v:{} for v in variants}
    for tag,subject in FOLDS.items():
        checkpoint=Path(f'outputs/stage6_ex/unet_holdout_{tag}/model.pt')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        ids=[s for s in truth if split_id(s)[0]!=subject]
        if state['held_out']!=subject or set(state['train_ids'])!=set(ids):raise ValueError('Wrong development split')
        folder=Path(f'outputs/stage6_ex/development_{tag}');folder.mkdir(exist_ok=True)
        signature={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'learning_sha256':hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest(),
                   'train_ids':ids,'held_out':subject}
        manifest=folder/'manifest.json'
        if args.generate:
            device=choose_device('mps');model=SmallUNet(state['width']).to(device);model.load_state_dict(state['state_dict']);model.eval()
            for sid in ids:
                image=read_image(image_path(root,'training',sid,'exvivo'))
                maps=probability_maps(model,image,1,device)
                np.savez_compressed(folder/f'{sid}_exvivo.npz',maps=maps.astype(np.float16))
                print('EX development',tag,sid,flush=True)
            manifest.write_text(json.dumps(signature,indent=2)+'\n');del model;torch.mps.empty_cache()
            continue
        if json.loads(manifest.read_text())!=signature:raise ValueError('Stale specialist development maps')
        dev=Path(f'outputs/stage3/{tag}')
        if json.loads((dev/'development_manifest.json').read_text())!=development_signature(f'outputs/stage2/unet_holdout_{tag}/model.pt'):
            raise ValueError('Stale baseline development maps')
        thresholds=(.3,.4,.5,.6,.7);scores={v:{t:[] for t in thresholds} for v in variants}
        for sid in ids:
            new=np.load(folder/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
            old=np.load(dev/'development_maps'/f'{sid}_exvivo.npz')['maps'].astype(np.float32)
            actual=parse_instances(truth[sid],'exvivo',new.shape[1:])
            for variant,maps in [('specialist',new),('ensemble',(old+new)/2)]:
                for threshold in thresholds:
                    instances=labels_to_instances(instances_from_maps(maps,threshold,.35),'EXP')
                    guess=parse_instances({'exvivo_instances':instances},'exvivo',maps.shape[1:])
                    scores[variant][threshold].append(segmentation_score(guess,actual)['pq'])
        for variant in variants:
            means={t:float(np.mean(s)) for t,s in scores[variant].items()}
            chosen=max(thresholds,key=lambda t:(means[t],-abs(t-.5)))
            reports[variant][subject]={'train_ids':ids,'threshold':chosen,'training_scores':means}
            print(tag,variant,chosen,means,flush=True)
            original=read_rows(f'outputs/stage6_ex/unet_holdout_{tag}.csv')
            for sid,row in original.items():
                if split_id(sid)[0]!=subject:raise ValueError('Wrong held-out predictions')
                maps=np.load(f'outputs/stage6_ex/unet_holdout_{tag}_maps/{sid}_exvivo.npz')['maps'].astype(np.float32)
                if variant=='ensemble':maps=(maps+np.load(f'outputs/stage2/unet_holdout_{tag}_maps/{sid}_exvivo.npz')['maps'].astype(np.float32))/2
                instances=labels_to_instances(instances_from_maps(maps,chosen,.35),'EXP')
                variants[variant].append({'sample_id':sid,'invivo_instances':row['invivo_instances'],
                                         'exvivo_instances':instances,'match_pairs':[]})
    if args.generate:return
    for variant,rows in variants.items():
        out=Path('outputs')/f'stage6_{variant}_calibrated';out.mkdir(exist_ok=True)
        path=out/'segmentation_oof.csv';write_rows(rows,path)
        (out/'calibration.json').write_text(json.dumps(reports[variant],indent=2)+'\n')
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
