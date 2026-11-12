"""Resumable, provenance-checked native-tile hidden-test probability ensembles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .learning import SmallUNet,choose_device,probability_maps
from .data import read_rows,read_image,image_path,split_id
from .stage3 import FOLDS


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--iv-family',choices=['refined','capacity'],default='refined')
    parser.add_argument('--modality',choices=['all','invivo','exvivo'],default='all');parser.add_argument('--device',default='mps');parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(2);device=choose_device(args.device);root=Path('data/raw/Project_2_Dataset')
    truth=read_rows(root/'training/train_ground_truth.csv');hidden=read_rows(root/'sample_submission.csv')
    if set(hidden)&set(truth):raise ValueError('Training and hidden IDs overlap')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    modalities=['invivo','exvivo'] if args.modality=='all' else [args.modality]
    learning_hash=hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest()
    for modality in modalities:
        for tag,subject in FOLDS.items():
            checkpoint=Path(f'outputs/stage6_ex/unet_holdout_{tag}/model.pt') if modality=='exvivo' else Path(f'outputs/stage8_iv/{tag}/model.pt' if args.iv_family=='capacity' else f'outputs/stage5_refinement/unet_holdout_{tag}/model.pt')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            expected={sid for sid in truth if split_id(sid)[0]!=subject}
            if state['held_out']!=subject or set(state['train_ids'])!=expected or set(state['train_ids'])&set(hidden):raise ValueError('Invalid fold checkpoint')
            folder=out/modality/tag;folder.mkdir(parents=True,exist_ok=True)
            signature={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'learning_sha256':learning_hash,'hidden_ids':sorted(hidden),'train_ids':sorted(expected),'held_out':subject,
                'tile':128,'overlap':64,'batch_size':8,'modality':modality}
            manifest=folder/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Different checkpoint/cache provenance; use a new output directory')
            if not manifest.exists():manifest.write_text(json.dumps(signature,indent=2)+'\n')
            model=SmallUNet(state['width']).to(device);model.load_state_dict(state['state_dict']);model.eval()
            for sid in hidden:
                target=folder/f'{sid}.npz';image=read_image(image_path(root,'hidden_test',sid,modality))
                if target.exists():
                    maps=np.load(target)['maps']
                    if maps.shape!=(2,*image.shape) or not np.isfinite(maps).all():raise ValueError(f'Invalid cached maps: {target}')
                else:
                    maps=probability_maps(model,image,int(modality=='exvivo'),device,tile=128,overlap=64,batch_size=8)
                    temporary=folder/f'{sid}.partial.npz';np.savez_compressed(temporary,maps=maps.astype(np.float16));temporary.replace(target)
                print('Hidden maps',modality,tag,sid,flush=True)
            del model;torch.mps.empty_cache() if device=='mps' else None
        average=out/'ensemble';average.mkdir(exist_ok=True)
        for sid in hidden:
            values=[np.load(out/modality/tag/f'{sid}.npz')['maps'].astype(np.float32) for tag in FOLDS]
            maps=np.mean(values,axis=0);np.savez_compressed(average/f'{sid}_{modality}.npz',maps=maps.astype(np.float16))
    completed=set(modalities)
    previous=out/'ensemble_manifest.json'
    if previous.exists():completed.update(json.loads(previous.read_text()).get('modalities_completed',[]))
    (out/'ensemble_manifest.json').write_text(json.dumps({'hidden_ids':sorted(hidden),'iv_family':args.iv_family,'ensemble':'Equal probability mean of the three whole-mouse-fold checkpoints','modalities_completed':sorted(completed),
        'note':'All hidden mice are unseen by every checkpoint. This final ensemble differs from single-held-out-fold validation and has no measured hidden-test score.'},indent=2)+'\n')

if __name__=='__main__':main()
