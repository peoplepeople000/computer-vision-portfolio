"""Cache refined IV development predictions with exact outer-fold provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .learning import SmallUNet,choose_device,probability_maps,instances_from_maps
from .data import read_rows,write_rows,read_image,image_path,split_id
from .rle import labels_to_instances
from .stage3 import FOLDS


def main():
    torch.set_num_threads(2);device=choose_device('mps');root=Path('data/raw/Project_2_Dataset')
    truth=read_rows(root/'training/train_ground_truth.csv')
    for tag,subject in FOLDS.items():
        checkpoint=Path(f'outputs/stage5_refinement/unet_holdout_{tag}/model.pt')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);expected={s for s in truth if split_id(s)[0]!=subject}
        if state['held_out']!=subject or set(state['train_ids'])!=expected:raise ValueError('Invalid training partition')
        model=SmallUNet(state['width']).to(device);model.load_state_dict(state['state_dict']);model.eval()
        out=Path(f'outputs/stage7_iv_development/{tag}');out.mkdir(parents=True,exist_ok=True);rows=[]
        for sid in sorted(expected):
            image=read_image(image_path(root,'training',sid,'invivo'));maps=probability_maps(model,image,0,device)
            np.savez_compressed(out/f'{sid}_invivo.npz',maps=maps.astype(np.float16))
            rows.append({'sample_id':sid,'invivo_instances':labels_to_instances(instances_from_maps(maps,.5,.35),'IVP'),
                         'exvivo_instances':{},'match_pairs':[]})
            print('Refined IV development',tag,sid,flush=True)
        write_rows(rows,out/'predictions.csv')
        (out/'manifest.json').write_text(json.dumps({'held_out':subject,'train_ids':sorted(expected),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'learning_sha256':hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest()},indent=2)+'\n')
        del model;torch.mps.empty_cache()


if __name__=='__main__':main()
