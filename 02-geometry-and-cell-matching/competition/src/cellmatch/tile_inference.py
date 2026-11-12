"""Compare training-sized inference tiles with frozen calibration and geometry."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .learning import SmallUNet,choose_device,probability_maps,instances_from_maps
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .rle import labels_to_instances
from .cell_features import extract_features
from .stage3 import FOLDS,quality_probabilities
from .registration import assign_pairs
from .cli import evaluate


def main():
    torch.set_num_threads(2);device=choose_device('mps');root=Path('data/raw/Project_2_Dataset')
    reference=read_rows('outputs/stage7_consensus/predicted_all.csv')
    transforms={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(Path('outputs/stage7_consensus/predicted_all_manifest.json').read_text())['regions'] if r.get('matrix')}
    calibration=json.loads(Path('outputs/stage6_specialist_calibrated/calibration.json').read_text())
    out=Path('outputs/stage7_tiles');out.mkdir(exist_ok=True);cache=out/'maps';cache.mkdir(exist_ok=True)
    results={'native':[],'mean':[]}
    for tag,subject in FOLDS.items():
        models=[]
        for stage in ['stage5_refinement','stage6_ex']:
            state=torch.load(f'outputs/{stage}/unet_holdout_{tag}/model.pt',map_location='cpu',weights_only=False)
            if state['held_out']!=subject or any(split_id(s)[0]==subject for s in state['train_ids']):raise ValueError('Invalid model fold')
            model=SmallUNet(state['width']).to(device);model.load_state_dict(state['state_dict']);models.append(model.eval())
        qstate=torch.load(f'outputs/stage6_specialist_quality/{tag}/model.pt',map_location='cpu',weights_only=False)
        q=nn.Linear(30,1);q.load_state_dict(qstate['quality']);fitted=(q.eval(),qstate['mean'],qstate['sd'])
        for sid,row in reference.items():
            if split_id(sid)[0]!=subject:continue
            variants={v:{'sample_id':sid,'match_pairs':[]} for v in results};masks_by={v:[] for v in results}
            for mode,modality in enumerate(['invivo','exvivo']):
                image=read_image(image_path(root,'training',sid,modality));new=probability_maps(models[mode],image,mode,device,tile=128,overlap=64,batch_size=8)
                np.savez_compressed(cache/f'{sid}_{modality}.npz',maps=new.astype(np.float16))
                stage='stage5_refinement' if mode==0 else 'stage6_ex'
                old=np.load(f'outputs/{stage}/unet_holdout_{tag}_maps/{sid}_{modality}.npz')['maps'].astype(np.float32)
                threshold=.5 if mode==0 else calibration[subject]['threshold']
                for variant,maps in [('native',new),('mean',(new+old)/2)]:
                    instances=labels_to_instances(instances_from_maps(maps,threshold,.35),'IVP' if mode==0 else 'EXP')
                    masks=parse_instances({modality+'_instances':instances},modality,image.shape)
                    if mode==1:
                        keys,_,features=extract_features(image,masks,maps);features[:,5:26]=0;p=quality_probabilities(fitted,features)
                        keep={k for k,v in zip(keys,p) if v>=qstate['threshold']}
                        masks={k:v for k,v in masks.items() if k in keep};instances={k:v for k,v in instances.items() if k in keep}
                    variants[variant][modality+'_instances']=instances;masks_by[variant].append(masks)
            for variant in results:
                variants[variant]['match_pairs']=assign_pairs(*masks_by[variant],transforms.get(sid),radius=5,ambiguity_ratio=.8)
                results[variant].append(variants[variant])
            print('Tile inference',sid,flush=True)
        del models;torch.mps.empty_cache()
    for variant,rows in results.items():
        folder=out/variant;folder.mkdir(exist_ok=True);path=folder/'predicted_all.csv';write_rows(rows,path)
        (folder/'manifest.json').write_text(json.dumps({'tile':128,'overlap':64,'variant':variant,
            'calibration':'Frozen stage-six foreground and quality calibration',
            'geometry':'Frozen stage-seven consensus transforms, recomputed assignment for current masks'},indent=2)+'\n')
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=folder/'predicted_all_metrics.json'))


if __name__=='__main__':main()
