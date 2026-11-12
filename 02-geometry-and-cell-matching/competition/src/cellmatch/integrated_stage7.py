"""Jointly compare fixed native-tile masks, geometry recipes and pair models."""
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
from .registration import assign_pairs
from .cli import evaluate


def main():
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset')
    rows=read_rows('outputs/stage7_tiles/native/predicted_all.csv')
    sources={'consensus':'stage7_consensus','support':'stage7_affine_support','excess':'stage7_affine_excess'}
    manifests={name:json.loads((Path('outputs')/folder/'predicted_all_manifest.json').read_text()) for name,folder in sources.items()}
    matrices={name:{r['sample_id']:np.asarray(r['matrix']) for r in manifest['regions'] if r.get('matrix') and r.get('status')=='registered'} for name,manifest in manifests.items()}
    variants={f'{g}_{m}':[] for g in sources for m in ['original','refined']}
    diagnostics={name:[] for name in variants}
    for tag,subject in FOLDS.items():
        models={}
        for label,folder in [('original','stage7_pair_quality'),('refined','stage7_pair_quality_refined')]:
            state=torch.load(Path('outputs')/folder/f'{tag}.pt',map_location='cpu',weights_only=False)
            if state['held_out']!=subject or any(split_id(s)[0]==subject for s in state['train_ids']):raise ValueError('Invalid pair model fold')
            model=nn.Linear(22,1);model.load_state_dict(state['state_dict']);models[label]=((model.eval(),state['mean'],state['sd']),state['threshold'])
        for sid,row in rows.items():
            if split_id(sid)[0]!=subject:continue
            masks=[];features=[]
            for modality in ['invivo','exvivo']:
                image=read_image(image_path(root,'training',sid,modality));m=parse_instances(row,modality,image.shape);masks.append(m)
                maps=np.load(f'outputs/stage7_tiles/maps/{sid}_{modality}.npz')['maps'].astype(np.float32)
                features.append(extract_features(image,m,maps)[2])
            for geometry in sources:
                matrix=matrices[geometry].get(sid)
                pairs=assign_pairs(*masks,matrix,radius=5,ambiguity_ratio=.8)
                x=pair_features(masks,features,pairs,matrix) if pairs else np.empty((0,22),np.float32)
                for label,(fitted,threshold) in models.items():
                    p=probabilities(fitted,x);name=f'{geometry}_{label}';value=copy.deepcopy(row)
                    value['match_pairs']=[pair for pair,prob in zip(pairs,p) if prob>=threshold];variants[name].append(value)
                    diagnostics[name].append({'sample_id':sid,'threshold':threshold,'candidate_pairs':pairs,'probabilities':p.tolist()})
            print('Integrated comparison',sid,flush=True)
    out=Path('outputs/stage7_integrated');out.mkdir(exist_ok=True)
    for name,predictions in variants.items():
        folder=out/name;folder.mkdir(exist_ok=True);path=folder/'predicted_all.csv';write_rows(predictions,path)
        geometry=name.split('_')[0];(folder/'predicted_all_manifest.json').write_text(json.dumps(manifests[geometry],indent=2)+'\n')
        (folder/'pipeline.json').write_text(json.dumps({'masks':'outputs/stage7_tiles/native/predicted_all.csv',
            'geometry':sources[geometry],'pair_model':name.split('_')[1],'regions':diagnostics[name]},indent=2)+'\n')
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=folder/'predicted_all_metrics.json'))


if __name__=='__main__':main()
