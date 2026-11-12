"""Rank geometry hypotheses with learned pair-credit estimates, without labels."""
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
from .registration_consensus import SOURCES
from .matching import centroids
from .cli import evaluate


def main():
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');rows=read_rows('outputs/stage7_tiles/native/predicted_all.csv')
    old={name:{r['sample_id']:r for r in json.loads((Path('outputs')/name/'predicted_all_manifest.json').read_text())['regions']} for name in SOURCES}
    variants={v:[] for v in ['credit','margin','consensus_credit']};reports={v:[] for v in variants}
    for tag,subject in FOLDS.items():
        state=torch.load(f'outputs/stage7_pair_quality_refined/{tag}.pt',map_location='cpu',weights_only=False)
        if state['held_out']!=subject or any(split_id(s)[0]==subject for s in state['train_ids']):raise ValueError('Invalid pair fold')
        model=nn.Linear(22,1);model.load_state_dict(state['state_dict']);fitted=(model.eval(),state['mean'],state['sd']);threshold=state['threshold']
        for sid,row in rows.items():
            if split_id(sid)[0]!=subject:continue
            masks=[];features=[]
            for modality in ['invivo','exvivo']:
                image=read_image(image_path(root,'training',sid,modality));m=parse_instances(row,modality,image.shape);masks.append(m)
                maps=np.load(f'outputs/stage7_tiles/maps/{sid}_{modality}.npz')['maps'].astype(np.float32)
                features.append(extract_features(image,m,maps)[2])
            points=centroids(masks[0])[1];anchor=points[np.linspace(0,len(points)-1,min(20,len(points))).astype(int)]
            originals=[(name,np.asarray(records[sid]['matrix'])) for name,records in old.items() if records[sid].get('status')=='registered' and records[sid].get('matrix')]
            bank=json.loads(Path(f'outputs/stage7_affine_candidates/{sid}.json').read_text())
            candidates=originals+[(f'bank_{i}',np.asarray(d['matrix'])) for i,d in enumerate(bank)]
            unique=[];projected=[]
            for name,matrix in candidates:
                p=anchor@matrix[:,:2].T+matrix[:,2]
                if any(np.median(np.linalg.norm(p-q,axis=1))<3 for q in projected):continue
                unique.append((name,matrix));projected.append(p)
            options=[]
            for (name,matrix),p in zip(unique,projected):
                pairs=assign_pairs(*masks,matrix,radius=5,ambiguity_ratio=.8)
                x=pair_features(masks,features,pairs,matrix) if pairs else np.empty((0,22),np.float32)
                probability=probabilities(fitted,x);kept=probability>=threshold
                votes=sum(np.median(np.linalg.norm(p-(anchor@m[:,:2].T+m[:,2]),axis=1))<10 for _,m in originals)
                options.append({'source':name,'matrix':matrix.tolist(),'credit':float(probability[kept].sum()),
                    'margin':float((probability[kept]-threshold).sum()),'votes':int(votes),
                    'pairs':[pair for pair,ok in zip(pairs,kept) if ok],'probabilities':probability[kept].tolist()})
            for variant in variants:
                key=(lambda d:(d['votes'],d['credit'])) if variant=='consensus_credit' else (lambda d:d[variant])
                selected=max(options,key=key) if options else {'matrix':None,'pairs':[],'source':'none','votes':0,'credit':0,'margin':0}
                value=copy.deepcopy(row);value['match_pairs']=selected['pairs'];variants[variant].append(value)
                diag={k:v for k,v in selected.items() if k!='pairs'};diag.update(sample_id=sid,status='registered' if selected['matrix'] else 'rejected',candidate_count=len(options))
                reports[variant].append(diag)
            print('Learned transform ranking',sid,len(options),flush=True)
    out=Path('outputs/stage7_ranked');out.mkdir(exist_ok=True)
    for variant,predictions in variants.items():
        folder=out/variant;folder.mkdir(exist_ok=True);path=folder/'predicted_all.csv';write_rows(predictions,path)
        (folder/'predicted_all_manifest.json').write_text(json.dumps({'regions':reports[variant],
            'masks':'outputs/stage7_tiles/native/predicted_all.csv','pair_models':'outputs/stage7_pair_quality_refined','policy':variant},indent=2)+'\n')
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=folder/'predicted_all_metrics.json'))


if __name__=='__main__':main()
