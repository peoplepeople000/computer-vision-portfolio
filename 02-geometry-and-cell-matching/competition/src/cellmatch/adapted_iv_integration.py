"""Compare fixed integrations of adapted IV masks using frozen training-only policies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows, write_rows, split_id, parse_instances
from .stage3 import FOLDS, quality_probabilities
from .ex_joint_calibration import morphology_features
from .star_pairing import describe
from .joint_pair_assignment import candidate_features, probability, assign_edges
from .cellpose_contour_consensus import blend
from .cli import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', choices=list(FOLDS), required=True)
    p.add_argument('--source',type=Path,default=Path('outputs/stage35_cpdino_iv_adapted'))
    p.add_argument('--output',type=Path,default=Path('outputs/stage39_adapted_iv'))
    a=p.parse_args();torch.set_num_threads(1)
    subject=FOLDS[a.fold]
    baseline_path=Path('outputs/stage32_contour_consensus/predicted_all.csv')
    base=read_rows(baseline_path)
    expected={s for s in base if split_id(s)[0]==subject}
    path=a.source/a.fold/'predictions.csv';candidate=read_rows(path)
    cfg_path=a.source/a.fold/'configuration.json';cfg=json.loads(cfg_path.read_text())
    if set(candidate)!=expected or cfg['held_out']!=subject or set(cfg['train_ids'])!=set(base)-expected:
        raise ValueError('Invalid excluded-mouse prediction partition')
    pair_path=Path(f'outputs/stage8_joint_iv_quality/{a.fold}.pt')
    quality_path=Path(f'outputs/stage8_iv_quality_capacity/{a.fold}.pt')
    pair=torch.load(pair_path,map_location='cpu',weights_only=False)
    quality=torch.load(quality_path,map_location='cpu',weights_only=False)
    for state in [pair,quality]:
        if state['held_out']!=subject or set(state['train_ids'])!=set(cfg['train_ids']):
            raise ValueError('Frozen classifier contains evaluation mouse')
    model=torch.nn.Linear(33,1);model.load_state_dict(pair['state_dict']);fitted=(model.eval(),pair['mean'],pair['sd'])
    qm=torch.nn.Linear(30,1);qm.load_state_dict(quality['quality']);qfit=(qm.eval(),quality['mean'],quality['sd'])
    geometry_path=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    geo=json.loads(geometry_path.read_text())
    matrices={r['sample_id']:np.array(r['matrix']) for r in geo['regions'] if r.get('matrix') and r.get('status')=='registered'}
    results={name:[] for name in ['consensus','replacement','filtered']}
    inputs=[path,cfg_path,baseline_path,pair_path,quality_path,geometry_path]
    for sid,new in candidate.items():
        iv_path=Path(f'outputs/stage8_iv/{a.fold}/maps/{sid}_invivo.npz')
        ex_path=Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')
        inputs.extend([iv_path,ex_path])
        maps=np.load(iv_path)['maps'].astype(np.float32)
        consensus,_=blend(base[sid],new,maps.shape[1:]);results['consensus'].append(consensus)
        for mode in ['replacement','filtered']:
            row=copy.deepcopy(base[sid]);row['invivo_instances']=copy.deepcopy(new['invivo_instances']);row['match_pairs']=[]
            if mode=='filtered':
                masks=parse_instances(row,'invivo',maps.shape[1:]);keys,features=morphology_features(masks,maps)
                keep={key for key,p in zip(keys,quality_probabilities(qfit,features)) if p>=quality['threshold']}
                row['invivo_instances']={k:v for k,v in row['invivo_instances'].items() if k in keep}
            masks,features=describe(row,iv_path,ex_path)
            pairs,x=candidate_features(masks,features,matrices.get(sid))
            row['match_pairs']=assign_edges(pairs,probability(fitted,x),pair['selected']['threshold'],x[:,18],pair['selected']['radius'])
            results[mode].append(row)
    out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True)
    for mode,rows in results.items():
        write_rows(rows,out/f'{mode}.csv')
        evaluate(SimpleNamespace(data=Path('data/raw/Project_2_Dataset'),predictions=out/f'{mode}.csv',subject=subject,output=out/f'{mode}_metrics.json'))
    (out/'manifest.json').write_text(json.dumps(dict(held_out=subject,train_ids=cfg['train_ids'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        policy='Three fixed integration recipes; no outer-label parameter tuning; compare complete folds before global selection.'),indent=2)+'\n')


if __name__=='__main__':main()
