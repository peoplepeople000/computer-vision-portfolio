"""Choose held-out transform hypotheses by agreement across prediction recipes."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .data import read_rows,write_rows,parse_instances,image_path,image_shape,split_id
from .matching import centroids,mutual_neighbors
from .registration import assign_pairs
from .cli import evaluate

SOURCES=['stage4_geometry','stage5_refinement_geometry','stage5_tta_geometry',
         'stage6_ex_geometry','stage6_specialist_quality_geometry','stage6_specialist_quality_conservative_geometry']


def choose_matrix(points,target,candidates):
    if not candidates:return None,{'status':'no_candidates'}
    projected=np.stack([points@m[:,:2].T+m[:,2] for _,m in candidates])
    distance=np.median(np.linalg.norm(projected[:,None]-projected[None,:],axis=-1),axis=-1)
    agreement=distance<10
    support=[]
    for p in projected:
        a,b,e=mutual_neighbors(p,target,5.)
        support.append(float(np.exp(-.5*(e/2.5)**2).sum()))
    votes=agreement.sum(1)
    index=max(range(len(candidates)),key=lambda i:(votes[i],support[i]))
    return candidates[index][1],{'status':'registered','selected_source':candidates[index][0],
        'votes':int(votes[index]),'sources':len(candidates),'support':support[index],
        'all_votes':votes.tolist(),'all_support':support,'distance_matrix':distance.tolist(),
        'matrix':candidates[index][1].tolist()}


def main():
    root=Path('data/raw/Project_2_Dataset');predictions=read_rows('outputs/stage6_specialist_quality_geometry/predicted_all.csv')
    manifests={};provenance={}
    for name in SOURCES:
        path=Path('outputs')/name/'predicted_all_manifest.json';manifest=json.loads(path.read_text())
        for subject,prior in manifest['priors'].items():
            if any(split_id(s)[0]==subject for s in prior['train_ids']):raise ValueError('Geometry prior leakage')
        manifests[name]={r['sample_id']:r for r in manifest['regions']}
        provenance[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    rows=[];diagnostics=[]
    for sid,row in predictions.items():
        masks=[parse_instances(row,m,image_shape(image_path(root,'training',sid,m))) for m in ['invivo','exvivo']]
        points=[centroids(m)[1] for m in masks]
        candidates=[(name,np.asarray(records[sid]['matrix'])) for name,records in manifests.items()
                    if records[sid].get('status')=='registered' and records[sid].get('matrix') is not None]
        matrix,diag=choose_matrix(*points,candidates)
        value=copy.deepcopy(row);value['match_pairs']=assign_pairs(*masks,matrix,radius=5,ambiguity_ratio=.8)
        rows.append(value);diag.update(sample_id=sid,pairs=len(value['match_pairs']));diagnostics.append(diag)
    out=Path('outputs/stage7_consensus');out.mkdir(exist_ok=True);path=out/'predicted_all.csv';write_rows(rows,path)
    (out/'predicted_all_manifest.json').write_text(json.dumps({'sources':provenance,'regions':diagnostics,
        'policy':'Six global recipe hypotheses; most transforms within median projection distance 10 px, then largest common-mask support',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'predicted_all_metrics.json'))


if __name__=='__main__':main()
