"""One image-only RANSAC update followed by guided EX re-prediction.

Use the existing frozen conservative refit acceptance rule. Calibration heads
and segmentation models remain unchanged, with no evaluation-label input.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,image_shape,image_path,split_id
from .stage3 import FOLDS
from .guided_ex_repaired import GuidedNet,infer
from .guided_ex_data import make_record
from .guided_ex_calibration_repaired import select
from .joint_pair_assignment import probability
from .pair_ransac_refinement import refine
from .preserve_pair_masks import merge_row
from .rle import Mask
from .cli import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/stage49_guided_refinement'));a=p.parse_args();torch.set_num_threads(2)
    device=torch.device('mps')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    root=Path('data/raw/Project_2_Dataset');basepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');base=read_rows(basepath);subject=FOLDS[a.fold]
    expected=sorted(s for s in base if split_id(s)[0]!=subject);source=Path('outputs/stage48_guided_ex')/a.fold;head=Path('outputs/stage48_guided_pairs')/a.fold/'model.pt'
    state=torch.load(source/'model.pt',map_location='cpu',weights_only=False);qstate=torch.load(head,map_location='cpu',weights_only=False)
    if state['config']['held_out']!=subject or state['config']['train_ids']!=expected or qstate['held_out']!=subject or qstate['train_ids']!=expected:raise ValueError('Invalid outer partition')
    model=GuidedNet(state['config']['width']);model.load_state_dict(state['state_dict']);model=model.to(device).eval()
    classifier=torch.nn.Linear(13,1);classifier.load_state_dict(qstate['state_dict']);qfit=classifier.eval(),qstate['mean'],qstate['sd'];threshold=qstate['selected']['threshold']
    geometry=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');matrices={r['sample_id']:np.array(r['matrix']) for r in json.loads(geometry.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    rows=[];diagnostics=[];files=[basepath,source/'model.pt',source/'configuration.json',source/'completion.json',head,geometry]
    out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True)
    if (out/'predictions.csv').exists():raise FileExistsError('Preserve previous refinement')
    for sid in sorted(set(base)-set(expected)):
        path=source/'proposals'/f'{sid}.json';files.append(path);proposals=json.loads(path.read_text())['proposals'];matrix=matrices.get(sid)
        x=np.asarray([p['features'] for p in proposals],np.float32).reshape(-1,13);scores=probability(qfit,x);chosen=select(proposals,scores,threshold)
        ivshape=image_shape(image_path(root,'training',sid,'invivo'));exshape=image_shape(image_path(root,'training',sid,'exvivo'))
        iv=parse_instances(base[sid],'invivo',ivshape)
        start=np.array([iv[proposals[n]['iv']].coordinates().mean(0) for n in chosen]).reshape(-1,2)
        target=np.array([Mask.parse(proposals[n]['rle'],exshape).coordinates().mean(0) for n in chosen]).reshape(-1,2)
        updated,report=refine(start,target,matrix)
        if report['accepted']:
            mpath=Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz');files.append(mpath);maps=np.load(mpath)['maps'].astype(np.float32)
            record=make_record(root,sid,base[sid],updated,maps);proposals=infer(model,record,device)['proposals']
            x=np.asarray([p['features'] for p in proposals],np.float32).reshape(-1,13);scores=probability(qfit,x);chosen=select(proposals,scores,threshold)
        paired=copy.deepcopy(base[sid]);paired['exvivo_instances']={};paired['match_pairs']=[]
        for n in chosen:
            key=f'GUIDED_{n:06d}';paired['exvivo_instances'][key]=proposals[n]['rle'];paired['match_pairs'].append([proposals[n]['iv'],key])
        rows.append(merge_row(paired,base[sid]));diagnostics.append(dict(sample_id=sid,refinement=report,matrix=None if updated is None else updated.tolist(),pairs=len(chosen),proposals=proposals))
        print('Guided refinement',a.fold,sid,report['accepted'],len(chosen),flush=True)
    write_rows(rows,out/'predictions.csv');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(held_out=subject,train_ids=expected,regions=diagnostics,input_sha256={str(p):sha(p) for p in files},source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['guided_ex_refinement.py','guided_ex_repaired.py','guided_ex_data.py','guided_ex_calibration_repaired.py','pair_ransac_refinement.py']},policy=__doc__),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=subject,output=out/'metrics.json'))


if __name__=='__main__':main()
