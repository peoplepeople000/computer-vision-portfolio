"""Rank image-only registration hypotheses by frozen guided-cell quality.

No annotation input. Original geometry is always a candidate. Other hypotheses
come from the Stage 52 image-only search and are deduplicated by projected corner
movement. A new geometry needs 25% more predicted pair utility to replace it.
"""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,split_id
from .stage3 import FOLDS
from .guided_ex_repaired import GuidedNet,infer
from .guided_ex_data import make_record,cell_record
from .guided_ex_bank import bank_record,choose,assemble
from .joint_pair_assignment import probability
from .cli import evaluate


def candidate_matrices(original,candidates,shape,limit=24):
    h,w=shape;points=np.array([[0,0],[w,0],[0,h],[w,h],[w/2,h/2]])
    matrices=[np.asarray(original)];positions=[points@matrices[0][:,:2].T+matrices[0][:,2]]
    for c in candidates:
        m=np.asarray(c['matrix']);projected=points@m[:,:2].T+m[:,2]
        if any(np.mean(np.linalg.norm(projected-q,axis=1))<4 for q in positions):continue
        matrices.append(m);positions.append(projected)
        if len(matrices)>=limit:break
    return matrices


def main():
    torch.set_num_threads(2);device=torch.device('mps');assert torch.backends.mps.is_available()
    root=Path('data/raw/Project_2_Dataset');out=Path('outputs/stage53_guided_geometry_rank');out.mkdir(parents=True,exist_ok=True);cache=out/'regions';cache.mkdir(exist_ok=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    bp=Path('outputs/stage32_contour_consensus/predicted_all.csv');pp=Path('outputs/stage51_guided_bank/full/predictions.csv');gp=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');npth=Path('outputs/stage52_translation_geometry/predicted_all_manifest.json')
    baseline=read_rows(bp);previous=read_rows(pp);original={r['sample_id']:r for r in json.loads(gp.read_text())['regions']};newgeo={r['sample_id']:r for r in json.loads(npth.read_text())['regions']}
    audit=json.loads(pp.with_name('audit.json').read_text());assert audit['status']=='passed' and audit['csv_sha256']==sha(pp)
    config=dict(source_sha256=sha(__file__),inputs_sha256={str(p):sha(p) for p in [bp,pp,gp,npth,pp.with_name('audit.json')]},policy='Maximum 24 unique matrices; >=4 pixel mean corner movement; original always included; new pair utility >max(1.5,1.25*original); >=5 selected pairs; no annotation input.')
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');rows=[];reports=[];files=[]
    for tag,subject in FOLDS.items():
        ids=sorted(s for s in baseline if split_id(s)[0]!=subject);mp=Path(f'outputs/stage48_guided_ex/{tag}/model.pt');qp=Path(f'outputs/stage51_guided_bank/{tag}/quality.pt');files.extend([mp,qp])
        state=torch.load(mp,map_location='cpu',weights_only=False);quality=torch.load(qp,map_location='cpu',weights_only=False)
        assert state['config']['held_out']==quality['held_out']==subject and state['config']['train_ids']==quality['train_ids']==ids
        assert sha(qp)==next(r['head_sha256'] for r in audit['folds'] if r['fold']==tag)
        model=GuidedNet(state['config']['width']).to(device);model.load_state_dict(state['state_dict']);model.eval();head=torch.nn.Linear(47,1);head.load_state_dict(quality['state_dict']);fitted=head.eval(),quality['mean'],quality['sd'];threshold=quality['selected']['threshold']
        for sid in sorted(set(baseline)-set(ids)):
            candidates=newgeo[sid].get('translation_candidates',[])
            if not candidates or original[sid].get('votes',0)>=4:rows.append(previous[sid]);continue
            path=cache/f'{sid}.json'
            paths=[Path(f'outputs/stage8_iv/{tag}/maps/{sid}_invivo.npz'),Path(f'outputs/stage35_cpdino_iv_adapted/{tag}/maps/{sid}_invivo.npz'),Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz'),Path(f'outputs/stage35_cpdino_iv_adapted/{tag}/regions/{sid}.json'),Path(f'outputs/stage48_guided_ex/{tag}/proposals/{sid}.json'),Path(f'outputs/stage51_guided_bank/{tag}/candidates/{sid}.json')];files.extend(paths)
            om,nm,em=[np.load(p)['maps'].astype(np.float32) for p in paths[:3]];old=baseline[sid];alt=copy.deepcopy(old);alt['invivo_instances']=json.loads(paths[3].read_text())['invivo_instances']
            matrices=candidate_matrices(original[sid]['matrix'],candidates,om.shape[1:])
            if path.exists():
                report=json.loads(path.read_text());assert report['matrices']==[m.tolist() for m in matrices]
            else:
                record=make_record(root,sid,old,matrices[0],em);cells=[[cell_record(k,m) for k,m in parse_instances(row,'invivo',om.shape[1:]).items()] for row in [old,alt]]
                trials=[]
                for index,matrix in enumerate(matrices):
                    if index==0:proposals=[json.loads(p.read_text())['proposals'] for p in paths[4:]]
                    else:
                        proposals=[]
                        for bankcells in cells:
                            rr=dict(record);rr['matrix']=matrix;rr['cells']=bankcells;proposals.append(infer(model,rr,device)['proposals'])
                    bank=bank_record([old,alt],[om,nm],proposals,em.shape[1:]);scores=probability(fitted,bank['x']);chosen=choose(bank,scores,threshold)
                    utility=float(np.maximum(scores[chosen]-threshold,0).sum());trials.append(dict(index=index,utility=utility,pairs=len(chosen),proposals=proposals))
                    print('Geometry trial',sid,index,len(matrices),round(utility,4),len(chosen),flush=True)
                best=max(trials,key=lambda t:t['utility']);accepted=best['index']!=0 and best['utility']>max(1.5,1.25*trials[0]['utility']) and best['pairs']>=5
                selected=best['index'] if accepted else 0
                report=dict(sample_id=sid,matrices=[m.tolist() for m in matrices],trials=trials,selected=selected,accepted=accepted)
                path.write_text(json.dumps(report)+'\n')
            selected=report['selected'];trial=report['trials'][selected]
            bank=bank_record([old,alt],[om,nm],trial['proposals'],em.shape[1:]);scores=probability(fitted,bank['x']);chosen=choose(bank,scores,threshold);row=assemble(old,bank,chosen)
            if selected==0:assert row==previous[sid]
            rows.append(row);reports.append(dict(sample_id=sid,selected=selected,old_utility=report['trials'][0]['utility'],new_utility=trial['utility'],pairs=len(chosen)));files.append(path)
            print('Geometry selected',reports[-1],flush=True)
        del model
    assert len(rows)==47;write_rows(rows,out/'predictions.csv')
    (out/'manifest.json').write_text(json.dumps(dict(config=config,regions=reports,input_sha256={str(p):sha(p) for p in files}),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=None,output=out/'metrics.json'))

if __name__=='__main__':main()
