"""Deliver guided EX with an IV contour bank and equal fold-quality averaging.

Final adaptation: all-training guided and IV networks, three equally averaged
quality heads, and their median training-only threshold. Hidden scores unknown.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,split_id,image_path,read_image,image_shape,validate_submission
from .learning import image_features
from .guided_ex_data import cell_record
from .guided_ex_repaired import GuidedNet,infer
from .guided_ex_bank import bank_record,choose,assemble
from .joint_pair_assignment import probability
from .stage3 import FOLDS

ROOT=Path('data/raw/Project_2_Dataset');OUT=Path('outputs/submission_stage51_guided_bank')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(infer_missing=False):
    torch.set_num_threads(2 if infer_missing else 1)
    prior=Path('outputs/submission_stage48_guided_ex');adapted=Path('outputs/stage35_cpdino_iv_adapted/all-training');heads=Path('outputs/stage51_guided_bank')
    audit=json.loads((heads/'full/audit.json').read_text());assert audit['status']=='passed' and audit['csv_sha256']==sha(heads/'full/predictions.csv')
    pa=json.loads((prior/'delivery_audit.json').read_text());assert pa['status']=='passed' and pa['csv_sha256']==sha(prior/'submission.csv')
    pm=json.loads((prior/'manifest.json').read_text());assert pm['model_sha256']==sha(prior/'model.pt')
    cfg=json.loads((prior/'configuration.json').read_text());acfg=json.loads((adapted/'configuration.json').read_text());truth=read_rows(ROOT/'training/train_ground_truth.csv')
    assert cfg['train_ids']==acfg['train_ids']==sorted(truth) and acfg['held_out'] is None
    assert acfg['source_sha256']==sha('src/cellmatch/cpdino_iv_adaptation.py')
    adone=json.loads((adapted/'completion.json').read_text());assert adone['checkpoint_sha256']==sha(adapted/'model.pt') and adone['csv_sha256']==sha(adapted/'predictions.csv')
    for path,digest in cfg['source_sha256'].items():assert sha(path)==digest
    bp=Path('outputs/submission_stage32_cellpose_consensus/submission.csv');base=read_rows(bp);alt=read_rows(adapted/'predictions.csv')
    official=read_rows(ROOT/'sample_submission.csv');assert set(base)==set(alt)==set(official) and not(set(base)&set(truth))
    files=[bp,ROOT/'sample_submission.csv',prior/'configuration.json',prior/'manifest.json',prior/'delivery_audit.json',prior/'model.pt',adapted/'configuration.json',adapted/'completion.json',adapted/'model.pt',adapted/'predictions.csv',heads/'full/audit.json']
    fitted=[];thresholds=[]
    for tag,subject in FOLDS.items():
        path=heads/tag/'quality.pt';files.append(path);expected=next(r['head_sha256'] for r in audit['folds'] if r['fold']==tag);assert sha(path)==expected
        state=torch.load(path,map_location='cpu',weights_only=False);assert state['held_out']==subject and state['train_ids']==sorted(s for s in truth if split_id(s)[0]!=subject)
        head=torch.nn.Linear(47,1);head.load_state_dict(state['state_dict']);fitted.append((head.eval(),state['mean'],state['sd']));thresholds.append(state['selected']['threshold'])
    threshold=float(np.median(thresholds));geo=Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json');files.append(geo)
    matrices={r['sample_id']:np.array(r['matrix']) for r in json.loads(geo.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    cache=OUT/'candidates';cache.mkdir(parents=True,exist_ok=True)
    model=None;device=None
    if infer_missing:
        assert torch.backends.mps.is_available();device=torch.device('mps');saved=torch.load(prior/'model.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
        model=GuidedNet(cfg['width']).to(device);model.load_state_dict(saved['state_dict']);model.eval()
    rows=[];choices=[]
    for sid,old in sorted(base.items()):
        paths=[Path(f'outputs/submission_ensemble_maps/ensemble/{sid}_invivo.npz'),adapted/'maps'/f'{sid}_invivo.npz',Path(f'outputs/submission_ensemble_maps/ensemble/{sid}_exvivo.npz'),prior/'proposals'/f'{sid}.json']
        files.extend(paths);files.extend(image_path(ROOT,'hidden_test',sid,m) for m in ['invivo','exvivo'])
        assert sha(paths[3])==pm['proposals_sha256'][sid]
        om,nm,em=[np.load(p)['maps'].astype(np.float32) for p in paths[:3]]
        new=copy.deepcopy(old);new['invivo_instances']=alt[sid]['invivo_instances'];path=cache/f'{sid}.json'
        if not path.exists():
            if not infer_missing:raise FileNotFoundError(path)
            if sid not in matrices:value=dict(sample_id=sid,proposals=[])
            else:
                iv=read_image(image_path(ROOT,'hidden_test',sid,'invivo'));ex=read_image(image_path(ROOT,'hidden_test',sid,'exvivo'))
                r=dict(sample_id=sid,subject=split_id(sid)[0],matrix=matrices[sid],iv_image=image_features(iv),ex_image=image_features(ex),ex_maps=em,labels=None,shape=ex.shape,cells=[cell_record(k,m) for k,m in parse_instances(new,'invivo',iv.shape).items()])
                value=infer(model,r,device)
            path.write_text(json.dumps(value)+'\n')
        files.append(path);ps=[json.loads(p.read_text())['proposals'] for p in [paths[3],path]]
        record=bank_record([old,new],[om,nm],ps,em.shape[1:]);scores=np.mean([probability(f,record['x']) for f in fitted],axis=0)
        chosen=choose(record,scores,threshold);row=assemble(old,record,chosen)
        for mode in ['invivo','exvivo']:
            shape=image_shape(image_path(ROOT,'hidden_test',sid,mode));occupied=np.zeros(shape,bool)
            for mask in parse_instances(row,mode,shape).values():
                xy=mask.coordinates();assert not occupied[xy[:,1],xy[:,0]].any();occupied[xy[:,1],xy[:,0]]=True
        rows.append(row);choices.append(dict(sample_id=sid,pairs=len(chosen),adapted_iv_pairs=sum(record['candidates'][n]['source']==1 for n in chosen)))
        if infer_missing:print('Final guided bank',choices[-1],flush=True)
    sourcepaths=[Path(__file__),Path('src/cellmatch/guided_ex_bank.py'),Path('src/cellmatch/guided_ex_repaired.py'),Path('src/cellmatch/guided_ex_data.py')]
    manifest=dict(policy=dict(threshold=threshold,quality='equal mean of three fold probabilities'),training_ids=sorted(truth),hidden_ids=sorted(base),regions=choices,input_sha256={str(p):sha(p) for p in files},source_sha256={str(p):sha(p) for p in sourcepaths},limitation=__doc__)
    return rows,manifest


def main():
    rows,manifest=build(infer_missing=True);path=OUT/'submission.csv'
    if path.exists():assert read_rows(path)=={r['sample_id']:r for r in rows}
    else:write_rows(rows,path)
    assert read_rows(path)=={r['sample_id']:r for r in rows}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validation=dict(status='passed',counts=validate_submission(ROOT,path),csv_sha256=sha(path),uploaded=False,public_score=None)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__':main()
