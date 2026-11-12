"""Choose paired IV/EX contours from baseline and adapted-IV guided candidates.

The guided network remains frozen. A training-only quality model combines its
EX evidence with IV contour evidence. Both masks must be pixel-disjoint and
one IV contour family can be selected once. All sources exclude the outer mouse.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,split_id
from .stage3 import FOLDS
from .guided_ex_data import training_records,make_record,cell_record
from .guided_ex_repaired import GuidedNet,infer
from .ex_joint_calibration import morphology_features
from .iv_mask_bank import family_features
from .joint_pair_assignment import fit,probability
from .preserve_pair_masks import pixels
from .rle import Mask,encode
from .metrics import segmentation_score
from .cli import evaluate


def row_from_cells(record):
    instances={}
    shape=record['iv_image'].shape[1:]
    for c in record['cells']:
        x,y=c['origin'];patch=c['patch']>0;canvas=np.zeros(shape,bool);canvas[y:y+patch.shape[0],x:x+patch.shape[1]]=patch;instances[c['key']]=encode(canvas)
    return dict(sample_id=record['sample_id'],invivo_instances=instances,exvivo_instances={},match_pairs=[])


def bank_record(rows,maps,proposals,shape,truth=None):
    banks=[parse_instances(row,'invivo',m.shape[1:]) for row,m in zip(rows,maps)]
    families,extra=family_features(banks);features=[];candidates=[];labels=[]
    actual_iv=parse_instances(truth,'invivo',maps[0].shape[1:]) if truth is not None else None
    actual_ex=parse_instances(truth,'exvivo',shape) if truth is not None else None
    verified=dict(truth['match_pairs']) if truth is not None else None
    for b,(row,bank,mp,ps) in enumerate(zip(rows,banks,maps,proposals)):
        keys,morph=morphology_features(bank,mp);morph=dict(zip(keys,morph))
        mapping=segmentation_score(bank,actual_iv)['pred_to_gt'] if truth is not None else None
        for p in ps:
            iv=p['iv'];key=f'{b}:{iv}';features.append(np.r_[p['features'],morph[iv],extra[key]])
            candidates.append(dict(iv=key,iv_rle=row['invivo_instances'][iv],ex_rle=p['rle'],family=families[key],source=b))
            if truth is not None:
                target=verified.get(mapping.get(iv));labels.append(bool(target and Mask.parse(p['rle'],shape).iou(actual_ex[target])>.75))
    return dict(sample_id=rows[0]['sample_id'],subject=split_id(rows[0]['sample_id'])[0],candidates=candidates,x=np.asarray(features,np.float32).reshape(-1,47),y=np.asarray(labels,bool))


def choose(record,scores,threshold):
    used=set();occupied_iv=set();occupied_ex=set();chosen=[]
    for n in sorted(range(len(scores)),key=lambda n:(-float(scores[n]),n)):
        c=record['candidates'][n]
        if scores[n]<=threshold or c['family'] in used:continue
        iv=pixels(c['iv_rle']);ex=pixels(c['ex_rle'])
        if iv&occupied_iv or ex&occupied_ex:continue
        used.add(c['family']);occupied_iv.update(iv);occupied_ex.update(ex);chosen.append(n)
    return chosen


def assemble(base,record,chosen):
    row=dict(sample_id=base['sample_id'],invivo_instances={},exvivo_instances={},match_pairs=[])
    for n in chosen:
        c=record['candidates'][n];i=c['iv'];e=f'GUIDED_BANK_{n:06d}'
        row['invivo_instances'][i]=c['iv_rle'];row['exvivo_instances'][e]=c['ex_rle'];row['match_pairs'].append([i,e])
    for mode in ['invivo','exvivo']:
        name=mode+'_instances';occupied=set()
        for rle in row[name].values():occupied.update(pixels(rle))
        for key,rle in base[name].items():
            pp=pixels(rle)
            if pp&occupied:continue
            row[name][f'BASE_{key}']=rle;occupied.update(pp)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args()
    torch.set_num_threads(2);device=torch.device('mps');assert torch.backends.mps.is_available()
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold]
    ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    source=Path('outputs/stage48_guided_ex')/a.fold;adapted=Path('outputs/stage35_cpdino_iv_adapted')/a.fold
    out=Path('outputs/stage51_guided_bank')/a.fold;out.mkdir(parents=True,exist_ok=True);cache=out/'candidates';cache.mkdir(exist_ok=True)
    cfg=json.loads((source/'configuration.json').read_text());acfg=json.loads((adapted/'configuration.json').read_text())
    for c in [cfg,acfg]:assert c['held_out']==subject and c['train_ids']==ids
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert cfg['source_sha256']==sha('src/cellmatch/guided_ex_repaired.py') and cfg['data_source_sha256']==sha('src/cellmatch/guided_ex_data.py')
    saved=torch.load(source/'model.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
    model=GuidedNet(cfg['width']).to(device);model.load_state_dict(saved['state_dict']);model.eval()
    done=json.loads((source/'completion.json').read_text());assert done['model_sha256']==sha(source/'model.pt')
    files=[source/'configuration.json',source/'completion.json',source/'model.pt',adapted/'configuration.json',adapted/'model.pt',root/'training/train_ground_truth.csv']
    development,loaded=training_records(root,truth,a.fold);assert loaded==ids
    records=[]
    def prepare(r,old,ivmap,exmap,actual):
        sid=r['sample_id'];newpath=adapted/'regions'/f'{sid}.json';nm=adapted/'maps'/f'{sid}_invivo.npz';pp=source/'proposals'/f'{sid}.json'
        files.extend([newpath,nm,pp,ivmap,exmap]);assert done['proposals'][sid]==sha(pp)
        new=copy.deepcopy(old);new['invivo_instances']=json.loads(newpath.read_text())['invivo_instances'];new['match_pairs']=[]
        maps=[np.load(path)['maps'].astype(np.float32) for path in [ivmap,nm]]
        path=cache/f'{sid}.json'
        if not path.exists():
            alt=dict(r);alt['cells']=[cell_record(k,m) for k,m in parse_instances(new,'invivo',maps[1].shape[1:]).items()];alt['labels']=None
            path.write_text(json.dumps(infer(model,alt,device))+'\n')
        proposals=[json.loads(pp.read_text())['proposals'],json.loads(path.read_text())['proposals']]
        rec=bank_record([old,new],maps,proposals,r['shape'],actual)
        print('Guided bank',a.fold,sid,len(rec['candidates']),int(rec['y'].sum()),flush=True)
        return rec
    for r in development:
        sid=r['sample_id'];old=row_from_cells(r)
        records.append(prepare(r,old,Path(f'outputs/stage8_iv/{a.fold}/maps/{sid}_invivo.npz'),Path(f'outputs/stage11_native_ex/{a.fold}/maps/{sid}_exvivo.npz'),truth[sid]))
    del development
    inner=[]
    for mouse in sorted({r['subject'] for r in records}):
        fitted=fit([r for r in records if r['subject']!=mouse]);inner.extend((r,probability(fitted,r['x'])) for r in records if r['subject']==mouse)
    total=sum(len(truth[s]['match_pairs']) for s in ids);measurements=[]
    for threshold in np.arange(.05,1.,.05):
        correct=count=0
        for r,scores in inner:
            chosen=choose(r,scores,threshold);correct+=int(r['y'][chosen].sum());count+=len(chosen)
        measurements.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
    selected=max(measurements,key=lambda r:r['f1']);fitted=fit(records)
    torch.save(dict(state_dict=fitted[0].state_dict(),mean=fitted[1],sd=fitted[2],selected=selected,train_ids=ids,held_out=subject),out/'quality.pt')
    bp=Path('outputs/stage32_contour_consensus/predicted_all.csv');gp=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');files.extend([bp,gp]);baseline=read_rows(bp)
    matrices={r['sample_id']:np.array(r['matrix']) for r in json.loads(gp.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    rows=[]
    for sid in sorted(set(truth)-set(ids)):
        if sid not in matrices:
            row=copy.deepcopy(baseline[sid]);row['match_pairs']=[];rows.append(row);continue
        exmap=Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz');exmaps=np.load(exmap)['maps'].astype(np.float32)
        r=make_record(root,sid,baseline[sid],matrices[sid],exmaps)
        rec=prepare(r,baseline[sid],Path(f'outputs/stage8_iv/{a.fold}/maps/{sid}_invivo.npz'),exmap,None)
        chosen=choose(rec,probability(fitted,rec['x']),selected['threshold']);rows.append(assemble(baseline[sid],rec,chosen))
    write_rows(rows,out/'predictions.csv')
    manifest=dict(held_out=subject,train_ids=ids,selected=selected,calibration=measurements,source_sha256=sha(__file__),inputs_sha256={str(p):sha(p) for p in files},candidates_sha256={p.stem:sha(p) for p in cache.glob('*.json')},limitation='Frozen Stage 48 network; development segmentation is in-sample; outer mouse excluded. Exploratory validation.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=subject,output=out/'metrics.json'))

if __name__=='__main__':main()
