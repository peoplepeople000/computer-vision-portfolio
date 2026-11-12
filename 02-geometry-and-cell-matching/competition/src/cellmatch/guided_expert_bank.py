"""Learn a same-outer choice between raw-image and map-guided EX candidates.

Image-only cross-expert IV/EX mask agreement augments per-candidate appearance.
Annotations are used exclusively for development records and final evaluation.
"""
import argparse,copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .stage3 import FOLDS
from .guided_ex_data import training_records
from .guided_ex_bank import bank_record,row_from_cells,choose,assemble
from .guided_appearance_quality import appearance_record
from .guided_endpoint_quality import endpoint_labels
from .joint_pair_assignment import fit,probability
from .preserve_pair_masks import pixels
from .learning import SmallUNet,image_features
from .iv_local_boundary import D4Average
from .ex_boundary import refine
from .cli import evaluate

IV_COLUMNS=np.r_[np.arange(13,43),np.arange(73,75)]
EX_COLUMNS=np.r_[np.arange(47,73),np.arange(73,78)]

def combine(records):
    assert len(records)==2 and records[0]['sample_id']==records[1]['sample_id'];supervised='endpoint' in records[0];assert ('endpoint' in records[1])==supervised
    sets=[[(pixels(c['iv_rle']),pixels(c['ex_rle'])) for c in r['candidates']] for r in records];features=[];candidates=[];labels=[];endpoints=[]
    for expert,r in enumerate(records):
        if not supervised:assert r['y'].size==0
        for n,c in enumerate(r['candidates']):
            iv,ex=sets[expert][n];agreements=[]
            for other_iv,other_ex in sets[1-expert]:
                v=len(iv&other_iv)/max(len(iv|other_iv),1)
                if v>.25:agreements.append((v,len(ex&other_ex)/max(len(ex|other_ex),1)))
            extra=[expert,max((v for v,e in agreements),default=0),max((e for v,e in agreements),default=0),max((min(v,e) for v,e in agreements),default=0),min(len(agreements),4)/4]
            value=copy.deepcopy(c);value['iv']=f'E{expert}_{c["iv"]}';value['family']=f'E{expert}_{c["family"]}';candidates.append(value);features.append(np.r_[r['x'][n],extra])
            if supervised:labels.append(r['y'][n]);endpoints.append(r['endpoint'][n])
    out=dict(sample_id=records[0]['sample_id'],subject=records[0]['subject'],candidates=candidates,x=np.asarray(features,np.float32).reshape(-1,78),y=np.asarray(labels,bool))
    if supervised:out['endpoint']=np.asarray(endpoints,bool).reshape(-1,2)
    return out


def fit_heads(records):
    return [fit([dict(r,x=r['x'][:,IV_COLUMNS],y=r['endpoint'][:,0]) for r in records]),fit([dict(r,x=r['x'][:,EX_COLUMNS],y=r['endpoint'][:,1]) for r in records]),fit(records)]

def scores(heads,x):return np.sqrt(probability(heads[0],x[:,IV_COLUMNS])*probability(heads[1],x[:,EX_COLUMNS]))*probability(heads[2],x)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args();torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold];ids=sorted(s for s in truth if split_id(s)[0]!=subject);out=Path('outputs/stage71_expert_bank')/a.fold;out.mkdir(parents=True,exist_ok=True);cache=out/'records';cache.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    oldroot=Path('outputs/stage51_guided_bank')/a.fold;oldmanifest=json.loads((oldroot/'manifest.json').read_text());assert oldmanifest['held_out']==subject and oldmanifest['train_ids']==ids
    for path,d in oldmanifest['inputs_sha256'].items():assert sha(path)==d
    rawroot=Path('outputs/stage62_endpoint_quality')/a.fold;rawcfg=json.loads((rawroot/'configuration.json').read_text());rawmanifest=json.loads((rawroot/'gated_joint/manifest.json').read_text());assert rawcfg['held_out']==subject and rawcfg['train_ids']==ids
    for n,d in rawcfg['source_sha256'].items():assert sha(Path(__file__).with_name(n))==d
    development,loaded=training_records(root,truth,a.fold);assert loaded==ids;oldrows={r['sample_id']:row_from_cells(r) for r in development};del development
    bp=Path('outputs/stage32_contour_consensus/predicted_all.csv');base=read_rows(bp);oldrows.update({s:r for s,r in base.items() if s not in ids});files=[oldroot/'manifest.json',rawroot/'configuration.json',rawroot/'gated_joint/manifest.json',bp];records={}
    for sid in sorted(oldrows):
        rp=rawroot/'records'/f'{sid}.pt';op=Path('outputs/stage48_guided_ex')/a.fold/'proposals'/f'{sid}.json';ap=oldroot/'candidates'/f'{sid}.json'
        if not rp.exists() or not op.exists() or not ap.exists():continue
        assert sha(rp)==rawmanifest['record_sha256'][sid] and sha(ap)==oldmanifest['candidates_sha256'][sid]
        raw=torch.load(rp,map_location='cpu',weights_only=False);ip,ep=[image_path(root,'training',sid,m) for m in ['invivo','exvivo']];iv,ex=read_image(ip),read_image(ep);altpath=Path(f'outputs/stage35_cpdino_iv_adapted/{a.fold}/regions/{sid}.json');mpaths=[Path(f'outputs/stage8_iv/{a.fold}/maps/{sid}_invivo.npz'),Path(f'outputs/stage35_cpdino_iv_adapted/{a.fold}/maps/{sid}_invivo.npz')];maps=[np.load(p)['maps'].astype(np.float32) for p in mpaths];alt=copy.deepcopy(oldrows[sid]);alt['invivo_instances']=json.loads(altpath.read_text())['invivo_instances'];alt['match_pairs']=[];rows=[oldrows[sid],alt];proposals=[json.loads(p.read_text())['proposals'] for p in [op,ap]]
        old=bank_record(rows,maps,proposals,ex.shape,truth[sid] if sid in ids else None);old=appearance_record(old,rows,maps,iv,ex)
        if sid in ids:old['endpoint']=endpoint_labels(old,truth[sid],[iv.shape,ex.shape])
        r=combine([raw,old]);assert ('endpoint' in r)==(sid in ids);records[sid]=r;torch.save(r,cache/f'{sid}.pt');files.extend([rp,op,ap,ip,ep,altpath,*mpaths]);print('Expert bank',a.fold,sid,len(r['x']),flush=True)
    config=dict(held_out=subject,train_ids=ids,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['guided_expert_bank.py','guided_ex_data.py','guided_ex_bank.py','guided_appearance_quality.py','guided_endpoint_quality.py','joint_pair_assignment.py']},inputs_sha256={str(p):sha(p) for p in files},policy=__doc__);(out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n');training=[r for s,r in records.items() if s in ids];inner=[]
    for mouse in sorted({r['subject'] for r in training}):
        heads=fit_heads([r for r in training if r['subject']!=mouse]);inner.extend((r,scores(heads,r['x'])) for r in training if r['subject']==mouse)
    measures=[];total=sum(len(truth[s]['match_pairs']) for s in ids)
    for threshold in np.arange(.05,1.,.05):
        count=correct=0
        for r,pred in inner:
            chosen=choose(r,pred,threshold);count+=len(chosen);correct+=int(r['y'][chosen].sum())
        measures.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
    selected=max(measures,key=lambda x:x['f1']);heads=fit_heads(training);torch.save(dict(heads=[dict(state_dict=h[0].state_dict(),mean=h[1],sd=h[2]) for h in heads],selected=selected,held_out=subject,train_ids=ids,configuration_sha256=sha(out/'configuration.json')),out/'quality.pt');(out/'calibration.json').write_text(json.dumps(dict(selected=selected,measurements=measures),indent=2)+'\n');results=[]
    for sid in sorted(set(truth)-set(ids)):
        if sid in records:
            r=records[sid];results.append(assemble(base[sid],r,choose(r,scores(heads,r['x']),selected['threshold'])))
        else:row=copy.deepcopy(base[sid]);row['match_pairs']=[];results.append(row)
    write_rows(results,out/'selected.csv');device=torch.device('mps');assert torch.backends.mps.is_available();cp=Path('outputs/stage57_iv_local')/a.fold/'model.pt';state=torch.load(cp,map_location='cpu',weights_only=False);assert state['config']['held_out']==subject and state['config']['train_ids']==ids;net=SmallUNet(16).to(device);net.load_state_dict(state['state_dict']);net.eval();model=D4Average(net).eval()
    for row in results:
        image=read_image(image_path(root,'training',row['sample_id'],'invivo'));row['invivo_instances'],_=refine(parse_instances(row,'invivo',image.shape),image_features(image),model,device)
    write_rows(results,out/'predictions.csv');evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=subject,output=out/'metrics.json'));(out/'manifest.json').write_text(json.dumps(dict(iv_model_sha256=sha(cp),quality_sha256=sha(out/'quality.pt'),selected_sha256=sha(out/'selected.csv'),csv_sha256=sha(out/'predictions.csv'),record_sha256={p.stem:sha(p) for p in cache.glob('*.pt')}),indent=2)+'\n')
if __name__=='__main__':main()
