"""Select EX contours per IV cell from four frozen foreground decoders.

Append foreground threshold and cross-threshold mask stability features. Fit IV,
EX and joint quality heads only on development mice. Preserve one IV contour
family and pixel-disjoint IV/EX masks through the original bank selector.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .stage3 import FOLDS
from .preserve_pair_masks import pixels
from .guided_ex_bank import choose,assemble
from .joint_pair_assignment import fit,probability
from .learning import SmallUNet,image_features
from .iv_local_boundary import D4Average
from .ex_boundary import refine
from .cli import evaluate

THRESHOLDS=(.35,.45,.55,.65)
EX_COLUMNS=np.r_[np.arange(47,73),np.arange(73,78)]

def combine(records):
    first=records[0];assert len(records)==len(THRESHOLDS) and all(r['sample_id']==first['sample_id'] for r in records)
    groups={}
    for r in records:
        for c in r['candidates']:groups.setdefault(c['iv'],[]).append(pixels(c['ex_rle']))
    candidates=[];features=[];labels=[];endpoints=[];supervised='endpoint' in first
    for threshold,r in zip(THRESHOLDS,records):
        assert ('endpoint' in r)==supervised
        for n,c in enumerate(r['candidates']):
            pp=pixels(c['ex_rle']);others=groups[c['iv']];ious=[len(pp&q)/max(len(pp|q),1) for q in others];median=np.median([len(q) for q in others]);extra=[threshold,np.mean(ious),min(ious),np.log(max(len(pp),1)/max(median,1)),len(others)/4]
            candidates.append(copy.deepcopy(c));features.append(np.r_[r['x'][n],extra])
            if supervised:labels.append(r['y'][n]);endpoints.append(r['endpoint'][n])
        if not supervised:assert r['y'].size==0
    result=dict(sample_id=first['sample_id'],subject=first['subject'],candidates=candidates,x=np.asarray(features,np.float32).reshape(-1,78),y=np.asarray(labels,bool))
    if supervised:result['endpoint']=np.asarray(endpoints,bool).reshape(-1,2)
    return result


def fit_heads(records):
    return [fit([dict(r,x=r['x'][:,13:43],y=r['endpoint'][:,0]) for r in records]),fit([dict(r,x=r['x'][:,EX_COLUMNS],y=r['endpoint'][:,1]) for r in records]),fit(records)]

def scores(heads,x):
    iv=probability(heads[0],x[:,13:43]);ex=probability(heads[1],x[:,EX_COLUMNS]);return np.sqrt(iv*ex)*probability(heads[2],x)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args();torch.set_num_threads(2);device=torch.device('mps');assert torch.backends.mps.is_available();root=Path('data/raw/Project_2_Dataset');source=Path('outputs/stage67_ex_threshold')/a.fold;out=Path('outputs/stage68_contour_bank')/a.fold;out.mkdir(parents=True,exist_ok=True);cache=out/'records';cache.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold];ids=sorted(s for s in truth if split_id(s)[0]!=subject);cfg=json.loads((source/'configuration.json').read_text());assert cfg['held_out']==subject and cfg['train_ids']==ids
    for n,d in cfg['source_sha256'].items():assert sha(Path(__file__).with_name(n))==d
    for path,d in cfg['input_sha256'].items():assert sha(path)==d
    manifests=[json.loads((source/f't{int(round(t*100))}'/'manifest.json').read_text()) for t in THRESHOLDS];sample_ids=set(manifests[0]['record_sha256']);assert all(set(m['record_sha256'])==sample_ids for m in manifests);records={};inputs=[source/'configuration.json']
    for sid in sorted(sample_ids):
        parts=[]
        for t,manifest in zip(THRESHOLDS,manifests):
            path=source/f't{int(round(t*100))}'/'records'/f'{sid}.pt';assert sha(path)==manifest['record_sha256'][sid];inputs.append(path);parts.append(torch.load(path,map_location='cpu',weights_only=False))
        r=combine(parts);assert ('endpoint' in r)==(sid in ids);records[sid]=r;torch.save(r,cache/f'{sid}.pt')
    config=dict(held_out=subject,train_ids=ids,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['guided_contour_bank.py','guided_ex_bank.py','joint_pair_assignment.py']},inputs_sha256={str(p):sha(p) for p in inputs},source_model_sha256=cfg['source_model_sha256'],thresholds=THRESHOLDS,policy=__doc__);(out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n');training=[r for s,r in records.items() if s in ids];inner=[]
    for mouse in sorted({r['subject'] for r in training}):
        heads=fit_heads([r for r in training if r['subject']!=mouse]);inner.extend((r,scores(heads,r['x'])) for r in training if r['subject']==mouse)
    measures=[];total=sum(len(truth[s]['match_pairs']) for s in ids)
    for threshold in np.arange(.05,1.,.05):
        count=correct=0
        for r,pred in inner:
            chosen=choose(r,pred,threshold);count+=len(chosen);correct+=int(r['y'][chosen].sum())
        measures.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
    selected=max(measures,key=lambda x:x['f1']);heads=fit_heads(training);torch.save(dict(heads=[dict(state_dict=h[0].state_dict(),mean=h[1],sd=h[2]) for h in heads],selected=selected,held_out=subject,train_ids=ids,configuration_sha256=sha(out/'configuration.json')),out/'quality.pt');(out/'calibration.json').write_text(json.dumps(dict(selected=selected,measurements=measures),indent=2)+'\n')
    bp=Path('outputs/stage32_contour_consensus/predicted_all.csv');base=read_rows(bp);results=[]
    for sid in sorted(set(truth)-set(ids)):
        if sid in records:
            r=records[sid];results.append(assemble(base[sid],r,choose(r,scores(heads,r['x']),selected['threshold'])))
        else:row=copy.deepcopy(base[sid]);row['match_pairs']=[];results.append(row)
    write_rows(results,out/'raw.csv');cp=Path('outputs/stage57_iv_local')/a.fold/'model.pt';state=torch.load(cp,map_location='cpu',weights_only=False);assert state['config']['held_out']==subject and state['config']['train_ids']==ids;net=SmallUNet(16).to(device);net.load_state_dict(state['state_dict']);net.eval();model=D4Average(net).eval()
    for row in results:
        sid=row['sample_id'];image=read_image(image_path(root,'training',sid,'invivo'));row['invivo_instances'],_=refine(parse_instances(row,'invivo',image.shape),image_features(image),model,device)
    write_rows(results,out/'predictions.csv');evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=subject,output=out/'metrics.json'));(out/'manifest.json').write_text(json.dumps(dict(baseline_sha256=sha(bp),iv_model_sha256=sha(cp),quality_sha256=sha(out/'quality.pt'),raw_csv_sha256=sha(out/'raw.csv'),csv_sha256=sha(out/'predictions.csv'),record_sha256={p.stem:sha(p) for p in cache.glob('*.pt')}),indent=2)+'\n')
if __name__=='__main__':main()
