"""Explicit IV validity, EX validity and conditional correspondence calibration.
Endpoint labels are derived only from other mice's annotations. Held-out records
contain no labels. Compare factorized scores and endpoint-gated joint scores.
"""
import argparse,copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,read_image,image_path,split_id
from .stage3 import FOLDS
from .cell_features import extract_features
from .guided_ex_bank import bank_record,choose,assemble
from .raw_guided_bundle import separate
from .joint_pair_assignment import fit,probability
from .rle import Mask
from .cli import evaluate
from .learning import SmallUNet,image_features
from .iv_local_boundary import D4Average
from .ex_boundary import refine

def endpoint_labels(record,truth,shapes):
    actual=[parse_instances(truth,m,shape) for m,shape in zip(['invivo','exvivo'],shapes)]
    labels=[]
    for c in record['candidates']:
        masks=[Mask.parse(c[key],shape) for key,shape in zip(['iv_rle','ex_rle'],shapes)]
        labels.append([max((mask.iou(g) for g in bank.values()),default=0)>.75 for mask,bank in zip(masks,actual)])
    return np.asarray(labels,bool).reshape(-1,2)


def factor_fit(records):
    heads=[]
    for endpoint,cols in [(0,slice(13,43)),(1,slice(47,73))]:
        heads.append(fit([dict(r,x=r['x'][:,cols],y=r['endpoint'][:,endpoint]) for r in records]))
    conditional=[]
    for r in records:
        valid=r['endpoint'].all(1)
        if valid.any():conditional.append(dict(r,x=r['x'][valid],y=r['y'][valid]))
    heads.append(fit(conditional));heads.append(fit(records));return heads


def factor_probability(heads,x,recipe):
    iv=probability(heads[0],x[:,13:43]);ex=probability(heads[1],x[:,47:73])
    if recipe=='factorized':return iv*ex*probability(heads[2],x)
    if recipe=='gated_joint':return np.sqrt(iv*ex)*probability(heads[3],x)
    raise ValueError(recipe)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args()
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');out=Path('outputs/stage62_endpoint_quality')/a.fold;out.mkdir(parents=True,exist_ok=True);cache=out/'records';cache.mkdir(exist_ok=True)
    truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold];ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    source=Path('outputs/stage56_raw_guided')/a.fold;cfg=json.loads((source/'configuration.json').read_text());manifest=json.loads((source/'manifest.json').read_text());assert cfg['held_out']==subject and cfg['train_ids']==ids
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert manifest['model_sha256']==sha(source/'model.pt')
    publicpath=Path('outputs/stage28_cpdino_iv_full/predictions.csv');public=read_rows(publicpath);basepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');base=read_rows(basepath)
    inputs=[publicpath,basepath,source/'configuration.json',source/'manifest.json',root/'training/train_ground_truth.csv'];records={}
    old=Path('outputs/stage61_appearance')/a.fold
    ocfg=json.loads((old/'configuration.json').read_text());assert ocfg['held_out']==subject and ocfg['train_ids']==ids
    for name,digest in ocfg['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    inputs.append(old/'configuration.json');cached=json.loads((old/'iv_only/manifest.json').read_text())['record_sha256']
    for rp in sorted((old/'records').glob('*.pt')):
        sid=rp.stem;assert sha(rp)==cached[sid];record=torch.load(rp,map_location='cpu',weights_only=False);inputs.append(rp)
        if sid in ids:
            shapes=[read_image(image_path(root,'training',sid,m)).shape for m in ['invivo','exvivo']]
            record['endpoint']=endpoint_labels(record,truth[sid],shapes)
        else:assert len(record['y'])==0
        records[sid]=record;torch.save(record,cache/f'{sid}.pt');print('Endpoint record',a.fold,sid,len(record['x']),flush=True)
    config=dict(held_out=subject,train_ids=ids,source_model_sha256=sha(source/'model.pt'),source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['guided_endpoint_quality.py','guided_appearance_quality.py','guided_ex_bank.py','cell_features.py','joint_pair_assignment.py']},input_sha256={str(p):sha(p) for p in inputs},policy=__doc__)
    (out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    for recipe in ['factorized','gated_joint']:
        recs=records;training=[r for s,r in recs.items() if s in ids];inner=[]
        for mouse in sorted({r['subject'] for r in training}):
            fitted=factor_fit([r for r in training if r['subject']!=mouse]);inner.extend((r,factor_probability(fitted,r['x'],recipe)) for r in training if r['subject']==mouse)
        measures=[];total=sum(len(truth[s]['match_pairs']) for s in ids)
        for threshold in np.arange(.05,1.,.05):
            count=correct=0
            for r,scores in inner:
                chosen=choose(r,scores,threshold);count+=len(chosen);correct+=int(r['y'][chosen].sum())
            measures.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
        selected=max(measures,key=lambda x:x['f1']);fitted=factor_fit(training);folder=out/recipe;folder.mkdir(exist_ok=True)
        torch.save(dict(heads=[dict(state_dict=h[0].state_dict(),mean=h[1],sd=h[2]) for h in fitted],recipe=recipe,selected=selected,held_out=subject,train_ids=ids,configuration_sha256=sha(out/'configuration.json')),folder/'quality.pt');(folder/'calibration.json').write_text(json.dumps(dict(selected=selected,measurements=measures),indent=2)+'\n')
        results=[]
        for sid in sorted(set(truth)-set(ids)):
            if sid in recs:
                r=recs[sid];results.append(assemble(base[sid],r,choose(r,factor_probability(fitted,r['x'],recipe),selected['threshold'])))
            else:
                row=copy.deepcopy(base[sid]);row['match_pairs']=[];results.append(row)
        write_rows(results,folder/'raw.csv')
        device=torch.device('mps');cp=Path('outputs/stage57_iv_local')/a.fold/'model.pt';state=torch.load(cp,map_location='cpu',weights_only=False);assert state['config']['held_out']==subject and state['config']['train_ids']==ids
        net=SmallUNet(16).to(device);net.load_state_dict(state['state_dict']);net.eval();model=D4Average(net).eval()
        for row in results:
            sid=row['sample_id'];image=read_image(image_path(root,'training',sid,'invivo'));masks=parse_instances(row,'invivo',image.shape);row['invivo_instances'],_=refine(masks,image_features(image),model,device)
        write_rows(results,folder/'predictions.csv');evaluate(SimpleNamespace(data=root,predictions=folder/'predictions.csv',subject=subject,output=folder/'metrics.json'))
        (folder/'manifest.json').write_text(json.dumps(dict(iv_model_sha256=sha(cp),quality_sha256=sha(folder/'quality.pt'),raw_csv_sha256=sha(folder/'raw.csv'),csv_sha256=sha(folder/'predictions.csv'),record_sha256={p.stem:sha(p) for p in cache.glob('*.pt')}),indent=2)+'\n');del net,model;torch.mps.empty_cache()
if __name__=='__main__':main()
