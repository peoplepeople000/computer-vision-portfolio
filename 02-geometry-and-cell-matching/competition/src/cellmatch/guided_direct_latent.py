"""Stage82: direct frozen image features, with fixed joint L2=.2.
All241 features enter the joint classifier without PCA. Original endpoint gates,
candidates, inner threshold protocol and IV refinement remain fixed.
"""
import argparse,copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from scipy import ndimage as ndi
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .stage3 import FOLDS
from .raw_guided_bundle import make_record
from .guided_ex_data import crop
from .guided_ex_repaired import GuidedNet
from .guided_regularized_quality import balanced_fit
from .guided_endpoint_quality import factor_fit as base_fit,factor_probability as base_probability
from .guided_ex_bank import choose,assemble
from .rle import Mask
from .learning import SmallUNet,image_features
from .iv_local_boundary import D4Average
from .ex_boundary import refine
from .cli import evaluate


def fit_records(records):
    # Preserve endpoint gates; only the augmented joint classifier changes.
    heads=base_fit([dict(r,x=r['x'][:,:73]) for r in records])
    heads[3]=balanced_fit(records,regularization=.2,balanced=False)
    return dict(heads=heads)


def score(fitted,x):return base_probability(fitted['heads'],x,'gated_joint')


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args();torch.set_num_threads(2);device=torch.device('mps')
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold];ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    out=Path('outputs/stage82_direct_latent')/a.fold;cache=out/'records';cache.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=Path('outputs/stage81_latent_quality')/a.fold;cfg=json.loads((source/'configuration.json').read_text());manifest=json.loads((source/'gated_joint/manifest.json').read_text());assert cfg['held_out']==subject and cfg['train_ids']==ids
    for n,d in cfg['source_sha256'].items():assert sha(Path(__file__).with_name(n))==d
    for path,d in cfg['input_sha256'].items():assert sha(path)==d,path
    mp=Path('outputs/stage56_raw_guided')/a.fold/'model.pt';assert cfg['source_model_sha256']==sha(mp)
    bp=Path('outputs/stage32_contour_consensus/predicted_all.csv');base=read_rows(bp);records={};files=[source/'configuration.json',source/'gated_joint/manifest.json',mp,bp,root/'training/train_ground_truth.csv']
    dependencies=['guided_direct_latent.py','guided_regularized_quality.py','guided_endpoint_quality.py','guided_ex_bank.py','joint_pair_assignment.py'];sourcehash={n:sha(Path(__file__).with_name(n)) for n in dependencies}
    for sid,digest in sorted(manifest['record_sha256'].items()):
        rp=source/'records'/f'{sid}.pt';assert sha(rp)==digest;record=torch.load(rp,map_location='cpu',weights_only=False)
        assert record['x'].shape==(len(record['candidates']),241) and np.isfinite(record['x']).all()
        if sid not in ids:assert len(record['y'])==0 and 'endpoint' not in record
        else:assert record['subject']!=subject and record['endpoint'].shape==(len(record['x']),2)
        (cache/rp.name).write_bytes(rp.read_bytes());files.append(rp);records[sid]=record
    config=dict(held_out=subject,train_ids=ids,source_model_sha256=sha(mp),source_sha256=sourcehash,input_sha256={str(p):sha(p) for p in files},policy=__doc__);(out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    training=[r for s,r in records.items() if s in ids];inner=[]
    for mouse in sorted({r['subject'] for r in training}):
        fitted=fit_records([r for r in training if r['subject']!=mouse]);inner.extend((r,score(fitted,r['x'])) for r in training if r['subject']==mouse)
    total=sum(len(truth[s]['match_pairs']) for s in ids);measurements=[]
    for threshold in np.arange(.05,1.,.05):
        count=correct=0
        for r,scores in inner:
            chosen=choose(r,scores,threshold);count+=len(chosen);correct+=int(r['y'][chosen].sum())
        measurements.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
    selected=max(measurements,key=lambda r:r['f1']);fitted=fit_records(training);folder=out/'gated_joint';folder.mkdir(exist_ok=True)
    torch.save(dict(heads=[dict(state_dict=h[0].state_dict(),mean=h[1],sd=h[2]) for h in fitted['heads']],selected=selected,held_out=subject,train_ids=ids,configuration_sha256=sha(out/'configuration.json')),folder/'quality.pt');(folder/'calibration.json').write_text(json.dumps(dict(selected=selected,measurements=measurements),indent=2)+'\n')
    results=[]
    for sid in sorted(set(truth)-set(ids)):
        if sid in records:
            r=records[sid];results.append(assemble(base[sid],r,choose(r,score(fitted,r['x']),selected['threshold'])))
        else:
            row=copy.deepcopy(base[sid]);row['match_pairs']=[];results.append(row)
    write_rows(results,folder/'raw.csv');ip=Path('outputs/stage57_iv_local')/a.fold/'model.pt';st=torch.load(ip,map_location='cpu',weights_only=False);assert st['config']['held_out']==subject and st['config']['train_ids']==ids
    net=SmallUNet(16).to(device);net.load_state_dict(st['state_dict']);net.eval();refiner=D4Average(net).eval()
    for row in results:
        image=read_image(image_path(root,'training',row['sample_id'],'invivo'));row['invivo_instances'],_=refine(parse_instances(row,'invivo',image.shape),image_features(image),refiner,device)
    write_rows(results,folder/'predictions.csv');evaluate(SimpleNamespace(data=root,predictions=folder/'predictions.csv',subject=subject,output=folder/'metrics.json'))
    (folder/'manifest.json').write_text(json.dumps(dict(iv_model_sha256=sha(ip),quality_sha256=sha(folder/'quality.pt'),raw_csv_sha256=sha(folder/'raw.csv'),csv_sha256=sha(folder/'predictions.csv'),record_sha256={p.stem:sha(p) for p in cache.glob('*.pt')}),indent=2)+'\n')
if __name__=='__main__':main()
