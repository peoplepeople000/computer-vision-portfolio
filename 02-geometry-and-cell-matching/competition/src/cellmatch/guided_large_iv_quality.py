"""Stage89: supplement Stage77 with public ViT-L IV prompts and frozen EX guide.
Original candidates and73 features remain exact. New candidates use public/adapted
agreement features and image-only overlap families. Same development-only head
fitting, threshold protocol and IV refinement; segmentation networks stay frozen.
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

from .guided_endpoint_quality import endpoint_labels,factor_fit,factor_probability
from .guided_appearance_quality import appearance_record
from .raw_guided_bundle import make_record
from .guided_ex_repaired import GuidedNet
from .guided_multicomponent_decode import infer
from .guided_large_iv_bank import supplement

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args()
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');out=Path('outputs/stage89_large_iv_guided')/a.fold;out.mkdir(parents=True,exist_ok=True);cache=out/'records';cache.mkdir(exist_ok=True);additional_cache=out/'additional_records';additional_cache.mkdir(exist_ok=True)
    truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold];ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    source=Path('outputs/stage56_raw_guided')/a.fold;cfg=json.loads((source/'configuration.json').read_text());manifest=json.loads((source/'manifest.json').read_text());assert cfg['held_out']==subject and cfg['train_ids']==ids
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert manifest['model_sha256']==sha(source/'model.pt')
    publicpath=Path('outputs/stage88_large_iv_full/predictions.csv');public=read_rows(publicpath);basepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');base=read_rows(basepath)
    inputs=[publicpath,basepath,source/'configuration.json',source/'manifest.json',root/'training/train_ground_truth.csv'];records={}
    gp=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');inputs.append(gp)
    geometry={r['sample_id']:np.array(r['matrix']) for r in json.loads(gp.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    auditpath=Path('outputs/stage88_large_iv_full/coverage_audit.json');audit=json.loads(auditpath.read_text());assert audit['summary']['status']=='passed' and audit['input_sha256'][str(publicpath)]==sha(publicpath);inputs.append(auditpath)
    original=Path('outputs/stage77_multi_ex')/a.fold;ocfg=json.loads((original/'configuration.json').read_text());om=json.loads((original/'gated_joint/manifest.json').read_text());assert ocfg['held_out']==subject and ocfg['train_ids']==ids
    for name,digest in ocfg['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    inputs.extend([original/'configuration.json',original/'gated_joint/manifest.json'])
    dependencies=['guided_large_iv_quality.py','guided_large_iv_bank.py','guided_multicomponent_decode.py','guided_endpoint_quality.py','guided_appearance_quality.py','guided_ex_bank.py','cell_features.py','joint_pair_assignment.py','raw_guided_bundle.py','guided_ex_data.py','guided_ex_repaired.py']
    sourcehashes={n:sha(Path(__file__).with_name(n)) for n in dependencies}
    device=torch.device('mps');saved=torch.load(source/'model.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
    for n,digest in cfg['source_sha256'].items():assert sha(Path(__file__).with_name(n))==digest
    network=GuidedNet(24).to(device);network.load_state_dict(saved['state_dict']);network.eval()
    adapted=Path('outputs/stage35_cpdino_iv_adapted')/a.fold
    acfg=json.loads((adapted/'configuration.json').read_text());assert acfg['held_out']==subject and acfg['train_ids']==ids
    inputs.extend([source/'model.pt',adapted/'configuration.json'])
    for sid in sorted(truth):
        pp=source/'proposals'/f'{sid}.json'
        if not pp.exists():continue
        if sid in ids and len(truth[sid]['match_pairs'])<5:continue
        if sid not in ids and sid not in geometry:continue
        assert sha(pp)==manifest['proposals_sha256'][sid]
        paths=[publicpath.parent/'maps'/f'{sid}_invivo.npz',adapted/'maps'/f'{sid}_invivo.npz',adapted/'regions'/f'{sid}.json']
        images=[image_path(root,'training',sid,m) for m in ['invivo','exvivo']];inputs.extend([pp,*paths,*images])
        banks=[public[sid],json.loads(paths[2].read_text())];maps=[np.load(p)['maps'].astype(np.float32) for p in paths[:2]];iv,ex=map(read_image,images)
        if sid in ids:
            masks=[parse_instances(truth[sid],m,shape) for m,shape in zip(['invivo','exvivo'],[iv.shape,ex.shape])]
            u=np.array([masks[0][i].coordinates().mean(0) for i,j in truth[sid]['match_pairs']]);v=np.array([masks[1][j].coordinates().mean(0) for i,j in truth[sid]['match_pairs']])
            matrix=np.linalg.lstsq(np.c_[u,np.ones(len(u))],v,rcond=None)[0].T
        else:matrix=geometry[sid]
        oldpath=original/'records'/f'{sid}.pt';assert sha(oldpath)==om['record_sha256'][sid];old=torch.load(oldpath,map_location='cpu',weights_only=False);inputs.append(oldpath)
        rp=cache/f'{sid}.pt';rc=cache/f'{sid}.json';ap=additional_cache/f'{sid}.pt'
        identity=dict(held_out=subject,train_ids=ids,matrix=matrix.tolist(),source_sha256=sourcehashes,input_sha256={str(p):sha(p) for p in [*inputs[:5],source/'model.pt',pp,*paths,*images,oldpath]},training_labels=sid in ids)
        if rp.exists():
            prior=json.loads(rc.read_text());assert prior['identity']==identity and prior['record_sha256']==sha(rp);record=torch.load(rp,map_location='cpu',weights_only=False);assert prior['additional_sha256']==sha(ap)
        else:
            raw=make_record(root,'training',sid,banks,matrix);raw['cells']=[c for c in raw['cells'] if c['key'].startswith('0:')]
            generated=infer(network,raw,device)['proposals']
            record=bank_record(banks,maps,separate(generated),ex.shape,truth[sid] if sid in ids else None)
            record=appearance_record(record,banks,maps,iv,ex)
            if sid in ids:record['endpoint']=endpoint_labels(record,truth[sid],[iv.shape,ex.shape])
            torch.save(record,ap);record=supplement(old,record,iv.shape)
            torch.save(record,rp);rc.write_text(json.dumps(dict(identity=identity,record_sha256=sha(rp),additional_sha256=sha(ap)),indent=2)+'\n')
        if sid not in ids:assert len(record['y'])==0 and 'endpoint' not in record
        records[sid]=record;inputs.append(rc);print('Low IV candidate record',a.fold,sid,len(record['x']),flush=True)
    del network;torch.mps.empty_cache()
    config=dict(held_out=subject,train_ids=ids,source_model_sha256=sha(source/'model.pt'),source_sha256=sourcehashes,input_sha256={str(p):sha(p) for p in inputs},policy=__doc__)
    (out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    for recipe in ['gated_joint']:
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
        (folder/'manifest.json').write_text(json.dumps(dict(iv_model_sha256=sha(cp),quality_sha256=sha(folder/'quality.pt'),raw_csv_sha256=sha(folder/'raw.csv'),csv_sha256=sha(folder/'predictions.csv'),record_sha256={p.stem:sha(p) for p in cache.glob('*.pt')},additional_sha256={p.stem:sha(p) for p in additional_cache.glob('*.pt')}),indent=2)+'\n');del net,model;torch.mps.empty_cache()
if __name__=='__main__':main()
