"""Stage 73: frozen raw guided endpoint quality ranks image-only geometry.
Reuse Stage 53's fixed acceptance rule and 24-hypothesis cap. No annotations are
read during inference; apply globally to every low-vote eligible region.
"""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .stage3 import FOLDS
from .guided_geometry_ranking import candidate_matrices
from .raw_guided_bundle import make_record,separate
from .guided_ex_repaired import GuidedNet,infer
from .guided_ex_bank import bank_record,choose,assemble
from .guided_appearance_quality import appearance_record
from .guided_endpoint_quality import factor_probability
from .learning import SmallUNet,image_features
from .iv_local_boundary import D4Average
from .ex_boundary import refine
from .cli import evaluate

def select_trial(trials):
    best=max(trials,key=lambda t:t['utility'])
    return best['index'] if best['index']!=0 and best['pairs']>=5 and best['utility']>max(1.5,1.25*trials[0]['utility']) else 0

def main():
    torch.set_num_threads(2);device=torch.device('mps');assert torch.backends.mps.is_available()
    root=Path('data/raw/Project_2_Dataset');out=Path('outputs/stage73_raw_geometry');out.mkdir(exist_ok=True);cache=out/'regions';cache.mkdir(exist_ok=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    bp=Path('outputs/stage32_contour_consensus/predicted_all.csv');pp=Path('outputs/stage62_endpoint_quality/full/gated_joint/predictions.csv');pub=Path('outputs/stage28_cpdino_iv_full/predictions.csv')
    gp=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');npth=Path('outputs/stage52_translation_geometry/predicted_all_manifest.json')
    baseline=read_rows(bp);previous=read_rows(pp);public=read_rows(pub)
    original={r['sample_id']:r for r in json.loads(gp.read_text())['regions']};alternative={r['sample_id']:r for r in json.loads(npth.read_text())['regions']}
    audit=json.loads(pp.with_name('audit.json').read_text());assert audit['status']=='passed' and audit['csv_sha256']==sha(pp)
    files=[bp,pp,pub,gp,npth,pp.with_name('audit.json')]
    dependencies=['raw_geometry_ranking.py','guided_geometry_ranking.py','raw_guided_bundle.py','guided_ex_repaired.py','guided_ex_data.py','guided_ex_bank.py','guided_appearance_quality.py','guided_endpoint_quality.py','cell_features.py','iv_local_boundary.py','ex_boundary.py','learning.py']
    config=dict(source_sha256={n:sha(Path(__file__).with_name(n)) for n in dependencies},input_sha256={str(p):sha(p) for p in files},policy=__doc__)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');rows=[];reports=[]
    for tag,subject in FOLDS.items():
        ids=sorted(s for s in baseline if split_id(s)[0]!=subject)
        mp=Path(f'outputs/stage56_raw_guided/{tag}/model.pt');qp=Path(f'outputs/stage62_endpoint_quality/{tag}/gated_joint/quality.pt');ip=Path(f'outputs/stage57_iv_local/{tag}/model.pt');files.extend([mp,qp,ip])
        state=torch.load(mp,map_location='cpu',weights_only=False);quality=torch.load(qp,map_location='cpu',weights_only=False);ivstate=torch.load(ip,map_location='cpu',weights_only=False)
        for cfg in [state['config'],quality,ivstate['config']]:assert cfg['held_out']==subject and cfg['train_ids']==ids
        sm=json.loads(Path(f'outputs/stage56_raw_guided/{tag}/manifest.json').read_text());qm=json.loads(qp.with_name('manifest.json').read_text());assert sm['model_sha256']==sha(mp) and qm['quality_sha256']==sha(qp) and qm['iv_model_sha256']==sha(ip)
        model=GuidedNet(24).to(device);model.load_state_dict(state['state_dict']);model.eval()
        net=SmallUNet(16).to(device);net.load_state_dict(ivstate['state_dict']);net.eval();refiner=D4Average(net).eval()
        fitted=[]
        for h in quality['heads']:
            head=torch.nn.Linear(len(h['mean']),1);head.load_state_dict(h['state_dict']);fitted.append((head.eval(),h['mean'],h['sd']))
        threshold=quality['selected']['threshold']
        for sid in sorted(set(baseline)-set(ids)):
            candidates=alternative[sid].get('translation_candidates',[])
            if not candidates or original[sid].get('votes',0)>=4:rows.append(previous[sid]);continue
            dest=cache/sid;dest.mkdir(exist_ok=True)
            paths=[Path(f'outputs/stage28_cpdino_iv_full/maps/{sid}_invivo.npz'),Path(f'outputs/stage35_cpdino_iv_adapted/{tag}/maps/{sid}_invivo.npz'),Path(f'outputs/stage35_cpdino_iv_adapted/{tag}/regions/{sid}.json'),Path(f'outputs/stage62_endpoint_quality/{tag}/records/{sid}.pt')]
            images=[image_path(root,'training',sid,m) for m in ['invivo','exvivo']];files.extend([*paths,*images])
            maps=[np.load(p)['maps'].astype(np.float32) for p in paths[:2]];banks=[public[sid],json.loads(paths[2].read_text())];iv,ex=map(read_image,images)
            matrices=candidate_matrices(original[sid]['matrix'],candidates,iv.shape,limit=24);record=make_record(root,'training',sid,banks,matrices[0]);trials=[];records=[]
            region_config=dict(matrices=[m.tolist() for m in matrices],input_sha256={str(p):sha(p) for p in [mp,qp,ip,*paths,*images]},configuration_sha256=sha(cp))
            rc=dest/'configuration.json'
            if rc.exists():assert json.loads(rc.read_text())==region_config
            rc.write_text(json.dumps(region_config,indent=2)+'\n');files.append(rc)
            for index,matrix in enumerate(matrices):
                rp=dest/f'{index:02d}.pt'
                if rp.exists():bank=torch.load(rp,map_location='cpu',weights_only=False)
                elif index==0:
                    assert sha(paths[3])==qm['record_sha256'][sid];bank=torch.load(paths[3],map_location='cpu',weights_only=False);torch.save(bank,rp)
                else:
                    proposal=infer(model,dict(record,matrix=matrix),device)['proposals']
                    bank=appearance_record(bank_record(banks,maps,separate(proposal),ex.shape),banks,maps,iv,ex);torch.save(bank,rp)
                assert len(bank['y'])==0 and 'endpoint' not in bank
                scores=factor_probability(fitted,bank['x'],'gated_joint');chosen=choose(bank,scores,threshold)
                trials.append(dict(index=index,pairs=len(chosen),utility=float(np.maximum(scores[chosen]-threshold,0).sum())))
                records.append(bank);files.append(rp);print('Trial',tag,sid,index,len(matrices),trials[-1],flush=True)
            selected=select_trial(trials)
            if selected==0:row=previous[sid]
            else:
                bank=records[selected];row=assemble(baseline[sid],bank,choose(bank,factor_probability(fitted,bank['x'],'gated_joint'),threshold))
                write_rows([row],dest/'raw.csv');files.append(dest/'raw.csv')
                masks=parse_instances(row,'invivo',iv.shape);row['invivo_instances'],_=refine(masks,image_features(iv),refiner,device)
            rows.append(row);report=dict(sample_id=sid,fold=tag,selected=selected,trials=trials);reports.append(report);(dest/'selection.json').write_text(json.dumps(report,indent=2)+'\n');files.append(dest/'selection.json');print('Selected',sid,selected,flush=True)
        del model,net,refiner;torch.mps.empty_cache()
    assert len(rows)==47;write_rows(rows,out/'predictions.csv')
    (out/'manifest.json').write_text(json.dumps(dict(configuration_sha256=sha(cp),input_sha256={str(p):sha(p) for p in files},regions=reports,csv_sha256=sha(out/'predictions.csv')),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=None,output=out/'metrics.json'))
if __name__=='__main__':main()
