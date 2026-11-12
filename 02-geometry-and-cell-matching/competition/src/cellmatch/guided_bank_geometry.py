"""Re-evaluate frozen Stage 51 models with new image-only global geometry."""
import argparse,copy,hashlib,json
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry',type=Path,default=Path('outputs/stage52_translation_geometry/predicted_all_manifest.json'));p.add_argument('--output',type=Path,default=Path('outputs/stage52_guided_geometry'));a=p.parse_args()
    torch.set_num_threads(2);device=torch.device('mps');assert torch.backends.mps.is_available()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();root=Path('data/raw/Project_2_Dataset');basepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');baseline=read_rows(basepath)
    prior=Path('outputs/stage51_guided_bank/full/predictions.csv');previous=read_rows(prior);audit=json.loads(prior.with_name('audit.json').read_text());assert audit['status']=='passed' and audit['csv_sha256']==sha(prior)
    geo=json.loads(a.geometry.read_text());regions={r['sample_id']:r for r in geo['regions']};assert set(regions)==set(baseline)
    a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'proposals';cache.mkdir(exist_ok=True)
    config=dict(source_sha256=sha(__file__),geometry_sha256=sha(a.geometry),baseline_sha256=sha(basepath),reference_sha256=sha(prior))
    cp=a.output/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');rows=[];files=[basepath,prior,prior.with_name('audit.json'),a.geometry];diagnostics=[]
    for tag,subject in FOLDS.items():
        ids=sorted(s for s in baseline if split_id(s)[0]!=subject);mp=Path(f'outputs/stage48_guided_ex/{tag}/model.pt');qp=Path(f'outputs/stage51_guided_bank/{tag}/quality.pt');files.extend([mp,qp])
        state=torch.load(mp,map_location='cpu',weights_only=False);quality=torch.load(qp,map_location='cpu',weights_only=False)
        assert state['config']['held_out']==quality['held_out']==subject and state['config']['train_ids']==quality['train_ids']==ids
        assert sha(qp)==next(r['head_sha256'] for r in audit['folds'] if r['fold']==tag)
        model=GuidedNet(state['config']['width']).to(device);model.load_state_dict(state['state_dict']);model.eval()
        head=torch.nn.Linear(47,1);head.load_state_dict(quality['state_dict']);fitted=head.eval(),quality['mean'],quality['sd']
        for sid in sorted(set(baseline)-set(ids)):
            region=regions[sid]
            if not region.get('translation_replaced'):
                rows.append(previous[sid]);continue
            paths=[Path(f'outputs/stage8_iv/{tag}/maps/{sid}_invivo.npz'),Path(f'outputs/stage35_cpdino_iv_adapted/{tag}/maps/{sid}_invivo.npz'),Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz'),Path(f'outputs/stage35_cpdino_iv_adapted/{tag}/regions/{sid}.json')];files.extend(paths)
            om,nm,em=[np.load(p)['maps'].astype(np.float32) for p in paths[:3]];old=baseline[sid];new=copy.deepcopy(old);new['invivo_instances']=json.loads(paths[3].read_text())['invivo_instances']
            record=make_record(root,sid,old,np.array(region['matrix']),em);proposals=[]
            for b,row in enumerate([old,new]):
                path=cache/f'{sid}_{b}.json'
                if not path.exists():
                    rr=dict(record);rr['cells']=[cell_record(k,m) for k,m in parse_instances(row,'invivo',om.shape[1:]).items()]
                    path.write_text(json.dumps(infer(model,rr,device))+'\n')
                proposals.append(json.loads(path.read_text())['proposals']);files.append(path)
            bank=bank_record([old,new],[om,nm],proposals,em.shape[1:]);chosen=choose(bank,probability(fitted,bank['x']),quality['selected']['threshold']);row=assemble(old,bank,chosen);rows.append(row)
            diagnostics.append(dict(sample_id=sid,old_pairs=len(previous[sid]['match_pairs']),new_pairs=len(chosen)));print('New geometry',diagnostics[-1],flush=True)
        del model
    assert len(rows)==47;write_rows(rows,a.output/'predictions.csv')
    (a.output/'manifest.json').write_text(json.dumps(dict(config=config,regions=diagnostics,input_sha256={str(p):sha(p) for p in files},policy='Only label-free geometry replaced; neural and quality models frozen; unchanged regions copied exactly.'),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=a.output/'predictions.csv',subject=None,output=a.output/'metrics.json'))

if __name__=='__main__':main()
