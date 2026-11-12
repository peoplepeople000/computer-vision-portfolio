"""D4 probability averaging of a frozen raw-guided model, with matched calibration.
Every outer fold reuses only its own excluded-mouse model. Quality fitting and
held-out prediction use identical rotation/reflection-averaged proposals.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .learning import image_features
from .guided_ex_data import cell_record,sample
from .guided_ex_repaired import GuidedNet,objective,infer
from .guided_ex_bank import bank_record,choose,assemble
from .cellpose_finetune import instance_labels
from .metrics import segmentation_score
from .joint_pair_assignment import fit,probability
from .stage3 import FOLDS
from .cli import evaluate


def make_record(root,split,sid,rows,matrix,truth=None):
    iv=read_image(image_path(root,split,sid,'invivo'));ex=read_image(image_path(root,split,sid,'exvivo'));cells=[]
    actual=parse_instances(truth,'invivo',iv.shape) if truth is not None else None
    pairs=dict(truth['match_pairs']) if truth is not None else {}
    numbers={k:i for i,k in enumerate(truth['exvivo_instances'],1)} if truth is not None else {}
    for b,row in enumerate(rows):
        masks=parse_instances(row,'invivo',iv.shape)
        mapping=segmentation_score(masks,actual)['pred_to_gt'] if truth is not None else {}
        cells.extend(cell_record(f'{b}:{k}',m,numbers.get(pairs.get(mapping.get(k)),0)) for k,m in masks.items())
    labels=instance_labels(truth['exvivo_instances'],ex.shape) if truth is not None else None
    return dict(sample_id=sid,subject=split_id(sid)[0],matrix=np.asarray(matrix),iv_image=image_features(iv),ex_image=image_features(ex),
        ex_maps=np.zeros((2,*ex.shape),np.float32),labels=labels,cells=cells,shape=ex.shape)


def separate(proposals):
    banks=[[],[]]
    for proposal in proposals:
        p=dict(proposal);source,key=p['iv'].split(':',1);p['iv']=key;banks[int(source)].append(p)
    return banks


class GuidedD4(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,x):
        maps=[];presence=[]
        for reflected in [False,True]:
            base=x.flip(-1) if reflected else x
            for k in range(4):
                logits,exists=self.model(torch.rot90(base,k,(-2,-1)))
                restored=torch.rot90(logits.sigmoid(),-k,(-2,-1))
                if reflected:restored=restored.flip(-1)
                maps.append(restored);presence.append(exists.sigmoid())
        return tuple(torch.logit(torch.stack(v).mean(0).clamp(1e-6,1-1e-6)) for v in [maps,presence])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=[*FOLDS,'all-training'],required=True);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--output',type=Path,default=Path('outputs/stage60_guided_d4'));p.add_argument('--adapted',type=Path,default=Path('outputs/stage35_cpdino_iv_adapted'))
    p.add_argument('--public',type=Path,default=Path('outputs/stage28_cpdino_iv_full'));p.add_argument('--public-hidden',type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(2081);rng=np.random.default_rng(2081);device=torch.device('mps');assert torch.backends.mps.is_available()
    assert a.fold!='all-training','Research-only wrapper: final deployment requires separately audited inputs'
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS.get(a.fold);ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    adapted=a.adapted/a.fold;acfg=json.loads((adapted/'configuration.json').read_text());assert acfg['held_out']==subject and acfg['train_ids']==ids
    public=read_rows(a.public/'predictions.csv');out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True);cache=out/'proposals';cache.mkdir(exist_ok=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();files=[a.public/'predictions.csv',adapted/'configuration.json',adapted/'model.pt',root/'training/train_ground_truth.csv']
    def sources(sid,publicrows,publicroot):
        rp=adapted/'regions'/f'{sid}.json';paths=[publicroot/'maps'/f'{sid}_invivo.npz',adapted/'maps'/f'{sid}_invivo.npz'];files.extend([rp,*paths])
        return [publicrows[sid],json.loads(rp.read_text())],[np.load(path)['maps'].astype(np.float32) for path in paths]
    records=[];train_rows={};train_maps={}
    for sid in ids:
        files.extend(image_path(root,'training',sid,m) for m in ['invivo','exvivo'])
        if len(truth[sid]['match_pairs'])<5:continue
        rows,maps=sources(sid,public,a.public);iv=parse_instances(truth[sid],'invivo',maps[0].shape[1:]);exshape=read_image(image_path(root,'training',sid,'exvivo')).shape;ex=parse_instances(truth[sid],'exvivo',exshape)
        u=np.array([iv[i].coordinates().mean(0) for i,j in truth[sid]['match_pairs']]);v=np.array([ex[j].coordinates().mean(0) for i,j in truth[sid]['match_pairs']]);matrix=np.linalg.lstsq(np.c_[u,np.ones(len(u))],v,rcond=None)[0].T
        records.append(make_record(root,'training',sid,rows,matrix,truth[sid]));train_rows[sid]=rows;train_maps[sid]=maps
        print('Raw guided training record',a.fold,sid,len(records[-1]['cells']),flush=True)
    dependencies=['guided_d4_bundle.py','guided_ex_data.py','guided_ex_repaired.py','guided_ex_bank.py','joint_pair_assignment.py']
    source=Path('outputs/stage56_raw_guided')/a.fold;final=source/'model.pt'
    state=torch.load(final,map_location='cpu',weights_only=False);scfg=json.loads((source/'configuration.json').read_text())
    assert scfg==state['config'] and scfg['held_out']==subject and scfg['train_ids']==ids
    assert state['step']==scfg['steps']==3000
    for name,digest in scfg['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    config=dict(held_out=subject,train_ids=ids,source_model_sha256=sha(final),source_configuration_sha256=sha(source/'configuration.json'),view='D4 probability average, eight transforms',source_sha256={n:sha(Path(__file__).with_name(n)) for n in dependencies},training_inputs_sha256={str(p):sha(p) for p in files},policy=__doc__)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n')
    neural=GuidedNet(scfg['width']).to(device);neural.load_state_dict(state['state_dict']);model=GuidedD4(neural.eval()).eval()
    model.eval();calibration=[]
    for record in records:
        sid=record['sample_id'];path=cache/f'{sid}.json'
        if not path.exists():path.write_text(json.dumps(infer(model,record,device))+'\n')
        rec=bank_record(train_rows[sid],train_maps[sid],separate(json.loads(path.read_text())['proposals']),record['shape'],truth[sid]);calibration.append(rec)
        print('Raw guided calibration',sid,len(rec['y']),int(rec['y'].sum()),flush=True)
    del records,train_rows,train_maps
    inner=[]
    for mouse in sorted({r['subject'] for r in calibration}):
        fitted=fit([r for r in calibration if r['subject']!=mouse]);inner.extend((r,probability(fitted,r['x'])) for r in calibration if r['subject']==mouse)
    measurements=[];total=sum(len(truth[s]['match_pairs']) for s in ids)
    for threshold in np.arange(.05,1.,.05):
        count=correct=0
        for rec,scores in inner:
            chosen=choose(rec,scores,threshold);count+=len(chosen);correct+=int(rec['y'][chosen].sum())
        measurements.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
    selected=max(measurements,key=lambda v:v['f1']);fitted=fit(calibration)
    torch.save(dict(state_dict=fitted[0].state_dict(),mean=fitted[1],sd=fitted[2],held_out=subject,train_ids=ids,selected=selected,segmenter_sha256=sha(final)),out/'quality.pt')
    (out/'calibration.json').write_text(json.dumps(dict(selected=selected,measurements=measurements,segmenter_sha256=sha(final),limitation='Inner classifier holdout only; neural training includes both development mice.'),indent=2)+'\n')
    if subject is None:
        if a.public_hidden is None:raise ValueError('Final inference requires explicitly supplied public hidden predictions/maps')
        split='hidden_test';basepath=Path('outputs/submission_stage32_cellpose_consensus/submission.csv');gp=Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json');prediction_public=read_rows(a.public_hidden/'predictions.csv');prediction_public_root=a.public_hidden
    else:
        split='training';basepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');gp=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');prediction_public=public;prediction_public_root=a.public
    base=read_rows(basepath);files.extend([basepath,gp]);matrices={r['sample_id']:np.array(r['matrix']) for r in json.loads(gp.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'};result=[]
    for sid in sorted(base):
        if subject is not None and split_id(sid)[0]!=subject:continue
        if sid not in matrices:
            row=copy.deepcopy(base[sid]);row['match_pairs']=[];result.append(row);continue
        rows,maps=sources(sid,prediction_public,prediction_public_root);record=make_record(root,split,sid,rows,matrices[sid]);path=cache/f'{sid}.json'
        if not path.exists():path.write_text(json.dumps(infer(model,record,device))+'\n')
        rec=bank_record(rows,maps,separate(json.loads(path.read_text())['proposals']),record['shape']);chosen=choose(rec,probability(fitted,rec['x']),selected['threshold']);result.append(assemble(base[sid],rec,chosen))
        print('Raw guided inference',sid,len(chosen),flush=True)
    write_rows(result,out/'predictions.csv');(out/'manifest.json').write_text(json.dumps(dict(config=config,model_sha256=sha(final),quality_sha256=sha(out/'quality.pt'),input_sha256={str(p):sha(p) for p in files},proposals_sha256={p.stem:sha(p) for p in cache.glob('*.json')},limitation='Paired branch has shared fit/inference code; unpaired legacy segmentation remains a separate inherited final-ensemble adaptation.'),indent=2)+'\n')
    if subject is not None:evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=subject,output=out/'metrics.json'))

if __name__=='__main__':main()
