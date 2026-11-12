"""Raw-image guided EX model with development-only misregistration negatives.
Use the same 3000-step, width-24, seed-2081 baseline initialization and optimizer.
For 20 percent of sampled verified cells, use a 40–100 pixel wrong projection,
checking that the true target is absent; unrelated cells are never positive.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .learning import image_features
from .guided_ex_data import cell_record
from .guided_ex_hard_negative_data import sample
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=[*FOLDS,'all-training'],required=True);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--output',type=Path,default=Path('outputs/stage70_raw_hard_negative'));p.add_argument('--adapted',type=Path,default=Path('outputs/stage35_cpdino_iv_adapted'))
    p.add_argument('--public',type=Path,default=Path('outputs/stage28_cpdino_iv_full'));p.add_argument('--public-hidden',type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(2081);rng=np.random.default_rng(2081);device=torch.device('mps');assert torch.backends.mps.is_available()
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
    dependencies=['raw_guided_hard_bundle.py','guided_ex_hard_negative_data.py','guided_ex_data.py','guided_ex_repaired.py','guided_ex_bank.py','joint_pair_assignment.py']
    config=dict(held_out=subject,train_ids=ids,steps=a.steps,width=24,seed=2081,initializer='fixed random seed, no dataset-pretrained initializer',supervised_ex_maps=False,batch=16,source_sha256={n:sha(Path(__file__).with_name(n)) for n in dependencies},training_inputs_sha256={str(p):sha(p) for p in files},policy=__doc__)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');model=GuidedNet(24).to(device);final=out/'model.pt'
    if final.exists():
        state=torch.load(final,map_location='cpu',weights_only=False);assert state['config']==config and state['step']==a.steps;model.load_state_dict(state['state_dict'])
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=3e-5);model.train();history=[];tick=time.perf_counter()
        for step in range(1,a.steps+1):
            x,y,exists=(v.to(device) for v in sample(records,rng));optimizer.zero_grad(set_to_none=True);logits,presence=model(x);loss=objective(logits,presence,y,exists);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                item=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.perf_counter()-tick);history.append(item);print('Raw guided training',a.fold,item,flush=True)
        torch.save(dict(config=config,step=a.steps,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),final);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        del optimizer,scheduler,x,y,exists,logits,presence,loss
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
