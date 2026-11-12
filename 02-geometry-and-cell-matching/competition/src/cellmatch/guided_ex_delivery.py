"""Train one final guided EX model on all annotated mice and build hidden CSV.

Final adaptation uses out-of-fold IV masks and EX maps on all training regions.
Calibration is a three-mouse classifier holdout; segmenter calibration is in-sample.
No hidden labels or public score tuning are used.
"""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,split_id,read_image,image_path,validate_submission
from .learning import image_features
from .guided_ex_data import make_record,cell_record,sample
from .guided_ex_repaired import GuidedNet,objective,infer
from .guided_ex_calibration_repaired import select
from .joint_pair_assignment import fit,probability
from .preserve_pair_masks import merge_row
from .rle import Mask

ROOT=Path('data/raw/Project_2_Dataset')
OUT=Path('outputs/submission_stage48_guided_ex')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assemble(base,proposals,fitted,threshold):
    x=np.asarray([p['features'] for p in proposals],np.float32).reshape(-1,13)
    chosen=select(proposals,probability(fitted,x),threshold)
    paired=copy.deepcopy(base);paired['exvivo_instances']={};paired['match_pairs']=[]
    for n in chosen:
        key=f'GUIDED_{n:06d}';paired['exvivo_instances'][key]=proposals[n]['rle'];paired['match_pairs'].append([proposals[n]['iv'],key])
    return merge_row(paired,base)


def main():
    torch.set_num_threads(2);torch.manual_seed(2078);rng=np.random.default_rng(2078)
    device=torch.device('mps');assert torch.backends.mps.is_available()
    OUT.mkdir(parents=True,exist_ok=True);proot=OUT/'proposals';proot.mkdir(exist_ok=True)
    truth=read_rows(ROOT/'training/train_ground_truth.csv');bp=Path('outputs/stage32_contour_consensus/predicted_all.csv');base=read_rows(bp)
    hiddenpath=Path('outputs/submission_stage32_cellpose_consensus/submission.csv');hidden=read_rows(hiddenpath)
    initializer=Path('outputs/stage6_ex/unet_holdout_5d/model.pt');initial=torch.load(initializer,map_location='cpu',weights_only=False)
    sources=[Path(__file__),Path('src/cellmatch/guided_ex_repaired.py'),Path('src/cellmatch/guided_ex_data.py'),Path('src/cellmatch/guided_ex_calibration_repaired.py')]
    inputs=[bp,hiddenpath,initializer,ROOT/'training/train_ground_truth.csv',Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json')]
    records=[]
    for sid in sorted(truth):
        inputs.extend(image_path(ROOT,'training',sid,m) for m in ['invivo','exvivo'])
        mp=Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz');inputs.append(mp)
        if len(truth[sid]['match_pairs'])<5:continue
        em=np.load(mp)['maps'].astype(np.float32)
        ivshape=read_image(image_path(ROOT,'training',sid,'invivo')).shape
        iv=parse_instances(truth[sid],'invivo',ivshape);ex=parse_instances(truth[sid],'exvivo',em.shape[1:])
        a=np.array([iv[i].coordinates().mean(0) for i,j in truth[sid]['match_pairs']]);b=np.array([ex[j].coordinates().mean(0) for i,j in truth[sid]['match_pairs']])
        matrix=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0].T
        records.append(make_record(ROOT,sid,base[sid],matrix,em,truth[sid]))
    for sid in sorted(hidden):
        inputs.extend(image_path(ROOT,'hidden_test',sid,m) for m in ['invivo','exvivo'])
        inputs.append(Path(f'outputs/submission_ensemble_maps/ensemble/{sid}_exvivo.npz'))
    config=dict(train_ids=sorted(truth),hidden_ids=sorted(hidden),steps=1500,seed=2078,batch=16,width=initial['width'],initializer=str(initializer),
        source_sha256={str(p):sha(p) for p in sources},inputs_sha256={str(p):sha(p) for p in inputs},
        final_adaptation='All-training guided segmenter, fixed 1500 updates; OOF IV masks and EX maps; training GT affines. Fixed first-fold EX initializer. Three-mouse inner classifier calibration with in-sample guided proposals. Hidden geometry is image-only.')
    cp=OUT/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n')
    model=GuidedNet(initial['width'],initial['state_dict']).to(device);final=OUT/'model.pt'
    if final.exists():
        saved=torch.load(final,map_location='cpu',weights_only=False);assert saved['config']==config and saved['step']==1500;model.load_state_dict(saved['state_dict'])
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1500,eta_min=3e-5)
        model.train();tick=time.perf_counter();history=[]
        for step in range(1,1501):
            x,y,exists=(v.to(device) for v in sample(records,rng));optimizer.zero_grad(set_to_none=True);logits,presence=model(x);loss=objective(logits,presence,y,exists)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                item=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.perf_counter()-tick);history.append(item);print(item,flush=True)
        torch.save(dict(config=config,step=1500,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),final)
        (OUT/'history.json').write_text(json.dumps(history,indent=2)+'\n');del optimizer,scheduler,x,y,exists,logits,presence,loss
    model.eval();calibration=[]
    for r in records:
        sid=r['sample_id'];path=proot/f'{sid}.json'
        if not path.exists():path.write_text(json.dumps(infer(model,r,device))+'\n')
        proposals=json.loads(path.read_text())['proposals'];targets={c['key']:c['target'] for c in r['cells']};gt=parse_instances(truth[sid],'exvivo',r['shape']);keys=list(gt)
        y=[bool(targets[p['iv']] and Mask.parse(p['rle'],r['shape']).iou(gt[keys[targets[p['iv']]-1]])>.75) for p in proposals]
        calibration.append(dict(sample_id=sid,subject=r['subject'],proposals=proposals,x=np.array([p['features'] for p in proposals],np.float32).reshape(-1,13),y=np.array(y,bool)))
        print('Final calibration proposals',sid,len(proposals),flush=True)
    del records
    inner=[]
    for subject in sorted({r['subject'] for r in calibration}):
        fitted=fit([r for r in calibration if r['subject']!=subject]);inner.extend((r,probability(fitted,r['x'])) for r in calibration if r['subject']==subject)
    total=sum(len(v['match_pairs']) for v in truth.values());measurements=[]
    for threshold in np.arange(.05,1.,.05):
        correct=count=0
        for r,score in inner:
            chosen=select(r['proposals'],score,threshold);correct+=int(r['y'][chosen].sum());count+=len(chosen)
        measurements.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
    selected=max(measurements,key=lambda v:v['f1']);fitted=fit(calibration)
    torch.save(dict(state_dict=fitted[0].state_dict(),mean=fitted[1],sd=fitted[2],selected=selected,train_ids=sorted(truth)),OUT/'quality.pt')
    (OUT/'calibration.json').write_text(json.dumps(dict(selected=selected,measurements=measurements,limitation='Classifier-only holdout; not a generalization score.'),indent=2)+'\n')
    geo=json.loads(Path('outputs/stage25_hidden_geometry/predicted_all_manifest.json').read_text());matrices={r['sample_id']:np.array(r['matrix']) for r in geo['regions'] if r.get('matrix') and r.get('status')=='registered'}
    rows=[]
    for sid,row in sorted(hidden.items()):
        path=proot/f'{sid}.json'
        if not path.exists():
            if sid not in matrices:value=dict(sample_id=sid,proposals=[])
            else:
                iv=read_image(image_path(ROOT,'hidden_test',sid,'invivo'));ex=read_image(image_path(ROOT,'hidden_test',sid,'exvivo'))
                maps=np.load(f'outputs/submission_ensemble_maps/ensemble/{sid}_exvivo.npz')['maps'].astype(np.float32)
                r=dict(sample_id=sid,subject=split_id(sid)[0],matrix=matrices[sid],iv_image=image_features(iv),ex_image=image_features(ex),ex_maps=maps,labels=None,shape=ex.shape,cells=[cell_record(k,m) for k,m in parse_instances(row,'invivo',iv.shape).items()])
                value=infer(model,r,device)
            path.write_text(json.dumps(value)+'\n')
        proposals=json.loads(path.read_text())['proposals'];rows.append(assemble(row,proposals,fitted,selected['threshold']));print('Final hidden',sid,len(rows[-1]['match_pairs']),flush=True)
    write_rows(rows,OUT/'submission.csv');counts=validate_submission(ROOT,OUT/'submission.csv')
    manifest=dict(model_sha256=sha(final),quality_sha256=sha(OUT/'quality.pt'),configuration_sha256=sha(cp),proposals_sha256={p.stem:sha(p) for p in proot.glob('*.json')},csv_sha256=sha(OUT/'submission.csv'),counts=counts,uploaded=False,public_score=None)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
