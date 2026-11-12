"""Reverse IV recovery initialized from a same-outer IV specialist.
Upweight positive training targets missed by public Cellpose by a fixed factor 4.
Retain the Stage 65 recovery/quality rules and fixed 3000-update evaluation budget.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .stage3 import FOLDS
from .learning import instances_from_maps
from .rle import labels_to_instances,Mask
from .reverse_guided_data import reverse_record
from .guided_ex_data import sample
from .guided_ex_repaired import GuidedNet,objective,infer
from .cell_features import extract_features
from .metrics import segmentation_score
from .joint_pair_assignment import fit,probability
from .preserve_pair_masks import pixels
from .cli import evaluate


def merge_prompts(instances,maps,guided):
    candidates=[];shape=maps.shape[1:]
    for key,rle in instances.items():
        mask=Mask.parse(rle,shape);xy=mask.coordinates();candidates.append((float(maps[0,xy[:,1],xy[:,0]].mean()),'N_'+key,rle))
    for n,p in enumerate(guided):candidates.append((float(p['presence']*p['features'][3]),f'G_{n:06d}',p['rle']))
    result={};occupied=set()
    for score,key,rle in sorted(candidates,reverse=True):
        pp=pixels(rle)
        if pp&occupied:continue
        result[key]=rle;occupied.update(pp)
    return result


def quality_record(sid,instances,proposals,ivshape,eximage,truth=None):
    masks=parse_instances({'exvivo_instances':instances},'exvivo',eximage.shape);keys,_,values=extract_features(eximage,masks);features=dict(zip(keys,values));candidates=[];xs=[];ys=[]
    if truth is not None:
        actualex=parse_instances(truth,'exvivo',eximage.shape);actualiv=parse_instances(truth,'invivo',ivshape);mapping=segmentation_score(masks,actualex)['pred_to_gt'];pairs={e:i for i,e in truth['match_pairs']}
    for p in proposals:
        key=p['iv'];candidates.append(dict(ex=key,ex_rle=instances[key],iv_rle=p['rle']));xs.append(np.r_[p['features'],features[key][:26]])
        if truth is not None:
            target=pairs.get(mapping.get(key));ys.append(bool(target and Mask.parse(p['rle'],ivshape).iou(actualiv[target])>.75))
    return dict(sample_id=sid,subject=split_id(sid)[0],candidates=candidates,x=np.asarray(xs,np.float32).reshape(-1,39),y=np.asarray(ys,bool))


def selected(record,scores,threshold):
    chosen=[];occupied=set();exused=set()
    for n in sorted(range(len(scores)),key=lambda n:(-float(scores[n]),n)):
        c=record['candidates'][n];pp=pixels(c['iv_rle'])
        if scores[n]<threshold or c['ex'] in exused or pp&occupied:continue
        chosen.append(n);exused.add(c['ex']);occupied.update(pp)
    return chosen


def recover(base,record,scores,threshold,ivshape):
    row=copy.deepcopy(base);paired_iv={i for i,e in base['match_pairs']};paired_ex={e for i,e in base['match_pairs']};locked=set()
    for key in paired_iv:locked.update(pixels(base['invivo_instances'][key]))
    additions=[]
    for n in selected(record,scores,threshold):
        c=record['candidates'][n];pp=pixels(c['iv_rle'])
        if c['ex'] in paired_ex or pp&locked:continue
        new=Mask.parse(c['iv_rle'],ivshape);remove=[];reject=False
        for key,rle in row['invivo_instances'].items():
            if not (pp&pixels(rle)):continue
            if key in paired_iv or new.iou(Mask.parse(rle,ivshape))<=.2:reject=True;break
            remove.append(key)
        if reject:continue
        for key in remove:del row['invivo_instances'][key]
        key=f'REVERSE_{n:06d}';assert key not in row['invivo_instances'];row['invivo_instances'][key]=c['iv_rle'];row['match_pairs'].append([key,c['ex']]);paired_iv.add(key);paired_ex.add(c['ex']);locked.update(pp);additions.append(dict(candidate=n,score=float(scores[n]),removed=remove))
    assert row['exvivo_instances']==base['exvivo_instances'] and row['match_pairs'][:len(base['match_pairs'])]==base['match_pairs']
    return row,additions


def emphasize_missing(record,missing_targets,factor=4):
    cells=[]
    for cell in record['cells']:
        copies=factor if cell['target']>0 and cell['target'] in missing_targets else 1
        cells.extend([cell]*copies)
    return dict(record,cells=cells)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(2087);rng=np.random.default_rng(2087);device=torch.device('mps');assert torch.backends.mps.is_available()
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold];ids=sorted(s for s in truth if split_id(s)[0]!=subject);out=Path('outputs/stage66_reverse_initialized')/a.fold;out.mkdir(parents=True,exist_ok=True);cache=out/'proposals';cache.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    exroot=Path('outputs/stage11_native_ex')/a.fold;source=Path('outputs/stage56_raw_guided')/a.fold;ecfg=json.loads((exroot/'training_config.json').read_text());scfg=json.loads((source/'configuration.json').read_text());sm=json.loads((source/'manifest.json').read_text())
    for cfg in [ecfg,scfg]:assert cfg['held_out']==subject and cfg['train_ids']==ids
    assert ecfg['checkpoint_sha256']==sha(ecfg['checkpoint']) and sm['model_sha256']==sha(source/'model.pt')
    initializer=Path('outputs/stage8_iv')/a.fold/'model.pt';initial=torch.load(initializer,map_location='cpu',weights_only=False);assert initial['held_out']==subject and initial['train_ids']==ids and initial['width']==24
    publicpath=Path('outputs/stage28_cpdino_iv_full/predictions.csv');public=read_rows(publicpath);training=[]
    files=[initializer,publicpath,exroot/'training_config.json',source/'configuration.json',source/'manifest.json',root/'training/train_ground_truth.csv'];records=[];prompts={};images={}
    for sid in ids:
        if len(truth[sid]['match_pairs'])<5:continue
        ip,ep=[image_path(root,'training',sid,m) for m in ['invivo','exvivo']];iv,ex=read_image(ip),read_image(ep);mp=exroot/'maps'/f'{sid}_exvivo.npz';maps=np.load(mp)['maps'].astype(np.float32);pp=source/'proposals'/f'{sid}.json';guided=[]
        if pp.exists():assert sha(pp)==sm['proposals_sha256'][sid];guided=json.loads(pp.read_text())['proposals'];files.append(pp)
        instances=merge_prompts(labels_to_instances(instances_from_maps(maps,.4,.35),'EX'),maps,guided);actualiv=parse_instances(truth[sid],'invivo',iv.shape);actualex=parse_instances(truth[sid],'exvivo',ex.shape)
        u=np.array([actualiv[i].coordinates().mean(0) for i,e in truth[sid]['match_pairs']]);v=np.array([actualex[e].coordinates().mean(0) for i,e in truth[sid]['match_pairs']]);matrix=np.linalg.lstsq(np.c_[u,np.ones(len(u))],v,rcond=None)[0].T
        records.append(reverse_record(sid,iv,ex,instances,matrix,truth[sid]));covered=set(segmentation_score(parse_instances(public[sid],'invivo',iv.shape),actualiv)['pred_to_gt'].values());missing={n for n,k in enumerate(truth[sid]['invivo_instances'],1) if k not in covered};training.append(emphasize_missing(records[-1],missing));prompts[sid]=instances;images[sid]=(iv.shape,ex);files.extend([ip,ep,mp]);print('Reverse training record',a.fold,sid,len(instances),flush=True)
    dependencies=['reverse_guided_initialized.py','reverse_guided_data.py','guided_ex_data.py','guided_ex_repaired.py','cell_features.py','joint_pair_assignment.py'];config=dict(held_out=subject,train_ids=ids,steps=3000,width=24,seed=2087,initializer_sha256=sha(initializer),hard_positive_factor=4,source_sha256={n:sha(Path(__file__).with_name(n)) for n in dependencies},training_inputs_sha256={str(p):sha(p) for p in files},policy=__doc__)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');model=GuidedNet(24,initial['state_dict']).to(device);final=out/'model.pt'
    if final.exists():
        state=torch.load(final,map_location='cpu',weights_only=False);assert state['config']==config and state['step']==3000;model.load_state_dict(state['state_dict'])
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,3000,eta_min=3e-5);model.train();history=[];tick=time.perf_counter()
        for step in range(1,3001):
            x,y,exists=(v.to(device) for v in sample(training,rng));optimizer.zero_grad(set_to_none=True);logits,presence=model(x);loss=objective(logits,presence,y,exists);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                item=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.perf_counter()-tick);history.append(item);print('Reverse training',a.fold,item,flush=True)
        torch.save(dict(config=config,step=3000,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),final);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n');del optimizer,scheduler,x,y,exists,logits,presence,loss
    model.eval();calibration=[]
    for record in records:
        sid=record['sample_id'];pp=cache/f'{sid}.json'
        if not pp.exists():pp.write_text(json.dumps(infer(model,record,device))+'\n')
        ivshape,ex=images[sid];calibration.append(quality_record(sid,prompts[sid],json.loads(pp.read_text())['proposals'],ivshape,ex,truth[sid]));print('Reverse calibration',sid,len(calibration[-1]['y']),int(calibration[-1]['y'].sum()),flush=True)
    del records,training,images,prompts
    inner=[]
    for mouse in sorted({r['subject'] for r in calibration}):
        fitted=fit([r for r in calibration if r['subject']!=mouse]);inner.extend((r,probability(fitted,r['x'])) for r in calibration if r['subject']==mouse)
    measurements=[];total=sum(len(truth[s]['match_pairs']) for s in ids)
    for threshold in np.arange(.05,1.,.05):
        count=correct=0
        for r,scores in inner:
            chosen=selected(r,scores,threshold);count+=len(chosen);correct+=int(r['y'][chosen].sum())
        measurements.append(dict(threshold=float(threshold),correct=correct,predicted=count,f1=2*correct/(count+total)))
    best=max(measurements,key=lambda x:x['f1']);fitted=fit(calibration);torch.save(dict(state_dict=fitted[0].state_dict(),mean=fitted[1],sd=fitted[2],selected=best,held_out=subject,train_ids=ids,segmenter_sha256=sha(final)),out/'quality.pt');(out/'calibration.json').write_text(json.dumps(dict(selected=best,measurements=measurements),indent=2)+'\n')
    bp=Path('outputs/stage62_endpoint_quality/full/gated_joint/predictions.csv');gp=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');base=read_rows(bp);matrices={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(gp.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'};held={}
    for sid in sorted(set(truth)-set(ids)):
        if sid not in matrices:continue
        iv,ex=[read_image(image_path(root,'training',sid,m)) for m in ['invivo','exvivo']];used={e for i,e in base[sid]['match_pairs']};instances={k:v for k,v in base[sid]['exvivo_instances'].items() if k not in used}
        if not instances:continue
        record=reverse_record(sid,iv,ex,instances,matrices[sid]);pp=cache/f'{sid}.json'
        if not pp.exists():pp.write_text(json.dumps(infer(model,record,device))+'\n')
        r=quality_record(sid,instances,json.loads(pp.read_text())['proposals'],iv.shape,ex);torch.save(r,out/f'held_{sid}.pt');held[sid]=(r,probability(fitted,r['x']),iv.shape);print('Reverse held inference',sid,len(r['candidates']),flush=True)
    reports={}
    for floor in [.5,.7]:
        rows=[];changes=[];threshold=max(floor,best['threshold']);name=f'floor{int(floor*10)}'
        for sid in sorted(set(truth)-set(ids)):
            if sid in held:r,scores,shape=held[sid];row,added=recover(base[sid],r,scores,threshold,shape)
            else:row,added=copy.deepcopy(base[sid]),[]
            rows.append(row);changes.append(dict(sample_id=sid,added=added))
        folder=out/name;folder.mkdir(exist_ok=True);write_rows(rows,folder/'predictions.csv');evaluate(SimpleNamespace(data=root,predictions=folder/'predictions.csv',subject=subject,output=folder/'metrics.json'));reports[name]=dict(threshold=threshold,regions=changes)
    (out/'manifest.json').write_text(json.dumps(dict(config=config,model_sha256=sha(final),quality_sha256=sha(out/'quality.pt'),baseline_sha256=sha(bp),geometry_sha256=sha(gp),proposals_sha256={p.stem:sha(p) for p in cache.glob('*.json')},reports=reports,limitation='Quality calibration uses all development prompts; held-out recovery uses only unmatched EX prompts, so conservative fixed probability floors are applied.'),indent=2)+'\n')
if __name__=='__main__':main()
