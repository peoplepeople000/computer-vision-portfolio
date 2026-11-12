"""Stage92: fixed local IV readout adaptation on frozen public ViT-L features.
Train only the196800-parameter readout for1200 updates. Sample valid native-view
tokens uniformly within images and balance development mice. No augmentation,
backbone fitting, held-out checkpoint selection or per-mouse recipe choice.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from cellpose import dynamics
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .cellpose_finetune import instance_labels
from .cellpose_baseline import confidence_maps
from .frozen_dino_readout import decode_tokens,pack_pixels,token_loss,tile_pixels,reassemble
from .rle import labels_to_instances
from .metrics import segmentation_score
from .stage3 import FOLDS

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare_training(root,truth,ids,cache,target_cache,completion):
    target_cache.mkdir(parents=True,exist_ok=True);records=[]
    for sid in ids:
        meta=json.loads((cache/f'{sid}.json').read_text());feature=cache/f'{sid}.npy'
        assert completion['records'][sid]==meta['feature_sha256']==sha(feature)
        layout=meta['identity']['layout'];image_file=image_path(root,'training',sid,'invivo')
        assert meta['identity']['image_sha256']==sha(image_file)
        labels=instance_labels(truth[sid]['invivo_instances'],tuple(layout['shape']))
        checksum=hashlib.sha256(labels.tobytes()).hexdigest();path=target_cache/f'{sid}_{checksum[:16]}.npz'
        if not path.exists():
            target=dynamics.labels_to_flows([labels.copy()],device=torch.device('cpu'))[0][1:].astype(np.float32)
            np.savez_compressed(path,target=target,checksum=checksum)
        with np.load(path) as saved:
            assert saved['checksum'].item()==checksum;target=saved['target'].copy()
        y=pack_pixels(torch.from_numpy(tile_pixels(target,layout))).reshape(-1,192).numpy()
        valid=pack_pixels(torch.from_numpy(tile_pixels(np.ones((1,*labels.shape),np.float32),layout))).reshape(-1,64).numpy()
        indices=np.flatnonzero(valid.sum(1)>0)
        x=np.load(feature,mmap_mode='r').reshape(-1,1024)
        records.append(dict(sample_id=sid,subject=split_id(sid)[0],x=x,y=y,valid=valid,indices=indices,target_sha256=sha(path)))
        print('Prepared readout training',sid,len(indices),flush=True)
    assert sorted(r['sample_id'] for r in records)==ids
    return records


def sample_tokens(records,rng):
    subjects=sorted({r['subject'] for r in records});xs=[];ys=[];vs=[]
    for subject in subjects:
        group=[r for r in records if r['subject']==subject]
        for _ in range(2):
            record=group[int(rng.integers(len(group)))];indices=rng.choice(record['indices'],64)
            xs.append(np.asarray(record['x'][indices]));ys.append(record['y'][indices]);vs.append(record['valid'][indices])
    return [torch.from_numpy(np.concatenate(v)) for v in [xs,ys,vs]]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fold',choices=list(FOLDS),required=True);a=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2092);rng=np.random.default_rng(2092);device=torch.device('mps');assert torch.backends.mps.is_available()
    root=Path('data/raw/Project_2_Dataset');truth_path=root/'training/train_ground_truth.csv';truth=read_rows(truth_path);held=FOLDS[a.fold]
    ids=sorted(s for s in truth if split_id(s)[0]!=held);test_ids=sorted(set(truth)-set(ids));cache=Path('outputs/stage91_frozen_features')
    completion=json.loads((cache/'completion.json').read_text());assert completion['status']=='complete' and completion['regions']==47 and completion['configuration_sha256']==sha(cache/'configuration.json') and completion['readout_sha256']==sha(cache/'public_readout.pt')
    cache_config=json.loads((cache/'configuration.json').read_text());assert cache_config['annotations_read'] is False
    cache_audit=json.loads((cache/'audit.json').read_text());assert cache_audit['status']=='passed' and cache_audit['completion_sha256']==sha(cache/'completion.json')
    for path,digest in cache_config['source_sha256'].items():assert sha(path)==digest
    out=Path('outputs/stage92_readout_adaptation')/a.fold;out.mkdir(parents=True,exist_ok=True)
    config=dict(policy=__doc__,held_out=held,train_ids=ids,test_ids=test_ids,steps=1200,seed=2092,batch_tokens=256,lr=1e-4,minimum_lr=1e-5,weight_decay=.1,cache_configuration_sha256=sha(cache/'configuration.json'),cache_completion_sha256=sha(cache/'completion.json'),public_readout_sha256=sha(cache/'public_readout.pt'),annotation_sha256=sha(truth_path),source_sha256={str(p):sha(p) for p in [Path(__file__),Path('src/cellmatch/frozen_dino_readout.py'),Path('vendor/cellpose4/cellpose/dynamics.py')]},decoder=dict(cellprob_threshold=0.,flow_threshold=.4,min_size=15,niter=200))
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n')
    head=torch.nn.Linear(1024,192).to(device);head.load_state_dict(torch.load(cache/'public_readout.pt',weights_only=True,map_location=device));final=out/'readout.pt'
    if final.exists():
        saved=torch.load(final,map_location='cpu',weights_only=False);assert saved['config']==config and saved['step']==1200;head.load_state_dict(saved['state_dict'])
    else:
        records=prepare_training(root,truth,ids,cache,Path('outputs/stage92_readout_adaptation/flow_targets'),completion)
        assert all(r['subject']!=held for r in records)
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.1);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1200,eta_min=1e-5)
        history=[];started=time.perf_counter()
        for step in range(1,1201):
            x,y,valid=[v.to(device) for v in sample_tokens(records,rng)]
            optimizer.zero_grad(set_to_none=True);loss=token_loss(head(x),y,valid);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                item=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.perf_counter()-started);history.append(item);print('Readout training',a.fold,item,flush=True)
        torch.save(dict(config=config,step=1200,state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},target_sha256={r['sample_id']:r['target_sha256'] for r in records}),final)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n');del records,optimizer,scheduler,x,y,valid,loss
    # Seal the fixed fitted model before computing any held-out evaluation metrics.
    sealed=sha(final);head.eval()
    reference=read_rows('outputs/stage89_large_iv_guided/full/gated_joint/predictions.csv')
    public=read_rows('outputs/stage74_iv_redecode/minus1/predictions.csv');large=read_rows('outputs/stage88_large_iv_full/predictions.csv')
    for name in ['regions','maps','flows']:(out/name).mkdir(exist_ok=True)
    rows=[];reports=[]
    for sid in test_ids:
        assert split_id(sid)[0]==held and sid not in ids
        meta=json.loads((cache/f'{sid}.json').read_text());fp=cache/f'{sid}.npy';assert sha(fp)==completion['records'][sid]==meta['feature_sha256'];layout=meta['identity']['layout']
        features=np.load(fp,mmap_mode='r');values=[];started=time.perf_counter()
        with torch.inference_mode():
            for tile in features:
                tensor=torch.from_numpy(np.array(tile,copy=True))[None].to(device)
                values.append(decode_tokens(head(tensor),384,384).cpu().numpy()[0])
        full=reassemble(np.stack(values),layout)
        labels=dynamics.resize_and_compute_masks(full[:2],full[2],niter=200,cellprob_threshold=0.,flow_threshold=.4,min_size=15,device=device)
        row=copy.deepcopy(reference[sid]);row['invivo_instances']=labels_to_instances(labels.astype(np.int32),'READOUT');row['match_pairs']=[]
        (out/'regions'/f'{sid}.json').write_text(json.dumps(row)+'\n');rows.append(row)
        np.savez_compressed(out/'flows'/f'{sid}_invivo.npz',dP=full[:2],cellprob=full[2],labels=labels)
        np.savez_compressed(out/'maps'/f'{sid}_invivo.npz',maps=confidence_maps(labels,full[2]).astype(np.float16))
        actual=parse_instances(truth[sid],'invivo',tuple(layout['shape']));verified={i for i,e in truth[sid]['match_pairs']}
        adapted=json.loads((Path('outputs/stage35_cpdino_iv_adapted')/a.fold/'regions'/f'{sid}.json').read_text());banks=dict(readout=row,public=public[sid],large=large[sid],adapted=adapted,best=reference[sid]);scores={};covered={}
        for name,bank in banks.items():
            metric=segmentation_score(parse_instances(bank,'invivo',tuple(layout['shape'])),actual);covered[name]=set(metric.pop('pred_to_gt').values());scores[name]=metric
        old=covered['public']|covered['adapted']|covered['large']
        reports.append(dict(sample_id=sid,metrics=scores,new_verified_vs_union=len((covered['readout']-old)&verified),lost_verified_vs_union=len((old-covered['readout'])&verified),readout_verified=len(covered['readout']&verified),seconds=time.perf_counter()-started))
        print('Readout evaluation',sid,round(scores['readout']['pq'],6),flush=True)
    assert sha(final)==sealed
    write_rows(rows,out/'predictions.csv')
    summary=dict(regions=len(rows),mean_pq={name:float(np.mean([r['metrics'][name]['pq'] for r in reports])) for name in banks},new_verified_vs_union=sum(r['new_verified_vs_union'] for r in reports),lost_verified_vs_union=sum(r['lost_verified_vs_union'] for r in reports),readout_verified=sum(r['readout_verified'] for r in reports))
    summary['expand_full_folds']=summary['mean_pq']['readout']>=summary['mean_pq']['adapted']+.005 or (summary['new_verified_vs_union']>=8 and summary['mean_pq']['readout']>=summary['mean_pq']['large'])
    (out/'evaluation.json').write_text(json.dumps(dict(summary=summary,regions=reports,policy='Fixed1200-step head, sealed before held-out scoring. Expansion gate predeclared in protocol.json; endpoint evaluation only, no matching-score claim.'),indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(status='complete',held_out=held,train_ids=ids,readout_sha256=sealed,csv_sha256=sha(out/'predictions.csv'),configuration_sha256=sha(cp)),indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
