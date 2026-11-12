"""Stage653 inference from raw TIFF pairs; no annotations or prediction archives.

Inputs: images/subject_name/region_name/{invivo,exvivo}.tif.
Learned parameters are frozen. Cache is created by this run, keyed by input hashes.
"""
import argparse,copy,gc,hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent if (HERE.parent/'src').exists() else HERE
sys.path[:0]=[str(SOURCE/'vendor/cellpose4'),str(SOURCE/'src')]
import cv2,numpy as np,torch
from cellmatch.data import read_image,parse_instances,write_rows
from cellmatch.learning import SmallUNet,probability_maps
from cellmatch.hidden_submission import raw_masks
from cellmatch.ex_joint_calibration import morphology_features
from cellmatch.stage3 import FOLDS,quality_probabilities
from cellmatch.geometry_prior import register
from cellmatch.registration_consensus import choose_matrix
from cellmatch.matching import centroids
from cellmatch.subject_registration import linear_bank,search,support
from cellmatch.joint_pair_assignment import candidate_features,probability,assign_edges
from cellmatch.relative_ex_quality import build
from cellmatch.preserve_pair_masks import merge_row
from cellmatch.cellpose_baseline import confidence_maps
from cellmatch.rle import labels_to_instances
from cellmatch.cellpose_contour_consensus import blend
from cellmatch.current_delivery import contours,candidates,load_heads
from cellmatch.iv_local_boundary import D4Average
from cellmatch.native_ex_refiner import refine_row
from cellmatch.iv_consensus_completion import complete
from cellmatch.refined_iv_completion import refine_added
from cellmatch.guided_ex_repaired import GuidedNet
from cellmatch.candidate_context import context_features
from cellmatch.guided_endpoint_quality import factor_probability


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def load(p):return json.loads(p.read_text())
def clean(device):
 gc.collect()
 if device=='mps':torch.mps.empty_cache()
def fitted(state,key):
 net=torch.nn.Linear(len(state['mean']),1);net.load_state_dict(state[key]);return net.eval(),state['mean'],state['sd']
def row(sid,instances):return dict(sample_id=sid,invivo_instances=instances,exvivo_instances={},match_pairs=[])
def native_identity(candidates,bank,shape,field):
 from cellmatch.rle import Mask
 names=list(bank);owner=np.zeros(np.prod(shape),np.int32);areas=[]
 for n,k in enumerate(names,1):
  m=Mask.parse(bank[k],shape);areas.append(m.area)
  for a,b in zip(m.starts,m.ends):
   if owner[a:b].any():raise ValueError('Overlapping segmentation')
   owner[a:b]=n
 areas=np.array(areas);result={}
 for c in candidates:
  rle=c[field]
  if rle in result:continue
  m=Mask.parse(rle,shape);pts=np.concatenate([np.arange(a,b) for a,b in zip(m.starts,m.ends)])
  inter=np.bincount(owner[pts],minlength=len(names)+1)[1:];union=m.area+areas-inter
  iou=np.divide(inter,union,out=np.zeros(len(names)),where=union>0);ix=np.flatnonzero(iou>.5)
  if len(ix)>1:raise ValueError('Ambiguous identity')
  result[rle]=names[ix[0]] if len(ix) else None
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--images',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--limit',type=int,default=0);p.add_argument('--phase',choices=['all','native','foundation','finish'],default='all');a=p.parse_args()
 torch.set_num_threads(2);cv2.setNumThreads(2);torch.manual_seed(2026)
 cfg=load(HERE/'config.json');images=a.images.resolve();w=a.weights.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 paths={f'{p.parent.parent.name}__{p.parent.name}':p.parent for p in sorted(images.glob('*/*/invivo.tif'))}
 if a.limit:paths=dict(list(paths.items())[:a.limit])
 if not paths:raise ValueError('No image pairs found')
 for folder in paths.values():
  if not (folder/'exvivo.tif').is_file():raise ValueError('Missing EX image')
 signature=dict(images={sid:{m:sha(folder/(m+'.tif')) for m in ['invivo','exvivo']} for sid,folder in paths.items()},config_sha256=sha(HERE/'config.json'),engine_sha256=sha(__file__),device=a.device)
 if (out/'inputs.json').exists() and load(out/'inputs.json')!=signature:raise ValueError('Changed input/configuration; use a new output directory')
 save(out/'inputs.json',signature)
 def state(name):return torch.load(w/name,map_location='cpu',weights_only=False)
 def image(sid,m):return read_image(paths[sid]/(m+'.tif'))
 def maps(sid,m,fold=None):return np.load(out/'native_maps'/(fold or 'ensemble')/f'{sid}_{m}.npz')['maps'].astype(np.float32)
 qualities={};pair=[]
 for tag in FOLDS:
  pair.append(fitted(state(f'outputs/stage8_joint_iv_quality/{tag}.pt'),'state_dict'))
  for m,loc in [('invivo',f'outputs/stage8_iv_quality_capacity/{tag}.pt'),('exvivo',f'outputs/stage6_specialist_quality/{tag}/model.pt')]:qualities[m,tag]=state(loc)
 if a.phase in ['all','native']:
  for m in ['invivo','exvivo']:
   for tag in FOLDS:
    name=f'outputs/stage8_iv/{tag}/model.pt' if m=='invivo' else f'outputs/stage6_ex/unet_holdout_{tag}/model.pt'
    st=state(name);net=SmallUNet(st['width']).to(a.device);net.load_state_dict(st['state_dict']);net.eval()
    folder=out/'native_maps'/tag;folder.mkdir(parents=True,exist_ok=True)
    for sid in paths:
     fp=folder/f'{sid}_{m}.npz'
     if not fp.exists():
      t=time.monotonic();v=probability_maps(net,image(sid,m),int(m=='exvivo'),a.device,tile=128,overlap=64,batch_size=8);np.savez_compressed(fp,maps=v.astype(np.float16));print('native maps',m,tag,sid,round(time.monotonic()-t,2),flush=True)
    del net,st;clean(a.device)
   folder=out/'native_maps/ensemble';folder.mkdir(exist_ok=True)
   for sid in paths:np.savez_compressed(folder/f'{sid}_{m}.npz',maps=np.mean([maps(sid,m,t) for t in FOLDS],axis=0).astype(np.float16))
  allrows={};geometry=[]
  for sid in paths:
   rp=out/'baseline8'/f'{sid}.json';gp=out/'initial_geometry'/f'{sid}.json'
   if rp.exists() and gp.exists():allrows[sid]=load(rp);geometry.append(load(gp));continue
   encoded={};raw={};filtered={}
   for m in ['invivo','exvivo']:
    s=cfg['native'][m];encoded[m],raw[m]=raw_masks(maps(sid,m),s['foreground'],s['seed'],m)
    keys,x=morphology_features(raw[m],maps(sid,m));pr=np.mean([quality_probabilities(fitted(qualities[m,t],'quality'),x) for t in FOLDS],axis=0)
    filtered[m]={k:raw[m][k] for k,v in zip(keys,pr) if v>=s['quality']}
   hypotheses=[('ensemble_filtered',filtered['invivo'],filtered['exvivo']),('ensemble_raw',raw['invivo'],raw['exvivo']),('ensemble_iv_filtered',filtered['invivo'],raw['exvivo'])]
   for tag,subject in FOLDS.items():
    mm=[]
    for m in ['invivo','exvivo']:
     q=qualities[m,tag];fg=q.get('foreground',.5) if m=='invivo' else cfg['ex_fold_foreground'][subject]
     mp=maps(sid,m,tag);_,pa=raw_masks(mp,fg,q.get('seed_threshold',.35),m);ks,xx=morphology_features(pa,mp);pr=quality_probabilities(fitted(q,'quality'),xx);mm.append({k:pa[k] for k,v in zip(ks,pr) if v>=q['threshold']})
    hypotheses.append((f'fold_{tag}',*mm))
   options=[]
   for name,iv,ex in hypotheses:
    cv2.setRNGSeed(2026);mat,diag=register(iv,ex,cfg['prior'])
    if mat is not None:options.append((name,mat))
   pts=[centroids(filtered[m])[1] for m in ['invivo','exvivo']]
   mat,diag=choose_matrix(*pts,options) if all(len(v) for v in pts) else (None,dict(status='insufficient_cells'))
   ms=[filtered[m] for m in ['invivo','exvivo']];ft=[morphology_features(filtered[m],maps(sid,m))[1] for m in ['invivo','exvivo']]
   pp,xx=candidate_features(ms,ft,mat);pr=np.mean([probability(f,xx) for f in pair],axis=0);pairs=assign_edges(pp,pr,cfg['native']['pairs']['threshold'],xx[:,18],cfg['native']['pairs']['radius'])
   r=dict(sample_id=sid,invivo_instances={k:v for k,v in encoded['invivo'].items() if k in filtered['invivo']},exvivo_instances={k:v for k,v in encoded['exvivo'].items() if k in filtered['exvivo']},match_pairs=pairs)
   diag.update(sample_id=sid);save(rp,r);save(gp,diag);allrows[sid]=r;geometry.append(diag);print('initial geometry',sid,diag['status'],flush=True)
  for rec in geometry:
   sid=rec['sample_id'];r=allrows[sid];new=copy.deepcopy(rec);bank=linear_bank(geometry,sid)
   if rec.get('votes',0)<4 and bank:
    mm=[parse_instances(r,m,image(sid,m).shape) for m in ['invivo','exvivo']];pts=[centroids(m)[1] for m in mm];cv2.setRNGSeed(2026);opts=search(*pts,image(sid,'exvivo').shape,bank);old=support(*pts,np.asarray(rec['matrix']))[0] if rec.get('matrix') else 0.
    if opts and opts[0]['support']>max(7.,1.2*old) and opts[0]['inliers']>=10:new.update(opts[0]);new['status']='registered'
   save(out/'geometry'/f'{sid}.json',new)
  qs=[state(f'outputs/stage13_native_relative/{tag}.pt') for tag in FOLDS]
  for sid,r in allrows.items():
   gg=load(out/'geometry'/f'{sid}.json');mat=np.array(gg['matrix']) if gg.get('matrix') is not None else None
   ms=[parse_instances(r,m,image(sid,m).shape) for m in ['invivo','exvivo']];ft=[morphology_features(mm,maps(sid,m))[1] for mm,m in zip(ms,['invivo','exvivo'])];pp,xx=candidate_features(ms,ft,mat);pr=np.mean([probability(f,xx) for f in pair],axis=0)
   paired=dict(r,match_pairs=assign_edges(pp,pr,cfg['relative']['pairs']['threshold'],xx[:,18],cfg['relative']['pairs']['radius']))
   enc,_,ks,xx=build(maps(sid,'exvivo'),cfg['relative']['foreground']);pr=np.mean([quality_probabilities(fitted(q,'quality'),xx) for q in qs],axis=0)
   seg=dict(r,match_pairs=[],exvivo_instances={k:enc[k] for k,v in zip(ks,pr) if v>=cfg['relative']['quality']});save(out/'baseline13'/f'{sid}.json',merge_row(paired,seg))
 if a.phase in ['all','foundation']:
  from cellpose import models,dynamics
  from cellmatch.dino_lora import install,load_adapter
  for kind in ['public','adapted','large','lora']:
   ck=w/'models/cellpose4'/('cpdino-vitb' if kind in ['public','adapted'] else 'cpdino')
   # Match original public CPU inference; other networks originally used MPS.
   dev='cpu' if kind=='public' else a.device
   model=models.CellposeModel(pretrained_model=str(ck),device=torch.device(dev),use_bfloat16=False)
   if kind=='adapted':model.net.load_state_dict(state('outputs/stage118_delivery/adapted/all-training/model.pt')['state_dict'])
   if kind=='lora':install(model.net,8,8);load_adapter(model.net,state('outputs/stage100_lora_iv/all-training/adapter.pt')['state_dict'])
   model.net.eval()
   for sid in paths:
    fp=out/kind/f'{sid}.json'
    if fp.exists():continue
    t=time.monotonic();kwargs=dict(batch_size=2 if kind=='adapted' else 1,diameter=None,min_size=12 if kind in ['adapted','lora'] else 15,flow_threshold=.4,cellprob_threshold=0.)
    if kind in ['adapted','lora']:kwargs['bsize']=128
    labels,flow,_=model.eval(image(sid,'invivo'),**kwargs)
    prefix={'public':'CP4','adapted':'DINO','large':'CP4','lora':'LORA'}[kind]
    save(fp,row(sid,labels_to_instances(labels.astype(np.int32),prefix)))
    np.savez_compressed(fp.with_suffix('.npz'),maps=confidence_maps(labels,np.asarray(flow[2])).astype(np.float16))
    if kind=='public':
     low=dynamics.resize_and_compute_masks(np.asarray(flow[1],np.float32),np.asarray(flow[2],np.float32),niter=200,cellprob_threshold=-1.,flow_threshold=.4,min_size=15,device=torch.device('cpu'))
     dest=out/'public_low'/f'{sid}.json';save(dest,row(sid,labels_to_instances(low,'CPLOW')));np.savez_compressed(dest.with_suffix('.npz'),maps=confidence_maps(low,np.asarray(flow[2])).astype(np.float16))
    print('foundation',kind,sid,round(time.monotonic()-t,2),flush=True)
   del model;clean(a.device)
 if a.phase in ['all','finish']:
  refiners=[]
  for loc in ['outputs/stage118_delivery/refiner/all-training/model.pt','outputs/stage233_native_ex_delivery/model.pt','outputs/stage247_positive_iv_delivery/model.pt']:
   st=state(loc);net=SmallUNet(16).to(a.device);net.load_state_dict(st['state_dict']);refiners.append(D4Average(net.eval()).eval())
  for sid in paths:
   iv=image(sid,'invivo');ex=image(sid,'exvivo');base=load(out/'baseline13'/f'{sid}.json');public=load(out/'public'/f'{sid}.json');lora=load(out/'lora'/f'{sid}.json')
   r,_=blend(base,public,iv.shape);save(out/'stage32'/f'{sid}.json',r)
   r=contours(r,lora,iv,refiners[0],a.device);r,_=refine_row(r,base,ex,refiners[1],a.device)
   r,sel=complete(r,load(out/'public_low'/f'{sid}.json'),lora,iv.shape);r,_,_,_=refine_added(r,[x['key'] for x in sel['added']],iv,refiners[2],a.device);save(out/'native_masks'/f'{sid}.json',r)
  del refiners,net;clean(a.device)
  st=state('outputs/stage118_delivery/guide/all-training/model.pt');guide=GuidedNet(24).to(a.device);guide.load_state_dict(st['state_dict']);guide.eval();heads=[load_heads(state(loc)) for loc in ['outputs/stage187_delivery/quality.pt','outputs/stage645_unseen_joint_delivery/quality.pt']];rows=[]
  for sid in paths:
   native=load(out/'native_masks'/f'{sid}.json');gg=load(out/'geometry'/f'{sid}.json');mat=gg.get('matrix')
   if mat is None:rows.append(dict(native,match_pairs=[]));continue
   banks=[load(out/k/f'{sid}.json') for k in ['public_low','adapted','large']];mp=[np.load(out/k/f'{sid}.npz')['maps'].astype(np.float32) for k in ['public_low','adapted','large']]
   rec=candidates(images.parent,images.name,sid,banks,mp,np.asarray(mat),guide,a.device,None)
   shapes=[image(sid,m).shape for m in ['invivo','exvivo']];xx=np.c_[rec['x'],context_features(rec,mat,shapes)];ss=[factor_probability(h,xx,'gated_joint') for h in heads];score=.5*(ss[0].astype(np.float64)+ss[1].astype(np.float64))
   mm=[native_identity(rec['candidates'],native[m+'_instances'],sh,f) for m,sh,f in zip(['invivo','exvivo'],shapes,['iv_rle','ex_rle'])];used=[set(),set()];pairs=[]
   for n in sorted(range(len(score)),key=lambda i:(-score[i],i)):
    c=rec['candidates'][n];i,e=mm[0][c['iv_rle']],mm[1][c['ex_rle']]
    if i is None or e is None or score[n]<=.25 or i in used[0] or e in used[1]:continue
    pairs.append([i,e]);used[0].add(i);used[1].add(e)
   final=dict(native,match_pairs=pairs);rows.append(final);save(out/'final'/f'{sid}.json',final);torch.save(rec,out/'final'/f'{sid}.pt');print('final',sid,len(pairs),flush=True)
  write_rows(rows,out/'submission.csv');save(out/'completion.json',dict(rows=len(rows),csv_sha256=sha(out/'submission.csv'),source_sha256=sha(__file__),labels_read=False))

if __name__=='__main__':main()
