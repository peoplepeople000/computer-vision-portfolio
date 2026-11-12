"""Stage83: image-only adapted IV decoding at fixed cellprob 0/-1.
Use the original MPS device for decoder equivalence. Generate all held-out regions; optional pilot uses first sorted region per mouse.
Default decode must exactly reproduce the existing adapted mask set.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,torch
from cellpose import dynamics
from .data import read_rows,write_rows,split_id
from .rle import labels_to_instances
from .stage3 import FOLDS
from .cellpose_baseline import confidence_maps

def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 out=Path('outputs/stage83_adapted_redecode_mps');out.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 base=read_rows('outputs/stage32_contour_consensus/predicted_all.csv');ids=sorted(base);tags={s:t for t,s in FOLDS.items()}
 if a.pilot:ids=[next(s for s in ids if split_id(s)[0]==m) for m in sorted(tags)]
 configs={t:json.loads(Path(f'outputs/stage35_cpdino_iv_adapted/{t}/configuration.json').read_text()) for t in FOLDS}
 for tag,subject in FOLDS.items():assert configs[tag]['held_out']==subject and configs[tag]['train_ids']==sorted(s for s in base if split_id(s)[0]!=subject)
 config=dict(source_sha256=sha(__file__),dynamics_sha256=sha(dynamics.__file__),thresholds=[0.,-1.],niter=200,flow_threshold=.4,min_size=12,device='mps',sample_ids=ids,policy=__doc__,models={t:sha(Path(f'outputs/stage35_cpdino_iv_adapted/{t}/model.pt')) for t in FOLDS})
 cp=out/('pilot_configuration.json' if a.pilot else 'configuration.json')
 if cp.exists():assert json.loads(cp.read_text())==config
 cp.write_text(json.dumps(config,indent=2)+'\n');reports=[];results={k:[] for k in ['zero','minus1']}
 for sid in ids:
  tag=tags[split_id(sid)[0]];source=Path('outputs/stage35_cpdino_iv_adapted')/tag;fp=source/'flows'/f'{sid}.npz';original=source/'regions'/f'{sid}.json';flow=np.load(fp);reference=set(labels_to_instances(flow['labels'],'x').values());assert reference==set(json.loads(original.read_text())['invivo_instances'].values())
  report=dict(sample_id=sid,fold=tag,flow_sha256=sha(fp),original_sha256=sha(original),original=len(reference),counts={})
  for key,threshold in [('zero',0.),('minus1',-1.)]:
   dest=out/key;(dest/'regions').mkdir(parents=True,exist_ok=True);(dest/'maps').mkdir(exist_ok=True);rp=dest/'regions'/f'{sid}.json';meta=rp.with_suffix('.meta.json');identity=dict(flow_sha256=sha(fp),source_sha256=sha(__file__),dynamics_sha256=sha(dynamics.__file__),threshold=threshold)
   if rp.exists():
    saved=json.loads(meta.read_text());assert saved['identity']==identity and saved['sha256']==sha(rp);row=json.loads(rp.read_text())
   else:
    decoded=dynamics.resize_and_compute_masks(flow['dP'],flow['cellprob'],niter=200,cellprob_threshold=threshold,flow_threshold=.4,min_size=12,device=torch.device('mps'))
    row=copy.deepcopy(base[sid]);row['invivo_instances']=labels_to_instances(decoded,'ADAPT_LOW');row['match_pairs']=[]
    np.savez_compressed(dest/'maps'/f'{sid}_invivo.npz',maps=confidence_maps(decoded,flow['cellprob']).astype(np.float16));rp.write_text(json.dumps(row)+'\n');meta.write_text(json.dumps(dict(identity=identity,sha256=sha(rp)),indent=2)+'\n')
   if key=='zero':assert set(row['invivo_instances'].values())==reference,'Default decoder mismatch: '+sid
   results[key].append(row);report['counts'][key]=len(row['invivo_instances']);print('Decoded',sid,key,report['counts'][key],flush=True)
  reports.append(report)
 for key,rows in results.items():write_rows(rows,out/key/('pilot_predictions.csv' if a.pilot else 'predictions.csv'))
 (out/('pilot_manifest.json' if a.pilot else 'manifest.json')).write_text(json.dumps(dict(configuration_sha256=sha(cp),regions=reports),indent=2)+'\n')
if __name__=='__main__':main()
