"""Stage 74: fixed public IV flow redecoding at cellprob 0, -1, -2.
Image-only candidate generation; no annotations or ground-truth-derived IDs read.
The zero-threshold branch verifies cached decoder equivalence before new masks.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,torch
from cellpose import dynamics
from .data import read_rows,write_rows,split_id
from .rle import labels_to_instances
from .cellpose_baseline import confidence_maps

def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 source=Path('outputs/stage28_cpdino_iv_full');out=Path('outputs/stage74_iv_redecode');out.mkdir(exist_ok=True)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=read_rows(source/'predictions.csv');ids=sorted(rows)
 if a.pilot:ids=[next(s for s in ids if split_id(s)[0]==m) for m in sorted({split_id(s)[0] for s in ids})]
 config=dict(source_sha256=sha(__file__),dynamics_sha256=sha(dynamics.__file__),thresholds=[0.,-1.,-2.],niter=200,flow_threshold=.4,min_size=15,device='cpu',prediction_sha256=sha(source/'predictions.csv'),sample_ids=ids,policy=__doc__)
 cp=out/('pilot_configuration.json' if a.pilot else 'configuration.json')
 if cp.exists():assert json.loads(cp.read_text())==config
 cp.write_text(json.dumps(config,indent=2)+'\n');reports=[];results={key:[] for key in ['zero','minus1','minus2']}
 for sid in ids:
  fp=source/'flows'/f'{sid}_invivo.npz';flow=np.load(fp);labels=flow['labels'];reference=set(labels_to_instances(labels,'x').values());report=dict(sample_id=sid,flow_sha256=sha(fp),original=len(reference),counts={})
  for key,threshold in [('zero',0.),('minus1',-1.),('minus2',-2.)]:
   dest=out/key;dest.mkdir(exist_ok=True);(dest/'regions').mkdir(exist_ok=True);(dest/'maps').mkdir(exist_ok=True);rp=dest/'regions'/f'{sid}.json'
   if rp.exists():row=json.loads(rp.read_text())
   else:
    decoded=dynamics.resize_and_compute_masks(flow['dP'],flow['cellprob'],niter=200,cellprob_threshold=threshold,flow_threshold=.4,min_size=15,device=torch.device('cpu'))
    masks=labels_to_instances(decoded,'CPLOW');row=copy.deepcopy(rows[sid]);row['invivo_instances']=masks;row['match_pairs']=[]
    np.savez_compressed(dest/'maps'/f'{sid}_invivo.npz',maps=confidence_maps(decoded,flow['cellprob']).astype(np.float16));rp.write_text(json.dumps(row)+'\n')
   if key=='zero':assert set(row['invivo_instances'].values())==reference,'Default decoder differs from cached masks: '+sid
   results[key].append(row);report['counts'][key]=len(row['invivo_instances']);print('Decoded',sid,key,report['counts'][key],flush=True)
  reports.append(report)
 for key,part in results.items():write_rows(part,out/key/('pilot_predictions.csv' if a.pilot else 'predictions.csv'))
 (out/('pilot_manifest.json' if a.pilot else 'manifest.json')).write_text(json.dumps(dict(configuration_sha256=sha(cp),regions=reports),indent=2)+'\n')
if __name__=='__main__':main()
