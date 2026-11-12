"""Outer-fold-safe development registration from predicted IV/EX masks.
The affine prior uses only outer-development labels. Per-image registration sees
predicted centroids only. This simpler search differs from Stage9 consensus.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
from .data import read_rows,parse_instances,split_id,image_shape,image_path
from .stage3 import FOLDS,quality_probabilities
from .learning import instances_from_maps
from .rle import labels_to_instances
from .ex_joint_calibration import morphology_features
from .geometry_prior import fit_prior,register
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True);a=p.parse_args()
 torch.set_num_threads(1);cv2.setNumThreads(1);cv2.setRNGSeed(2026)
 root=Path('data/raw/Project_2_Dataset');alltruth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold]
 truth={s:r for s,r in alltruth.items() if split_id(s)[0]!=subject};ids=sorted(truth)
 out=Path('outputs/stage95_geometry_parity')/a.fold;out.mkdir(parents=True,exist_ok=True)
 exdir=Path('outputs/stage11_native_ex')/a.fold;ivdir=Path('outputs/stage35_cpdino_iv_adapted')/a.fold
 paths=[exdir/'training_config.json',exdir/'quality.pt',exdir/'model.pt',ivdir/'configuration.json',ivdir/'model.pt']
 exconfig=json.loads(paths[0].read_text());ivconfig=json.loads(paths[3].read_text())
 for config in [exconfig,ivconfig]:assert config['held_out']==subject and sorted(config['train_ids'])==ids
 assert exconfig['checkpoint_sha256']==sha(exdir/'model.pt')
 for mp in [exdir/'model.pt',ivdir/'model.pt']:
  state=torch.load(mp,map_location='cpu',weights_only=False);cfg=state.get('config',state)
  assert cfg['held_out']==subject and sorted(cfg['train_ids'])==ids
 state=torch.load(exdir/'quality.pt',map_location='cpu',weights_only=False);assert state['held_out']==subject and sorted(state['train_ids'])==ids
 head=torch.nn.Linear(len(state['mean']),1);head.load_state_dict(state['quality']);fitted=head.eval(),state['mean'],state['sd']
 prior=fit_prior(root,truth,subject);assert set(prior['train_ids'])<=set(ids)
 source={n:sha(Path(__file__).with_name(n)) for n in ['development_image_geometry.py','geometry_prior.py','learning.py','ex_joint_calibration.py','stage3.py']}
 inputs={str(p):sha(p) for p in paths};regions=[]
 for sid in ids:
  ep=exdir/'maps'/f'{sid}_exvivo.npz';ip=ivdir/'regions'/f'{sid}.json';inputs.update({str(p):sha(p) for p in [ep,ip]})
  maps=np.load(ep)['maps'].astype(np.float32);labels=instances_from_maps(maps,state['foreground'],state['seed_threshold'])
  ex=parse_instances({'exvivo_instances':labels_to_instances(labels,'EX')},'exvivo',maps.shape[1:]);keys,features=morphology_features(ex,maps)
  keep={k for k,v in zip(keys,quality_probabilities(fitted,features)) if v>=state['threshold']};ex={k:v for k,v in ex.items() if k in keep}
  iv=parse_instances(json.loads(ip.read_text()),'invivo',image_shape(image_path(root,'training',sid,'invivo')))
  matrix,diag=register(iv,ex,prior)
  regions.append(dict(sample_id=sid,matrix=None if matrix is None else matrix.tolist(),diagnostic=diag,iv_count=len(iv),ex_count=len(ex)))
  print('Development image registration',a.fold,sid,diag['status'],flush=True)
 report=dict(policy=__doc__,held_out=subject,train_ids=ids,prior=prior,source_sha256=source,input_sha256=inputs,regions=regions)
 (out/'development_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Registered',sum(r['matrix'] is not None for r in regions),'of',len(regions),flush=True)
if __name__=='__main__':main()
