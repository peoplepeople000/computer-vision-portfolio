"""Second-pass Cellpose inference using diameters from its first-pass masks.

No annotations enter scale estimation. Clamp the predicted median diameter to
7.5--60 pixels to bound resizing, and use native scale when no cells are found.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from .data import read_rows,write_rows,split_id,read_image,image_path
from .stage3 import FOLDS
from .rle import labels_to_instances
from .cellpose_baseline import confidence_maps


def estimate_diameter(instances):
    areas=[sum(map(int,rle.split()[1::2])) for rle in instances.values()]
    if not areas:return 30.
    if min(areas)<=0:raise ValueError("Invalid predicted cell areas")
    return float(np.clip(np.median(2*np.sqrt(np.array(areas)/np.pi)),7.5,60.))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['cpdino-vitb','cpsam_v2'],required=True)
    p.add_argument('--native-root',type=Path,required=True);p.add_argument('--split',choices=['training','hidden_test'],default='training');p.add_argument('--output',type=Path,required=True);p.add_argument('--modality',choices=['invivo','exvivo','both'],default='invivo');p.add_argument('--limit-per-subject',type=int,default=1);p.add_argument('--device',choices=['cpu','mps'],default='mps');a=p.parse_args()
    from cellpose import models
    if version('cellpose')!='4.2.1':raise ValueError('Use pinned isolated Cellpose 4 runtime')
    torch.set_num_threads(2);torch.manual_seed(2068);device=torch.device(a.device)
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    root=Path('data/raw/Project_2_Dataset')
    ref_path=Path('outputs/stage13_native_relative/preserved/predicted_all.csv' if a.split=='training' else 'outputs/submission_stage8/submission.csv');reference=read_rows(ref_path)
    native_path=a.native_root/'predictions.csv';native=read_rows(native_path)
    native_config=json.loads((a.native_root/'configuration.json').read_text())
    if native_config['model']!=a.model or native_config['modality']!=a.modality or native_config.get('split','training')!=a.split:
        raise ValueError('First-pass model, modality, or split differs')
    modalities=['invivo','exvivo'] if a.modality=='both' else [a.modality]
    selected=[]
    for subject in sorted({split_id(s)[0] for s in reference}):
        ids=sorted(s for s in reference if split_id(s)[0]==subject)
        selected.extend(ids[:a.limit_per_subject] if a.limit_per_subject else ids)
    if not set(selected)<=set(native):raise ValueError('Incomplete native first-pass coverage')
    diameter={sid:{m:estimate_diameter(native[sid][m+'_instances']) for m in modalities} for sid in selected}
    checkpoint=Path('models/cellpose4')/a.model;digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    known=next(r for r in json.loads(Path('models/cellpose4/provenance.json').read_text()) if r['model']==a.model)
    if digest(checkpoint)!=known['sha256']:raise ValueError('Public checkpoint changed')
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'source_sha256':digest(__file__),'reference_sha256':digest(ref_path),
        'checkpoint_sha256':digest(checkpoint),'cellpose_version':version('cellpose'),'torch_version':torch.__version__,'scales':diameter,'native_csv_sha256':digest(native_path),'native_configuration_sha256':digest(a.native_root/'configuration.json'),'training':'Official public checkpoint, no competition-data fitting',
        'policy':'Per-image median area-equivalent cell diameter from native first-pass predicted masks, clamped to 7.5--60 pixels. No annotations. Fixed official pretrained weights, default normalization, flow 0.4, cell probability 0, minimum 15 original pixels, float32 batch one.',
        'batch_size':1,'use_bfloat16':False,'fine_tuning':False}
    out=a.output;out.mkdir(parents=True,exist_ok=True);config_path=out/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Output configuration changed')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    for name in ['regions','maps','flows']:(out/name).mkdir(exist_ok=True)
    model=models.CellposeModel(pretrained_model=str(checkpoint),device=device,use_bfloat16=False);rows=[]
    for sid in selected:
        saved=out/'regions'/f'{sid}.json'
        if saved.exists():rows.append(json.loads(saved.read_text()));continue
        row=copy.deepcopy(reference[sid]);row['match_pairs']=[]
        for modality in modalities:
            started=time.perf_counter();image=read_image(image_path(root,a.split,sid,modality));diam=diameter[sid][modality]
            labels,flows,_=model.eval(image,batch_size=1,diameter=diam,min_size=15,flow_threshold=.4,cellprob_threshold=0.,normalize=True)
            if labels.shape!=image.shape or np.asarray(flows[2]).shape!=image.shape:raise ValueError('Wrong full-resolution output')
            row[modality+'_instances']=labels_to_instances(labels.astype(np.int32),'CP4S')
            np.savez_compressed(out/'maps'/f'{sid}_{modality}.npz',maps=confidence_maps(labels,np.asarray(flows[2])).astype(np.float16))
            np.savez_compressed(out/'flows'/f'{sid}_{modality}.npz',dP=np.asarray(flows[1],dtype=np.float32),cellprob=np.asarray(flows[2],dtype=np.float32),labels=labels.astype(np.int32))
            print('Auto-scaled Cellpose',a.model,sid,modality,'diameter',diam,'cells',len(row[modality+'_instances']),'seconds',round(time.perf_counter()-started,2),flush=True)
            if a.device=='mps':torch.mps.empty_cache()
        saved.write_text(json.dumps(row)+'\n');rows.append(row)
    write_rows(rows,out/'predictions.csv');(out/'completion.json').write_text(json.dumps({'regions':len(rows),'sample_ids':selected,'full_coverage':set(selected)==set(reference),'csv_sha256':digest(out/'predictions.csv')},indent=2)+'\n')

if __name__=='__main__':main()
