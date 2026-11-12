"""Resume guided inference with explicit NumPy scalar JSON serialization.

The trained model and hashed training source are preserved byte-for-byte. This
entry point only repairs serialization of the already defined proposal output.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from .guided_ex import GuidedNet,infer
from .guided_ex_data import training_records,make_record
from .data import read_rows,split_id
from .stage3 import FOLDS


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON output type: {type(value).__name__}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True)
    p.add_argument('--source',type=Path,default=Path('outputs/stage47_guided_ex'));a=p.parse_args();torch.set_num_threads(2)
    device=torch.device('mps')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    folder=a.source/a.fold;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    config=json.loads((folder/'configuration.json').read_text());saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    if saved['config']!=config or saved['step']!=config['steps']:raise ValueError('Checkpoint provenance mismatch')
    if config['source_sha256']!=sha(Path(__file__).with_name('guided_ex.py')) or config['data_source_sha256']!=sha(Path(__file__).with_name('guided_ex_data.py')):raise ValueError('Trained source changed')
    for path,digest in config['development_inputs_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Development input changed: {path}')
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold]
    ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    if config['held_out']!=subject or config['train_ids']!=ids:raise ValueError('Invalid outer partition')
    model=GuidedNet(config['width']);model.load_state_dict(saved['state_dict']);model=model.to(device).eval();del saved
    records,loaded=training_records(root,truth,a.fold)
    if loaded!=ids:raise ValueError('Development IDs differ')
    proposals=folder/'proposals';proposals.mkdir(exist_ok=True);development=[]
    for record in records:
        sid=record['sample_id'];development.append(sid);path=proposals/f'{sid}.json'
        if not path.exists():path.write_text(json.dumps(infer(model,record,device),default=json_default)+'\n')
        print('Guided EX development inference',sid,flush=True)
    del records
    baselinepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');baseline=read_rows(baselinepath)
    geometry=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    matrices={r['sample_id']:np.array(r['matrix']) for r in json.loads(geometry.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    for sid in sorted(set(truth)-set(ids)):
        path=proposals/f'{sid}.json'
        if path.exists():continue
        value=dict(sample_id=sid,proposals=[])
        if sid in matrices:
            maps=np.load(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz')['maps'].astype(np.float32)
            value=infer(model,make_record(root,sid,baseline[sid],matrices[sid],maps),device)
        path.write_text(json.dumps(value,default=json_default)+'\n');print('Guided EX held-out inference',sid,len(value['proposals']),flush=True)
    (folder/'completion.json').write_text(json.dumps(dict(model_sha256=sha(folder/'model.pt'),development_ids=development,held_out=subject,train_ids=ids,
        inference_source_sha256=sha(__file__),serialization='NumPy scalar values converted with .item(); trained source unchanged.',geometry_sha256=sha(geometry),baseline_sha256=sha(baselinepath),heldout_maps_sha256={s:sha(f'outputs/stage7_tiles/maps/{s}_exvivo.npz') for s in sorted(set(truth)-set(ids))},proposals={p.stem:sha(p) for p in proposals.glob('*.json')}),indent=2)+'\n')


if __name__=='__main__':main()
