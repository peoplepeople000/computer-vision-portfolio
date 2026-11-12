"""Fixed equal mixture of linear geometry and nonlinear appearance pair credit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from . import nonlinear_pairing as nonlinear
from . import joint_pair_assignment as linear
from .data import read_rows, write_rows, split_id, read_image, image_path, parse_instances
from .cell_features import extract_features
from .stage3 import FOLDS
from .cli import evaluate


class BlendedPairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance = nonlinear.make_model()
        self.geometry = nn.Linear(33, 1)

    def forward(self, x):
        probability = .5 * (self.appearance(x).sigmoid() + self.geometry(x[:, :33]).sigmoid())
        return torch.logit(probability.clamp(1e-6, 1-1e-6))


def fit(records):
    learned, mean, sd = nonlinear.fit(records)
    geometry, gmean, gsd = linear.fit([{**r, 'x':r['x'][:, :33]} for r in records])
    if not np.allclose(mean[:33], gmean) or not np.allclose(sd[:33], gsd):
        raise ValueError('Incompatible feature normalization')
    model = BlendedPairModel()
    model.appearance.load_state_dict(learned.state_dict())
    model.geometry.load_state_dict(geometry.state_dict())
    return model.eval(), mean, sd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/stage38_blended_pairing'))
    a = p.parse_args(); torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    base_path = Path('outputs/stage32_contour_consensus/predicted_all.csv')
    baseline = read_rows(base_path)
    geo_path = Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    geometry = json.loads(geo_path.read_text())
    matrices = {r['sample_id']:np.asarray(r['matrix']) for r in geometry['regions'] if r.get('matrix') and r.get('status')=='registered'}
    rows, report = [], {}
    for tag, mouse in FOLDS.items():
        path = Path(f'outputs/stage37_nonlinear_pairing/{tag}_development_records.pt')
        records = torch.load(path, map_location='cpu', weights_only=False)
        ids = sorted(s for s in truth if split_id(s)[0] != mouse)
        state = torch.load(path.with_name(f'{tag}.pt'), map_location='cpu', weights_only=False)
        if state['held_out'] != mouse or set(state['train_ids']) != set(ids):
            raise ValueError('Invalid outer fold')
        if any(r['sample_id'] not in ids or r['subject']==mouse for r in records):
            raise ValueError('Excluded mouse in pair training')
        inner = []
        for subject in sorted({r['subject'] for r in records}):
            model = fit([r for r in records if r['subject'] != subject])
            inner.extend((r, nonlinear.probability(model,r['x'])) for r in records if r['subject']==subject)
        measured = []
        verified = sum(len(truth[s]['match_pairs']) for s in ids)
        for radius in (4.,5.,6.,8.):
            for threshold in (.05,.1,.15,.2,.25,.3,.35,.4,.45,.5):
                correct = count = 0
                for r, prob in inner:
                    chosen=linear.assign_edges(r['pairs'],prob,threshold,r['x'][:,18],radius)
                    positive={tuple(pair) for pair,y in zip(r['pairs'],r['y']) if y}
                    correct+=sum(tuple(pair) in positive for pair in chosen);count+=len(chosen)
                measured.append(dict(radius=radius,threshold=threshold,f1=2*correct/(count+verified)))
        policy=max(measured,key=lambda x:x['f1']);model=fit(records)
        torch.save(dict(state_dict=model[0].state_dict(),mean=model[1],sd=model[2],held_out=mouse,
                        train_ids=ids,selected=policy,feature_count=75,architecture='equal_linear33_mlp75x32'),a.output/f'{tag}.pt')
        report[mouse]=dict(selected=policy,inner_scores=measured,train_ids=ids,
                          records_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for sid, base in baseline.items():
            if split_id(sid)[0] != mouse:continue
            masks,features=[],[]
            for modality in ['invivo','exvivo']:
                image=read_image(image_path(root,'training',sid,modality))
                masks.append(parse_instances(base,modality,image.shape))
                cache=Path(f'outputs/stage8_iv/{tag}/maps') if modality=='invivo' else Path('outputs/stage7_tiles/maps')
                maps=np.load(cache/f'{sid}_{modality}.npz')['maps'].astype(np.float32)
                features.append(extract_features(image,masks[-1],maps)[2])
            pairs,x=nonlinear.candidate_features(masks,features,matrices.get(sid))
            row=copy.deepcopy(base)
            row['match_pairs']=linear.assign_edges(pairs,nonlinear.probability(model,x),policy['threshold'],x[:,18],policy['radius'])
            rows.append(row)
        print('Blended pairs',tag,policy,flush=True)
    write_rows(rows,a.output/'predicted_all.csv')
    report['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path(nonlinear.__file__),Path(linear.__file__),base_path,geo_path]}
    (a.output/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=a.output/'predicted_all.csv',subject=None,output=a.output/'metrics.json'))


if __name__=='__main__':main()
