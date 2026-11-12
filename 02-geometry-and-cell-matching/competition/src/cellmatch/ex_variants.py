"""Frozen specialist comparisons with fixed IV masks and optional EX rejection."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .data import read_rows, write_rows, parse_instances, split_id
from .learning import instances_from_maps
from .rle import labels_to_instances
from .stage3 import FOLDS
from .transfer_filter import transfer_rejections
from .cli import evaluate


def main():
    root=Path('data/raw/Project_2_Dataset')
    original=read_rows('outputs/stage2/unet_oof.csv')
    retained=read_rows('outputs/stage4_conservative/quality_oof.csv')
    specialist=read_rows('outputs/stage6_ex/segmentation_oof.csv')
    if not set(original)==set(retained)==set(specialist): raise ValueError('Wrong prediction IDs')
    variants={name:[] for name in ['ex_filtered','ex_ensemble','ex_ensemble_filtered']}
    sources=[]; decisions=[]
    for tag, subject in FOLDS.items():
        manifest_path=Path(f'outputs/stage6_ex/unet_holdout_{tag}.csv.manifest.json')
        manifest=json.loads(manifest_path.read_text())
        if manifest['held_out']!=subject or any(split_id(s)[0]==subject for s in manifest['train_ids']):
            raise ValueError('Specialist includes held-out training data')
        baseline_manifest=json.loads(Path(f'outputs/stage2/unet_holdout_{tag}.csv.manifest.json').read_text())
        if baseline_manifest['held_out']!=subject or set(baseline_manifest['train_ids'])!=set(manifest['train_ids']):
            raise ValueError('Mismatched baseline provenance')
        sources.append(manifest)
        for sid,row in specialist.items():
            if split_id(sid)[0]!=subject: continue
            old=np.load(f'outputs/stage2/unet_holdout_{tag}_maps/{sid}_exvivo.npz')['maps'].astype(np.float32)
            new=np.load(f'outputs/stage6_ex/unet_holdout_{tag}_maps/{sid}_exvivo.npz')['maps'].astype(np.float32)
            if old.shape!=new.shape: raise ValueError('Map shape mismatch')
            combined=copy.deepcopy(row)
            combined['exvivo_instances']=labels_to_instances(instances_from_maps((old+new)/2,.5,.35),'EXP')
            variants['ex_ensemble'].append(combined)
            before=parse_instances(original[sid],'exvivo',new.shape[1:])
            for name,source in [('ex_filtered',row),('ex_ensemble_filtered',combined)]:
                target=copy.deepcopy(source)
                after=parse_instances(target,'exvivo',new.shape[1:])
                removed=transfer_rejections(before,retained[sid]['exvivo_instances'],after)
                target['exvivo_instances']={k:v for k,v in target['exvivo_instances'].items() if k not in removed}
                variants[name].append(target); decisions.append({'sample_id':sid,'variant':name,'removed':len(removed)})
    for name,rows in variants.items():
        folder=Path('outputs')/f'stage6_{name}';folder.mkdir(exist_ok=True)
        path=folder/'segmentation_oof.csv';write_rows(rows,path)
        (folder/'manifest.json').write_text(json.dumps({'variant':name,'sources':sources,
            'specialist_csv_sha256':hashlib.sha256(Path('outputs/stage6_ex/segmentation_oof.csv').read_bytes()).hexdigest(),
            'frozen_rejections':'outputs/stage4_conservative/quality_oof.csv','transfer_iou':.5,
            'decisions':[d for d in decisions if d['variant']==name]},indent=2)+'\n')
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=folder/'segmentation_oof_metrics.json'))


if __name__=='__main__': main()
