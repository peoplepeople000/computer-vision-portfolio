"""Mouse-disjoint quality calibration for frozen public Cellpose IV masks."""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows,write_rows,parse_instances,split_id,read_image,image_path
from .ex_joint_calibration import morphology_features
from .cell_features import extract_features
from .relative_ex_quality import relative_features
from .stage3 import FOLDS,calibrate_quality,quality_probabilities
from .metrics import segmentation_score
from .preserve_pair_masks import merge_row
from .cli import evaluate


def quality_features(masks,maps,representation,image=None):
    if representation.startswith('appearance'):
        if image is None:raise ValueError('Appearance features need an image')
        features=extract_features(image,masks,maps)[2]
    else:features=morphology_features(masks,maps)[1]
    return relative_features(features) if representation.endswith('relative') else features


def preserve_iv_pairs(paired,segmentation):
    def swap(row):
        return {'sample_id':row['sample_id'],'invivo_instances':row['exvivo_instances'],
                'exvivo_instances':row['invivo_instances'],'match_pairs':[[b,a] for a,b in row['match_pairs']]}
    return swap(merge_row(swap(paired),swap(segmentation)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    baseline_path=Path('outputs/stage13_native_relative/preserved/predicted_all.csv');baseline=read_rows(baseline_path)
    predictions=read_rows(a.source/'predictions.csv')
    if set(predictions)!=set(truth) or len(truth)!=47:raise ValueError('Full 47-region coverage required')
    configuration=json.loads((a.source/'configuration.json').read_text())
    if configuration['training']!='Official public checkpoint, no competition-data fitting' or configuration['modality']!='invivo':raise ValueError('Expected fixed pretrained IV inference')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use a fresh output')
    a.output.mkdir(parents=True,exist_ok=True)
    records={};encoded={};files=[baseline_path,a.source/'predictions.csv',a.source/'configuration.json']
    for sid,row in predictions.items():
        path=a.source/'maps'/f'{sid}_invivo.npz';files.append(path)
        maps=np.load(path)['maps'].astype(np.float32);masks=parse_instances(row,'invivo',maps.shape[1:]);actual=parse_instances(truth[sid],'invivo',maps.shape[1:])
        keys,features=morphology_features(masks,maps)
        image_file=image_path(root,'training',sid,'invivo');files.append(image_file)
        appearance=extract_features(read_image(image_file),masks,maps)[2]
        mapping=segmentation_score(masks,actual)['pred_to_gt']
        records[sid]={'sample_id':sid,'subject':split_id(sid)[0],'features':[None,features],'quality':np.array([k in mapping for k in keys]),'gt_count':len(actual),
                      'appearance':appearance,'ious':np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0 for k in keys]),'keys':keys}
        encoded[sid]=row['invivo_instances']
    rows=[];preserved=[];reports={}
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject};choices=[]
        for representation in ['absolute','relative','appearance','appearance_relative']:
            training=[]
            for sid in sorted(ids):
                r=copy.deepcopy(records[sid])
                if representation.startswith('appearance'):r['features'][1]=r['appearance']
                if representation.endswith('relative'):r['features'][1]=relative_features(r['features'][1])
                training.append(r)
            fitted,threshold,measurements=calibrate_quality(training)
            choices.append((max(m['mean_pq'] for m in measurements),representation,fitted,threshold,measurements))
        _,representation,fitted,threshold,measurements=max(choices,key=lambda c:c[0])
        torch.save({'held_out':subject,'train_ids':sorted(ids),'quality':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],'threshold':threshold,'representation':representation},a.output/f'{tag}.pt')
        reports[tag]={'held_out':subject,'train_ids':sorted(ids),'representation':representation,'threshold':threshold,
                      'choices':[{ 'representation':c[1],'inner_pq':c[0],'threshold':c[3],'measurements':c[4]} for c in choices]}
        for sid in sorted(set(truth)-ids):
            record=records[sid];f=record['features'][1]
            if representation.startswith('appearance'):f=record['appearance']
            if representation.endswith('relative'):f=relative_features(f)
            keep={k for k,v in zip(record['keys'],quality_probabilities(fitted,f)) if v>=threshold}
            row=copy.deepcopy(baseline[sid]);row['match_pairs']=[];row['invivo_instances']={k:v for k,v in encoded[sid].items() if k in keep}
            rows.append(row);preserved.append(preserve_iv_pairs(baseline[sid],row))
        print('Cellpose IV quality',tag,representation,threshold,flush=True)
    for name,values in [('segmentation',rows),('preserved',preserved)]:
        path=a.output/f'{name}.csv';write_rows(values,path)
        evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=a.output/f'{name}_metrics.json'))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    reports['provenance']={'source_sha256':digest(__file__),'inputs_sha256':{str(p):digest(p) for p in files},
        'policy':'Fixed public pretrained IV masks, quality/representation selection via inner mouse validation on only two other mice. Preserve baseline pair endpoints exactly.',
        'dependencies_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['stage3.py','ex_joint_calibration.py','relative_ex_quality.py','preserve_pair_masks.py','metrics.py','cell_features.py','segmentation.py']}}
    (a.output/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
