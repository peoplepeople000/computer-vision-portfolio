"""Add independently scored Cellpose IV cells without altering baseline masks.

Addition thresholds are chosen on the two outer-training mice, using quality
predictions from the other development mouse and same-outer IV baseline models.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import read_rows,write_rows,parse_instances,split_id,read_image,image_path
from .stage3 import FOLDS,fit_quality,quality_probabilities
from .ex_joint_calibration import morphology_features
from .relative_ex_quality import relative_features
from .preserve_pair_masks import pixels
from .metrics import segmentation_score
from .cli import evaluate
from .cellpose4_quality import quality_features


def available(baseline,candidate):
    occupied=set()
    for value in baseline['invivo_instances'].values():occupied.update(pixels(value))
    return np.array([not bool(pixels(value)&occupied) for value in candidate['invivo_instances'].values()])


def augment_iv(baseline,candidate,probabilities,threshold):
    if baseline['sample_id']!=candidate['sample_id']:raise ValueError('Different samples')
    if len(probabilities)!=len(candidate['invivo_instances']):raise ValueError('Wrong probability count')
    free=available(baseline,candidate);result=copy.deepcopy(baseline)
    for (key,value),p,ok in zip(candidate['invivo_instances'].items(),probabilities,free):
        if ok and p>=threshold:
            new='CP4_ADDED_'+key
            if new in result['invivo_instances']:raise ValueError('ID collision')
            result['invivo_instances'][new]=value
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--quality',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');pred=read_rows(a.source/'predictions.csv')
    reference_path=Path('outputs/stage13_native_relative/preserved/predicted_all.csv');reference=read_rows(reference_path)
    if set(pred)!=set(truth) or len(pred)!=47:raise ValueError('Complete 47 regions required')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Preserve previous experiment')
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[];reports={};files=[reference_path,a.source/'predictions.csv',a.quality/'calibration.json']
    for tag,subject in FOLDS.items():
        ids={s for s in truth if split_id(s)[0]!=subject}
        qpath=a.quality/f'{tag}.pt';qstate=torch.load(qpath,map_location='cpu',weights_only=False)
        native_root=Path('outputs/stage8_iv')/tag
        native_quality=Path(f'outputs/stage8_iv_quality_capacity/{tag}.pt')
        native_state=torch.load(native_quality,map_location='cpu',weights_only=False)
        native_model=torch.load(native_root/'model.pt',map_location='cpu',weights_only=False)
        for state in [qstate,native_state,native_model]:
            if set(state['train_ids'])!=ids or state['held_out']!=subject:raise ValueError('Invalid outer training partition')
        files.extend([qpath,native_quality,native_root/'model.pt',native_root/'development.csv'])
        native=read_rows(native_root/'development.csv')
        if set(native)!=ids:raise ValueError('Invalid development predictions')
        model=torch.nn.Linear(30,1);model.load_state_dict(native_state['quality']);native_fitted=model.eval(),native_state['mean'],native_state['sd']
        records=[]
        for sid in sorted(ids):
            path=a.source/'maps'/f'{sid}_invivo.npz';native_path=native_root/'maps'/f'{sid}_invivo.npz';files.extend([path,native_path])
            maps=np.load(path)['maps'].astype(np.float32);native_maps=np.load(native_path)['maps'].astype(np.float32)
            masks=parse_instances(pred[sid],'invivo',maps.shape[1:]);actual=parse_instances(truth[sid],'invivo',maps.shape[1:])
            keys=list(masks);image_file=image_path(root,'training',sid,'invivo');files.append(image_file)
            features=quality_features(masks,maps,qstate['representation'],read_image(image_file))
            mapping=segmentation_score(masks,actual)['pred_to_gt']
            native_masks=parse_instances(native[sid],'invivo',maps.shape[1:]);nk,nf=morphology_features(native_masks,native_maps)
            keep={k for k,v in zip(nk,quality_probabilities(native_fitted,nf)) if v>=native_state['threshold']}
            base=copy.deepcopy(native[sid]);base['invivo_instances']={k:v for k,v in base['invivo_instances'].items() if k in keep}
            score=segmentation_score({k:v for k,v in native_masks.items() if k in keep},actual)
            records.append({'sample_id':sid,'subject':split_id(sid)[0],'features':[None,features],
                'quality':np.array([k in mapping for k in keys]),'free':available(base,pred[sid]),
                'ious':np.array([masks[k].iou(actual[mapping[k]]) if k in mapping else 0 for k in keys]),
                'baseline_iou':score['iou_sum'],'denominator':.5*(len(actual)+len(keep))})
        inner=[]
        for mouse in sorted({r['subject'] for r in records}):
            fitted=fit_quality([r for r in records if r['subject']!=mouse])
            inner.extend((r,quality_probabilities(fitted,r['features'][1])) for r in records if r['subject']==mouse)
        measurements=[]
        for threshold in [.65,.75,.85,.95,1.01]:
            scores=[]
            for r,p in inner:
                keep=r['free']&(p>=threshold)
                scores.append((r['baseline_iou']+r['ious'][keep].sum())/(r['denominator']+.5*keep.sum()))
            measurements.append({'threshold':threshold,'mean_pq':float(np.mean(scores))})
        selected=max(measurements,key=lambda r:r['mean_pq'])
        reports[tag]={'held_out':subject,'train_ids':sorted(ids),'selected':selected,'measurements':measurements,'representation':qstate['representation']}
        qmodel=torch.nn.Linear(30,1);qmodel.load_state_dict(qstate['quality']);fitted=qmodel.eval(),qstate['mean'],qstate['sd']
        for sid in sorted(set(truth)-ids):
            path=a.source/'maps'/f'{sid}_invivo.npz';files.append(path);maps=np.load(path)['maps'].astype(np.float32)
            image_file=image_path(root,'training',sid,'invivo');files.append(image_file)
            features=quality_features(parse_instances(pred[sid],'invivo',maps.shape[1:]),maps,qstate['representation'],read_image(image_file))
            rows.append(augment_iv(reference[sid],pred[sid],quality_probabilities(fitted,features),selected['threshold']))
        print('Cellpose additive calibration',tag,selected,flush=True)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    reports['provenance']={'source_sha256':digest(__file__),'input_sha256':{str(p):digest(p) for p in set(files)},'policy':'Preserve every baseline mask/pair. Add only nonoverlapping IV masks; training-only conservative threshold calibration. No EX or matching changes.'}
    (a.output/'calibration.json').write_text(json.dumps(reports,indent=2)+'\n')
    write_rows(rows,a.output/'predicted_all.csv');evaluate(SimpleNamespace(data=root,predictions=a.output/'predicted_all.csv',subject=None,output=a.output/'metrics.json'))

if __name__=='__main__':main()
