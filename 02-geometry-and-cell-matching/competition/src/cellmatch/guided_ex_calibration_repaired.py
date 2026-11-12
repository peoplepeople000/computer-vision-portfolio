"""Calibrate guided EX proposal credit on development mice only."""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,split_id
from .guided_ex_data import training_records
from .joint_pair_assignment import fit,probability
from .preserve_pair_masks import pixels,merge_row
from .rle import Mask
from .stage3 import FOLDS
from .cli import evaluate


def select(proposals,scores,threshold):
    selected=[];occupied=set();used=set()
    for n in sorted(range(len(proposals)),key=lambda n:(-float(scores[n]),n)):
        p=proposals[n]
        if scores[n]<=threshold or p['iv'] in used:continue
        mask=pixels(p['rle'])
        if mask&occupied:continue
        occupied.update(mask);used.add(p['iv']);selected.append(n)
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True)
    p.add_argument('--source',type=Path,default=Path('outputs/stage48_guided_ex'));p.add_argument('--output',type=Path,default=Path('outputs/stage48_guided_pairs'));a=p.parse_args();torch.set_num_threads(1)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold]
    ids=sorted(s for s in truth if split_id(s)[0]!=subject);folder=a.source/a.fold
    config=json.loads((folder/'configuration.json').read_text());complete=json.loads((folder/'completion.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if config['held_out']!=subject or config['train_ids']!=ids or complete['model_sha256']!=sha(folder/'model.pt'):raise ValueError('Invalid guided model provenance')
    if config['source_sha256']!=sha(Path(__file__).with_name('guided_ex_repaired.py')) or config['data_source_sha256']!=sha(Path(__file__).with_name('guided_ex_data.py')):raise ValueError('Guided source changed')
    development,loaded=training_records(root,truth,a.fold)
    if loaded!=ids:raise ValueError('Invalid development partition')
    records=[]
    for r in development:
        sid=r['sample_id'];path=folder/'proposals'/f'{sid}.json'
        if sha(path)!=complete['proposals'][sid]:raise ValueError('Changed development proposals')
        proposals=json.loads(path.read_text())['proposals'];targets={c['key']:c['target'] for c in r['cells']}
        gt=parse_instances(truth[sid],'exvivo',r['shape']);keys=list(gt)
        labels=[]
        for proposal in proposals:
            target=targets[proposal['iv']]
            labels.append(bool(target and Mask.parse(proposal['rle'],r['shape']).iou(gt[keys[target-1]])>.75))
        records.append(dict(sample_id=sid,subject=r['subject'],proposals=proposals,x=np.asarray([p['features'] for p in proposals],np.float32).reshape(-1,13),y=np.array(labels,bool)))
    del development
    if sum(int(r['y'].sum()) for r in records)==0:raise ValueError('No strict-IoU-valid development proposals; guided pilot collapsed')
    inner=[]
    for mouse in sorted({r['subject'] for r in records}):
        model=fit([r for r in records if r['subject']!=mouse])
        inner.extend((r,probability(model,r['x'])) for r in records if r['subject']==mouse)
    measurements=[];total=sum(len(truth[s]['match_pairs']) for s in ids)
    for threshold in np.arange(.05,1.,.05):
        correct=count=0
        for r,scores in inner:
            chosen=select(r['proposals'],scores,threshold);count+=len(chosen);correct+=int(r['y'][chosen].sum())
        measurements.append(dict(threshold=float(threshold),f1=2*correct/(count+total),correct=correct,predicted=count))
    selected=max(measurements,key=lambda v:v['f1']);fitted=fit(records)
    out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True)
    if (out/'model.pt').exists():raise FileExistsError('Preserve previous calibration')
    torch.save(dict(state_dict=fitted[0].state_dict(),mean=fitted[1],sd=fitted[2],selected=selected,held_out=subject,train_ids=ids),out/'model.pt')
    basepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');baseline=read_rows(basepath);rows=[]
    for sid in sorted(set(truth)-set(ids)):
        path=folder/'proposals'/f'{sid}.json'
        if sha(path)!=complete['proposals'][sid]:raise ValueError('Changed held-out proposals')
        proposals=json.loads(path.read_text())['proposals'];x=np.asarray([p['features'] for p in proposals],np.float32).reshape(-1,13)
        chosen=select(proposals,probability(fitted,x),selected['threshold'])
        paired=copy.deepcopy(baseline[sid]);paired['match_pairs']=[];paired['exvivo_instances']={}
        for n in chosen:
            key=f'GUIDED_{n:06d}';paired['exvivo_instances'][key]=proposals[n]['rle'];paired['match_pairs'].append([proposals[n]['iv'],key])
        rows.append(merge_row(paired,baseline[sid]))
    write_rows(rows,out/'predictions.csv')
    (out/'manifest.json').write_text(json.dumps(dict(held_out=subject,train_ids=ids,calibration=measurements,selected=selected,
        input_sha256={str(p):sha(p) for p in [folder/'configuration.json',folder/'completion.json',folder/'model.pt',basepath]},
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['guided_ex_calibration_repaired.py','guided_ex_repaired.py','guided_ex_data.py']},
        limitation='Inner classifier mouse holdout only; guided segmenter trained on both development mice. Outer labels excluded from all fitting.'),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=out/'predictions.csv',subject=subject,output=out/'metrics.json'))


if __name__=='__main__':main()
