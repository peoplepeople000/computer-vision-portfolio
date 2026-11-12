"""Reject unpaired baseline IV masks unsupported by independent Cellpose masks.

A fixed lenient IoU > 0.10 is a detection-support check, not a strict segmentation
match. All existing paired IV masks and every EX mask are retained unchanged.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from .data import read_rows,write_rows,parse_instances,image_shape,image_path,validate_submission
from .cli import evaluate


def supported_row(baseline,cellpose,shape):
    if baseline['sample_id']!=cellpose['sample_id']:raise ValueError('Different samples')
    protected={a for a,b in baseline['match_pairs']}
    old=parse_instances(baseline,'invivo',shape);new=parse_instances(cellpose,'invivo',shape)
    keep={k for k,m in old.items() if k in protected or any(m.iou(n)>.10 for n in new.values())}
    result=copy.deepcopy(baseline);result['invivo_instances']={k:v for k,v in baseline['invivo_instances'].items() if k in keep}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['training','hidden_test'],default='training');a=p.parse_args()
    root=Path('data/raw/Project_2_Dataset');test=a.split=='hidden_test'
    baseline_path=Path('outputs/submission_stage8/submission.csv' if test else 'outputs/stage13_native_relative/preserved/predicted_all.csv')
    baseline=read_rows(baseline_path);source_path=a.source/'predictions.csv';source=read_rows(source_path)
    if set(source)!=set(baseline) or len(source)!=(29 if test else 47):raise ValueError('Complete split required')
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[supported_row(row,source[sid],image_shape(image_path(root,a.split,sid,'invivo'))) for sid,row in baseline.items()]
    path=a.output/('submission.csv' if test else 'predicted_all.csv');write_rows(rows,path)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps({'source_sha256':digest(__file__),'inputs_sha256':{str(p):digest(p) for p in [baseline_path,source_path,a.source/'configuration.json']},'policy':__doc__,'fitting':'None','split':a.split},indent=2)+'\n')
    if test:(a.output/'validation.json').write_text(json.dumps({'status':'passed','counts':validate_submission(root,path),'csv_sha256':digest(path),'uploaded':False,'hidden_score':None},indent=2)+'\n')
    else:evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=a.output/'metrics.json'))

if __name__=='__main__':main()
