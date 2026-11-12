"""Keep original baseline assignments unless predicted-pair geometry improves."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from .data import read_rows,write_rows,validate_submission
from .cli import evaluate


def select(baseline,candidate,records):
    if set(baseline)!=set(candidate) or set(records)!=set(baseline):raise ValueError('Different coverage')
    rows=[];changed=[]
    for sid in baseline:
        accepted=any(r['accepted'] for r in records[sid]['iterations'])
        if any(baseline[sid][key]!=candidate[sid][key] for key in ['invivo_instances','exvivo_instances']):raise ValueError('Masks changed')
        rows.append(candidate[sid] if accepted else baseline[sid])
        if accepted:changed.append(sid)
    return rows,changed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest_path=a.source/'manifest.json';manifest=json.loads(manifest_path.read_text());is_test=manifest['split']=='hidden_test'
    base_path=Path('outputs/submission_stage8/submission.csv' if is_test else 'outputs/stage13_native_relative/preserved/predicted_all.csv')
    candidate_path=a.source/('submission.csv' if is_test else 'predicted_all.csv')
    baseline=read_rows(base_path);candidate=read_rows(candidate_path)
    rows,changed=select(baseline,candidate,{r['sample_id']:r for r in manifest['regions']})
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/candidate_path.name;write_rows(rows,path)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report={'source_sha256':digest(__file__),'inputs_sha256':{str(p):digest(p) for p in [base_path,candidate_path,manifest_path]},'changed_regions':changed,'policy':'Use candidate assignments only when at least one predicted-pair RANSAC refinement was accepted; otherwise retain baseline row. Masks remain identical.'}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    root=Path('data/raw/Project_2_Dataset')
    if is_test:(a.output/'validation.json').write_text(json.dumps({'status':'passed','counts':validate_submission(root,path),'csv_sha256':digest(path),'uploaded':False,'hidden_score':None},indent=2)+'\n')
    else:evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=a.output/'metrics.json'))

if __name__=='__main__':main()
