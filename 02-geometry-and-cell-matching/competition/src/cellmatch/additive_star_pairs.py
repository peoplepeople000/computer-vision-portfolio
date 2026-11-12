"""Preserve baseline pairs and add spatially nonconflicting radial pairs.

No labels enter this policy. Existing baseline assignments and their exact masks
are retained; radial pairs can rescue only unused IV cells and nonoverlapping EX
cells. Unpaired EX segmentation is filled by the baseline afterward.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from .data import read_rows,write_rows,parse_instances,image_shape,image_path,split_id
from .preserve_pair_masks import merge_row
from .stage3 import FOLDS
from .cli import evaluate


def augment(baseline,radial,shape):
    if baseline['sample_id']!=radial['sample_id'] or baseline['invivo_instances']!=radial['invivo_instances']:
        raise ValueError('Different IV masks or sample IDs')
    old=parse_instances(baseline,'exvivo',shape);new=parse_instances(radial,'exvivo',shape)
    used_iv={i for i,j in baseline['match_pairs']};used_ex={j for i,j in baseline['match_pairs']}
    occupied=[old[j] for j in used_ex]
    row=copy.deepcopy(baseline)
    row['exvivo_instances']={'BASE_'+j:baseline['exvivo_instances'][j] for j in sorted(used_ex)}
    row['match_pairs']=[[i,'BASE_'+j] for i,j in baseline['match_pairs']]
    for i,j in radial['match_pairs']:
        if i in used_iv or any(new[j].iou(m)>0 for m in occupied):continue
        key='RAD_'+j;row['exvivo_instances'][key]=radial['exvivo_instances'][j]
        row['match_pairs'].append([i,key]);used_iv.add(i);occupied.append(new[j])
    return merge_row(row,baseline)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',required=True,choices=list(FOLDS));a=p.parse_args()
    root=Path('data/raw/Project_2_Dataset');subject=FOLDS[a.fold]
    baseline_path=Path('outputs/stage13_native_relative/preserved/predicted_all.csv')
    radial_path=Path(f'outputs/stage25_pair_ransac/{a.fold}/merged.csv')
    baseline=read_rows(baseline_path);radial=read_rows(radial_path)
    if set(radial)!={s for s in baseline if split_id(s)[0]==subject}:raise ValueError('Incomplete fold')
    out=Path(f'outputs/stage27_additive_pairs/{a.fold}');out.mkdir(parents=True,exist_ok=True)
    rows=[augment(baseline[s],radial[s],image_shape(image_path(root,'training',s,'exvivo'))) for s in sorted(radial)]
    path=out/'merged.csv';write_rows(rows,path)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps({'held_out':subject,'source_sha256':digest(__file__),'input_sha256':{str(p):digest(p) for p in [baseline_path,radial_path]},'policy':'Retain every baseline pair and its exact masks; add only unused-IV/nonoverlapping-EX radial pairs; fill unpaired EX with baseline segmentation. No labels or fitted parameters.'},indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=subject,output=out/'metrics.json'))

if __name__=='__main__':main()
