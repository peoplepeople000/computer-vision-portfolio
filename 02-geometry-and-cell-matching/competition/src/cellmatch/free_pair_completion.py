"""Add mutually nearest unpaired detections using frozen image-only geometry.

No labels are read during prediction. Global distance and ambiguity recipes are
predeclared; existing segmentation and pair identities are preserved exactly.
"""
import copy,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from .data import read_rows,write_rows,parse_instances,image_path,image_shape

RECIPES={'strict':(4.,2.),'moderate':(7.,2.),'wide':(10.,2.5)}

def complete(row,shapes,matrix,max_distance,ratio):
    result=copy.deepcopy(row);used=[{p[n] for p in row['match_pairs']} for n in [0,1]]
    banks=[parse_instances(row,m,s) for m,s in zip(['invivo','exvivo'],shapes)]
    keys=[[k for k in bank if k not in u] for bank,u in zip(banks,used)]
    if not all(keys):return result,[]
    points=[np.array([bank[k].coordinates().mean(0) for k in ks]) for bank,ks in zip(banks,keys)]
    projected=points[0]@matrix[:,:2].T+matrix[:,2];d=cdist(projected,points[1]);forward=np.argmin(d,axis=1);reverse=np.argmin(d,axis=0);added=[]
    for i,j in enumerate(forward):
        if reverse[j]!=i or d[i,j]>max_distance:continue
        others_i=np.delete(d[i],j);others_j=np.delete(d[:,j],i)
        second=min(others_i.min(initial=np.inf),others_j.min(initial=np.inf))
        if second<max(2.,ratio*d[i,j]):continue
        pair=[keys[0][i],keys[1][j]];result['match_pairs'].append(pair);added.append(dict(pair=pair,distance=float(d[i,j]),second_distance=float(second)))
    return result,added

def main():
    root=Path('data/raw/Project_2_Dataset');bp=Path('outputs/stage57_iv_local/full/stage56_d4/predictions.csv');gp=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');base=read_rows(bp)
    geo={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(gp.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    out=Path('outputs/stage59_free_pairs');out.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(dict(recipes=RECIPES,source_sha256=sha(__file__),base_sha256=sha(bp),geometry_sha256=sha(gp),policy=__doc__),indent=2)+'\n')
    for name,(distance,ratio) in RECIPES.items():
        rows=[];reports=[]
        for sid,row in base.items():
            if sid in geo:
                shapes=[image_shape(image_path(root,'training',sid,m)) for m in ['invivo','exvivo']];new,added=complete(row,shapes,geo[sid],distance,ratio)
            else:new,added=copy.deepcopy(row),[]
            rows.append(new);reports.append(dict(sample_id=sid,added=added))
        folder=out/name;folder.mkdir(exist_ok=True);write_rows(rows,folder/'predictions.csv');(folder/'manifest.json').write_text(json.dumps(dict(regions=reports,csv_sha256=sha(folder/'predictions.csv')),indent=2)+'\n')
if __name__=='__main__':main()
