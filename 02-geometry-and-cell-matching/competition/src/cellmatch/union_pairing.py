"""Learn assignment across overlapping native and radial EX proposals.

Connected overlap components are assignment columns: different contours of the
same cell compete, rather than being counted as separate matched cells.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .data import read_rows,write_rows,parse_instances,split_id
from .stage3 import FOLDS,quality_probabilities
from .ex_joint_calibration import morphology_features
from .relative_ex_quality import build
from .joint_pair_assignment import candidate_features,fit,probability
from .preserve_pair_masks import merge_row
from .pair_ransac_refinement import refine
from .metrics import segmentation_score
from .cli import evaluate


def overlap_groups(masks):
    keys=list(masks);parent=list(range(len(keys)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(keys):
        for j in range(i):
            b=keys[j]
            if a[:2]==b[:2]:continue
            if masks[a].iou(masks[b])>0:
                x,y=find(i),find(j)
                if x!=y:parent[x]=y
    return {k:find(i) for i,k in enumerate(keys)}


def assign(pairs,probabilities,threshold,distances,radius,groups):
    if not pairs:return []
    left=sorted({a for a,b in pairs});right=sorted({groups[b] for a,b in pairs})
    ai={k:i for i,k in enumerate(left)};bi={k:i for i,k in enumerate(right)}
    utility=np.zeros((len(left),len(right)+len(left)));utility[:,:len(right)]=-1e6;chosen={}
    for n,((a,b),p) in enumerate(zip(pairs,probabilities)):
        i,j=ai[a],bi[groups[b]]
        if distances[n]<radius and p>threshold and p-threshold>utility[i,j]:
            utility[i,j]=p-threshold;chosen[i,j]=[a,b]
    ii,jj=linear_sum_assignment(-utility)
    return [chosen[i,j] for i,j in zip(ii,jj) if j<len(right) and utility[i,j]>0]


def union_features(row,iv_maps,native_maps,star_maps):
    masks=[parse_instances(row,'invivo',iv_maps.shape[1:]),parse_instances(row,'exvivo',native_maps.shape[1:])]
    iv_features=morphology_features(masks[0],iv_maps)[1]
    ex_features=[]
    for prefix,maps in [('S_',star_maps),('B_',native_maps)]:
        subset={k:v for k,v in masks[1].items() if k.startswith(prefix)}
        ex_features.extend(morphology_features(subset,maps)[1])
    return masks,[iv_features,np.asarray(ex_features,dtype=np.float32).reshape(-1,30)]


def edges(masks,features,matrix):
    pairs,x=candidate_features(masks,features,matrix)
    family=np.array([float(b.startswith('S_')) for a,b in pairs],dtype=np.float32)
    return pairs,np.c_[x,family]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/stage26_union_pairs'));a=p.parse_args()
    torch.set_num_threads(2)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    subject=FOLDS[a.fold];ids={s for s in truth if split_id(s)[0]!=subject}
    reference_path=Path('outputs/stage13_native_relative/preserved/predicted_all.csv');reference=read_rows(reference_path)
    iv_root=Path('outputs/stage8_iv')/a.fold;star_root=Path('outputs/stage20_star_ex')/a.fold
    native_root=Path('outputs/stage11_native_ex')/a.fold
    model_paths=[iv_root/'model.pt',star_root/'model.pt',native_root/'model.pt',Path(f'outputs/stage8_iv_quality_capacity/{a.fold}.pt'),Path(f'outputs/stage13_native_relative/{a.fold}.pt')]
    for path in model_paths:
        state=torch.load(path,map_location='cpu',weights_only=False)
        if set(state['train_ids'])!=ids or state['held_out']!=subject:raise ValueError('Invalid model partition')
    def quality(path):
        state=torch.load(path,map_location='cpu',weights_only=False);model=torch.nn.Linear(30,1);model.load_state_dict(state['quality'])
        return state,(model.eval(),state['mean'],state['sd'])
    iv_state,iv_q=quality(model_paths[-2]);ex_state,ex_q=quality(model_paths[-1])
    development=read_rows(iv_root/'development.csv')
    if set(development)!=ids:raise ValueError('Incomplete development set')
    geometry_path=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json')
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(geometry_path.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    out=a.output/a.fold
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    files=model_paths+[reference_path,geometry_path,iv_root/'development.csv']
    def inputs(sid,training):
        iv_path=iv_root/'maps'/f'{sid}_invivo.npz'
        native_path=(native_root/'maps' if training else Path('outputs/stage7_tiles/maps'))/f'{sid}_exvivo.npz'
        star_path=star_root/'maps'/f'{sid}_exvivo.npz';star_region=star_root/'regions'/f'{sid}.json'
        files.extend([iv_path,native_path,star_path,star_region])
        iv_maps=np.load(iv_path)['maps'].astype(np.float32);native_maps=np.load(native_path)['maps'].astype(np.float32);star_maps=np.load(star_path)['maps'].astype(np.float32)
        row=copy.deepcopy(development[sid] if training else reference[sid]);row['match_pairs']=[]
        if training:
            parsed=parse_instances(row,'invivo',iv_maps.shape[1:]);keys,f=morphology_features(parsed,iv_maps)
            keep={k for k,v in zip(keys,quality_probabilities(iv_q,f)) if v>=iv_state['threshold']}
            row['invivo_instances']={k:v for k,v in row['invivo_instances'].items() if k in keep}
            encoded,_,keys,f=build(native_maps,ex_state['foreground'])
            keep={k for k,v in zip(keys,quality_probabilities(ex_q,f)) if v>=ex_state['threshold']}
            native={k:v for k,v in encoded.items() if k in keep}
        else:native=row['exvivo_instances']
        star=json.loads(star_region.read_text())['exvivo_instances']
        row['exvivo_instances']={**{'S_'+k:v for k,v in star.items()},**{'B_'+k:v for k,v in native.items()}}
        masks,features=union_features(row,iv_maps,native_maps,star_maps)
        return row,masks,features,overlap_groups(masks[1])
    records=[]
    for sid in sorted(ids):
        row,masks,features,groups=inputs(sid,True)
        if len(truth[sid]['match_pairs'])<5:continue
        actual=[parse_instances(truth[sid],m,next(iter(masks[i].values())).shape) for i,m in enumerate(['invivo','exvivo'])]
        verified=set(map(tuple,truth[sid]['match_pairs']))
        source=np.array([actual[0][i].coordinates().mean(0) for i,j in sorted(verified)])
        target=np.array([actual[1][j].coordinates().mean(0) for i,j in sorted(verified)])
        matrix=np.linalg.lstsq(np.c_[source,np.ones(len(source))],target,rcond=None)[0].T
        pairs,x=edges(masks,features,matrix)
        iv_mapping=segmentation_score(masks[0],actual[0])['pred_to_gt'];ex_mapping={}
        for prefix in ['S_','B_']:
            ex_mapping.update(segmentation_score({k:v for k,v in masks[1].items() if k.startswith(prefix)},actual[1])['pred_to_gt'])
        y=np.array([(iv_mapping.get(i),ex_mapping.get(j)) in verified for i,j in pairs])
        records.append({'sample_id':sid,'subject':split_id(sid)[0],'pairs':pairs,'x':x,'y':y,'groups':groups})
        print('Union development',sid,len(pairs),int(y.sum()),flush=True)
    inner=[]
    for mouse in sorted({r['subject'] for r in records}):
        fitted=fit([r for r in records if r['subject']!=mouse])
        inner.extend((r,probability(fitted,r['x'])) for r in records if r['subject']==mouse)
    measurements=[];verified_count=sum(len(truth[s]['match_pairs']) for s in ids)
    for radius in [4.,5.,6.,8.]:
        for threshold in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]:
            count=correct=0
            for r,values in inner:
                chosen=assign(r['pairs'],values,threshold,r['x'][:,18],radius,r['groups']);positive={tuple(pair) for pair,y in zip(r['pairs'],r['y']) if y}
                count+=len(chosen);correct+=sum(tuple(pair) in positive for pair in chosen)
            measurements.append({'radius':radius,'threshold':threshold,'f1':2*correct/(count+verified_count)})
    selected=max(measurements,key=lambda m:m['f1']);fitted=fit(records)
    torch.save({'state_dict':fitted[0].state_dict(),'mean':fitted[1],'sd':fitted[2],'feature_count':34,'train_ids':sorted(ids),'held_out':subject,'selected':selected},out/'pairs.pt')
    (out/'calibration.json').write_text(json.dumps({'selected':selected,'measurements':measurements},indent=2)+'\n')
    rows=[];diagnostics=[]
    for sid in sorted(set(truth)-ids):
        row,masks,features,groups=inputs(sid,False);matrix=matrices.get(sid);history=[]
        centers=[{k:v.coordinates().mean(0) for k,v in m.items()} for m in masks]
        for iteration in range(2):
            pairs,x=edges(masks,features,matrix);chosen=assign(pairs,probability(fitted,x),selected['threshold'],x[:,18],selected['radius'],groups)
            source=np.asarray([centers[0][i] for i,j in chosen]).reshape(-1,2);target=np.asarray([centers[1][j] for i,j in chosen]).reshape(-1,2)
            matrix,report=refine(source,target,matrix);history.append(report)
            if not report['accepted']:break
        pairs,x=edges(masks,features,matrix)
        row['match_pairs']=assign(pairs,probability(fitted,x),selected['threshold'],x[:,18],selected['radius'],groups)
        rows.append(merge_row(row,reference[sid]));diagnostics.append({'sample_id':sid,'matrix':None if matrix is None else matrix.tolist(),'iterations':history})
        print('Union heldout',sid,len(row['match_pairs']),flush=True)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report={'held_out':subject,'train_ids':sorted(ids),'source_sha256':digest(__file__),'input_sha256':{str(p):digest(p) for p in set(files)},'regions':diagnostics,'policy':'34-feature pair classifier; overlapping EX proposal components share one assignment column; inner mouse threshold selection; predicted-pair RANSAC and pair-preserving merge.'}
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    write_rows(rows,out/'merged.csv');evaluate(SimpleNamespace(data=root,predictions=out/'merged.csv',subject=subject,output=out/'metrics.json'))

if __name__=='__main__':main()
