"""Calibrated choice between native and radial EX contours for each pair.

EX alternatives share assignment families. Every selected EX family and IV cell
is used once, with explicit EX pixel nonoverlap. Baseline unpaired EX masks are
retained. All training and threshold calibration exclude the evaluation mouse.

Additional 33 source-by-feature interactions permit source-specific confidence slopes.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from .data import read_rows,write_rows,parse_instances,split_id
from .stage3 import FOLDS,quality_probabilities
from .ex_joint_calibration import morphology_features
from .relative_ex_quality import relative_features
from .pair_quality import training_record
from .joint_pair_assignment import candidate_features,fit,probability
from .iv_mask_bank import family_features,choose
from .cellpose_contour_consensus import blend
from .preserve_pair_masks import merge_row
from .metrics import segmentation_score
from .cli import evaluate


def bank_record(rows,ivmaps,exmaps,matrix,actual=None,verified=None):
    banks=[parse_instances(row,'exvivo',v.shape[1:]) for row,v in zip(rows,exmaps)]
    iv=parse_instances(rows[0],'invivo',ivmaps.shape[1:]);features=morphology_features(iv,ivmaps)[1]
    families,extra=family_features(banks);pairs=[];xs=[];ys=[];all_masks={}
    ivmap=segmentation_score(iv,actual[0])['pred_to_gt'] if actual is not None else None
    for b,(masks,v) in enumerate(zip(banks,exmaps)):
        edges,x=candidate_features([iv,masks],[features,morphology_features(masks,v)[1]],matrix)
        all_masks.update({f'{b}:{k}':m for k,m in masks.items()})
        # Reverse only assignment IDs: the original feature order stays IV, EX.
        pairs.extend([[f'{b}:{j}',i] for i,j in edges])
        xs.append(np.c_[x,np.array([extra[f'{b}:{j}'] for i,j in edges]).reshape(-1,4),x*b].astype(np.float32))
        if actual is not None:
            mapping=segmentation_score(masks,actual[1])['pred_to_gt']
            ys.extend((ivmap.get(i),mapping.get(j)) in verified for i,j in edges)
    return dict(pairs=pairs,x=np.concatenate(xs),y=np.asarray(ys,bool),families=families,masks=all_masks)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=list(FOLDS),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/stage45_ex_mask_bank_interactions'));a=p.parse_args();torch.set_num_threads(1)
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS[a.fold]
    ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    newroot=Path('outputs/stage20_star_ex')/a.fold;ivroot=Path('outputs/stage8_iv')/a.fold;exroot=Path('outputs/stage11_native_ex')/a.fold
    files=[newroot/'training_config.json',newroot/'model.pt',ivroot/'training_config.json',ivroot/'model.pt',ivroot/'development.csv',exroot/'training_config.json']
    for path in [newroot/'training_config.json',ivroot/'training_config.json',exroot/'training_config.json']:
        cfg=json.loads(path.read_text())
        if cfg['held_out']!=subject or sorted(cfg['train_ids'])!=ids:raise ValueError('Invalid outer partition')
    publicpath=Path('outputs/stage28_cpdino_iv_full/predictions.csv');public=read_rows(publicpath);files.append(publicpath)
    baselinepath=Path('outputs/stage32_contour_consensus/predicted_all.csv');baseline=read_rows(baselinepath);files.append(baselinepath)
    def quality(path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['held_out']!=subject or sorted(state['train_ids'])!=ids:raise ValueError('Invalid quality partition')
        model=torch.nn.Linear(30,1);model.load_state_dict(state['quality']);files.append(path)
        return model.eval(),state['mean'],state['sd'],state['threshold']
    iq=quality(Path(f'outputs/stage8_iv_quality_capacity/{a.fold}.pt'));eq=quality(Path(f'outputs/stage13_native_relative/{a.fold}.pt'))
    cp=Path('outputs/stage13_native_relative/calibration.json');cal=json.loads(cp.read_text())[subject];files.append(cp)
    if sorted(cal['train_ids'])!=ids:raise ValueError('Invalid foreground calibration')
    development=read_rows(ivroot/'development.csv')
    if set(development)!=set(ids):raise ValueError('Invalid development coverage')
    records=[]
    for sid in ids:
        paths=[ivroot/'maps'/f'{sid}_invivo.npz',exroot/'maps'/f'{sid}_exvivo.npz',newroot/'maps'/f'{sid}_exvivo.npz',newroot/'regions'/f'{sid}.json'];files.extend(paths)
        im,om,nm=[np.load(path)['maps'].astype(np.float32) for path in paths[:3]]
        old=copy.deepcopy(development[sid]);masks=parse_instances(old,'invivo',im.shape[1:]);keys,xf=morphology_features(masks,im)
        keep={k for k,q in zip(keys,quality_probabilities(iq[:3],xf)) if q>=iq[3]}
        old['invivo_instances']={k:v for k,v in old['invivo_instances'].items() if k in keep};old['match_pairs']=[]
        old,_=blend(old,public[sid],im.shape[1:])
        result=training_record(root,sid,truth[sid],old,om,cal['foreground'],eq,quality_transform=relative_features)
        if result is None:continue
        old,_,actual,matrix=result;new=copy.deepcopy(old);new['exvivo_instances']=json.loads(paths[3].read_text())['exvivo_instances']
        record=bank_record([old,new],im,[om,nm],matrix,actual,set(map(tuple,truth[sid]['match_pairs'])))
        record.update(sample_id=sid,subject=split_id(sid)[0]);records.append(record)
        print('EX bank development',a.fold,sid,len(record['pairs']),int(record['y'].sum()),flush=True)
    inner=[]
    for mouse in sorted({r['subject'] for r in records}):
        fitted=fit([r for r in records if r['subject']!=mouse])
        inner.extend((r,probability(fitted,r['x'])) for r in records if r['subject']==mouse)
    measurements=[];total=sum(len(truth[s]['match_pairs']) for s in ids)
    for radius in (4.,5.,6.,8.):
        for threshold in (.05,.1,.15,.2,.25,.3,.35,.4,.45,.5):
            correct=count=0
            for r,prob in inner:
                chosen=choose(r,prob,threshold,radius);correct+=int(r['y'][chosen].sum());count+=len(chosen)
            measurements.append(dict(radius=radius,threshold=threshold,f1=2*correct/(count+total)))
    selected=max(measurements,key=lambda x:x['f1']);fitted=fit(records)
    out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True)
    if (out/'model.pt').exists():raise FileExistsError('Preserve existing model')
    torch.save(dict(state_dict=fitted[0].state_dict(),mean=fitted[1],sd=fitted[2],train_ids=ids,held_out=subject,selected=selected),out/'model.pt')
    geometry=Path('outputs/stage9_subject_geometry/predicted_all_manifest.json');files.append(geometry)
    matrices={r['sample_id']:np.asarray(r['matrix']) for r in json.loads(geometry.read_text())['regions'] if r.get('matrix') and r.get('status')=='registered'}
    rows=[];choices=[0,0]
    for sid in sorted(set(truth)-set(ids)):
        paths=[ivroot/'maps'/f'{sid}_invivo.npz',Path(f'outputs/stage7_tiles/maps/{sid}_exvivo.npz'),newroot/'maps'/f'{sid}_exvivo.npz',newroot/'regions'/f'{sid}.json'];files.extend(paths)
        im,om,nm=[np.load(path)['maps'].astype(np.float32) for path in paths[:3]]
        old=copy.deepcopy(baseline[sid]);new=copy.deepcopy(old);new['exvivo_instances']=json.loads(paths[3].read_text())['exvivo_instances']
        record=bank_record([old,new],im,[om,nm],matrices.get(sid));prob=probability(fitted,record['x'])
        chosen=choose(record,prob,selected['threshold'],selected['radius'])
        paired=copy.deepcopy(old);paired['exvivo_instances']={f'{b}:{k}':v for b,row in enumerate([old,new]) for k,v in row['exvivo_instances'].items()}
        paired['match_pairs']=[[record['pairs'][n][1],record['pairs'][n][0]] for n in chosen]
        for _,ex in paired['match_pairs']:choices[int(ex[0])]+=1
        rows.append(merge_row(paired,old))
    path=out/'predictions.csv';write_rows(rows,path);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(held_out=subject,train_ids=ids,selected=selected,calibration=measurements,choices=choices,input_sha256={str(p):sha(p) for p in files},source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['ex_mask_bank_interactions.py','iv_mask_bank.py','joint_pair_assignment.py','preserve_pair_masks.py']},limitation='Development segmentation is in-sample; outer mouse excluded. Repeated exploratory validation.'),indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=subject,output=out/'metrics.json'))


if __name__=='__main__':main()
