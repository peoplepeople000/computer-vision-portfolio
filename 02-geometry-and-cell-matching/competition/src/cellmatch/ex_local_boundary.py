"""EX boundary refinement on same-outer raw-guided proposals with fixed pairs.
Training uses only other mice, plus synthetic perturbed training contours.
All-training fitting remains unsupported until matched development inputs exist.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from scipy import ndimage as ndi
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .learning import SmallUNet,image_features,loss_function
from .ex_boundary import origin,local_mask,example,patch_rle,sample,refine
from .rle import Mask
from .stage3 import FOLDS
from .cli import evaluate


class D4Average(torch.nn.Module):
    """Average foreground/center probabilities over rotations and reflections."""
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,x):
        values=[]
        for reflected in [False,True]:
            base=x.flip(-1) if reflected else x
            for k in range(4):
                probability=self.model(torch.rot90(base,k,(-2,-1))).sigmoid()
                restored=torch.rot90(probability,-k,(-2,-1))
                if reflected:restored=restored.flip(-1)
                values.append(restored)
        return torch.logit(torch.stack(values).mean(0).clamp(1e-6,1-1e-6))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=[*FOLDS,'all-training'],required=True);p.add_argument('--steps',type=int,default=3000);p.add_argument('--output',type=Path,default=Path('outputs/stage58_ex_local'));a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2083);rng=np.random.default_rng(2083);device=torch.device('mps');assert torch.backends.mps.is_available()
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS.get(a.fold);ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    if subject is None:raise ValueError('Final fitting requires explicitly prepared matching all-training proposal inputs.')
    source=Path('outputs/stage56_raw_guided')/a.fold
    scfg=json.loads((source/'configuration.json').read_text());assert scfg['held_out']==subject and scfg['train_ids']==ids
    manifest=json.loads((source/'manifest.json').read_text())
    assert hashlib.sha256((source/'model.pt').read_bytes()).hexdigest()==manifest['model_sha256']
    assert manifest['config']==scfg
    inputs=[root/'training/train_ground_truth.csv',source/'configuration.json',source/'manifest.json',source/'model.pt']
    for sid in ids:
        pp=source/'proposals'/f'{sid}.json'
        if pp.exists():
            assert hashlib.sha256(pp.read_bytes()).hexdigest()==manifest['proposals_sha256'][sid]
            inputs.append(pp)
        inputs.append(image_path(root,'training',sid,'exvivo'))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True)
    config=dict(held_out=subject,train_ids=ids,steps=a.steps,seed=2083,width=16,batch=32,patch=32,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['ex_local_boundary.py','ex_boundary.py','learning.py']},input_sha256={str(p):sha(p) for p in inputs},policy=__doc__)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');model=SmallUNet(16).to(device);final=out/'model.pt'
    if final.exists():
        state=torch.load(final,map_location='cpu',weights_only=False);assert state['config']==config and state['step']==a.steps;model.load_state_dict(state['state_dict'])
    else:
        records=[];generation=np.random.default_rng(2083)
        for sid in ids:
            image=read_image(image_path(root,'training',sid,'exvivo'));features=image_features(image);actual=parse_instances(truth[sid],'exvivo',image.shape)
            pp=source/'proposals'/f'{sid}.json'
            proposals=json.loads(pp.read_text())['proposals'] if pp.exists() else []
            for item in proposals:
                proposal=Mask.parse(item['rle'],image.shape)
                score,key=max(((proposal.iou(gt),k) for k,gt in actual.items()),default=(0,None));target=actual[key] if score>.2 else None
                x,y=example(features,proposal,target);records.append((split_id(sid)[0],target is not None,x,y))
            for gt in actual.values():
                x0,y0=origin(gt);patch=local_mask(gt,x0,y0).astype(bool);kind=int(generation.integers(3))
                if kind==0:patch=ndi.binary_dilation(patch)
                elif kind==1:
                    eroded=ndi.binary_erosion(patch)
                    if eroded.any():patch=eroded
                patch=ndi.shift(patch.astype(float),generation.integers(-1,2,2),order=0,mode='constant')>0
                if not patch.any():continue
                proposal=Mask.parse(patch_rle(patch,x0,y0,image.shape),image.shape);x,y=example(features,proposal,gt);records.append((split_id(sid)[0],True,x,y))
            print('EX local examples',a.fold,sid,len(records),flush=True)
        groups={key:[r for r in records if (r[0],r[1])==key] for key in sorted({(r[0],r[1]) for r in records})};del records
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=1e-4);model.train();tick=time.perf_counter();history=[]
        for step in range(1,a.steps+1):
            x,y=sample(groups,rng);optimizer.zero_grad(set_to_none=True);loss=loss_function(model(x.to(device)),y.to(device));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                item=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.perf_counter()-tick);history.append(item);print('EX local training',a.fold,item,flush=True)
        torch.save(dict(config=config,step=a.steps,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),final);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n');del groups,optimizer,scheduler,x,y,loss
    model.eval();reports=[]
    for recipe,bp in [('stage57',Path('outputs/stage57_iv_local/full/stage56_d4/predictions.csv'))]:
        if not bp.exists():raise FileNotFoundError('Complete Stage 57 before EX boundary comparison')
        for view,inference_model in [('single',model),('d4',D4Average(model).eval())]:
            baseline=read_rows(bp);rows=[];changes=[]
            for sid,row in baseline.items():
                if split_id(sid)[0]!=subject:continue
                image=read_image(image_path(root,'training',sid,'exvivo'));masks=parse_instances(row,'exvivo',image.shape)
                instances,accepted=refine(masks,image_features(image),inference_model,device);result=copy.deepcopy(row);result['exvivo_instances']=instances
                assert result['match_pairs']==row['match_pairs'] and result['invivo_instances']==row['invivo_instances'] and set(instances)==set(row['exvivo_instances'])
                occupied=np.zeros(image.shape,bool)
                for mask in parse_instances(result,'exvivo',image.shape).values():
                    xy=mask.coordinates();assert not occupied[xy[:,1],xy[:,0]].any();occupied[xy[:,1],xy[:,0]]=True
                rows.append(result);changes.append(dict(sample_id=sid,accepted=accepted));print('EX local inference',recipe,sid,accepted,flush=True)
            folder=out/f'{recipe}_{view}';folder.mkdir(exist_ok=True);write_rows(rows,folder/'predictions.csv');evaluate(SimpleNamespace(data=root,predictions=folder/'predictions.csv',subject=subject,output=folder/'metrics.json'))
            reports.append(dict(recipe=recipe,view=view,baseline_sha256=sha(bp),regions=changes))
    (out/'manifest.json').write_text(json.dumps(dict(config=config,model_sha256=sha(final),reports=reports,policy='Fixed .5 foreground, overlap component, no invasion of existing neighbors, .5-1.7 area ratio, IoU with original >.4; all pairs and IV fixed.'),indent=2)+'\n')

if __name__=='__main__':main()
