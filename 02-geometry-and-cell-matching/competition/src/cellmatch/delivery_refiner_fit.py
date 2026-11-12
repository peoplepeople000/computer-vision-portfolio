"""Mask-conditioned IV contour refinement with fixed IDs, pairs and geometry.

Train only on the other mice's public/adapted/U-Net predictions and synthetic
imperfect training contours. Never rerun registration after changing a contour.
Final fitting refuses to substitute OOF masks for missing all-training development masks.
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=[*FOLDS,'all-training'],required=True);p.add_argument('--steps',type=int,default=3000);p.add_argument('--output',type=Path,default=Path('outputs/stage118_delivery/refiner'));a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2082);rng=np.random.default_rng(2082);device=torch.device('mps');assert torch.backends.mps.is_available()
    root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv');subject=FOLDS.get(a.fold);ids=sorted(s for s in truth if split_id(s)[0]!=subject)
    if subject is not None:raise ValueError('This entry point only fits the final all-training bundle')
    publicpath=Path('outputs/stage28_cpdino_iv_full/predictions.csv');oldpath=Path(f'outputs/stage118_delivery/unet/{a.fold}/development.csv');adapted=Path(f'outputs/stage118_delivery/adapted/{a.fold}')
    public=read_rows(publicpath);old=read_rows(oldpath);assert set(old)==set(ids)
    inputs=[publicpath,oldpath,root/'training/train_ground_truth.csv',adapted/'configuration.json',Path(f'outputs/stage118_delivery/unet/{a.fold}/training_config.json')]
    for path in inputs[-2:]:
        cfg=json.loads(path.read_text());assert cfg['held_out']==subject and cfg['train_ids']==ids
    for sid in ids:inputs.extend([adapted/'regions'/f'{sid}.json',image_path(root,'training',sid,'invivo')])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();out=a.output/a.fold;out.mkdir(parents=True,exist_ok=True)
    config=dict(held_out=subject,train_ids=ids,steps=a.steps,seed=2082,width=16,batch=32,patch=32,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['delivery_refiner_fit.py','iv_local_boundary.py','ex_boundary.py','learning.py']},input_sha256={str(p):sha(p) for p in inputs},policy=__doc__)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    cp.write_text(json.dumps(config,indent=2)+'\n');model=SmallUNet(16).to(device);final=out/'model.pt'
    if final.exists():
        state=torch.load(final,map_location='cpu',weights_only=False);assert state['config']==config and state['step']==a.steps;model.load_state_dict(state['state_dict'])
    else:
        records=[];generation=np.random.default_rng(2082)
        for sid in ids:
            image=read_image(image_path(root,'training',sid,'invivo'));features=image_features(image);actual=parse_instances(truth[sid],'invivo',image.shape)
            new=json.loads((adapted/'regions'/f'{sid}.json').read_text());banks=[old[sid],public[sid],new]
            for row in banks:
                for proposal in parse_instances(row,'invivo',image.shape).values():
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
            print('IV local examples',a.fold,sid,len(records),flush=True)
        groups={key:[r for r in records if (r[0],r[1])==key] for key in sorted({(r[0],r[1]) for r in records})};del records
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=1e-4);model.train();tick=time.perf_counter();history=[]
        for step in range(1,a.steps+1):
            x,y=sample(groups,rng);optimizer.zero_grad(set_to_none=True);loss=loss_function(model(x.to(device)),y.to(device));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                item=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.perf_counter()-tick);history.append(item);print('IV local training',a.fold,item,flush=True)
        torch.save(dict(config=config,step=a.steps,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),final);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n');del groups,optimizer,scheduler,x,y,loss
    (out/'completion.json').write_text(json.dumps(dict(status='complete',model_sha256=sha(final),configuration_sha256=sha(cp)),indent=2)+'\n')

if __name__=='__main__':main()
