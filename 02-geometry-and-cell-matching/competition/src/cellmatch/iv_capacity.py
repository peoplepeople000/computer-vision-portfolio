"""Larger native-resolution IV specialist with fixed-update outer-fold training."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .learning import SmallUNet,load_training,sample_batch,loss_function,choose_device,probability_maps,instances_from_maps
from .data import read_rows,write_rows,read_image,image_path,split_id
from .rle import labels_to_instances
from .stage3 import FOLDS
from .cli import evaluate


def main():
    torch.set_num_threads(2);device=choose_device('mps');root=Path('data/raw/Project_2_Dataset')
    truth=read_rows(root/'training/train_ground_truth.csv');reference=read_rows('outputs/stage7_tiles/native/predicted_all.csv')
    out=Path('outputs/stage8_iv');out.mkdir(exist_ok=True);allrows=[]
    for tag,subject in FOLDS.items():
        folder=out/tag;folder.mkdir(exist_ok=True)
        if (folder/'model.pt').exists():raise FileExistsError('Preserving larger IV model')
        items,ids=load_training(root,subject);items=[r for r in items if r['mode']==0]
        if any(r['subject']==subject for r in items):raise ValueError('Training leakage')
        torch.manual_seed(2032);rng=np.random.default_rng(2032);model=SmallUNet(24).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,3000,eta_min=.0001)
        config={'held_out':subject,'train_ids':ids,'width':24,'steps':3000,'seed':2032,'patch':128,
            'mode':'IV only, native resolution, subject-balanced baseline sampling','parameters':sum(p.numel() for p in model.parameters()),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (folder/'training_config.json').write_text(json.dumps(config,indent=2)+'\n');model.train();history=[];start=time.perf_counter()
        for step in range(1,3001):
            x,y=sample_batch(items,8,128,rng);optimizer.zero_grad(set_to_none=True)
            loss=loss_function(model(x.to(device)),y.to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();scheduler.step();value=float(loss.detach().cpu())
            if not np.isfinite(value):raise RuntimeError('Nonfinite loss')
            if step==1 or step%200==0:
                history.append({'step':step,'loss':value,'seconds':time.perf_counter()-start});print('Larger IV',tag,history[-1],flush=True)
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'width':24,'held_out':subject,'train_ids':ids,'config':config},folder/'model.pt')
        (folder/'history.json').write_text(json.dumps(history,indent=2)+'\n');del items;model.eval()
        cache=folder/'maps';cache.mkdir(exist_ok=True);development=[];heldout=[]
        for sid in truth:
            image=read_image(image_path(root,'training',sid,'invivo'));maps=probability_maps(model,image,0,device,tile=128,overlap=64,batch_size=8)
            np.savez_compressed(cache/f'{sid}_invivo.npz',maps=maps.astype(np.float16))
            row={'sample_id':sid,'invivo_instances':labels_to_instances(instances_from_maps(maps,.5,.35),'IVP'),
                 'exvivo_instances':reference[sid]['exvivo_instances'] if split_id(sid)[0]==subject else {},'match_pairs':[]}
            (heldout if split_id(sid)[0]==subject else development).append(row)
            print('Larger IV inference',tag,sid,flush=True)
        write_rows(development,folder/'development.csv');write_rows(heldout,folder/'heldout.csv');allrows.extend(heldout)
        evaluate(SimpleNamespace(data=root,predictions=folder/'heldout.csv',subject=subject,output=folder/'heldout_metrics.json'))
        del model;torch.mps.empty_cache()
    path=out/'segmentation_oof.csv';write_rows(allrows,path)
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
