"""Evaluate existing EX local contour models while freezing global geometry."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .data import read_rows,write_rows,read_image,image_path,parse_instances,split_id
from .learning import SmallUNet,image_features
from .ex_boundary import refine
from .stage3 import FOLDS
from .cli import evaluate


def main():
    torch.set_num_threads(2);root=Path('data/raw/Project_2_Dataset');truth=read_rows(root/'training/train_ground_truth.csv')
    rows=read_rows('outputs/stage7_tiles/native/predicted_all.csv');results=[];diagnostics=[]
    for tag,subject in FOLDS.items():
        state=torch.load(f'outputs/stage6_boundary/{tag}/model.pt',map_location='cpu',weights_only=False)
        if state['held_out']!=subject or set(state['train_ids'])!={s for s in truth if split_id(s)[0]!=subject}:raise ValueError('Wrong boundary fold')
        model=SmallUNet(state['width']);model.load_state_dict(state['state_dict']);model.eval()
        for sid,row in rows.items():
            if split_id(sid)[0]!=subject:continue
            image=read_image(image_path(root,'training',sid,'exvivo'));masks=parse_instances(row,'exvivo',image.shape)
            instances,count=refine(masks,image_features(image),model,'cpu');value=copy.deepcopy(row);value['exvivo_instances']=instances;value['match_pairs']=[];results.append(value)
            diagnostics.append({'sample_id':sid,'updated':count,'total':len(masks)});print('Revisited boundary',sid,count,flush=True)
    out=Path('outputs/stage8_boundary_frozen');out.mkdir(exist_ok=True);path=out/'segmentation_oof.csv';write_rows(results,path)
    (out/'manifest.json').write_text(json.dumps({'models':'stage6_boundary','input':'stage7_tiles/native','regions':diagnostics},indent=2)+'\n')
    evaluate(SimpleNamespace(data=root,predictions=path,subject=None,output=out/'segmentation_oof_metrics.json'))

if __name__=='__main__':main()
