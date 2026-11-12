"""Transfer conservative rejection decisions onto recalibrated mask boundaries.

Uses only predicted masks. A recalibrated instance is rejected only when its
IoU with a rejected original instance is > 0.5. Uncertain identities are kept.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from .data import read_rows,write_rows,parse_instances,image_path,image_shape
from .cli import evaluate


def transfer_rejections(original,retained,calibrated):
    rejected={k:m for k,m in original.items() if k not in retained}
    return {k for k,mask in calibrated.items() if any(mask.iou(old)>.5 for old in rejected.values())}


def main():
    data=Path('data/raw/Project_2_Dataset')
    original=read_rows('outputs/stage2/unet_oof.csv')
    retained=read_rows('outputs/stage4_conservative/quality_oof.csv')
    calibrated=read_rows('outputs/stage4_calibrated/segmentation_oof.csv')
    if not set(original)==set(retained)==set(calibrated):raise ValueError('Prediction ID mismatch')
    result=[];diagnostics=[]
    for sid,row in calibrated.items():
        shape=image_shape(image_path(data,'training',sid,'exvivo'))
        before=parse_instances(original[sid],'exvivo',shape)
        after=parse_instances(row,'exvivo',shape)
        removed=transfer_rejections(before,retained[sid]['exvivo_instances'],after)
        value=copy.deepcopy(row);value['exvivo_instances']={k:v for k,v in row['exvivo_instances'].items() if k not in removed}
        result.append(value);diagnostics.append({'sample_id':sid,'removed_calibrated_instances':len(removed)})
    folder=Path('outputs/stage4_combined');folder.mkdir(exist_ok=True)
    path=folder/'segmentation_oof.csv';write_rows(result,path)
    (folder/'manifest.json').write_text(json.dumps({'identity_iou_threshold':.5,'policy':'Keep uncertain identities','regions':diagnostics},indent=2)+'\n')
    evaluate(SimpleNamespace(data=data,predictions=path,subject=None,output=folder/'segmentation_oof_metrics.json'))


if __name__=='__main__':main()
