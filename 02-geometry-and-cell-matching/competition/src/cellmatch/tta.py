"""Four-view flip probability averaging for frozen held-out baseline models."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .learning import SmallUNet, choose_device, probability_maps, instances_from_maps
from .data import read_rows, read_image, image_path, write_rows, split_id
from .rle import labels_to_instances
from .stage3 import FOLDS
from .cli import evaluate


class FlipAverage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        probabilities = []
        for dims in ((), (-1,), (-2,), (-2, -1)):
            view = torch.flip(x, dims) if dims else x
            p = self.model(view).sigmoid()
            probabilities.append(torch.flip(p, dims) if dims else p)
        return torch.logit(torch.stack(probabilities).mean(0).clamp(1e-6, 1-1e-6))


def main():
    torch.set_num_threads(4)
    device = choose_device('mps')
    root = Path('data/raw/Project_2_Dataset'); output = Path('outputs/stage5_tta')
    cache = output/'maps'; cache.mkdir(parents=True, exist_ok=True)
    truth = read_rows(root/'training/train_ground_truth.csv'); rows = []; manifests = []
    for tag, heldout in FOLDS.items():
        path = Path(f'outputs/stage2/unet_holdout_{tag}/model.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        ids = [sid for sid in truth if split_id(sid)[0] == heldout]
        if checkpoint['held_out'] != heldout or set(ids) & set(checkpoint['train_ids']):
            raise ValueError('Invalid held-out checkpoint')
        model = SmallUNet(checkpoint['width']); model.load_state_dict(checkpoint['state_dict'])
        model = FlipAverage(model).to(device).eval()
        for sid in ids:
            row = {'sample_id': sid, 'match_pairs': []}
            for mode, modality in enumerate(('invivo', 'exvivo')):
                image = read_image(image_path(root, 'training', sid, modality))
                maps = probability_maps(model, image, mode, device)
                np.savez_compressed(cache/f'{sid}_{modality}.npz', maps=maps.astype(np.float16))
                row[modality+'_instances'] = labels_to_instances(instances_from_maps(maps, .5, .35), 'IVP' if mode == 0 else 'EXP')
            rows.append(row)
            print('TTA', sid, flush=True)
        manifests.append({'held_out': heldout, 'train_ids': checkpoint['train_ids'],
                          'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    path = output/'segmentation_oof.csv'; write_rows(rows, path)
    (output/'manifest.json').write_text(json.dumps({'views': ['identity', 'horizontal', 'vertical', 'both'],
        'combination': 'Mean probabilities before watershed', 'threshold': .5, 'seed_threshold': .35,
        'folds': manifests, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+'\n')
    evaluate(SimpleNamespace(data=root, predictions=path, subject=None, output=output/'segmentation_oof_metrics.json'))


if __name__ == '__main__': main()
