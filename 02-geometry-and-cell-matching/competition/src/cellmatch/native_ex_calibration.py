"""Match EX development inference tiles to the frozen 128-pixel validation tiles.

Existing outer-fold networks are unchanged. Only development predictions and
training-only foreground/quality calibration are recomputed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .learning import SmallUNet, probability_maps, choose_device
from .data import read_rows, read_image, image_path, split_id, write_rows
from .stage3 import FOLDS
from .ex_capacity import calibrate
from .cli import evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--output', default='outputs/stage11_native_ex')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = choose_device(args.device)
    root = Path('data/raw/Project_2_Dataset')
    truth = read_rows(root/'training/train_ground_truth.csv')
    reference = read_rows('outputs/stage8_iv_quality_capacity/joint/predicted_all.csv')
    prior_audit = Path('outputs/stage8_iv_quality_capacity/joint/audit.json')
    audited_models = {m['path']: m['sha256'] for m in json.loads(prior_audit.read_text())['models']}
    out = Path(args.output); out.mkdir(exist_ok=True)
    results = []
    for tag, subject in FOLDS.items():
        checkpoint = Path(f'outputs/stage6_ex/unet_holdout_{tag}/model.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ids = sorted(s for s in truth if split_id(s)[0] != subject)
        if state['held_out'] != subject or set(state['train_ids']) != set(ids):
            raise ValueError('EX checkpoint training partition is invalid')
        if audited_models[str(checkpoint)] != digest(checkpoint):
            raise ValueError('Legacy EX checkpoint changed since validation audit')
        heldout_maps = {s: Path(f'outputs/stage7_tiles/maps/{s}_exvivo.npz')
                        for s in truth if split_id(s)[0] == subject}
        config = {'held_out': subject, 'train_ids': ids, 'checkpoint': str(checkpoint),
                  'checkpoint_sha256': digest(checkpoint), 'tile': 128, 'overlap': 64, 'batch_size': 8,
                  'device': device, 'purpose': 'Recalibration of existing EX specialist with consistent development/heldout tile sizes.',
                  'source_sha256': digest(__file__), 'learning_sha256': digest(Path(__file__).with_name('learning.py')),
                  'heldout_cache_sha256': {s: digest(p) for s, p in heldout_maps.items()},
                  'heldout_cache_provenance': 'stage7 tile_inference.py, unchanged stage6 models verified against stage8 audit'}
        folder = out/tag; folder.mkdir(exist_ok=True)
        config_path = folder/'training_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Changed native calibration configuration')
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        model_link = folder/'model.pt'
        if not model_link.exists():
            model_link.symlink_to(checkpoint.resolve())
        if digest(model_link) != digest(checkpoint):
            raise ValueError('Wrong EX model reference')
        cache = folder/'maps'; cache.mkdir(exist_ok=True)
        model = SmallUNet(state['width']).to(device)
        model.load_state_dict(state['state_dict']); model.eval()
        for sid in truth:
            destination = cache/f'{sid}_exvivo.npz'
            if destination.exists():
                continue
            if sid in heldout_maps:
                destination.symlink_to(heldout_maps[sid].resolve())
                continue
            image = read_image(image_path(root, 'training', sid, 'exvivo'))
            maps = probability_maps(model, image, 1, device, tile=128, overlap=64, batch_size=8)
            np.savez_compressed(destination, maps=maps.astype(np.float16))
            print('Native EX development', tag, sid, flush=True)
        del model
        if device == 'mps':
            torch.mps.empty_cache()
        if not (folder/'heldout.csv').exists():
            calibrate(folder, tag, truth, reference)
            evaluate(SimpleNamespace(data=root, predictions=folder/'heldout.csv', subject=subject, output=folder/'metrics.json'))
        results.extend(read_rows(folder/'heldout.csv').values())
    write_rows(results, out/'segmentation_oof.csv')
    evaluate(SimpleNamespace(data=root, predictions=out/'segmentation_oof.csv', subject=None, output=out/'segmentation_oof_metrics.json'))


if __name__ == '__main__':
    main()
