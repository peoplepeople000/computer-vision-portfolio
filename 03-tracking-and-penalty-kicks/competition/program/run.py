"""Run every trained component; cached inputs contain no predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier
from human_rgb_adaptation import DirectionHead
from sam_core import TokenHead

HERE = Path(__file__).resolve().parent
SEEDS = (6643, 6644, 6645)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(value):
    value = np.asarray(value, np.float64)
    value = np.exp(value - value.max(-1, keepdims=True))
    return value / value.sum(-1, keepdims=True)


def tree(path, features, prior):
    model = CatBoostClassifier()
    model.load_model(str(path))
    assert model.tree_count_ == 450
    return softmax(model.predict(features, prediction_type='RawFormulaVal', thread_count=4) + prior)


def predict(features, models):
    features, models = Path(features), Path(models)
    a = dict(np.load(features / 'private_trees.npz', allow_pickle=False))
    masks = dict(np.load(features / 'availability.npz', allow_pickle=False))
    sam = dict(np.load(features / 'sam_private.npz', allow_pickle=False))
    clips = a['clips']; n = len(clips)
    assert n == 900 and len(set(clips)) == n
    np.testing.assert_array_equal(clips, masks['clips'])
    np.testing.assert_array_equal(clips, sam['clips'])
    assert set(a) == {'clips', 'keeper_static', 'keeper_late3', 'keeper_valid',
                      'fold0', 'fold2', 'fold0_body_valid', 'fold2_body_valid',
                      'fold0_field_valid', 'fold2_field_valid'}
    for name, value in a.items():
        if value.dtype.kind not in 'US':
            assert np.isfinite(value).all(), name
    priors = np.load(models / 'prior_tables.npz')
    geometry = []
    for bank in ['fold0', 'fold2']:
        margin = priors['kick'][a[bank][:, 186:189].argmax(1)]
        for seed in SEEDS:
            previous = None
            for level, width in [('2d', 189), ('body', 312), ('field', 435)]:
                p = tree(models / f'trees/{bank}_{level}/{seed}/model.cbm', a[bank][:, :width], margin)
                if level != '2d':
                    valid = a[f'{bank}_{level}_valid']
                    p[~valid] = previous[~valid]
                previous = p
            geometry.append(previous)
    geometry = np.mean(geometry, axis=0)
    keeper_margin = priors['keeper'][a['keeper_static'][:, 186:189].argmax(1)]
    eligible = (sam['mask'].sum(1) >= 3) & sam['mask'][:, -1]
    kick, keep, base, sam_probs = [], [], [], []
    torch.set_num_threads(4)
    for seed in SEEDS:
        state = torch.load(models / f'rgb/{seed}/checkpoint.pt', map_location='cpu', weights_only=True)
        head = DirectionHead().eval(); head.load_state_dict(state['head'], strict=True)
        rgb_features = np.load(features / f'rgb_{seed}.npy')
        assert rgb_features.shape == (n, 5120)
        with torch.inference_mode():
            # Preserve the submitted classifier's one-observation arithmetic.
            logits = np.array([head(torch.from_numpy(rgb_features[i:i+1].copy()),
                                     torch.from_numpy(masks['rgb_available'][i:i+1].copy())).numpy()[0]
                               for i in range(n)])
        rgb_p = softmax(logits)
        rgb_p[masks['flips']] = rgb_p[masks['flips']][:, [2, 1, 0]]
        base_kick = .75 * geometry + .25 * rgb_p
        base_kick[~masks['rgb_available']] = geometry[~masks['rgb_available']]
        base.append(base_kick)
        saved = torch.load(models / f'sam/{seed}.pt', map_location='cpu', weights_only=True)
        temporal = TokenHead().eval(); temporal.load_state_dict(saved['head'], strict=True)
        values = []
        with torch.inference_mode():
            for off in range(0, n, 64):
                args = [torch.from_numpy(sam[k][off:off+64].copy()) for k in ['tokens', 'mask', 'times']]
                values.append(temporal(*args, torch.zeros(len(args[0]), 3), 'B_sequence').numpy())
        sam_p = softmax(np.concatenate(values))
        sam_p[masks['flips']] = sam_p[masks['flips']][:, [2, 1, 0]]
        sam_probs.append(sam_p)
        fused = .75 * base_kick + .25 * sam_p
        fused[~eligible] = base_kick[~eligible]
        kick.append(fused)
        kp = tree(models / f'trees/keeper/{seed}/model.cbm', a['keeper_late3'], keeper_margin)
        fallback = tree(models / f'trees/control/{seed}/model.cbm', a['keeper_static'], keeper_margin)
        kp[~a['keeper_valid']] = fallback[~a['keeper_valid']]
        keep.append(kp)
    probabilities = np.stack([np.mean(kick, 0), np.mean(keep, 0)], axis=1)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-7)
    return dict(clips=clips, probabilities=probabilities, prediction=probabilities.argmax(-1),
                kick_seed=np.array(kick), keeper_seed=np.array(keep), base_kick_seed=np.array(base),
                SAM_seed=np.array(sam_probs), eligible=eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=HERE / 'assets/features')
    parser.add_argument('--models', type=Path, default=HERE / 'assets/models')
    parser.add_argument('--out', type=Path, default=HERE.parent / 'output/reproduced')
    parser.add_argument('--verify', action='store_true', help='Compare outputs only after running inference')
    args = parser.parse_args()
    result = predict(args.features, args.models)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'submission.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'kick_direction', 'goalkeeper_direction'])
        for clip, (kick, keeper) in zip(result['clips'], result['prediction']):
            writer.writerow([clip, ['left', 'middle', 'right'][kick],
                             ['left', 'stands in the middle', 'right'][keeper]])
    np.savez_compressed(args.out / 'probabilities.npz', **result)
    report = dict(rows=len(result['clips']), models_executed=30, SAM_eligible=int(result['eligible'].sum()),
                  SAM_fallback=int((~result['eligible']).sum()), submission_sha256=sha(path),
                  private_labels_used=False, stored_predictions_used_as_input=False)
    if args.verify:
        contract = json.loads((HERE / 'assets/model_contract.json').read_text())
        assert sha(path) == contract['csv_sha256'], 'CSV differs from the submitted reference'
        ref = np.load(HERE.parent / 'output/reference_probabilities.npz')
        np.testing.assert_array_equal(result['clips'], ref['clips'])
        np.testing.assert_array_equal(result['prediction'], ref['prediction'])
        np.testing.assert_allclose(result['probabilities'], ref['probabilities'], atol=2e-6, rtol=2e-6)
        report.update(exact_submitted_csv=True,
                      max_probability_error=float(abs(result['probabilities'] - ref['probabilities']).max()))
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
