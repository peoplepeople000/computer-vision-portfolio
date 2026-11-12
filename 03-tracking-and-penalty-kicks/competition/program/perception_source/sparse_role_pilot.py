"""Static, pose-free role selection from existing detector boxes; no inference rerun."""
import argparse
import hashlib
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import ball_goal_pilot as baseline
import rgb_gru_data as data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/kaggle_backbone/sparse_role_pilot'
CONFIG = {'version': 1, 'sample_sources': 24, 'sample_salt': 'B1-static-6643',
          'challenge_clips': ['1012_public.mp4', '0642_public.mp4', '0649_public.mp4', '0363_public.mp4'],
          'methods': ['per_anchor', 'final_objects'],
          'role_config': {k: v for k, v in baseline.ROLE_CONFIG.items() if k != 'tracking'},
          'person_proxy': 'bounding-box bottom centre; discard all pose fields',
          'sampling': 'reuse existing three tail anchors; no CSRT or motion-based ranking',
          'scope': 'training folds only; historical detector-pilot sources; not unseen-source accuracy'}


def select(anchors, method):
    if method not in CONFIG['methods']:
        raise ValueError(method)
    if not anchors or any(b['timestamp'] <= a['timestamp'] for a, b in zip(anchors, anchors[1:])):
        raise ValueError('Expected chronological anchors')
    if len({(a['width'], a['height']) for a in anchors}) != 1:
        raise ValueError('Coordinate reuse requires equal frame dimensions')
    result = []
    for anchor in anchors:
        clean = {key: anchor[key] for key in ['width', 'height']}
        clean['objects'] = (anchors[-1] if method == 'final_objects' else anchor)['objects']
        clean['persons'] = [{'bbox': p['bbox']} for p in anchor['persons']]
        candidates = baseline.role_candidates(clean)
        best = candidates[0] if candidates else None
        result.append({'frame_index': anchor['frame_index'], 'timestamp': anchor['timestamp'],
                       'selection': best,
                       'roles': {role: clean['persons'][best[role + '_index']]['bbox'] if best else None
                                 for role in ['kicker', 'keeper']}})
    return result


def sample(out):
    folds = pd.read_csv(data.DEFAULT_OUT / 'public_folds.csv')
    paths = list((ROOT / 'outputs/ball_goal_pilot/detections').glob('*.json'))
    paths += list((ROOT / 'outputs/keeper_geometry_pilot/fresh_baseline/ball_goal/detections').glob('*.json'))
    lookup = {p.name.removesuffix('.json'): p for p in paths}
    rows = folds[(folds.fold != 0) & folds['clip'].isin(lookup)].copy()
    rank = lambda value: hashlib.sha256((CONFIG['sample_salt'] + value).encode()).hexdigest()
    rows['rank'] = rows['clip'].map(rank)
    rows = rows.sort_values('rank').drop_duplicates('source_id').head(CONFIG['sample_sources'])
    rows['group'] = 'fixed_sample'
    extra = folds[(folds.fold != 0) & folds['clip'].isin(CONFIG['challenge_clips'])
                  & ~folds['clip'].isin(rows['clip'])].copy()
    extra['group'] = 'challenge_only'
    rows = pd.concat([rows, extra], ignore_index=True)
    result = rows[['clip', 'source_id', 'fold', 'fps', 'group']].copy()
    result['detection_path'] = result['clip'].map(lambda c: str(lookup[c].relative_to(ROOT)))
    result['detection_sha256'] = result['clip'].map(lambda c: data.sha(lookup[c]))
    data.immutable_json(out / 'protocol.json', {'config': CONFIG, 'sample': result.to_dict('records'),
                        'fold_sha256': data.sha(data.DEFAULT_OUT / 'public_folds.csv'),
                        'selector_sha256': data.sha(baseline.__file__), 'pilot_sha256': data.sha(__file__)})
    result.to_csv(out / 'sample.csv', index=False)
    return result


def annotated(frame, roles, title):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for role, color in [('kicker', '#18ed6c'), ('keeper', '#ffce32')]:
        box = roles[role]
        if box is not None:
            draw.rectangle(box, outline=color, width=4)
            draw.text((box[0], max(0, box[1] - 14)), role, fill=color)
    image.thumbnail((640, 360))
    tile = Image.new('RGB', (640, 390), '#f3f4f6')
    tile.paste(image, (0, 30)); ImageDraw.Draw(tile).text((8, 8), title, fill='black')
    return tile


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = sample(out)
    metrics, cards = [], []
    for row in rows.itertuples():
        cached = json.loads((ROOT / row.detection_path).read_text())
        anchors = sorted(cached['anchors'], key=lambda a: a['frame_index'])
        assert len(anchors) == 3
        frames = {i: frame for i, _, frame in baseline.read_tail(data.DEFAULT_DATA / 'train' / row.clip)}
        started = time.perf_counter()
        predictions = {method: select(anchors, method) for method in CONFIG['methods']}
        elapsed = time.perf_counter() - started
        data.dump(out / 'predictions' / (row.clip + '.json'),
                  {'clip': row.clip, 'detection_sha256': row.detection_sha256,
                   'selector_seconds': elapsed, 'predictions': predictions})
        sheet = Image.new('RGB', (1920, 780), 'white')
        for mi, (method, records) in enumerate(predictions.items()):
            for ai, record in enumerate(records):
                frame = frames[record['frame_index']]
                sheet.paste(annotated(frame, record['roles'],
                                     f'{method} | t={record["timestamp"]:.3f}'), (640 * ai, 390 * mi))
            metrics.append({'clip': row.clip, 'group': row.group, 'source_id': row.source_id,
                            'method': method, 'available_pairs': sum(r['selection'] is not None for r in records),
                            'anchors': len(records), 'selector_seconds': elapsed / 2})
        (out / 'examples').mkdir(exist_ok=True)
        sheet.save(out / 'examples' / (row.clip + '.jpg'), quality=90)
        # Actual native-resolution crops illustrate the information available to B0.
        last = predictions['per_anchor'][-1]
        rgb = cv2.cvtColor(frames[last['frame_index']], cv2.COLOR_BGR2RGB)
        crops = Image.new('RGB', (672, 250), '#edf0f3')
        for i, role in enumerate(['scene', 'kicker', 'keeper']):
            view = data.letterbox(rgb) if role == 'scene' else data.crop_view(rgb, last['roles'][role])
            if view is not None:
                crops.paste(Image.fromarray(view), (224 * i, 26))
            ImageDraw.Draw(crops).text((224 * i + 5, 6), role, fill='black')
        crops.save(out / 'examples' / (row.clip + '_crops.jpg'), quality=95)
        cards.append(f'<article><h2>{html.escape(row.clip)} · {row.group}</h2>'
                     f'<img src="examples/{row.clip}.jpg"><img class="crops" src="examples/{row.clip}_crops.jpg"></article>')
        print(row.clip, {k: sum(r['selection'] is not None for r in v) for k, v in predictions.items()}, flush=True)
    pd.DataFrame(metrics).to_csv(out / 'coverage.csv', index=False)
    coverage = pd.DataFrame(metrics).groupby(['group', 'method']).agg(
        available_pairs=('available_pairs', 'sum'), total_anchors=('anchors', 'sum')).reset_index().to_dict('records')
    data.dump(out / 'summary.json', {'clips': len(rows), 'coverage_not_accuracy': coverage,
              'new_neural_inference_calls': 0, 'used_pose': False, 'used_tracking': False,
              'note': 'Cached detector costs excluded. Manual identity review required; no classification scores yet.'})
    (out / 'gallery.html').write_text('''<!doctype html><meta charset="utf-8"><title>稀疏人物裁切 pilot</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;background:#f8fafc;color:#18212b}article{background:white;padding:20px;margin:24px 0;border:1px solid #ddd;border-radius:12px}img{max-width:100%;height:auto}.crops{display:block;width:672px}p{line-height:1.7}</style>
<h1>稀疏人物裁切：先檢查選人</h1><p>綠框＝踢球者；黃框＝守門員。上排：每個時刻的球／門框；下排：共用最後一幀的球／門框。最後一列是實際 224×224 全景與人物裁切。</p>
<p>固定 24 個訓練來源，另列既知困難案例。沿用既有偵測快取；不讀姿態或追蹤結果。框的可用率不代表選人正確率，舊 pilot 來源也不算全新驗證。</p>
''' + ''.join(cards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    run(parser.parse_args().out)
