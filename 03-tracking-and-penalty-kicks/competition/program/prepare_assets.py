"""Fetch pinned official pretrained resources outside the submission folder."""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(url, dest, expected):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix('.download')
        urllib.request.urlretrieve(url, tmp)
        if expected and sha(tmp) != expected:
            raise ValueError(f'Official asset checksum mismatch: {dest.name}')
        tmp.replace(dest)
    if expected and sha(dest) != expected:
        raise ValueError(f'Wrong asset: {dest.name}')
    return dest


def safe_extract(archive, dest):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe archive member')
        z.extractall(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='External pretrained-resource directory')
    parser.add_argument('--perception', action='store_true', help='Also fetch detector, pose, NLF and field calibration resources')
    parser.add_argument('--inspect', action='store_true', help='Print requirements without a download')
    args = parser.parse_args()
    spec = json.loads((HERE / 'assets/sam_contract.json').read_text())['feature_contract']
    official = spec['official_model']
    if args.inspect:
        print(json.dumps(dict(SAM_weights_bytes=official['checkpoint']['bytes'] + official['mhr']['bytes'],
                              gated_access_required=True, dataset_downloaded=False,
                              SAM_revision=official['huggingface_revision'], DINO_revision=spec['dinov3_revision']), indent=2))
        return
    from huggingface_hub import hf_hub_download
    args.out.mkdir(parents=True, exist_ok=True)
    # Uses an existing authorized login/environment token. Never prints it,
    # embeds it in URLs, accepts model terms or writes it into this package.
    for filename, expected in [('model.ckpt', official['checkpoint']['sha256']),
                               ('assets/mhr_model.pt', official['mhr']['sha256']),
                               ('model_config.yaml', spec['config_sha256'])]:
        target = args.out / filename
        if not target.exists():
            hf_hub_download(official['huggingface_repository'], filename,
                            revision=official['huggingface_revision'], local_dir=str(args.out))
        assert sha(target) == expected, filename
    flat = args.out / 'mhr_model.pt'
    if not flat.exists():
        os.link(args.out / 'assets/mhr_model.pt', flat)
    url = f'https://github.com/facebookresearch/sam-3d-body/archive/{official["source_revision"]}.zip'
    archive = fetch(url, args.out / 'sam_body_source.zip', spec['official_source_sha256'])
    safe_extract(archive, args.out / 'sam_body_source')
    roots = [p for p in (args.out / 'sam_body_source').iterdir() if (p / 'sam_3d_body').is_dir()]
    assert len(roots) == 1
    (args.out / 'sam_source_path.txt').write_text(str(roots[0].resolve()) + '\n')
    if args.perception:
        field = json.loads((HERE / 'assets/field_contract.json').read_text())
        for item in field['assets']:
            fetch(item['browser_download_url'], args.out / 'field/weights' / item['name'], field['weights'][item['name']])
        archive = fetch(f'https://github.com/mguti97/PnLCalib/archive/{field["upstream_commit"]}.zip',
                        args.out / 'pnlcalib_source.zip', None)
        safe_extract(archive, args.out / 'pnlcalib_source')
        fetch('https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.patch6.torchscript',
              args.out / 'nlf.torchscript', '2f9e377817acdd47e35224aeb70299b3f53ab574e0829372a094920515352a65')
        for name, url, expected in [
            ('yolox', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
             '3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23'),
            ('rtmw', 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip',
             '9a9bc17c13ff0a1e37507f45a037a21bc410cd08374b0e664c57d0080843aa58')]:
            archive = fetch(url, args.out / (name + '.zip'), None)
            with zipfile.ZipFile(archive) as z:
                members = [n for n in z.namelist() if n.endswith('.onnx')]
                assert len(members) == 1
                blob = z.read(members[0]); assert hashlib.sha256(blob).hexdigest() == expected
                (args.out / (name + '.onnx')).write_bytes(blob)
        checkpoints = args.out / 'models/rtmlib/hub/checkpoints'
        checkpoints.mkdir(parents=True, exist_ok=True)
        for source, name in [('yolox.onnx', 'yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),
                             ('rtmw.onnx', 'rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx')]:
            if not (checkpoints / name).exists():
                os.link(args.out / source, checkpoints / name)
        for name, expected in json.loads((HERE / 'assets/grounding_contract.json').read_text()).items():
            target = args.out / 'models/grounding-dino-tiny' / name
            if not target.exists():
                hf_hub_download('IDEA-Research/grounding-dino-tiny', name,
                                local_dir=str(target.parent))
            assert sha(target) == expected, 'Grounding DINO upstream changed; require the frozen checksum'
    print('Official resources verified. No competition videos were downloaded.')


if __name__ == '__main__':
    main()
