"""Isolated official UNETR decoder loader, without GUI or segmentation packages.

Reference sources and licenses are preserved in vendor/microsam-apg-reference.
Only unused relative imports are omitted; decoder class bodies are unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path
import torch

REFERENCE = Path(__file__).resolve().parents[2] / 'vendor/microsam-apg-reference'


def official_classes():
    provenance = json.loads((REFERENCE/'provenance.json').read_text())
    for name, record in provenance.items():
        assert hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest() == record['sha256']
    namespace = {'__name__': 'isolated_official_unetr'}
    exec(compile((REFERENCE/'unet.py').read_text(), str(REFERENCE/'unet.py'), 'exec'), namespace)
    tree = ast.parse((REFERENCE/'unetr.py').read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    exec(compile(tree, str(REFERENCE/'unetr.py'), 'exec'), namespace)
    tree = ast.parse((REFERENCE/'instance_segmentation.py').read_text())
    adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DecoderAdapter')
    exec(compile(ast.Module(body=[adapter], type_ignores=[]), str(REFERENCE/'instance_segmentation.py'), 'exec'), namespace)
    return namespace['UNETR'], namespace['DecoderAdapter']


def load_decoder(image_encoder, checkpoint, device='cpu'):
    unetr_class, adapter_class = official_classes()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    transposed = any('.block.' in k for k in state if k.startswith('decoder.samplers'))
    model = unetr_class(backbone='sam', encoder=image_encoder, out_channels=3,
                        use_sam_stats=True, final_activation='Sigmoid',
                        use_skip_connection=False, resize_input=True,
                        use_conv_transpose=transposed)
    current = model.state_dict()
    assert {k for k in current if not k.startswith('encoder.')} == set(state)
    for key, value in state.items():
        assert current[key].shape == value.shape
        current[key] = value
    model.load_state_dict(current, strict=True)
    return adapter_class(model).to(device).eval()
