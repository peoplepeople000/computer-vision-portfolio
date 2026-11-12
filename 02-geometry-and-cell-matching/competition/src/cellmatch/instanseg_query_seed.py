"""Fixed image-only nearest official cell seed peak within a query prior."""
import numpy as np
import torch
from instanseg.utils.loss.instanseg_loss import torch_peak_local_max
from instanseg.utils.tiling import _instanseg_padding, _recover_padding


def nearest_cell_seed(model, image96, prior32, centroid_seed):
    """Keep official preprocessing/peak policy; never use query annotations."""
    assert image96.shape == (1, 96, 96) and prior32.shape == (32, 32)
    original = np.asarray(centroid_seed, np.int64)
    with torch.inference_mode():
        image = image96[None].clone().clamp_(-2, 3)
        padded, pad = _instanseg_padding(image, extra_pad=0)
        fields = _recover_padding(model.fcn(padded)[0, 5:10], pad)
        assert fields.shape == (5, 96, 96)
        seed_map = fields[4] / 15 + .5
        # TorchScript exposes ScalarType::Long as integer4 in Python.
        assert model.index_dtype == 4 or model.index_dtype == torch.int64
        peaks = torch_peak_local_max(seed_map, neighbourhood_size=int(model.default_peak_distance),
                                    minimum_value=float(model.default_seed_threshold), dtype=torch.int64)
    peaks = peaks.cpu().numpy().astype(np.int64)
    expanded = np.repeat(np.repeat(prior32.astype(bool), 3, axis=0), 3, axis=1)
    eligible = [p for p in peaks if expanded[p[0], p[1]] and int(((p - original) ** 2).sum()) <= 12**2]
    if not eligible:
        return original, dict(fallback=True, full_field_peaks=len(peaks), eligible_peaks=0, shift_native=0.)
    ranked = sorted(eligible, key=lambda p: (int(((p - original) ** 2).sum()),
                                          -float(seed_map[p[0], p[1]]), int(p[0]), int(p[1])))
    chosen = ranked[0]
    return chosen, dict(fallback=False, full_field_peaks=len(peaks), eligible_peaks=len(eligible),
                        shift_native=float(np.linalg.norm(chosen - original) / 3))
