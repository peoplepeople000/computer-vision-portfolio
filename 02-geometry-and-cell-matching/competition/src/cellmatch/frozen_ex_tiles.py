"""Exact native Cellpose normalization/tiling for selected development tiles."""
import numpy as np
from .frozen_dino_readout import tile_layout,tile_pixels


def native_tiles(image):
    from cellpose import transforms,models
    converted=transforms.convert_image(image,channel_axis=None,z_axis=None,do_3D=False)
    assert converted.ndim==3
    normalized=transforms.normalize_img(converted[None],**dict(models.normalize_default))[0]
    layout=tile_layout(image.shape)
    return tile_pixels(normalized.transpose(2,0,1),layout),layout


def select_training_tiles(target_tiles,rng,limit=4):
    """Two foreground-containing tiles plus two spatially uniform remaining tiles."""
    count=len(target_tiles);positive=np.flatnonzero(target_tiles[:,0].reshape(count,-1).any(1))
    chosen=rng.choice(positive,min(2,len(positive)),replace=False).tolist()
    remaining=np.array([i for i in range(count) if i not in chosen])
    chosen.extend(rng.choice(remaining,min(limit-len(chosen),len(remaining)),replace=False).tolist())
    return sorted(chosen)
