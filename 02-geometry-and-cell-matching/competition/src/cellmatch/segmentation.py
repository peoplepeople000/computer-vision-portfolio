"""Untuned classical baseline: local contrast, blob seeds, watershed."""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


def normalize(image):
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(values, [1, 99.8])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def segment(image, config):
    normalized = normalize(image)
    smooth = ndi.gaussian_filter(normalized, config["smooth_sigma"])
    contrast = np.maximum(smooth - ndi.gaussian_filter(smooth, config["background_sigma"]), 0)
    response = np.zeros_like(contrast)
    for sigma in config["blob_sigmas"]:
        response = np.maximum(response, -sigma**2 * ndi.gaussian_laplace(contrast, sigma))
    foreground = contrast > config["foreground_threshold"]
    coordinates = peak_local_max(response, min_distance=config["min_distance"],
                                 threshold_abs=config["seed_threshold"],
                                 labels=foreground.astype(np.uint8), exclude_border=False)
    markers = np.zeros(image.shape, dtype=np.int32)
    if not len(coordinates):
        return markers
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    labels = watershed(-contrast, markers, mask=foreground)
    out = np.zeros_like(labels)
    next_id = 1
    for region in regionprops(labels):
        if config["min_area"] <= region.area <= config["max_area"] and region.eccentricity <= config["max_eccentricity"]:
            out[region.slice][region.image] = next_id
            next_id += 1
    return out
