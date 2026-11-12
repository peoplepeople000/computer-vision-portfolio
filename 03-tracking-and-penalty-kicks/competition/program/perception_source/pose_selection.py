"""Select an observed pose within the supplied clip; never extrapolate motion."""
import numpy as np


def select_pose(pose, mask, timestamps, latest_observed):
    pose = np.asarray(pose, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    assert pose.shape == (3, 2, 23, 4) and mask.shape == (3, 2)
    assert timestamps.shape == (3,) and np.all(np.diff(timestamps) > 0)
    assert np.isfinite(pose).all() and np.isfinite(timestamps).all()
    selected = np.zeros((2, 23, 4), np.float32)
    ages = np.zeros(2, np.float32)
    indices = np.full(2, -1, np.int64)
    usable = mask & (pose[..., 3] > 0).any(-1)
    for role in range(2):
        candidates = np.flatnonzero(usable[:, role]) if latest_observed else np.flatnonzero(usable[:, role] & (np.arange(3) == 2))
        if len(candidates):
            index = candidates[-1]
            selected[role] = pose[index, role]
            ages[role] = timestamps[-1] - timestamps[index]
            indices[role] = index
    return np.r_[selected.flatten(), ages], indices
