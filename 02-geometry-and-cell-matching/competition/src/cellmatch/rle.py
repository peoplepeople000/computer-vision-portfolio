"""Zero-based, C-order run-length encoding (not COCO/Fortran RLE)."""
from dataclasses import dataclass
from functools import cached_property
import numpy as np


def encode(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("A mask must be two-dimensional")
    edges = np.flatnonzero(np.diff(np.r_[False, mask.ravel(order="C"), False]))
    return " ".join(f"{start} {end-start}" for start, end in edges.reshape(-1, 2))


@dataclass
class Mask:
    shape: tuple
    starts: np.ndarray
    ends: np.ndarray

    @classmethod
    def parse(cls, value, shape):
        if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
            raise ValueError("Expected positive (height, width)")
        if not isinstance(value, str):
            raise ValueError("RLE must be a string")
        try:
            values = np.array([int(v) for v in value.split()], dtype=np.int64)
        except (ValueError, OverflowError) as error:
            raise ValueError("Invalid RLE integers") from error
        if len(values) % 2:
            raise ValueError("RLE must contain start/length pairs")
        starts, lengths = values[::2], values[1::2]
        size = int(shape[0]) * int(shape[1])
        if (np.any(starts < 0) or np.any(starts >= size) or np.any(lengths <= 0)
                or np.any(lengths > size - starts)):
            raise ValueError("RLE run outside image or nonpositive length")
        ends = starts + lengths
        if np.any(starts[1:] < ends[:-1]):
            raise ValueError("RLE runs overlap or are unsorted")
        return cls(tuple(shape), starts, ends)

    @cached_property
    def area(self):
        return int(np.sum(self.ends - self.starts))

    @cached_property
    def bbox(self):
        if not self.area:
            return (0, 0, 0, 0)
        width = self.shape[1]
        first_rows = self.starts // width
        last_rows = (self.ends - 1) // width
        if np.any(first_rows != last_rows):
            xmin, xmax = 0, width - 1
        else:
            xmin, xmax = int((self.starts % width).min()), int(((self.ends - 1) % width).max())
        return (xmin, int(first_rows.min()), xmax, int(last_rows.max()))

    def decode(self):
        flat = np.zeros(int(np.prod(self.shape)), dtype=bool)
        for start, end in zip(self.starts, self.ends):
            flat[start:end] = True
        return flat.reshape(self.shape)

    def coordinates(self):
        indices = (np.concatenate([np.arange(s, e) for s, e in zip(self.starts, self.ends)])
                   if self.area else np.array([], dtype=np.int64))
        return np.column_stack((indices % self.shape[1], indices // self.shape[1]))

    def iou(self, other):
        if self.shape != other.shape:
            raise ValueError("Mask dimensions differ")
        if not self.area or not other.area:
            return 0.0
        a, b = self.bbox, other.bbox
        if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
            return 0.0
        if self.ends[-1] <= other.starts[0] or other.ends[-1] <= self.starts[0]:
            return 0.0
        i = j = intersection = 0
        while i < len(self.starts) and j < len(other.starts):
            intersection += max(0, min(self.ends[i], other.ends[j]) - max(self.starts[i], other.starts[j]))
            if self.ends[i] <= other.ends[j]:
                i += 1
            else:
                j += 1
        return float(intersection / (self.area + other.area - intersection))


def labels_to_instances(labels, prefix):
    """Encode each component from its bounding box; avoid N full-image scans."""
    from scipy.ndimage import find_objects
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Expected a nonnegative integer label image")
    result = {}
    width = labels.shape[1]
    for label, box in enumerate(find_objects(labels), 1):
        if box is None:
            continue
        local = labels[box] == label
        runs = []
        for y, row in enumerate(local):
            edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
            for start, end in edges.reshape(-1, 2):
                runs.extend(((y + box[0].start) * width + box[1].start + int(start), int(end-start)))
        result[f"{prefix}_{len(result)+1:06d}"] = " ".join(map(str, runs))
    return result
