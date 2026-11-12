"""Known-location recovery under a controlled degradation of supplied data."""

from __future__ import annotations

import csv
from pathlib import Path
import unittest

import cv2
import numpy as np

from nocpt.data import parse_cell
from nocpt.matching import coarse_search, fine_search


class SyntheticMatchingTests(unittest.TestCase):
    def test_recovers_degraded_training_crop(self) -> None:
        root = Path("participant")
        with (root / "train_ground_truth.csv").open(newline="") as handle:
            row = next(item for item in csv.DictReader(handle)
                       if item["Id"] == "taurus")
        truth = next(value for number in range(1, int(row["n_patches"]) + 1)
                     if (value := parse_cell(row[f"patch_{number:02}"])) is not None
                     and value[2] == 1)
        x, y, _ = truth
        image = cv2.imread(str(root / "train" / "taurus" / "taurus_image.png"), 0)
        crop = image[y-16:y+16, x-16:x+16]
        transform = cv2.getRotationMatrix2D((15.5, 15.5), 17, 1.08)
        transform[:, 2] += (0.35, -0.4)
        query = cv2.warpAffine(crop, transform, (32, 32),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        yy, xx = np.mgrid[-16:16, -16:16]
        query = query.astype(np.float32) * (1 + 0.07*xx/16 - 0.04*yy/16)
        query = cv2.GaussianBlur(query, (0, 0), 0.8)
        generator = np.random.default_rng(6643)
        query = np.clip(query*0.92 + 4 + generator.normal(0, 2.5, (32, 32)),
                        0, 255).astype(np.uint8)
        half = cv2.resize(image, (1500, 1500), interpolation=cv2.INTER_AREA)
        coarse = coarse_search(half, query)
        fine = fine_search(image, query, coarse)
        self.assertTrue(any(np.hypot(candidate.x-x, candidate.y-y) <= 12
                            for candidate in fine))


if __name__ == "__main__":
    unittest.main()
