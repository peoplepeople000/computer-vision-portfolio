"""Check the direction and accuracy of NCC peak interpolation."""

import unittest

import cv2
import numpy as np

from nocpt.matching import Candidate
from nocpt.subpixel import refine_center


class SubpixelTests(unittest.TestCase):
    def test_fractional_translation(self) -> None:
        rng = np.random.default_rng(6643)
        patch = cv2.GaussianBlur(
            rng.uniform(30, 180, (32, 32)).astype(np.float32), (0, 0), 0.7
        ).astype(np.uint8)
        shift_x, shift_y = 0.3, -0.2
        matrix = np.array([[1, 0, 34+shift_x], [0, 1, 34+shift_y]],
                          dtype=np.float32)
        image = cv2.warpAffine(patch, matrix, (100, 100),
                               flags=cv2.INTER_LINEAR)
        candidate = Candidate(50, 50, 0, 1, 0.9, "raw", fine_score=0.9)
        result = refine_center(image, patch, candidate)
        self.assertLess(abs(result.x-(50+shift_x)), 0.12)
        self.assertLess(abs(result.y-(50+shift_y)), 0.12)


if __name__ == "__main__":
    unittest.main()
