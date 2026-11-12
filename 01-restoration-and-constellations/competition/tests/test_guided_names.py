"""Check the global evidence gate and membership update."""

import unittest

import numpy as np

from apply_guided_names import should_use_guide, update_membership


class GuidedNameTests(unittest.TestCase):
    def test_switch_requires_strong_independent_geometry(self):
        rankings = [
            {"pattern": "centaurus", "score": "6.4", "support": "13",
             "coverage_fraction": "0.76"},
            {"pattern": "crater", "score": "1.1", "support": "6",
             "coverage_fraction": "0.75"},
        ]
        self.assertTrue(should_use_guide("corona-australis", 0.2, 2, rankings))
        self.assertFalse(should_use_guide("corona-australis", 8.0, 6, rankings))
        self.assertFalse(should_use_guide("centaurus", 0.2, 2, rankings))
        rankings[0]["score"] = "1.5"
        self.assertFalse(should_use_guide("corona-australis", 0.2, 2, rankings))

    def test_membership_follows_guided_pose(self):
        row = {"n_patches": "3", "patch_01": "(10, 20, 0)",
               "patch_02": "(80, 80, 1)", "patch_03": "-1"}
        nodes = np.asarray([[10., 20.], [40., 50.]])
        pose = {"matrix": [[1., 0.], [0., 1.]], "translation": [0., 0.]}
        update_membership(row, nodes, pose)
        self.assertEqual(row["patch_01"], "(10, 20, 1)")
        self.assertEqual(row["patch_02"], "(80, 80, 0)")
        self.assertEqual(row["patch_03"], "-1")


if __name__ == "__main__":
    unittest.main()
