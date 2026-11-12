"""Checks for the published scoring formula and submission parser."""

from __future__ import annotations

import unittest

from nocpt.data import parse_cell
from nocpt.scoring import score_scene


class ScoringTests(unittest.TestCase):
    def make_row(self, first: str, second: str, name: str = "taurus") -> dict[str, str]:
        return {"Id": "example", "n_patches": "2", "patch_01": first,
                "patch_02": second, "constellation": name}

    def test_perfect_prediction(self) -> None:
        truth = self.make_row("(100, 100, 1)", "-1")
        self.assertEqual(score_scene(truth, truth)["score"], 1.0)

    def test_all_absent_and_wrong_name(self) -> None:
        truth = self.make_row("(100, 100, 1)", "-1")
        prediction = self.make_row("-1", "-1", "unknown")
        score = score_scene(truth, prediction)
        self.assertAlmostEqual(score["presence"], 1/3)
        self.assertEqual(score["localization"], 0)
        self.assertEqual(score["geometric_recovery"], 0)
        self.assertEqual(score["identification"], 0)

    def test_localization_ramp_and_membership_independence(self) -> None:
        truth = self.make_row("(100, 100, 1)", "-1")
        inside = self.make_row("(112, 100, 0)", "-1")
        halfway = self.make_row("(124, 100, 0)", "-1")
        outside = self.make_row("(136, 100, 0)", "-1")
        self.assertEqual(score_scene(truth, inside)["localization"], 1)
        self.assertEqual(score_scene(truth, inside)["geometric_recovery"], 1)
        self.assertEqual(score_scene(truth, halfway)["localization"], 0.5)
        self.assertEqual(score_scene(truth, outside)["localization"], 0)

    def test_one_to_one_geometry(self) -> None:
        truth = self.make_row("(100, 100, 1)", "(101, 100, 1)")
        prediction = self.make_row("(100, 100, 0)", "-1")
        self.assertEqual(score_scene(truth, prediction)["geometric_recovery"], 0.5)

    def test_lenient_cells(self) -> None:
        self.assertEqual(parse_cell("100 200 1"), (100, 200, 1))
        self.assertEqual(parse_cell("(100, 200)"), (100, 200, 0))
        self.assertIsNone(parse_cell("-1"))


if __name__ == "__main__":
    unittest.main()
