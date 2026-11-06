import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.scorecard import agreement, drop_ball_chunks, label_change, structure  # noqa: E402


class StructureTest(unittest.TestCase):
    def test_counts(self):
        # H x100, A x25 (1 s), D x50, H x75, X x10 (ungraded)
        labels = np.array(list("H" * 100 + "A" * 25 + "D" * 50 + "H" * 75 + "X" * 10))
        graded = np.array([True] * 250 + [False] * 10)
        s = structure(labels, graded, fps=25)
        self.assertEqual(s["possessions"], 3)
        self.assertEqual(s["short"], 1)
        self.assertAlmostEqual(s["median_s"], 3.0)
        self.assertAlmostEqual(s["dead_pct"], 20.0)
        self.assertAlmostEqual(s["switches_per_min"], 2 / (250 / 25 / 60))


class DropTest(unittest.TestCase):
    def setUp(self):
        n = 2000
        self.ball = np.random.default_rng(5).random((n, 2)).astype(np.float32)
        self.ball[:100] = np.nan
        self.graded = np.ones(n, bool)
        self.graded[1500:] = False
        self.data = dict(ball=self.ball, players=np.zeros((n, 2, 2)))

    def test_removes_about_frac_deterministic_and_copies(self):
        orig = self.ball.copy()
        a = drop_ball_chunks(self.data, self.graded, 0.10, 0)
        b = drop_ball_chunks(self.data, self.graded, 0.10, 0)
        self.assertTrue(np.array_equal(a["ball"], b["ball"], equal_nan=True))
        self.assertTrue(np.array_equal(self.ball, orig, equal_nan=True))
        vis = self.graded & ~np.isnan(orig).any(axis=1)
        gone = vis & np.isnan(a["ball"]).any(axis=1)
        self.assertGreaterEqual(gone.sum(), 0.10 * vis.sum())
        self.assertLess(gone.sum(), 0.10 * vis.sum() + 51)  # overshoot < one chunk
        self.assertIs(a["players"], self.data["players"])
        self.assertFalse(np.array_equal(
            a["ball"], drop_ball_chunks(self.data, self.graded, 0.10, 1)["ball"], equal_nan=True))


class AgreementTest(unittest.TestCase):
    def test_label_change_and_agreement(self):
        a = np.array(list("HHAAXX"))
        b = np.array(list("HAAAXX"))
        g = np.array([True] * 4 + [False] * 2)
        self.assertAlmostEqual(label_change(a, b, g), 25.0)
        ref = np.array([0, 0, -1, 0, 1, 1], dtype=np.int8)
        # frames 0,1,3 qualify; labels H,H,A vs H,H,H -> 2/3
        self.assertAlmostEqual(agreement(a, ref, g), 100 * 2 / 3)


if __name__ == "__main__":
    unittest.main()
