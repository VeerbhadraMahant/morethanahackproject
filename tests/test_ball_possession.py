import unittest

import numpy as np

from synx.ball_possession import control, nearest_dist

TEAM = np.array([0, 0, 1, 1], np.int8)


def frame(ball, pts):
    return np.array([ball], float), np.array([pts], float)


class BallPossessionTest(unittest.TestCase):
    def test_nearest_dist(self):
        b, p = frame([0, 0], [[1, 0], [3, 0], [0, 2], [5, 0]])
        np.testing.assert_allclose(nearest_dist(b, p, TEAM), [[1, 2]])
        self.assertEqual(nearest_dist(b, p, TEAM).dtype, np.float32)

    def test_nan_ball_and_missing_team(self):
        b, p = frame([np.nan, np.nan], [[1, 0], [3, 0], [0, 2], [5, 0]])
        self.assertTrue(np.isnan(nearest_dist(b, p, TEAM)).all())
        b, p = frame([0, 0], [[1, 0], [3, 0], [np.nan, np.nan], [np.nan, np.nan]])
        d = nearest_dist(b, p, TEAM)
        self.assertEqual(d[0, 0], 1)
        self.assertTrue(np.isnan(d[0, 1]))
        self.assertEqual(control(b, p, TEAM, np.array([1.0]))[0], 0)

    def test_control_basic(self):
        b, p = frame([0, 0], [[1, 0], [9, 0], [3, 0], [9, 9]])
        self.assertEqual(control(b, p, TEAM, np.array([2.0]))[0], 0)
        self.assertEqual(control(b, p, TEAM, np.array([np.nan]))[0], 0)  # NaN speed allowed

    def test_radius(self):
        b, p = frame([0, 0], [[2, 0], [9, 0], [9, 9], [9, 8]])
        self.assertEqual(control(b, p, TEAM, np.array([1.0]))[0], -1)

    def test_duel(self):
        b, p = frame([0, 0], [[1, 0], [9, 0], [1.3, 0], [9, 9]])
        self.assertEqual(control(b, p, TEAM, np.array([1.0]))[0], -1)

    def test_fast_ball(self):
        b, p = frame([0, 0], [[1, 0], [9, 0], [3, 0], [9, 9]])
        self.assertEqual(control(b, p, TEAM, np.array([12.0]))[0], -1)

    def test_nan_ball_control(self):
        b, p = frame([np.nan, np.nan], [[1, 0], [9, 0], [3, 0], [9, 9]])
        self.assertEqual(control(b, p, TEAM, np.array([1.0]))[0], -1)


if __name__ == "__main__":
    unittest.main()
