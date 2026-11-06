import unittest

import numpy as np

from synx.missing_ball import ball_speed, fill_ball


def track(n, gaps):
    b = np.stack([np.arange(n, dtype=np.float32), np.zeros(n, np.float32)], axis=1)
    for a, z in gaps:
        b[a:z] = np.nan
    return b


class FillBall(unittest.TestCase):
    def test_gap_of_max_gap_filled(self):
        b = track(40, [(5, 30)])  # 25 frames
        out, src = fill_ball(b, max_gap=25)
        self.assertTrue((src[5:30] == 1).all())
        np.testing.assert_allclose(out[5:30, 0], np.arange(5, 30))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(src.dtype, np.int8)

    def test_gap_over_max_gap_not_filled(self):
        b = track(40, [(5, 31)])  # 26 frames
        out, src = fill_ball(b, max_gap=25)
        self.assertTrue((src[5:31] == 2).all())
        self.assertTrue(np.isnan(out[5:31]).all())

    def test_leading_trailing_not_filled(self):
        b = track(20, [(0, 3), (17, 20)])
        out, src = fill_ball(b)
        self.assertTrue((src[:3] == 2).all() and (src[17:] == 2).all())
        self.assertTrue((src[3:17] == 0).all())

    def test_all_missing_and_single_obs(self):
        _, src = fill_ball(np.full((5, 2), np.nan))
        self.assertTrue((src == 2).all())
        b = np.full((5, 2), np.nan)
        b[2] = 1
        _, src = fill_ball(b)
        self.assertEqual(list(src), [2, 2, 0, 2, 2])

    def test_half_nan_row_is_missing(self):
        b = track(6, [])
        b[2, 1] = np.nan
        out, src = fill_ball(b)
        self.assertEqual(src[2], 1)
        self.assertAlmostEqual(float(out[2, 1]), 0.0)


class BallSpeed(unittest.TestCase):
    def test_constant_velocity(self):
        # 0.4 m/frame at 25 fps = 10 m/s
        p = np.stack([np.arange(20) * 0.4, np.zeros(20)], axis=1).astype(np.float32)
        s = ball_speed(p)
        np.testing.assert_allclose(s[2:-2], 10.0, rtol=1e-5)
        self.assertTrue(np.isnan(s[:2]).all() and np.isnan(s[-2:]).all())
        self.assertEqual(s.dtype, np.float32)

    def test_nan_propagates(self):
        p = np.zeros((10, 2), np.float32)
        p[5] = np.nan
        s = ball_speed(p)
        # windows touching frame 5 as an end: centres 3 and 7
        self.assertTrue(np.isnan(s[3]) and np.isnan(s[7]))
        self.assertFalse(np.isnan(s[5]))

    def test_short_array(self):
        self.assertTrue(np.isnan(ball_speed(np.zeros((3, 2)))).all())


if __name__ == "__main__":
    unittest.main()
