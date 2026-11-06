import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.stoppage import stoppage_dead  # noqa: E402

N = 2000


def run(gap, speed=3.0, sp=(), out=None, graded=None, observed=None):
    """gap = (start, end) exclusive of missing ball frames."""
    obs = np.ones(N, bool) if observed is None else observed
    if gap:
        obs[gap[0]:gap[1]] = False
    out = np.zeros(N, bool) if out is None else out
    graded = np.ones(N, bool) if graded is None else graded
    return stoppage_dead(obs, out, np.full((N, 3), speed), np.array(sp, dtype=int), graded, {1: 0})


class Stoppage(unittest.TestCase):
    def test_set_piece_before_gap(self):
        d = run((500, 600), sp=[450])  # 2 s before start is within sp_before
        self.assertTrue(d[500:600].all() and not d[499] and not d[600])

    def test_set_piece_after_gap(self):
        self.assertTrue(run((500, 600), sp=[699])[500:600].all())  # ~4 s after end
        self.assertFalse(run((500, 600), sp=[799])[500:600].any())  # 8 s after end

    def test_quiet_long_gap(self):
        self.assertTrue(run((500, 800), speed=0.5)[500:800].all())  # 12 s, slow
        self.assertFalse(run((500, 800), speed=3.0)[500:800].any())  # 12 s, fast
        self.assertFalse(run((500, 700), speed=0.5)[500:700].any())  # 8 s, slow

    def test_short_out_run_ignored(self):
        out = np.zeros(N, bool)
        out[300:303] = True
        self.assertFalse(run(None, out=out)[300:303].any())
        out[500:506] = True
        self.assertTrue(run(None, out=out)[500:506].all())

    def test_pre_kickoff_dead(self):
        graded = np.ones(N, bool)
        graded[:100] = False
        d = stoppage_dead(np.ones(N, bool), np.zeros(N, bool), np.full((N, 3), 3.0),
                          np.array([], int), graded, {1: 300})
        self.assertTrue(d[100:300].all() and not d[300:].any() and not d[:100].any())

    def test_gap_does_not_spill_across_ungraded(self):
        graded = np.ones(N, bool)
        graded[1000:1100] = False
        obs = np.ones(N, bool)
        obs[900:1000] = False  # two 100-frame gaps split by ungraded frames: each too short
        obs[1100:1200] = False
        self.assertFalse(run(None, speed=0.5, graded=graded, observed=obs).any())


if __name__ == "__main__":
    unittest.main()
