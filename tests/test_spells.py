import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.spells import build_spells, dead_ball, graded_mask  # noqa: E402


def lab(control, dead=None, graded=None, kt=None, sp=None, confirm=3):
    control = np.array(control, dtype=np.int8)
    n = len(control)
    dead = np.zeros(n, bool) if dead is None else np.array(dead, bool)
    graded = np.ones(n, bool) if graded is None else np.array(graded, bool)
    return "".join(build_spells(control, dead, graded, kt or {1: 0, 2: 0}, sp, confirm))


class BuildSpells(unittest.TestCase):
    def test_brief_opponent_touch_does_not_switch(self):
        self.assertEqual(lab([0, 0, 1, 1, 0, 0]), "HHHHHH")

    def test_sustained_switch_is_back_dated(self):
        # opponent first touches at index 3, confirmed at index 5
        self.assertEqual(lab([0, 0, -1, 1, -1, 1, 1, 1]), "HHHAAAAA")

    def test_dead_and_ungraded(self):
        out = lab([0, 0, 0, 0, 0], dead=[0, 0, 1, 1, 0], graded=[0, 1, 1, 1, 1])
        self.assertEqual(out, "XHDDH")

    def test_restart_team_from_set_piece(self):
        sp = (np.array([3]), np.array([1]))
        out = lab([0, 0, 0, -1, -1, 0], dead=[0, 0, 1, 0, 0, 0], sp=sp)
        self.assertEqual(out, "HHDAAA")

    def test_restart_team_first_control_filled_back(self):
        out = lab([0, 0, 0, -1, -1, 1, 1], dead=[0, 0, 1, 0, 0, 0, 0], sp=None)
        self.assertEqual(out, "HHDAAAA")

    def test_kickoff_team_at_half_start(self):
        self.assertEqual(lab([-1, -1, 0], kt={1: 1, 2: 0}), "AAA")

    def test_second_half_uses_its_kickoff_team(self):
        out = lab([-1, -1, -1, -1], graded=[1, 0, 1, 1], kt={1: 0, 2: 1})
        self.assertEqual(out, "HXAA")

    def test_no_x_on_graded_live_frames(self):
        out = lab([-1] * 6, graded=[1] * 6, kt={1: -1, 2: -1})
        self.assertNotIn("X", out)


class DeadBall(unittest.TestCase):
    def test_pre_kickoff_dead(self):
        graded = np.array([0, 1, 1, 1, 1, 1], bool)
        dead = dead_ball(np.zeros(6, bool), [], graded, {"1st Half": 4})
        self.assertEqual(dead.tolist(), [False, True, True, True, False, False])

    def test_out_run_ended_by_set_piece(self):
        out = np.array([0, 1, 1, 0, 0, 0, 0, 0], bool)
        dead = dead_ball(out, np.array([6]), np.ones(8, bool), {}, max_restart=10)
        self.assertEqual(dead.tolist(), [False, True, True, True, True, True, False, False])

    def test_out_run_ended_by_ball_back_when_set_piece_too_late(self):
        out = np.array([0, 1, 1, 0, 0, 0, 0, 0], bool)
        dead = dead_ball(out, np.array([7]), np.ones(8, bool), {}, max_restart=3)
        self.assertEqual(dead.tolist(), [False, True, True, False, False, False, False, False])


class Graded(unittest.TestCase):
    def test_mask_shape(self):
        m = graded_mask()
        self.assertEqual(len(m), 199319)
        self.assertTrue(m[11185] and not m[11184] and not m[100000])


if __name__ == "__main__":
    unittest.main()
