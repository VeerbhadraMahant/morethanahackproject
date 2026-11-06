import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.hmm import decode_hmm, tag_evidence  # noqa: E402


def tag(code, start, end, team):
    return {"code": code, "start": start, "end": end, "team": team, "type": None, "half": None}


def dec(control, dead=None, graded=None, kt=None, sp=None, tags=None, **kw):
    control = np.array(control, dtype=np.int8)
    n = len(control)
    dead = np.zeros(n, bool) if dead is None else np.array(dead, bool)
    graded = np.ones(n, bool) if graded is None else np.array(graded, bool)
    sp = (np.empty(0, int), np.empty(0, int)) if sp is None else sp
    return "".join(decode_hmm(control, dead, graded, kt or {1: 0, 2: 0}, sp,
                              np.full(n, -1, np.int8) if tags is None else tags, **kw))


class TagEvidence(unittest.TestCase):
    def test_attacking_defending_and_ignored(self):
        tags = [tag("BUILD UP", 0, 3, 0), tag("DEFENDING IN MIDDLE THIRD", 3, 6, 0),
                tag("BALL IN THE BOX", 6, 8, None), tag("SET PIECES", 8, 10, 1),
                tag("DEFENSIVE TRANSITION", 10, 12, 1)]
        self.assertEqual(tag_evidence(tags, 14).tolist(), [0] * 3 + [1] * 3 + [-1] * 4 + [0] * 2 + [-1] * 2)

    def test_conflicting_tags_are_ambiguous(self):
        tags = [tag("BUILD UP", 0, 4, 0), tag("PROGRESSION", 2, 6, 1)]
        self.assertEqual(tag_evidence(tags, 6).tolist(), [0, 0, -1, -1, 1, 1])


class DecodeHmm(unittest.TestCase):
    def test_single_frame_flicker_does_not_switch(self):
        self.assertEqual(dec([0] * 5 + [1] + [0] * 5), "H" * 11)

    def test_sustained_opponent_control_switches_at_onset(self):
        self.assertEqual(dec([0] * 20 + [1] * 20), "H" * 20 + "A" * 20)

    def test_tag_alone_decides_missing_ball(self):
        n = 30
        tags = np.full(n, 1, np.int8)
        self.assertEqual(dec([-1] * n, kt={1: 0, 2: 0}, tags=tags, switch=0.0), "A" * n)

    def test_control_beats_contradicting_tag(self):
        n = 30
        tags = np.full(n, 1, np.int8)  # one second of tag per frame of control: 30 * 0.04 < 30
        self.assertEqual(dec([0] * n, tags=tags), "H" * n)

    def test_tag_decides_ball_missing_stretch_between_controls(self):
        # control H then A, tags say A through a 100-frame ball-missing hole: the switch moves into the hole
        control = [0] * 20 + [-1] * 100 + [1] * 20
        tags = np.array([-1] * 20 + [1] * 100 + [-1] * 20, np.int8)
        self.assertEqual(dec(control, tags=tags), "H" * 20 + "A" * 120)

    def test_kickoff_prior(self):
        self.assertEqual(dec([-1] * 5, kt={1: 1, 2: 0}), "AAAAA")

    def test_set_piece_prior_after_stoppage(self):
        sp = (np.array([8]), np.array([1]))
        out = dec([0] * 5 + [-1] * 5 + [-1] * 5, dead=[0] * 5 + [1] * 5 + [0] * 5, sp=sp)
        self.assertEqual(out, "HHHHHDDDDDAAAAA")

    def test_set_piece_unknown_team_is_no_prior(self):
        sp = (np.array([8]), np.array([-1]))
        out = dec([0] * 5 + [-1] * 5 + [-1] * 5, dead=[0] * 5 + [1] * 5 + [0] * 5, sp=sp)
        self.assertEqual(out, "HHHHHDDDDDHHHHH")

    def test_dead_and_ungraded(self):
        out = dec([0] * 5, dead=[0, 0, 1, 1, 0], graded=[0, 1, 1, 1, 1])
        self.assertEqual(out, "XHDDH")

    def test_no_x_on_graded_live_frames(self):
        control = np.random.default_rng(0).integers(-1, 2, 300)
        out = dec(control, graded=[0] * 10 + [1] * 290)
        self.assertEqual(out[:10], "X" * 10)
        self.assertEqual(set(out[10:]), {"H", "A"})

    def test_half_boundary_uses_each_halfs_kickoff(self):
        graded = [1] * 5 + [0] * 3 + [1] * 5
        self.assertEqual(dec([-1] * 13, graded=graded, kt={1: 0, 2: 1}), "HHHHHXXXAAAAA")


if __name__ == "__main__":
    unittest.main()
