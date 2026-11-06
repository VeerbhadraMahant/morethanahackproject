import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402
from synx.ball_possession import control  # noqa: E402
from synx.missing_ball import ball_speed, fill_ball  # noqa: E402
from synx.pipeline import predict  # noqa: E402
from synx.pitch import ball_out, kickoff_team, set_piece_starts  # noqa: E402
from synx.spells import build_spells, graded_mask  # noqa: E402
from synx.stoppage import stoppage_dead  # noqa: E402

N = 199319  # predict grades against the real game's blocks, so the fake dataset has full length


def fake():
    t = np.arange(N)
    ball = np.stack([50 + 10 * np.sin(t / 500), np.full(N, 34.0)], axis=1).astype(np.float32)
    ball[20000:20300] = np.nan
    ball[70000:70100, 0] = 110.0  # leaves the pitch
    players = np.zeros((N, 4, 2), dtype=np.float32)  # slots 0-1 team 0, 2-3 team 1, far unless on the ball
    players[:, :, 1] = 0.0
    on = ball + np.float32(0.5)
    players[:50000, 0] = on[:50000]
    players[50000:, 2] = on[50000:]
    data = {"ball": ball, "players": players, "team": np.array([0, 0, 1, 1], np.int8),
            "pitch": np.array([105.3, 68.0]), "player_speed": np.full((N, 4), 2.0, np.float32)}
    tags = [
        {"code": "SET PIECES", "start": 11185, "end": 11300, "team": 0, "type": "Kick Off (Start)", "half": 1},
        {"code": "SET PIECES", "start": 113742, "end": 113800, "team": 1, "type": "Kick Off (Start)", "half": 2},
        {"code": "SET PIECES", "start": 20100, "end": 20200, "team": 1, "type": "Throw In", "half": 1},
        {"code": "BUILD UP", "start": 30000, "end": 30500, "team": 0, "type": None, "half": 1},
    ]
    return data, tags


class Predict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.tags = fake()

    def test_v2_equals_old_chain(self):
        data, tags = self.data, self.tags
        ball, _ = fill_ball(data["ball"])
        ctrl = control(ball, data["players"], data["team"], ball_speed(ball))
        sp_frames, sp_teams = set_piece_starts(tags)
        graded = graded_mask()
        out = ball_out(data["ball"], data["pitch"])
        dead = stoppage_dead(~np.isnan(data["ball"]).any(axis=1), out, data["player_speed"], sp_frames,
                             graded, KICKOFF_VIDEO_FRAMES)
        expected = build_spells(ctrl, dead, graded, kickoff_team(tags), (sp_frames, sp_teams))
        self.assertTrue(np.array_equal(predict(data, tags, decoder="v2"), expected))

    def test_hmm_is_valid_labelling(self):
        v2 = predict(self.data, self.tags, decoder="v2")
        hmm = predict(self.data, self.tags, decoder="hmm")
        graded = graded_mask()
        self.assertEqual(hmm.shape, (N,))
        self.assertTrue(np.array_equal(hmm == "X", ~graded))  # X exactly on ungraded frames
        self.assertTrue(np.array_equal(hmm == "D", v2 == "D"))  # dead frames are the same
        self.assertEqual(hmm[15000], "H")  # team 0 on the ball
        self.assertEqual(hmm[60000], "A")  # team 1 on the ball

    def test_unknown_decoder(self):
        with self.assertRaises(ValueError):
            predict(self.data, self.tags, decoder="nope")


if __name__ == "__main__":
    unittest.main()
