import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.pitch import ball_out, kickoff_team, load_tags, set_piece_starts  # noqa: E402


def inst(code, start, end, **tags):
    labels = "".join("<label><group>{}</group><text>{}</text></label>".format(k, v)
                     for k, v in tags.items())
    return "<instance><code>{}</code><start>{}</start><end>{}</end>{}</instance>".format(
        code, start, end, labels)


XML = "<file><ALL_INSTANCES>" + "".join([
    inst("SET PIECES", 100, 105, Team="Atletic Club", Type="Kick Off (Start)", Half="2nd Half"),
    inst("Set Pieces", 10, 15, team="barcelona", type="Kick Off (Start)", half="1st Half"),
    inst("SET PIECES", 50, 52, Team="N/A", Type="Throw In", Half="1st Half"),
    inst("BUILD UP", 20, 30, Team="Barcelona"),
]) + "</ALL_INSTANCES></file>"


class PitchTest(unittest.TestCase):
    def test_tags(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "tactical"))
            with open(os.path.join(d, "tactical", "BAR-ATH_tactical_smart_tagging.xml"), "w") as f:
                f.write(XML)
            tags = load_tags(data_dir=d)
        self.assertEqual(tags[0], {"code": "SET PIECES", "start": 2500, "end": 2625,
                                   "team": 1, "type": "Kick Off (Start)", "half": 2})
        self.assertEqual((tags[1]["code"], tags[1]["team"], tags[1]["half"]),
                         ("SET PIECES", 0, 1))
        self.assertIsNone(tags[2]["team"])
        self.assertIsNone(tags[3]["type"])
        self.assertIsNone(tags[3]["half"])

        frames, team = set_piece_starts(tags)
        self.assertEqual(frames.dtype, np.int64)
        self.assertEqual(team.dtype, np.int8)
        self.assertEqual(frames.tolist(), [250, 1250, 2500])
        self.assertEqual(team.tolist(), [0, -1, 1])
        self.assertEqual(kickoff_team(tags), {1: 0, 2: 1})

    def test_empty_and_unknown(self):
        frames, team = set_piece_starts([])
        self.assertEqual((len(frames), len(team)), (0, 0))
        self.assertEqual(kickoff_team([]), {1: -1, 2: -1})
        tags = [{"code": "SET PIECES", "start": 1, "end": 2, "team": None,
                 "type": "Kick Off (Start)", "half": 1}]
        self.assertEqual(kickoff_team(tags), {1: -1, 2: -1})

    def test_ball_out(self):
        pitch = np.array([105.3, 68.0])
        ball = np.array([[50, 30], [-0.4, 30], [-0.6, 30], [105.7, 10], [105.9, 10],
                         [50, 68.4], [50, 68.6], [50, -1], [np.nan, np.nan]])
        self.assertEqual(ball_out(ball, pitch).tolist(),
                         [False, False, True, False, True, False, True, True, False])


if __name__ == "__main__":
    unittest.main()
