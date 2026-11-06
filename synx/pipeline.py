"""Tracking + tags -> labels: the full chain, no I/O."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402
from synx.ball_possession import control  # noqa: E402
from synx.hmm import decode_hmm, tag_evidence  # noqa: E402
from synx.missing_ball import ball_speed, fill_ball  # noqa: E402
from synx.pitch import ball_out, kickoff_team, set_piece_starts  # noqa: E402
from synx.spells import build_spells, graded_mask  # noqa: E402
from synx.stoppage import stoppage_dead  # noqa: E402


def predict(data, tags, decoder="v2"):
    """labels (N,) of 'H' 'A' 'D' 'X' from a load_tracking dict and a load_tags list."""
    ball, _ = fill_ball(data["ball"])
    speed = ball_speed(ball)
    ctrl = control(ball, data["players"], data["team"], speed)

    sp_frames, sp_teams = set_piece_starts(tags)
    graded = graded_mask()
    out = ball_out(data["ball"], data["pitch"])  # observed ball only: no interpolated exits
    dead = stoppage_dead(~np.isnan(data["ball"]).any(axis=1), out, data["player_speed"], sp_frames,
                         graded, KICKOFF_VIDEO_FRAMES)
    if decoder == "v2":
        return build_spells(ctrl, dead, graded, kickoff_team(tags), (sp_frames, sp_teams))
    if decoder == "hmm":
        return decode_hmm(ctrl, dead, graded, kickoff_team(tags), (sp_frames, sp_teams),
                          tag_evidence(tags, len(ctrl)))
    raise ValueError("decoder must be 'v2' or 'hmm'")
