"""Smart Tagging set pieces and touchline / goal-line geometry."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from starter import load_smart_tagging  # noqa: E402

TEAMS = {"barcelona": 0, "atletic club": 1}


def _tag(event, group):
    """Tag text for a group (case-insensitive group name), stripped, or None."""
    for key, text in event["tags"].items():
        if (key or "").strip().lower() == group and text is not None:
            return text.strip()
    return None


def load_tags(game="BAR-ATH", data_dir=None):
    """Smart Tagging instances as {code, start, end, team, type, half}; frames, end exclusive."""
    out = []
    for e in load_smart_tagging.load(game, data_dir=data_dir):
        half = (_tag(e, "half") or "").lower()
        out.append({
            "code": (e["code"] or "").strip().upper(),
            "start": e["start_frame"],
            "end": e["end_frame"],
            "team": TEAMS.get((_tag(e, "team") or "").lower()),  # "N/A" / absent -> None
            "type": _tag(e, "type"),
            "half": int(half[0]) if half[:1] in ("1", "2") else None,
        })
    return out


def _set_pieces(tags):
    return sorted((t for t in tags if t["code"] == "SET PIECES"), key=lambda t: t["start"])


def set_piece_starts(tags):
    """(start frames int64, team int8 with -1 unknown) of SET PIECES, sorted by start."""
    sp = _set_pieces(tags)
    frames = np.array([t["start"] for t in sp], dtype=np.int64)
    team = np.array([-1 if t["team"] is None else t["team"] for t in sp], dtype=np.int8)
    return frames, team


def kickoff_team(tags):
    """{1: team, 2: team} of the 'Kick Off (Start)' set piece per half, -1 if not found."""
    out = {1: -1, 2: -1}
    for t in reversed(_set_pieces(tags)):  # reversed so the earliest tag wins
        if (t["type"] or "").lower() == "kick off (start)" and t["half"] in out:
            out[t["half"]] = -1 if t["team"] is None else t["team"]
    return out


def ball_out(ball, pitch, margin=0.5):
    """Ball observed and > margin m beyond a touchline/goal line (whole ball over the line
    plus tracking noise). NaN -> False."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return ((x < -margin) | (x > pitch[0] + margin)
                | (y < -margin) | (y > pitch[1] + margin))
