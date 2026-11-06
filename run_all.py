"""Full pipeline: tracking -> labels -> Synx.csv. Prints aggregates only, never rows."""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402
from synx.ball_possession import control  # noqa: E402
from synx.data import load_tracking  # noqa: E402
from synx.missing_ball import ball_speed, fill_ball  # noqa: E402
from synx.pitch import ball_out, kickoff_team, load_tags, set_piece_starts  # noqa: E402
from synx.spells import build_spells, graded_mask  # noqa: E402
from synx.stoppage import stoppage_dead  # noqa: E402


def main():
    data = load_tracking(GAME)
    ball, _ = fill_ball(data["ball"])
    speed = ball_speed(ball)
    ctrl = control(ball, data["players"], data["team"], speed)

    tags = load_tags(GAME)
    sp_frames, sp_teams = set_piece_starts(tags)
    graded = graded_mask(GAME)
    out = ball_out(data["ball"], data["pitch"])  # observed ball only: no interpolated exits
    dead = stoppage_dead(~np.isnan(data["ball"]).any(axis=1), out, data["player_speed"], sp_frames,
                         graded, KICKOFF_VIDEO_FRAMES)
    labels = build_spells(ctrl, dead, graded, kickoff_team(tags), (sp_frames, sp_teams))

    path = os.path.join(ROOT, "Synx.csv")
    with open(path, "w", newline="") as f:
        f.write("frame,label\n")
        f.write("".join("{},{}\n".format(i, c) for i, c in enumerate(labels)))

    live = labels[graded]
    team = live[(live == "H") | (live == "A")]
    possessions = int(np.sum((labels == "H") & (np.concatenate((["X"], labels[:-1])) != "H"))
                      + np.sum((labels == "A") & (np.concatenate((["X"], labels[:-1])) != "A")))
    switches = int(np.sum(team[1:] != team[:-1]))
    fps = GAMES[GAME]["fps"]
    for letter in "HADX":
        print("{}: {}".format(letter, int(np.sum(labels == letter))))
    print("possessions: {}".format(possessions))
    print("mean possession length (s): {:.1f}".format(len(team) / fps / max(possessions, 1)))
    print("switches per minute: {:.2f}".format(switches / (len(live) / fps / 60)))
    print("graded frames dead: {:.1f}%".format(100.0 * np.mean(live == "D")))

    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path],
                   check=True)


if __name__ == "__main__":
    main()
