"""Full pipeline: tracking -> labels -> Synx.csv. Prints aggregates only, never rows."""

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from synx.data import load_tracking  # noqa: E402
from synx.pipeline import predict  # noqa: E402
from synx.pitch import load_tags  # noqa: E402
from synx.spells import graded_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", choices=("v2", "hmm"), default="hmm")
    labels = predict(load_tracking(GAME), load_tags(GAME), parser.parse_args().decoder)
    graded = graded_mask(GAME)

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
