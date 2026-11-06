"""Label-free scorecard: structure, robustness to lost ball, evidence agreement. No ground truth.

`python -m synx.scorecard` prints aggregates only, never rows.
"""

import numpy as np

from games import GAME, GAMES

FPS = GAMES[GAME]["fps"]


def _possession_lengths(labels):
    """Frame lengths of maximal runs of one team letter (D/X or a switch ends a possession)."""
    team = (labels == "H") | (labels == "A")
    edge = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1], [True])))
    return np.array([b - a for a, b in zip(edge[:-1], edge[1:]) if team[a]], dtype=int)


def structure(labels, graded, fps=FPS):
    """Dict: possessions, median_s, p10_s, short (< 2 s), dead_pct, switches_per_min."""
    lengths = _possession_lengths(labels)
    live = labels[graded]
    team = live[(live == "H") | (live == "A")]
    switches = int(np.sum(team[1:] != team[:-1]))
    secs = lengths / fps if len(lengths) else np.zeros(1)
    return dict(possessions=len(lengths), median_s=float(np.median(secs)),
                p10_s=float(np.percentile(secs, 10)), short=int(np.sum(secs < 2.0)),
                dead_pct=100.0 * float(np.mean(live == "D")),
                switches_per_min=switches / max(len(live) / fps / 60, 1e-9))


def drop_ball_chunks(data, graded, frac, seed, chunk=25):
    """Copy of data with the ball removed on random 1-second chunks (a ball hidden for about a
    second is what a real tracking loss looks like) until frac of graded ball-visible frames
    are gone. Windows are sampled without replacement; the input is not modified."""
    ball = data["ball"].copy()
    visible = graded & ~np.isnan(ball).any(axis=1)
    target = frac * visible.sum()
    removed = 0
    for s in np.random.default_rng(seed).permutation(np.flatnonzero(visible)):
        if removed >= target:
            break
        window = ball[s:s + chunk]
        removed += int(visible[s:s + chunk][~np.isnan(window).any(axis=1)].sum())
        ball[s:s + chunk] = np.nan
    return dict(data, ball=ball)


def label_change(a, b, graded):
    """% of graded frames whose label differs between a and b."""
    return 100.0 * float(np.mean(a[graded] != b[graded]))


def agreement(labels, ref, graded):
    """% of live (H/A) graded frames with ref >= 0 where the label letter equals ref's team."""
    mask = graded & ((labels == "H") | (labels == "A")) & (ref >= 0)
    letter = np.where(ref[mask] == 0, "H", "A")
    return 100.0 * float(np.mean(labels[mask] == letter)) if mask.any() else float("nan")


def _control(data):
    from synx.ball_possession import control
    from synx.missing_ball import ball_speed, fill_ball
    ball, _ = fill_ball(data["ball"])
    return control(ball, data["players"], data["team"], ball_speed(ball))


def main():
    from synx.data import load_tracking
    from synx.hmm import tag_evidence
    from synx.pipeline import predict
    from synx.pitch import load_tags
    from synx.spells import graded_mask

    data = load_tracking(GAME)
    tags = load_tags(GAME)
    graded = graded_mask(GAME)
    ctrl = _control(data)
    tag_team = tag_evidence(tags, len(graded))
    for decoder in ("v2", "hmm"):
        labels = predict(data, tags, decoder)
        s = structure(labels, graded)
        runs = [predict(drop_ball_chunks(data, graded, 0.10, seed), tags, decoder)
                for seed in (0, 1, 2)]
        change = np.mean([label_change(labels, r, graded) for r in runs])
        dposs = np.mean([structure(r, graded)["possessions"] - s["possessions"] for r in runs])
        print("== {} ==".format(decoder))
        print("possessions: {}".format(s["possessions"]))
        print("possession length median / p10 (s): {:.1f} / {:.1f}".format(s["median_s"], s["p10_s"]))
        print("possessions < 2 s: {}".format(s["short"]))
        print("graded frames dead: {:.1f}%".format(s["dead_pct"]))
        print("switches per minute: {:.2f}".format(s["switches_per_min"]))
        print("robustness: label change {:.2f}% of graded frames, possession count change {:+.1f}"
              .format(change, dposs))
        print("agreement with control (live, control >= 0): {:.1f}%".format(agreement(labels, ctrl, graded)))
        print("agreement with tag evidence (live, tagged){}: {:.1f}%".format(
            " [decoder input, not a fair comparison]" if decoder == "hmm" else "",
            agreement(labels, tag_team, graded)))


if __name__ == "__main__":
    main()
