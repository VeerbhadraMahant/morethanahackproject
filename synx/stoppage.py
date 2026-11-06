"""Dead-ball detection from stoppage evidence: restart tags, quiet long gaps, ball out."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.spells import _runs, dead_ball  # noqa: E402


def stoppage_dead(observed, out, player_speed, set_piece_frames, graded, kickoffs,
                  sp_before=50, sp_after=125, long_gap=250, walk_speed=1.5, min_out=5):
    """bool (N,): ball not in play.

    sp_before=50 (2 s): the set-piece tag starts a little before the ball goes missing.
    sp_after=125 (5 s): the ball reappears when it is put back in play, shortly after the tag.
    long_gap=250 (10 s): live play rarely hides the ball that long.
    walk_speed=1.5 m/s: players walking on average means play is stopped.
    min_out=5 (0.2 s): a shorter observed-out run is tracking noise, not the ball leaving.
    Gaps are maximal runs of ~observed inside one graded block (never spanning ungraded frames).
    """
    sp = np.asarray(set_piece_frames)
    dead = np.zeros(len(observed), dtype=bool)
    for block_start, block_end in _runs(graded):
        for s, e in _runs(~observed[block_start:block_end]):
            s, e = s + block_start, e + block_start  # e exclusive
            near_sp = ((sp >= s - sp_before) & (sp < e - 1 + sp_after)).any()
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN gap -> NaN -> not quiet
                quiet = e - s >= long_gap and np.nanmean(player_speed[s:e]) < walk_speed
            dead[s:e] = near_sp or quiet
    out_f = out.copy()
    for s, e in _runs(out & observed):
        # noise only if tracking sees the ball back in play right after; out-then-lost is a real exit
        if e - s < min_out and e < len(observed) and observed[e] and graded[e]:
            out_f[s:e] = False
    return dead | dead_ball(out_f, observed, sp, graded, kickoffs)
