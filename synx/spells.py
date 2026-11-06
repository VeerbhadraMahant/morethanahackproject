"""Turn per-frame control and dead-ball flags into H/A/D/X possession labels."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

TEAM_LETTER = "HA"


def graded_mask(game=GAME):
    """bool (N,): True on frames inside a half (the only frames that are scored)."""
    mask = np.zeros(GAMES[game]["n_video_frames"], dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def _runs(flag):
    """[(start, end)] of maximal True runs, end exclusive."""
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def dead_ball(out, set_piece_frames, graded, kickoffs, max_restart=1500):
    """bool (N,): ball not in play.

    max_restart=1500 (60 s): restarts are taken within a minute, so a set piece that soon after
    the ball left ends the stoppage; otherwise the stoppage ends when the ball is back in play.
    kickoffs: {half: video frame of the kickoff}; graded frames before it are dead (pre-match).
    shortcut: fouls/offsides with the ball on the pitch are not detected in v1.
    """
    dead = np.zeros(len(out), dtype=bool)
    for kick in (kickoffs.values() if isinstance(kickoffs, dict) else kickoffs):
        not_graded = np.flatnonzero(~graded[:kick])
        first = not_graded[-1] + 1 if len(not_graded) else 0  # start of the half's graded block
        dead[first:kick] = graded[first:kick]
    sp = np.asarray(set_piece_frames)
    for start, end in _runs(out):
        i = np.searchsorted(sp, start, side="right")  # first set piece after the ball went out
        if i < len(sp) and sp[i] - start <= max_restart:
            end = sp[i]
        dead[start:end] = True
    return dead


def build_spells(control, dead, graded, kickoff_team, set_pieces=None, confirm=25):
    """array (N,) of 'H' 'A' 'D' 'X'.

    confirm=25 (1 s): a touch or deflection by the opponent is shorter than a real change of
    possession, so a switch needs a second of opponent control. The switch is back-dated to the
    first opponent control frame, because the spell really changed hands there.
    """
    labels = np.full(len(control), "X", dtype="<U1")
    labels[graded & dead] = "D"
    sp_frames, sp_teams = (np.empty(0, int), np.empty(0, int)) if set_pieces is None else set_pieces
    sp_frames = np.asarray(sp_frames)
    run_id = np.cumsum(graded & ~np.concatenate(([False], graded[:-1])))
    seen_runs = set()
    last_team = 0  # only used if a live stretch has no team information at all
    for a, b in _runs(graded & ~dead):
        run = int(run_id[a])
        team = -1
        if run not in seen_runs:  # first live stretch of a half: the kickoff team starts
            seen_runs.add(run)
            team = kickoff_team.get(run, -1)
        else:  # restart after a stoppage: the set piece's team if its tag starts near here
            near = np.flatnonzero(np.abs(sp_frames - a) <= confirm)
            if len(near):
                team = int(sp_teams[near[0]])
        if team == -1:  # first team seen in control, filled back to the restart frame
            seen = control[a:b][control[a:b] >= 0]
            team = int(seen[0]) if len(seen) else last_team
        cur, count, first_opp = team, 0, a
        labels[a:b] = TEAM_LETTER[cur]
        for t in range(a, b):
            c = int(control[t])
            if c == cur:
                count = 0
            elif c >= 0:
                if count == 0:
                    first_opp = t
                count += 1
                if count >= confirm:
                    cur, count = c, 0
                    labels[first_opp:b] = TEAM_LETTER[cur]
        last_team = cur
    return labels
