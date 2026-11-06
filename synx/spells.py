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


def dead_ball(out, observed, set_piece_frames, graded, kickoffs, max_restart=1500):
    """bool (N,): ball not in play.

    max_restart=1500 (60 s): restarts are taken within a minute, so a set piece that soon after
    the ball left ends the stoppage; otherwise the stoppage ends when the ball is back in play.
    kickoffs: {half: video frame of the kickoff}; graded frames before it are dead (pre-match).
    shortcut: fouls/offsides with the ball on the pitch are not detected in v1.
    """
    n = len(out)
    obs = observed | ~graded  # ungraded frames reset the state, so halftime cannot leak
    idx = np.maximum.accumulate(np.where(obs, np.arange(n), 0))
    out = out[idx] & graded  # forward-fill out across unobserved (NaN) frames
    dead = np.zeros(n, dtype=bool)
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


def build_spells(control, dead, graded, kickoff_team, set_pieces=None, confirm=25,
                 min_touch=3, expire=75):
    """array (N,) of 'H' 'A' 'D' 'X'.

    confirm=25 (1 s): a touch or deflection by the opponent is shorter than a real change of
    possession, so a switch needs a second of accumulated opponent control.
    min_touch=3 (0.12 s): a real touch lasts a few frames; a 1-2 frame proximity flicker is not one.
    expire=75 (3 s): a pass between teammates takes at most 3 s, so opponent evidence older than
    that without opponent control is stale. Two separate touches within it are a completed pass.
    The switch is back-dated to the first counted opponent frame, because the spell really
    changed hands there.
    """
    labels = np.full(len(control), "X", dtype="<U1")
    labels[graded & dead] = "D"
    sp_frames, sp_teams = (np.empty(0, int), np.empty(0, int)) if set_pieces is None else set_pieces
    sp_frames, sp_teams = np.asarray(sp_frames), np.asarray(sp_teams)
    run_id = np.cumsum(graded & ~np.concatenate(([False], graded[:-1])))
    seen_runs = set()
    last_team = 0  # only used if a live stretch has no team information at all
    for a, b in _runs(graded & ~dead):
        run = int(run_id[a])
        team = -1
        if run not in seen_runs:  # first live stretch of a half: the kickoff team starts
            seen_runs.add(run)
            team = kickoff_team.get(run, -1)
            last_team = max(team, 0)  # fallback for this half, not the previous half's team
        else:  # restart: latest set piece from 2 s before the dead stretch to `confirm` after it
            s = a
            while s > 0 and dead[s - 1] and graded[s - 1]:
                s -= 1
            near = np.flatnonzero((sp_frames >= s - 50) & (sp_frames <= a + confirm))
            if len(near):
                team = int(sp_teams[near[np.argmax(sp_frames[near])]])
        if team == -1:  # first team seen in control, filled back to the restart frame
            seen = control[a:b][control[a:b] >= 0]
            team = int(seen[0]) if len(seen) else last_team
        cur = team
        labels[a:b] = TEAM_LETTER[cur]
        count, touches, first_opp, last_opp, run_len, run_start = 0, [], 0, None, 0, 0
        for t in range(a, b):
            c = int(control[t])
            if c == cur:
                count, touches, last_opp, run_len = 0, [], None, 0
            elif c >= 0:
                if run_len == 0:
                    if last_opp is not None and t - last_opp > expire:  # stale evidence
                        count, touches = 0, []
                    run_start = t
                    if count == 0:
                        first_opp = t
                run_len += 1
                count += 1
                last_opp = t
                if run_len == min_touch:
                    touches.append(run_start)
                if count >= confirm or len(touches) >= 2:
                    start = first_opp if count >= confirm else touches[0]
                    cur = c
                    labels[start:b] = TEAM_LETTER[cur]
                    count, touches, last_opp, run_len = 0, [], None, 0
            else:
                run_len = 0
        last_team = cur
    return labels
