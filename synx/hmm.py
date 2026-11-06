"""Viterbi decoder: H vs A on live frames, from ball control and Smart Tagging phase evidence."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.spells import TEAM_LETTER, _runs  # noqa: E402

ATTACKING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
             "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL"}
DEFENDING = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def tag_evidence(tags, n):
    """int8 (N,): team in possession implied by phase tags, -1 none or conflicting.

    Attacking phases belong to the tag's team, defending phases to the other team.
    Tags without a team are ignored; frames implying both teams are ambiguous.
    """
    implied = np.zeros((n, 2), dtype=bool)
    for t in tags:
        if t["team"] is None:
            continue
        if t["code"] in ATTACKING:
            team = t["team"]
        elif t["code"] in DEFENDING:
            team = 1 - t["team"]
        else:
            continue
        implied[max(t["start"], 0):max(t["end"], 0), team] = True
    out = np.full(n, -1, dtype=np.int8)
    out[implied[:, 0] & ~implied[:, 1]] = 0
    out[implied[:, 1] & ~implied[:, 0]] = 1
    return out


def _viterbi(c0, c1, switch, prior):
    """Best 0/1 path minimising frame costs c0, c1 plus `switch` per change; prior = start team or -1."""
    n = len(c0)
    back = np.zeros((n, 2), dtype=np.int8)  # back[t, k] = state at t-1 on the best path into k at t
    d = [switch if prior == 1 else 0.0, switch if prior == 0 else 0.0]
    for t in range(n):
        if t:  # a change must beat staying strictly, so ties keep the previous state
            n0, n1 = d
            if n1 + switch < n0:
                d[0], back[t, 0] = n1 + switch, 1
            if n0 + switch < n1:
                d[1], back[t, 1] = n0 + switch, 0
            else:
                back[t, 1] = 1
        d[0] += c0[t]
        d[1] += c1[t]
    state = int(d[1] < d[0]) if d[0] != d[1] else max(prior, 0)
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path


def decode_hmm(control, dead, graded, kickoff_team, set_pieces, tag_team,
               c_ctrl=1.0, c_tag=0.04, switch=10.0):
    """array (N,) of 'H' 'A' 'D' 'X'; dead frames stay D, ungraded X, live frames decoded.

    c_ctrl=1.0: one frame of a player on the ball is the unit of evidence.
    c_tag=0.04 (1/25): a second of phase tag counts like one frame of control, because tags
    are automatic, ~1 s resolution and imperfect, so they decide only where the ball is missing.
    switch=10.0: the opponent needs about 10 frames (0.4 s, two clean touches) more evidence
    than the current team to take over; shorter flickers are touches or deflections.
    Start prior: a half's first live stretch starts with the kickoff team, a restart with the
    team of the latest known-team set piece in [dead_start - 50, live_start + 25].
    """
    labels = np.full(len(control), "X", dtype="<U1")
    labels[graded & dead] = "D"
    sp_frames, sp_teams = (np.asarray(a) for a in set_pieces)
    half = np.cumsum(graded & ~np.concatenate(([False], graded[:-1])))
    seen = set()
    for a, b in _runs(graded & ~dead):
        if half[a] not in seen:
            seen.add(half[a])
            prior = kickoff_team.get(int(half[a]), -1)
        else:
            s = a
            while s > 0 and dead[s - 1] and graded[s - 1]:
                s -= 1
            near = np.flatnonzero((sp_frames >= s - 50) & (sp_frames <= a + 25) & (sp_teams >= 0))
            prior = int(sp_teams[near[np.argmax(sp_frames[near])]]) if len(near) else -1
        ctrl, tag = control[a:b], tag_team[a:b]
        cost0 = c_ctrl * (ctrl == 1) + c_tag * (tag == 1)  # cost of being in state H
        cost1 = c_ctrl * (ctrl == 0) + c_tag * (tag == 0)  # cost of being in state A
        path = _viterbi(cost0.tolist(), cost1.tolist(), switch, prior)
        labels[a:b] = np.array(list(TEAM_LETTER))[path]
    return labels
