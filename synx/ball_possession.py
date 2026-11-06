"""Per-frame ball control from ball and player positions."""
import warnings

import numpy as np


def nearest_dist(ball, players, team):
    """Distance (m) from the ball to the nearest player of each team, (N,2) float32.

    NaN where the ball is NaN or the team has no tracked player in that frame.
    """
    d = np.linalg.norm(players - ball[:, None, :], axis=2).astype(np.float32)  # (N,S)
    out = np.full((len(ball), 2), np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected
        for k in (0, 1):
            if (team == k).any():
                out[:, k] = np.nanmin(d[:, team == k], axis=1)
    return out


def control(ball, players, team, speed, radius=1.5, max_speed=9.0, margin=0.5):
    """Team in control of the ball per frame (0/1), -1 for nobody, int8 (N,).

    radius: one stride - a player this close can touch the ball.
    margin: nearest opponent must be this much farther, else it is a duel (nobody controls).
    max_speed: m/s, faster than a sprinting player means the ball is travelling, not controlled.
    NaN speed is allowed (unknown speed does not block control).
    """
    d = nearest_dist(ball, players, team)
    out = np.full(len(ball), -1, np.int8)
    with np.errstate(invalid="ignore"):  # NaN comparisons are False
        slow = ~(speed >= max_speed)  # NaN speed -> True
        for k in (0, 1):
            opp = d[:, 1 - k]
            wins = (d[:, k] <= radius) & (np.isnan(opp) | (opp - d[:, k] >= margin)) & slow
            out[wins] = k
    return out
