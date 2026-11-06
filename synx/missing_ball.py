"""Fill short gaps in the ball track and derive ball speed."""
import numpy as np


def fill_ball(ball, max_gap=25):
    """Linearly interpolate interior NaN gaps of at most `max_gap` frames.

    max_gap=25 (1 s): a ball travels roughly straight that long, so a line is a fair guess.
    Returns (ball_filled float32 (N,2), source int8 (N,)): 0 observed, 1 interpolated, 2 missing.
    """
    ball = np.asarray(ball, dtype=np.float32)
    out = ball.copy()
    n = len(ball)
    obs = ~np.isnan(ball).any(axis=1)
    source = np.where(obs, 0, 2).astype(np.int8)
    idx = np.flatnonzero(obs)
    if len(idx) < 2:
        return out, source
    t = np.arange(n)
    # Neighbouring observations of every frame; gap = frames strictly between them.
    pos = np.searchsorted(idx, t)  # first observed index >= t
    interior = (pos > 0) & (pos < len(idx)) & ~obs
    prev = idx[np.clip(pos - 1, 0, len(idx) - 1)]
    nxt = idx[np.clip(pos, 0, len(idx) - 1)]
    fill = interior & (nxt - prev - 1 <= max_gap)
    for c in range(2):
        out[fill, c] = np.interp(t[fill], idx, ball[idx, c])
    source[fill] = 1
    return out, source


def ball_speed(ball_filled, fps=25, half_window=2):
    """Ball speed in m/s from a centred difference over +-`half_window` frames.

    half_window=2 (0.08 s): long enough to smooth tracking jitter, short enough to catch a pass.
    NaN where either end of the window is NaN or outside the array.
    """
    p = np.asarray(ball_filled, dtype=np.float32)
    h = half_window
    speed = np.full(len(p), np.nan, dtype=np.float32)
    if len(p) > 2 * h:
        d = np.linalg.norm(p[2 * h:] - p[:-2 * h], axis=1)
        speed[h:len(p) - h] = d / (2 * h / fps)
    return speed
