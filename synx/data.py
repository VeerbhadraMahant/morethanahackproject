"""Load the ATD tracking into numpy arrays indexed by video frame (frame_id - 1)."""

import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")


def team_of_slots(meta_path):
    """int8 per track slot: 0 = home (first Team in the XML), 1 = away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def _group(text):
    """(x, y, speed) from 'x,y,speed'; all NaN when missing. Speed NaN if the group has none."""
    if text[:1] == "N":
        return np.nan, np.nan, np.nan
    v = text.split(",")
    return float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else np.nan


def _parse(raw_path, n_slots, n_frames, pitch):
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, np.float32)
    ball_speed = np.full(n_frames, np.nan, np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    player_speed = np.full((n_frames, n_slots), np.nan, np.float32)
    bad = 0
    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                fid, plist, bl = line.strip().split(":")
                i = int(fid) - 1
                groups = plist.split(";")
                if not 0 <= i < n_frames or len(groups) != n_slots:
                    raise ValueError
                parsed = [_group(g) for g in groups]
                bx, by, bs = _group(bl)
            except (ValueError, IndexError):
                bad += 1
                continue
            for s, (x, y, sp) in enumerate(parsed):
                players[i, s] = (x * length, y * width)
                player_speed[i, s] = sp
            ball[i] = (bx * length, by * width)
            ball_speed[i] = bs
    return dict(ball=ball, ball_speed=ball_speed, players=players, player_speed=player_speed,
                n_malformed=bad)


def load_tracking(game=GAME, use_cache=True, paths=None, n_frames=None, cache_dir=None):
    """Tracking dict: ball (N,2), ball_speed (N,), players (N,S,2), player_speed (N,S),
    team (S,), pitch (2,), n_malformed (int). Metres, indexed by video frame, NaN = missing.

    paths = (meta_xml, raw_txt) overrides the data bundle (used by tests); n_frames overrides
    the video length. Cached in cache/<game>_tracking.npz, rebuilt if the raw file's size or
    mtime changed.
    """
    meta_path, raw_path = paths or require(["atd_meta", "atd_raw"], game)
    n_frames = n_frames or GAMES[game]["n_video_frames"]
    pitch = np.array(GAMES[game]["pitch"], dtype=np.float32)
    st = os.stat(raw_path)
    stamp = np.array([st.st_size, st.st_mtime_ns, n_frames], dtype=np.int64)
    cache = os.path.join(cache_dir or CACHE_DIR, "{}_tracking.npz".format(game))

    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            if np.array_equal(z["stamp"], stamp):
                out = {k: z[k] for k in z.files if k != "stamp"}
                out["n_malformed"] = int(out["n_malformed"])
                return out

    team = team_of_slots(meta_path)
    out = _parse(raw_path, len(team), n_frames, pitch)
    out.update(team=team, pitch=pitch)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, stamp=stamp, **out)
    return out


def main():
    t0 = time.time()
    d = load_tracking()
    dt = time.time() - t0
    n = len(d["ball"])
    S = d["players"].shape[1]
    has_ball = ~np.isnan(d["ball"][:, 0])
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    ng = int(graded.sum())
    tracked = (~np.isnan(d["players"][:, :, 0])).sum(1)
    mean_tracked = float(tracked[graded].mean())
    b_all, b_gr = int(has_ball.sum()), int((has_ball & graded).sum())

    print("ball {} players {} player_speed {} team {} pitch {}".format(
        d["ball"].shape, d["players"].shape, d["player_speed"].shape, d["team"].shape,
        tuple(float(v) for v in d["pitch"])))
    print("ball_speed present on {} frames".format(int((~np.isnan(d["ball_speed"])).sum())))
    print("load time {:.1f}s, malformed lines {}".format(dt, d["n_malformed"]))
    print("ball overall {}/{} ({:.1f}%)".format(b_all, n, 100.0 * b_all / n))
    print("ball graded  {}/{} ({:.1f}%)".format(b_gr, ng, 100.0 * b_gr / ng))
    print("mean tracked players over graded frames {:.2f}".format(mean_tracked))

    checks = [
        ("ball overall == 113488 / 199319", (b_all, n) == (113488, 199319)),
        ("ball graded == 94214 / 158857", (b_gr, ng) == (94214, 158857)),
        ("mean tracked players ~ 20.5", abs(mean_tracked - 20.5) < 0.1),
        ("S == 46", S == 46),
    ]
    for name, ok in checks:
        print("{}  {}".format("PASS" if ok else "FAIL", name))


if __name__ == "__main__":
    main()
