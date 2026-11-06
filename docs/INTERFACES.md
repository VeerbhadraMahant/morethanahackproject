# Module interfaces (v1)

The contract between modules. Change it only by agreement — every module codes against it.

**Conventions.** All per-frame arrays have length `N = 199319` and are indexed by **video frame**
(= submission row = ATD `frame_id - 1`). Positions are metres from `synx.data.load_tracking()`:
`x` in `[0, 105.3]` along the pitch, `y` in `[0, 68.0]`. Missing = `NaN`. Team codes:
`HOME = 0` (Barcelona, `H`), `AWAY = 1` (Athletic, `A`), `NONE = -1`. 25 fps. numpy + stdlib only.

Every parameter default below carries a one-line football reason in its docstring. Defaults are
**not** chosen by score on the public ground truth (rules forbid tuning on it).

## A — `synx/data.py` (done)
`load_tracking(game="BAR-ATH") -> dict`: `ball (N,2)`, `ball_speed (N,)` (mostly NaN — do not use),
`players (N,S,2)`, `player_speed (N,S)`, `team (S,) int8`, `pitch (2,)`, `n_malformed`.

## B — `synx/pitch.py`
- `load_tags(game="BAR-ATH") -> list[dict]`, built on `starter.load_smart_tagging.load`. Each:
  `code` (upper case), `start`, `end` (video frames, end exclusive), `team` (0 / 1 / None —
  "Barcelona" → 0, "Atletic Club" → 1, "N/A" or absent → None; case-insensitive), `type`
  (str or None), `half` (1, 2 or None).
- `set_piece_starts(tags) -> (frames int64 (K,), team int8 (K,))`: `SET PIECES` instances, sorted
  by start frame; team `-1` when unknown.
- `kickoff_team(tags) -> dict {1: team, 2: team}`: team of the `Kick Off (Start)` set piece per
  half, `-1` if not found.
- `ball_out(ball, pitch, margin=0.5) -> bool (N,)`: ball observed and more than `margin` metres
  outside a touchline or goal line (whole ball over the line + tracking noise). NaN → False.
- Homography: not used in v1 (tactical is valid on 100% of frames; nothing here needs pixels).

## C — `synx/ball_possession.py`
- `nearest_dist(ball, players, team) -> float32 (N,2)`: column k = distance from the ball to the
  nearest player of team k. NaN where the ball is NaN or team k has no tracked player.
- `control(ball, players, team, speed, radius=1.5, max_speed=9.0, margin=0.5) -> int8 (N,)`:
  team k when its nearest player is within `radius` (one stride), at least `margin` closer than
  the nearest opponent (else it is a duel — nobody), and the ball is slower than `max_speed`
  (faster than a sprinting player = ball travelling, not controlled; NaN speed = allowed).
  Otherwise `-1`.

## D — `synx/missing_ball.py`
- `fill_ball(ball, max_gap=25) -> (ball_filled float32 (N,2), source int8 (N,))`: linear
  interpolation across interior gaps of at most `max_gap` frames (1 s — a ball travels roughly
  straight that long), bounded by observations on both sides. `source`: 0 observed,
  1 interpolated, 2 missing.
- `ball_speed(ball_filled, fps=25, half_window=2) -> float32 (N,)`: m/s, centred difference
  `|p[t+h] - p[t-h]| / (2h / fps)`; NaN if either end is NaN.

## E — `synx/spells.py` + `run_all.py`
- `graded_mask(game="BAR-ATH") -> bool (N,)` from `games.half_video_blocks`.
- `dead_ball(out, observed, set_piece_frames, graded, kickoffs, max_restart=1500) -> bool (N,)`
  (`observed`: raw ball present; `out` is forward-filled across unobserved frames, never across ungraded):
  - graded frames before each half's kickoff (`starter.load_smart_tagging.KICKOFF_VIDEO_FRAMES`)
    are dead;
  - each run of `out` frames is dead from its first frame until the next set-piece start if that
    is within `max_restart` frames (60 s — restarts are taken within a minute), else until the
    ball is next observed inside the pitch.
  - `shortcut:` fouls/offsides with the ball on the pitch are not detected in v1.
- `build_spells(control, dead, graded, kickoff_team, set_pieces=None, confirm=25) -> array (N,)`
  of `'H' 'A' 'D' 'X'`:
  - ungraded → `X`; dead → `D`; never `X` on a graded live frame;
  - after a dead stretch the restarting team is the set piece's team if known, else the first team
    seen in control (filled back to the restart frame); at a half's start, `kickoff_team`;
  - a switch needs `confirm` frames (1 s) of opponent control without the current team regaining
    it — filters touches and deflections. The switch is **back-dated** to the first opponent
    control frame after the current team's last control frame (late switches cost spellF1).
- `run_all.py` (repo root, `python run_all.py`): load → fill → speed → control → tags → dead →
  spells → write `Synx.csv` (`frame,label`, 199,319 rows) → run
  `evaluation/validate_submission.py Synx.csv` → print aggregates only: label counts, number of
  possessions, mean possession length (s), switches per minute, % graded frames dead.
