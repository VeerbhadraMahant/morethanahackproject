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

# v2 changes

Motivation (aggregate diagnostics of v1, no ground truth involved): 57 of 59 set pieces fell inside
"live" stretches; 99.9% of missing-ball frames sit in gaps longer than 3 s, and the longer the gap
the slower the players walk (0.93 m/s in gaps over 30 s vs 2.16 m/s with the ball visible).

## F — `synx/stoppage.py` (new)
- `stoppage_dead(observed, out, player_speed, set_piece_frames, graded, kickoffs,
  sp_before=50, sp_after=125, long_gap=250, walk_speed=1.5, min_out=5) -> bool (N,)`.
  A graded frame is dead if any of:
  - **(a) restart gap**: it lies in a ball-missing gap (maximal run of `~observed` within one
    graded block, from the frame after the last sighting to the frame before the next one) and a
    set-piece start falls in `[gap_start - sp_before, gap_end + sp_after)` — 2 s before / 5 s
    after: the tag brackets the restart, and the ball reappears when it is put back in play.
  - **(b) quiet gap**: it lies in a ball-missing gap of at least `long_gap` frames (10 s — live
    play rarely hides the ball that long) whose mean player speed (`nanmean` over slots and
    frames of `player_speed`) is below `walk_speed` (1.5 m/s — players walking: play stopped).
  - **(c) out**: `spells.dead_ball(out_f, observed, set_piece_frames, graded, kickoffs)` where
    `out_f` is `out` with runs of observed-out frames shorter than `min_out` (0.2 s, tracking
    noise) removed. This also keeps the pre-kickoff rule.
  - gaps of 3–10 s with no set piece and fast players stay live (no guess).

## E — `build_spells` v2 (same signature plus `min_touch=3, expire=75`)
- **restart team**: the latest set piece whose start lies in
  `[dead_start - 50, live_start + confirm]` of the dead stretch just ended (the tag can start
  several seconds before the ball reappears), its team if known; else first control team as v1.
- **switch**: the opponent takes over when, since the current team's last control frame, it has
  either `confirm` (1 s) accumulated control frames, **or** 2 separate touches — maximal runs of
  opponent control of at least `min_touch` frames (0.12 s, a real touch, not a one-frame
  proximity flicker) separated by at least one non-control frame: a completed pass between
  teammates. Evidence expires after `expire` frames (3 s — the longest a pass between teammates
  takes) without opponent control. Back-dated to the first frame of the first counted touch.
- `run_all.py` uses `stoppage_dead` instead of `dead_ball`.

# v3 changes

Motivation (aggregate diagnostics of v2, no ground truth): only 11% of live time has the ball
missing, but there our team agrees with the Smart Tagging phase team on just 70% of frames (80%
elsewhere). Two candidate rules were checked and **dropped**: "restart team = opponent of the last
toucher" agreed with the tagged set-piece team on 9/21 restarts (chance level), and visible-ball
stoppages cover only 1.2% of live time.

## G — `synx/hmm.py` (new): Viterbi decoder for H vs A inside live stretches
Dead-ball frames stay exactly as `stoppage_dead` decides (D); the decoder only chooses H or A on
graded live frames. Ungraded → X. Each live stretch (maximal run of graded & ~dead) is decoded
independently with a 2-state Viterbi minimising total cost:
- `tag_evidence(tags, n) -> int8 (N,)`: per-frame team implied by Smart Tagging phases, -1 none.
  Attacking codes give their `team`: BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING
  TRANSITION, BALL IN FINAL THIRD, BALL IN THE BOX, PLAYERS IN THE BOX, LONG BALL. Defending
  codes give the *other* team: DEFENDING IN ATTACKING THIRD, DEFENDING IN MIDDLE THIRD,
  DEFENDING IN DEFENSIVE THIRD, DEFENSIVE TRANSITION. Tags with team None are ignored; frames
  where tags imply both teams → -1. SET PIECES and GOALS are not used here.
- `decode_hmm(control, dead, graded, kickoff_team, set_pieces, tag_team, c_ctrl=1.0, c_tag=0.04,
  switch=10.0) -> array (N,)` of 'H' 'A' 'D' 'X':
  - per-frame cost of being in state k: `c_ctrl` if `control` names the other team; `c_tag` if
    `tag_team` names the other team; 0 otherwise.
    `c_ctrl=1.0`: one frame of a player on the ball is the unit of evidence.
    `c_tag=0.04` (= 1/25): one second of phase tag counts like one frame of control — tags are
    automatic, phase-level (~1 s resolution) and imperfect, so they decide only where there is no
    control evidence (ball missing).
  - `switch=10.0`: cost of a possession change — the opponent needs about 10 frames (0.4 s, two
    clean touches) more evidence than the current team to take over; shorter flickers are
    touches/deflections. Viterbi places the switch at the cost-optimal frame (no separate
    back-dating rule).
  - start prior: at a half's first live stretch add `switch` to every frame-0 state that is not
    `kickoff_team[half]`; after a stoppage, the same with the set-piece team whose start lies in
    `[dead_start - 50, live_start + 25]` (latest wins, team known). No prior otherwise.
  - Ties broken towards the previous state (no gratuitous switch).
- `shortcut:` H/A emission is binary per signal; graded confidences (distance, duel) are a v4 idea.

## H — `synx/pipeline.py` (new) and `run_all.py`
- `predict(data, tags, decoder="v2") -> labels (N,)`: the full chain from a `load_tracking` dict
  and `load_tags` list (fill → speed → control → ball_out → stoppage_dead → `build_spells` for
  "v2" or `decode_hmm` for "hmm"). No I/O, no printing. Deterministic.
- `run_all.py [--decoder v2|hmm]` (default **hmm** once merged) calls `predict`, writes
  `Synx.csv`, runs the validator, prints aggregates only (as before).

## I — `synx/scorecard.py` (new): label-free evaluation, no ground truth
`python -m synx.scorecard` runs `predict` for both decoders and prints, per decoder, aggregates only:
- **structure**: possessions, median and 10th-percentile possession length (s), possessions < 2 s,
  % graded dead, switches per minute;
- **robustness**: re-run `predict` on a copy of the tracking with the ball additionally removed
  on random 1-second chunks covering 10% of graded ball-visible frames (3 fixed seeds:
  0, 1, 2); report mean % of graded frames whose label changes and mean change in possession
  count. A good method barely moves when evidence is lost;
- **evidence agreement**: % of live frames with `control >= 0` where the label equals the control
  team; % of tag-covered live frames where it equals `tag_evidence` — **flagged as a decoder input
  for hmm (not a fair comparison)**.

## Pre-registered selection rule (written before any v3 result)
Submit `hmm` instead of `v2` only if, on the scorecard: robustness label change is lower,
possessions < 2 s are not higher, and possessions and % dead stay within football-plausible
ranges (150–350 possessions; 35–50% dead). Otherwise submit `v2`. The public 5-minute window is
used once, afterwards, only to confirm the chosen file runs through the official scorer.
