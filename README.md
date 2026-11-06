# Team Synx - More Than a Hack 2027 (Phase 1: ball possession)

Task: label each of the 199,319 tactical frames of the match as H (home possession), A (away possession), D (dead ball) or X (abstain). Submissions are ranked on spell F1 (`spellF1`, mean possession-spell F1 over IoU 0.3-0.7). The organisers' baseline `baseline_smooth50` scores 0.2433.

## Setup

Python 3.13.

    pip install -r requirements.txt

`requirements-starter.txt` is optional (kloppy, matplotlib, notebook for the starter notebooks).

## Data

Put the match data under `data/tactical/...`, or point `MTAH_DATA_DIR` at the data directory:

    PowerShell: $env:MTAH_DATA_DIR = "path\to\data"
    bash:       export MTAH_DATA_DIR=path/to/data

## Commands

    python -m synx.data                                   # build the tracking cache (once)
    python run_all.py                                     # regenerate Synx.csv and validate it
    python run_all.py --decoder hmm                       # same, with the experimental HMM decoder (default: v2)
    python -m synx.scorecard                              # label-free evaluation of both decoders (aggregates only)
    python starter/baseline.py                            # regenerate and score the organisers' baselines
    python evaluation/run_evaluation.py                   # score everything in evaluation/results
    python evaluation/validate_submission.py Synx.csv     # format check before submitting

`python run_all.py` is the one command that produces `Synx.csv`. The method is described in `docs/Synx.md`.

## Data rules

`data/` is licensed match data. It is gitignored and must never be committed, uploaded, or sent to any AI service. The same goes for generated label CSVs derived from it.

## Credits

`games.py`, `evaluation/` and `starter/` are the organisers' scaffold (MIT, see `LICENSE-metrica-starter`).
