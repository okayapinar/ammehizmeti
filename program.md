# autoresearch: NBA spread / total

This is an experiment to have the LLM do its own research on predicting NBA betting outcomes.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct2`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**. The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data loading, the online evaluation harness. Do not modify.
   - `train.py` — the file you modify. Features, model, hyperparameters.
   - `nba_2008-2026.csv` — the data (glance at the header only). `divisions.json` — team → division/conference.
4. **Verify the environment**: `.venv/Scripts/python prepare.py` should print the game counts.
5. **Initialize results.tsv**: create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good, then kick off the experimentation.

## The task

Each row of the CSV is one NBA game with the closing betting line. Two binary targets:

- `id_spread`: 1 if the favorite (`whos_favored`) covered `spread`, 0 if not, 2 = push.
- `id_total`: 1 if the combined score went over `total`, 0 if under, 2 = push.

These are close to coin flips against the market — the baseline is ~0.50. Any real, reproducible edge is valuable (52.4% is roughly the break-even point at standard -110 odds).

## Experimentation

Each experiment is one run: `.venv/Scripts/python train.py > run.log 2>&1`. The harness streams all games day by day (2008 → 2026): for each day it calls `Model.predict(pre)` for every game using only pre-game info, then `Model.learn(pre, result)` with the scores. It aborts if the whole stream takes longer than `TIME_BUDGET` (300 s).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: feature engineering (team form, rolling margins, Elo variants, rest/back-to-backs, travel, line movement proxies, residuals vs. the line, ...), model choice (any river model, ensembles, logistic regression, periodically refit sklearn models, hand-written rules), hyperparameters, separate models per target.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It holds the fixed evaluation, the val split (`VAL_START_SEASON = 2024`) and the time budget.
- Read the CSV directly from `train.py` or otherwise peek at results before predicting. All information must come through `predict(pre)` / `learn(pre, result)`. Leaking future results makes the experiment worthless.
- Install new packages. Use only what is already in `.venv` (pandas, numpy, river, scikit-learn).

**The goal is simple: get the highest `val_acc`.** `val_logloss` (lower is better) is a secondary signal and the tie-breaker.

**Noise warning**: there are only ~3950 val games, so the standard error of `val_acc` is about 0.005. A change of +0.002 is almost certainly noise. Prefer changes that improve both `val_acc` and `val_logloss`, and be skeptical of tiny gains from added complexity. Do not tune a single random seed to win — that is overfitting to the val set.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_acc:          0.498102
val_acc_spread:   0.499494
val_acc_total:    0.496711
val_logloss:      0.723895
num_val_games:    3952
eval_seconds:     66.7
```

You can extract the key metrics from the log file:

```
grep "^val_acc:\|^val_logloss:\|^eval_seconds:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_acc	val_logloss	status	description
```

1. git commit hash (short, 7 chars)
2. val_acc achieved (e.g. 0.512345) — use 0.000000 for crashes
3. val_logloss (e.g. 0.6921) — use 0.0000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_acc	val_logloss	status	description
a1b2c3d	0.498102	0.7239	keep	baseline
b2c3d4e	0.507500	0.6935	keep	logistic regression on spread/total + elo diff
c3d4e5f	0.501000	0.7010	discard	add rolling 10-game margin
d4e5f6g	0.000000	0.0000	crash	sklearn refit too slow (timeout)
```

Do not commit `results.tsv` — leave it untracked (it is in `.gitignore`).

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/oct2`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `.venv/Scripts/python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^val_acc:\|^val_logloss:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv.
8. If `val_acc` improved (higher), you "advance" the branch, keeping the git commit.
9. If `val_acc` is equal or worse, `git reset --hard HEAD~1` back to where you started.

**Timeout**: The harness itself aborts after 300 s (`TimeoutError`). Treat that as a crash and make the idea cheaper (fewer trees, refit less often, etc.).

**Crashes**: If a run crashes because of something dumb and easy to fix (a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder: re-read the data, combine previous near-misses, try more radical changes (different model families, per-target models, calibrating against the line). The loop runs until the human interrupts you.
