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

Each experiment is one run: `.venv/Scripts/python train.py > run.log 2>&1`. The harness streams all games day by day (2008 → 2026): for each day it calls `Model.predict(pre)` for every game using only pre-game info, then `Model.learn(pre, result)` with the scores. It aborts if the whole stream takes longer than `TIME_BUDGET` (600 s / 10 minutes). Longer training runs (e.g. 2 to 8 minutes) are completely fine — the 10-minute budget is intentionally provided to allow richer, more principled architectures.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: feature engineering (team form, rolling margins, Elo variants, rest/back-to-backs, travel, line movement proxies, residuals vs. the line, ...), model choice (any river model, ensembles, logistic regression, periodically refit sklearn models, hand-written rules), hyperparameters, separate models per target.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It holds the fixed evaluation, the val split (`VAL_START_SEASON = 2024`) and the time budget.
- Read the CSV directly from `train.py` or otherwise peek at results before predicting. All information must come through `predict(pre)` / `learn(pre, result)`. Leaking future results makes the experiment worthless.
- Install new packages. Use only what is already in `.venv` (pandas, numpy, river, scikit-learn, scipy, xgboost).

**The goal is simple: get the highest `val_acc`.** `val_logloss` (lower is better) is a secondary signal and the tie-breaker.

**Noise warning**: there are only ~3950 val games, so the standard error of `val_acc` is about 0.005. A change of +0.002 is almost certainly noise. Prefer changes that improve both `val_acc` and `val_logloss`, and be skeptical of tiny gains from added complexity. Do not tune a single random seed to win — that is overfitting to the val set.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win.

## Research Strategy: Sensible Architectures Over Micro-Tuning

**ABSOLUTE RULE: NO MICRO-PARAMETER HILL CLIMBING**
- **DO NOT** make experiments that merely nudge arbitrary constants or hyperparameters (e.g., changing `ELO_HOME_ADV = 55 -> 57`, `MAX_REST_DAYS = 20 -> 21`, learning rates by `0.0003`, loss weights `0.12 -> 0.11`, clipping bounds `0.42 -> 0.415`, thresholds `60 -> 56`).
- That is pure noise-chasing on the ~3950 validation games. It produces brittle, overfitted, unreadable code and wastes hundreds of cycles without learning anything structural.
- If an architectural idea does not show promise with standard, sensible defaults, **move to a different architectural hypothesis immediately**. Do NOT spend iterations micro-tuning its constants.

**PRIORITIZE SENSIBLE ARCHITECTURAL EXPLORATION (Training duration being long is NOT a problem — up to 10 minutes is allowed!):**
Take full advantage of the 10-minute (600 s) runtime limit to explore real, principled machine learning and domain-grounded architectures:

1. **Continuous Score Regression Architecture (Expected Margin & Total Points)**:
   - Binary betting outcomes are coarse thresholds of continuous game scores: `margin = score_home - score_away` and `game_total = score_home + score_away`.
   - Instead of direct binary classification, model margin and total as continuous regression targets (predicting expected $\hat{\mu}_{margin}$ and $\hat{\mu}_{total}$).
   - Convert continuous predictions to cover probabilities using standard normal CDF: $P(\text{cover}) = 1 - \Phi\left(\frac{\text{spread} - \hat{\mu}}{\sigma}\right)$ or calibrated logistic transforms.
   - Continuous scores carry vastly more variance and information than binary 0/1, making regression fundamentally better conditioned.

2. **Diverse Model Families & Periodic Batch Refitting**:
   - You are not confined to streaming online linear models (River SGD).
   - With 10 minutes available, periodically refit (e.g., every 15, 30, or 60 days, or once per month) powerful tabular models on accumulated pre-game feature matrices:
     * `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier`
     * `XGBoost` (`xgb.XGBClassifier` / `xgb.XGBRegressor`)
     * `RandomForestRegressor` / `RandomForestClassifier`
     * Regularized `Ridge` / `Lasso` / `LogisticRegression`
     * `MLPRegressor` / `MLPClassifier`
   - Test hybrid models: fast online state / ratings + periodic tree-based or gradient boosted refits.

3. **Decoupled Asymmetric Architectures (Spread vs. Total)**:
   - Spread and Total are completely different betting markets:
     * **Spread**: Relative team quality, matchup advantages, net ratings, home court, rest differential vs. the closing handicap.
     * **Total**: Combined pace (possessions per 48 minutes) and offensive/defensive scoring efficiency.
   - Build completely separate, tailored model pipelines and feature sets for Spread vs. Total rather than forcing both into identical model classes.

4. **Domain-Sound Feature Architectures**:
   - **Possession & Pace**: Estimate game pace and offensive/defensive ratings per 100 possessions.
   - **Four Factors**: Effective Field Goal % (eFG%), Turnover Rate (TOV%), Offensive Rebound % (ORB%), Free Throw Rate (FTR).
   - **Fatigue & Schedule**: 3-in-4 nights, back-to-backs with travel distance, altitude (Denver/Utah).
   - **Market Residuals**: Directly model the market inefficiency $\Delta_{spread} = \text{actual\_margin} - \text{spread}$ and $\Delta_{total} = \text{actual\_total} - \text{total}$.

5. **Simplicity and Readability as Core Priorities**:
   - The user prioritizes simplicity and readability ("gereksiz karmasikliktan kacin", "Simplicity and readability are my priorities").
   - Code in `train.py` should be clean, modular, and easy to read.
   - Remove dead code, redundant branches, and ugly ad-hoc heuristic patches when pivoting to cleaner architectures.

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

## Do not keep scratch files

A one-off check may write a helper script, a dump, a diff, or a pickle (`scratch_*.py`, `scratch_*.sh`, `r*_out*`, `*.txt`, `*.diff`, `*.patch`, `*.pkl`, extra folders). Read what you need, log the result in `results.tsv`, then delete those files in the same step. The directory keeps only the project files plus `results.tsv`, `run.log`, and `agent.log`. Do not commit the scratch files.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/oct2`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `.venv/Scripts/python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^val_acc:\|^val_logloss:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv. Delete any scratch file written for this run.
8. If `val_acc` improved (higher), you "advance" the branch, keeping the git commit.
9. If `val_acc` is equal or worse, `git reset --hard HEAD~1` back to where you started.

**Timeout**: The harness itself aborts after 600 s / 10 minutes (`TimeoutError`). Longer training runs (e.g. 2 to 8 minutes) are completely fine and welcome for substantial architectures. Only treat it as a crash if it genuinely exceeds 10 minutes (600 s), in which case make the architecture slightly cheaper (e.g., refit every 30 days instead of 7 days, reduce tree depth or max_iter).

**Crashes**: If a run crashes because of something dumb and easy to fix (a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder: re-read the data, combine previous near-misses, try more radical changes (different model families, per-target models, calibrating against the line). The loop runs until the human interrupts you.
