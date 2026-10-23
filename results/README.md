# Results database

A shared record of everyone's runs: what was run, on which machine, when (UTC), how long it took, a short description, and its results table. It is committed to git; the run outputs themselves stay in `outputs/` and are never committed.

```
results/runs/<run_id>/meta.json     metadata (fields below)
results/runs/<run_id>/results.csv   the run's results table: aggregate numbers only
```

Each run is its own folder with a unique ID (`<start UTC>_<dataset_feature-set>_<6 hex>`), so two people recording runs at the same time never touch the same file and never get a merge conflict. There is no committed index file; `list` and `export` build it from the folders.

## Recording a run

Use `scripts/results_db.py`; never write the files by hand.

**A pipeline run** (stages through `df_analyze_report` finished). `--from-run` reads `reports/model_a.json` and `reports/model_a_metrics.csv` (every tuned model's shared-CV and test metrics becomes `results.csv`). `--run-log` reads the `run.py` invocation's log folder: start time, command, git commit, package versions and settings (classifiers, trials, split repeat, GNN settings). Use the log of the invocation that ran `df_analyze`, which is usually the newest folder in `outputs/<dataset>[_repeat<k>]/runs/`.

```bash
uv run scripts/results_db.py add \
    --from-run outputs/amazon/pilot_m1_own \
    --run-log outputs/amazon/runs/20261019T140501123456Z \
    --device rtx4060-laptop \
    --description "Pilot m1, split 0, lgbm/lr/catboost, 10 trials" \
    --tag pilot
```

The finish time is taken from when `model_a.json` was written; override it with `--finished-utc`, or give `--duration-min`, when that is wrong (for example after a `--resume` that only rebuilt the report).

**Anything else** (a probe, a comparison page's contrasts, `repeat_summary.csv`, a check): give the summary CSV and the times yourself.

```bash
uv run scripts/results_db.py add --kind compare \
    --csv outputs/amazon/compare/repeat_summary.csv \
    --dataset amazon --started-utc 2026-10-20T01:40:00Z --finished-utc 2026-10-20T04:48:00Z \
    --device rtx4060-laptop --description "3-split verdicts, pilot ladder, both variants"
```

Then commit the new folder:

```bash
uv run scripts/results_db.py check
git add results/runs/<run_id> && git commit -m "Results: <short description>"
```

Add `--dry-run` to see the entry without writing it. Times must carry a time zone (`...Z` for UTC); a time without one is refused so local time is never stored as UTC.

## Fields (`meta.json`)

| Field | Meaning |
|---|---|
| `run_id` | the folder name |
| `device` | your short label for the machine (e.g. `rtx4060-laptop`, `macbook-m2`); keep it the same for every run on that machine |
| `device_details` | detected automatically: OS, platform, WSL or not, Python, CPU, NVIDIA GPUs (name, driver, memory). No host or user names |
| `started_utc`, `finished_utc`, `duration_min` | wall-clock run time, UTC |
| `recorded_utc` | when the entry was made |
| `description` | one line, at most 200 characters |
| `kind` | `pipeline`, `probe`, `compare`, `check` or `other` |
| `status` | `ok`, `failed` or `partial` (record failed runs too: their time and device are useful) |
| `dataset`, `feature_set`, `split_repeat`, `seed` | what was run |
| `settings` | from the run log: classifiers, trials, tuning metric, df-analyze commit, blocks, GNN and neighbour settings |
| `headline` | for pipeline runs: Model A, test PR-AUC with its 95% interval, AUROC, no-skill PR-AUC, shared-CV PR-AUC ± SD, F1 at the out-of-fold threshold, train/test sizes, any sanity warning |
| `command`, `git_commit`, `git_dirty`, `packages` | how to reproduce it |
| `tags`, `notes`, `supersedes`, `source` | free labels; anything else; the `run_id` this entry corrects; where the table came from |

## Rules

- **Aggregate results only.** `results.csv` may hold one row per model, contrast, fold or setting, never per node. The script refuses ID, label and prediction columns (`node_id`, `label`, `group_id`, `split`, `target`, `prob`, …), tables over 1,000 rows and files over 256 kB, and `check` (also run by the test suite) refuses entries that break these rules.
- **Append-only.** Never edit or delete someone else's entry. To correct one, add a new entry with `--supersedes <run_id>` and say why in `--notes`.
- **Record every run whose numbers anyone might quote**, including pilots and failed runs, right after it finishes.
- Pilot and full-budget runs are different runs: tag pilots `--tag pilot` and never compare a pilot rung with a full one.

## Reading it

```bash
uv run scripts/results_db.py list                       # every entry, oldest first
uv run scripts/results_db.py list --dataset amazon --tag pilot
uv run scripts/results_db.py show 20261021T1405         # one entry (a unique prefix is enough)
uv run scripts/results_db.py export                     # outputs/results_db/index.csv and all_results.csv
```

`export` writes `index.csv` (one row per entry with the headline numbers flattened) and `all_results.csv` (every results table stacked, with the run ID, device, dataset and feature set in front). Both go under `outputs/`, so they are never committed.
