# Phase 1: Amazon, Model A end to end, shareable bundle

Plan for Claude Code. Written 2026-10-13. Before starting, read `AGENTS.md` and `docs/RESEARCH_PLAN.md` **on the `feat/review-fraud-pivot` branch** (the copies on `main` still describe CFPB). If anything here conflicts with them, this plan wins on the points listed under "Decisions"; for anything else, stop and ask.

## Goal

On the Amazon dataset (CARE-GNN `.mat`): download it, split it (seeded, stratified, grouped), run the **full** df-analyze Model A configuration, and pack the results with the existing `pack_artifacts.py`. A group member who unpacks the bundle must **never have to retrain anything**: every pipeline stage skips, and the trained models load and predict.

## Decisions (2026-10-13, from the user)

1. **Build on `feat/review-fraud-pivot`**, not on `main`. `main@3336d87` is still the CFPB/AMLworld pipeline. The pivot branch (based on `3336d87`) already has `download`, `load`, `split`, `features`, `audit`, `df_analyze_input`, `df_analyze`, `select_model`, `df_analyze_report`, `web_report` and per-dataset packing, and a 10-trial CPU pilot on Amazon passed (RESEARCH_PLAN §11). YelpChi code stays but is **not run** in phase 1.
2. **Split:** as on the branch. Seeded, stratified 60/40 over the 8,639 labelled users; each set of exact-duplicate feature rows (293 rows) stays on one side (`StratifiedGroupKFold`, 2 of 5 folds as test); 5 grouped CV folds on train. `split.n_repeats: 1`.
3. **Features:** keep all 25. The flag on `own__f13`, `own__f15`, `own__f18`, `own__f19` (univariate AUROC 0.86–0.88) is **GitHub issue #36**. Do not drop or change any feature. Reference #36 in the report, README and PR, and leave it open.
4. **Machine:** the user's Windows laptop with an NVIDIA GPU, **inside WSL2**, with the repo cloned in the WSL filesystem (`~/...`), not `/mnt/c/...`.
5. **Bundle:** keep **every tuned model** df-analyze saved, plus an exported, verified Model A; drop only caches and scratch files.
6. **Teammates** will install df-analyze's environment, so loading models and `predict.py` run in df-analyze's environment. Nothing needs to load in the project environment.
7. **Git end state:** when everything passes, **push the branch and open a PR to `main`** with `gh pr create`. This overrides RESEARCH_PLAN §0's "do not push". Never merge it.

## Background: the bundle already contains the trained models

Checked 2026-10-13 in df-analyze's source at the pinned commit `199e563`:
- `pack_artifacts.py` copies the whole latest df-analyze run directory.
- In it, `results/eval_htune_results_jsonpickle.json` is written by `EvaluationResults.save` (`hypertune.py`) with `jsonpickle.encode(..., unpicklable=True)`.
- Each `HtuneResult.model` in that file is the tuned model **refit on the full training set**: `htune_optuna` calls `refit_tuned` on all of `X_train`, and `htune_eval` restores that fit after its 5-fold loop.
- So every tuned (model, feature selection) combination is already in the bundle; this is a large part of its size.
- A quick check showed LightGBM, logistic regression and CatBoost survive jsonpickle encode → decode with identical probabilities.
- **Unverified risks:**
  - If encoding raises `TypeError`, df-analyze falls back to `fail_safe=str`, which **silently replaces** unencodable objects with strings. GANDALF (PyTorch) is the most likely case.
  - Decoding needs the same package versions as the run, which means df-analyze's environment.
- df-analyze also writes a `__JOBLIB_CACHE__` folder inside its run directory, and the bundle currently includes it.
- Nothing in this pipeline currently loads the saved models.

## Ground rules

- Git: work on `feat/review-fraud-pivot`. Small, focused commits. Never commit to `main`, never merge, never force-push, never tag. Before every commit, run `git status` and `git diff --staged`.
- Never commit anything under `data/` or `outputs/`, any `fraud_artifacts_*.tgz`, model files (`*.pkl`, `*.joblib`, `*.cbm`, `*.pt`, `*.ckpt`, …), or `PR_DESCRIPTION.md`. Add `.gitignore` patterns for any new model file types in the same commit that introduces them.
- Check the GPU with `nvidia-smi` in WSL and `torch.cuda.is_available()`; never hard-code `cuda`.
- The full run is pre-approved (decision log, 2026-10-08). It takes over 1 hour, so give the user the estimate and wait for "go" before starting it.

## Tasks

### Task 0: Set up and check the branch (WSL2)
- `git fetch origin && git checkout feat/review-fraud-pivot && git pull --ff-only && uv sync`
- Check uv is 0.9.16 or newer.
- Set up df-analyze at the pinned commit:
  ```
  git -C ../df-analyze fetch origin
  git -C ../df-analyze checkout 199e5638620693c267dac715784f1fd0e33fa796
  uv sync --locked --python '>=3.13.11,<3.14' --directory ../df-analyze
  ```
- Check that both environments see CUDA (project: `torch.cuda.is_available()`; df-analyze: the same inside its env).
- Run `uv run ruff check . && uv run ruff format --check . && uv run pytest`. Everything must pass before any change.
- Check `gh auth status` works in WSL (needed in Task 7). If it does not, tell the user now.
- Add `.gitattributes` in its own commit (`* text=auto eol=lf`; `*.sh text eol=lf`). The user's Windows checkout showed whole-file CRLF diffs. Afterwards, `git status` must be clean.

**Done when:** tests pass in WSL2 and both environments see the GPU.

### Task 1: Download
- Run `uv run run.py --dataset amazon --only download`. It writes to `data/raw/care_gnn/` and verifies the SHA-256 of `Amazon.zip` (`c1b7a1f8…a5b9`) and `Amazon.mat` (`4b7e3f9c…9d63`).

**Done when:** both checksums match. On a mismatch, stop and report. Do not fetch the data from anywhere else.

### Task 2: Load, split, features, audit
- Run `uv run run.py --dataset amazon --to audit`. It takes minutes.
- Check `outputs/amazon/reports/audit.md` and `split_summary.json` against RESEARCH_PLAN §2:
  - **Nodes:** 11,944 nodes, 8,639 labelled, 821 positives. Nodes 0–3,304 have `is_labelled = False`.
  - **Features and edges:** 25 features; edges U-P-U 175,608, U-S-U 3,566,479, U-V-U 1,036,737.
  - **Split:** about 5,183 train / 3,456 test, and no group crosses train/test or the CV folds.
  - **Shortcut flags:** f13, f15, f18 and f19 are flagged again. This is expected (#36).

**Done when:** every count matches. On a mismatch, stop and report.

### Task 3: Full df-analyze run (Model A)
- Tell the user the estimate: 1–3 h on the GPU (the 10-trial CPU pilot took 24 min). Wait for "go".
- Run `uv run run.py --dataset amazon`. Use the config as it is on the branch:
  - classifiers `lgbm, lr, catboost, gandalf, rf, knn`, plus the automatic dummy;
  - 100 tuning trials, tuned on `bal-acc`;
  - `--filter-pred-classify auroc`;
  - no wrapper feature selection.
- If the run is interrupted, continue it with `uv run run.py --dataset amazon --resume`.
- When it finishes, check:
  - `split_check.json` and `type_check.json` exist and pass;
  - `model_selection_cv.csv` scores every tuned combination;
  - the report has no sanity-band warning (Amazon ≈ 0.91 PR-AUC; no-skill 0.095).
- Record in RESEARCH_PLAN §11:
  - the wall-clock time of each stage and the GPU model;
  - Model A: its name and feature selection, shared-CV PR-AUC ± SD, and test PR-AUC and AUROC with bootstrap CIs.

**Done when:** all stages have finished, the checks pass, and the results are recorded. Never tune or rerun anything because of a test-set result (invariant 4).

### Task 4: `export_model` stage: verified, loadable models
- Add the stage `export_model` at the **end** of `STAGES`, after `web_report`, so that `run.py` never reruns df-analyze to reach it. Add a matching wrapper `scripts/NN_export_model.py`.
- Its work runs in **df-analyze's environment** via a new `scripts/dfa/export_model.py`, called the same way as `scripts/dfa/cv_select.py`.

**Steps:**
1. **Decode every saved model.** Decode `eval_htune_results_jsonpickle.json`. For **every** tuned combination, record whether its model decoded to a real fitted estimator (not a string) and whether it reproduces df-analyze's saved test probabilities for that combination.
   - Report the max abs difference.
   - Use a tolerance for GPU models (CatBoost, GANDALF) and record it.
   - Write `outputs/amazon/m1_own/model/models_check.csv`: model, selection, decoded (yes/no), max abs diff, ok.
2. **Model A:** take the combination chosen by `df_analyze_report` (by shared-CV PR-AUC only).
   - **If it decoded and reproduces its probabilities:** save it in its native format: `model_a.joblib` for sklearn/LightGBM, `.cbm` for CatBoost, a checkpoint directory for GANDALF.
   - **If it did not decode:** refit it with df-analyze's `refit_tuned` and its tuned params on the full training set, then verify it the same way.
   - **If it still fails:** stop and ask.
3. **Other combinations that did not decode:** do not refit them. List them in `models_check.csv` and in the report, and tell the user.
4. **Write `model_a.json`:**
   - model name and feature selection;
   - the ordered input columns (after selection);
   - the tuned params;
   - the decision threshold (OOF max-F1 rule, from `df_analyze_report`);
   - the seed and the df-analyze commit;
   - package versions (python, numpy, sklearn, lightgbm, catboost, torch, jsonpickle);
   - whether the model was decoded or refit.
5. **Write outputs atomically.** The stage counts as done only when every output file exists.

**Also add `scripts/dfa/predict.py`** (df-analyze env):
- By default it loads `model_a.*` and `model_a.json`. With `--model <name> --selection <sel>`, it loads any decoded combination straight from the jsonpickle file.
- It reads a feature table and checks the column names and order, failing loudly on a mismatch.
- It writes probabilities and labels at the saved threshold.
- It warns if the installed package versions differ from those in `model_a.json`.

**Tests (synthetic only, in `tests/synthetic.py`):**
- a column or order mismatch fails loudly;
- `model_a.json` round-trips;
- `export_model` is the last stage;
- a string placeholder in place of a model triggers the refit path;
- a tiny fitted sklearn model round-trips through jsonpickle → `export_model` → `predict.py`.

These need df-analyze's environment. If the test suite cannot reach it, put the pure logic (column checks, metadata, choosing the refit path) in `src/` and test that in the project environment.

**Done when:** on the real run, Model A is saved and verified, `models_check.csv` is written, and the tests pass.

### Task 5: Trim and pack the bundle
- **Size breakdown:** list the 20 largest files and folders in the latest df-analyze run directory and in `outputs/amazon/`, and record them in RESEARCH_PLAN §11.
- **Caches out:** exclude `__JOBLIB_CACHE__` and any other cache or scratch folder from `collect_files`, but only after checking that no stage, `predict.py` or `unpack_artifacts.py` reads it. Add a test.
- **Models in:** keep `eval_htune_results_jsonpickle.json` (every tuned model) and `outputs/amazon/m1_own/model/`. Make sure `collect_files` includes both.
- **Pack:** `uv run pack_artifacts.py --dataset amazon`. Record the bundle size before and after trimming.

**Done when:** the bundle has been made, it contains the models and no caches, and it is not tracked by git.

### Task 6: Round-trip test as a teammate would
In a **fresh clone** in a temporary WSL folder: check out the branch, run `uv sync`, and use the same df-analyze clone.
1. Run `uv run unpack_artifacts.py` with the bundle. The checksums must pass.
2. Run `uv run run.py --dataset amazon`. **Every stage must skip, including `download`** (the bundle has no raw data). If `download` or `load` wants the raw `.mat`, fix the skip logic so a clone that only has the bundle skips them. Do not add the raw data to the bundle.
3. Run `scripts/dfa/predict.py` on the unpacked test table. It must reproduce the saved Model A test probabilities. Do the same for one other combination via `--model`/`--selection`.
4. Automate steps 1–2 as a test on a synthetic dataset: pack → unpack into `tmp_path` → zero stages pending.

Then delete the temporary clone.

**Done when:** all four steps pass.

### Task 7: Docs, PR, hand-off
- **README:** Amazon phase-1 status and results next to the no-skill baseline. Update "Sharing outputs with the group" for `--dataset amazon`, covering what the bundle contains (all tuned models + Model A), that teammates need df-analyze's environment, and how to use `predict.py`. Add a note on #36.
- **AGENTS.md decision log, 2026-10-13:**
  - phase 1 is Amazon-only on this branch;
  - the split stays as it is, with `n_repeats: 1`;
  - all 25 features are kept pending #36;
  - the bundle keeps all tuned models and drops caches;
  - the `export_model` stage and `predict.py` were added;
  - the branch goes to `main` through a PR.
- **RESEARCH_PLAN §11/§12:** results, timings, size breakdown, and next steps (one YelpChi validation run, then Model 2). Update §0 to say the branch has now been pushed.
- **Checks:** run `ruff`, `ruff format --check` and `pytest` a final time, then `git status` to confirm nothing from the never-commit list is staged. Commit.
- **PR:** push `feat/review-fraud-pivot` and open a PR to `main` with `gh pr create`. The description covers what changed (the whole CFPB → YelpChi/Amazon pivot plus phase 1), why, how it was tested (WSL2 + NVIDIA GPU; macOS not tested), the leakage checklist, and "Refs #36". Never merge.
- **Tell the user:**
  - the PR link;
  - the bundle path and size;
  - Model A's headline numbers next to the no-skill baseline;
  - which tuned models failed to decode, if any.
  Remind them to share the bundle through OneDrive or Teams, never through the repository.

## Out of scope (ask first)
- Running YelpChi; GNN code, PyG or DGL; Model 2 neighbour features.
- Dropping or editing features (#36); changing the split, `n_repeats`, classifiers, trials, metrics or the threshold rule.
- Refitting tuned models other than Model A.
- Merging the PR.

## Stop and ask if
- a checksum or count does not match;
- df-analyze's split or type verification fails;
- Model A, whether decoded or refit, does not reproduce df-analyze's test probabilities;
- the run looks like it will take well over 3 h;
- a stage turns out to need the joblib cache or another file you meant to drop;
- anything would require committing data, outputs, models or the bundle.
