# AGENTS.md

Instructions for AI coding agents (primarily Claude Code) working in this repository. Read this file fully before making changes. `README.md` describes the research design; this file describes how to work on it. **[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) is the working research plan:** the model ladder, verified dataset facts, splits, interfaces, milestones with acceptance criteria, and the open decisions. Read it before picking up any work item. If it conflicts with this file, stop and ask.

## Project in one paragraph

We ask: **does using a graph improve fraud classification over the same kind of model without one, and can a GNN's representations improve a tabular AutoML pipeline (df-analyze)?** The datasets are two graph fraud benchmarks with tabular node features: **YelpChi** (main; nodes are hotel and restaurant reviews, labelled spam by Yelp's filter, three review–review relations) and **Amazon** (replication; nodes are users, labelled fraudulent or not, three user–user relations). Both come from the preprocessed `.mat` files of the CARE-GNN repository. The model ladder (plan §1) is: (1) df-analyze on each node's own features (Model A); (2) df-analyze plus label-free neighbour aggregates per relation; (3) GraphSAGE with all edges removed; (4) heterogeneous GraphSAGE over all relations; (5) df-analyze plus self-supervised GNN embeddings; (6) df-analyze plus supervised, out-of-fold GNN features (needs a rule change). Every model on a dataset is trained on the same split and scored on the same frozen, grouped test set. The headline metric is PR-AUC, always reported next to the no-skill baseline (the test positive rate). Model 1 and the interfaces the later rungs need (node table, relation edge lists, feature blocks, feature sets) were implemented first. Phase 2 ([`docs/PHASE2_GNN_PLAN.md`](docs/PHASE2_GNN_PLAN.md)) adds Models 2 and 5 as feature blocks (`nbr`, `gnnssl`) and a comparison page; GraphSAGE (Models 3–4) comes later. **Phase 3 ([`docs/PHASE3_FEEDBACK_PLAN.md`](docs/PHASE3_FEEDBACK_PLAN.md)) is the current plan:** it answers the professor's feedback of 2026-10-21 (embedding size and DeepTune, whether training the GNN helps, the 4 important features, more datasets).

## Current state (2026-10-23): read this first

**New on 2026-10-21:** the professor's feedback and the plan that answers it ([`docs/PHASE3_FEEDBACK_PLAN.md`](docs/PHASE3_FEEDBACK_PLAN.md); its open questions D1–D5 need the group); the [results database](#results-database) (`results/runs/`); and commits no longer need a PR ([Git workflow](#git-workflow)).

**New on 2026-10-23 (branch `exp/phase3-feedback`):** phase 3 Tasks 1, 3a and 5 done at the pilot budget (README, "Results (phase 3 pilots)"; RESEARCH_PLAN §11). Added: the untrained-GNN block `gnnrand`, `{keep: [...]}` feature variants (`only36`, `top4`), and the scripts `top_features.py`, `feedback_probes.py`, `embedding_sizes.py` and `repeat_summary.py --pairs`. All 24 runs and 9 probes or summaries are in the results database (tag `phase3`).

The details are in [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) §11 (decision log and every result) and §12 (next steps); a short summary is in the README ("Results").

**What has been run (Amazon only, seed 555, one RTX 4060 laptop under WSL2).** Every df-analyze run passed the split and column-type checks.

| Run or check | Configuration | Outcome (test PR-AUC unless stated) |
|---|---|---|
| Full Model A `m1_own` | 6 classifiers + dummy, 100 trials, split 0 | 0.919 [0.890, 0.942] |
| Full Model A `m1_own_no36` (issue #36) | same, without `own__f13`, `own__f15`, `own__f18`, `own__f19` | 0.518 [0.465, 0.576]; paired Δ −0.401 |
| Pilot ladder `pilot_m1/m2/m5` × all features / `_no36` | 10 trials; LightGBM, logistic regression, CatBoost; splits 0, 1, 2 (18 runs) | all features: m1 0.917–0.919, m2 0.923–0.938, m5 0.928–0.931; without #36: m1 0.524–0.574, m2 0.709–0.711, m5 0.665–0.688 |
| Pre-registered 3-split verdicts (`scripts/repeat_summary.py`) | Δ > 0 on all splits, mean > SD, Nadeau–Bengio p < 0.05 over 15 CV folds | all features: m2 − m1 +0.010 supported; m5 − m1 +0.011 not distinguishable; m5 − m2 ≈ 0. Without #36: m2 − m1 +0.162 and m5 − m1 +0.129 supported; m5 − m2 −0.033 worse |
| Training-only checks (`scripts/probe_blocks.py`, `progress_checks.py`, `feature_identity_checks.py`) | grouped CV folds of split 0 | random graph keeps most of the all-features gain and about half of the without-#36 gain; f19 alone 0.893 balanced accuracy; vote-ratio test negative; f13/f15/f18 consistent with rating statistics; embedding defaults mid-grid |
| Phase 3 pilots (2026-10-23): `pilot_m5r_own_gnnrand[_no36]`, `pilot_m{1,2,5}_*_only36`, `pilot_m{1,2,5}_*_top4` | pilot settings, splits 0, 1, 2 (24 runs) | m5 − m5r (DGI training) −0.004 all features, +0.012 without #36, neither significant; m1_only36 0.887 vs m1 0.918 (worse); m1_top4 0.922 (not distinguishable from m1); m5_top4 0.941–0.944 |
| Phase 3 checks (`top_features.py`, `feedback_probes.py`, `embedding_sizes.py`) | training CV folds, split 0 | permutation top 4 = `f19`, `f00`, `f23`, `f18` (≠ #36; the 4th is `f12` on splits 1–2); DGI's 64 dimensions per relation have effective rank 8–27 |

**Known:**
- The four #36 features carry most of Amazon's signal, but they are not the best four: `f13` and `f15` add nothing once `f18` and `f19` are known, and four features (`f00`, `f18`, `f19`, `f23`) match all 25.
- Without the #36 features, graph features help clearly on all 3 splits, and about half of that gain needs the real edges.
- Simple neighbour means beat untuned DGI embeddings, and DGI training adds nothing measurable over the same encoder with random weights.
- Published Amazon numbers use all 25 features.

**Unknown:**
- What f19 is. It is probably a helpfulness-vote count; the label is the share of helpful votes (< 20% fraud, > 80% benign), and `Amazon.mat` holds no votes.
- Whether other columns are vote-derived.
- Full-budget and tuned-embedding results for the graph methods.
- YelpChi.
- GraphSAGE (Models 3–4).

**Not run yet:** #36 checks (c)–(e) in RESEARCH_PLAN §12, step 0; a variant without only the vote-derived columns; full-budget m2/m5; YelpChi beyond a smoke test; Models 3, 4, 6.

**Where things are:**
- **Code:** on `main`, except the phase 3 plan (branch `claude/awesome-wozniak-wm548a`) and the phase 3 work built on it (branch `exp/phase3-feedback`), neither merged yet.
- **Results database:** `results/runs/` (committed; see [Results database](#results-database)). It holds the phase 3 runs of 2026-10-23; the earlier runs above have not been recorded in it (D5).
- **Overnight queue (2026-10-23):** the job runner and its logs live in the WSL clone's `outputs/amazon/phase3_logs/` (never committed): `bin/runner.sh` runs each line of `queue.txt` not yet in `done.txt`, `bin/pilot.sh` runs one pilot rung and records it, and `bin/summary.sh` builds the comparison pages and a verdict table.
- **Results:** in the WSL clone's `outputs/amazon*` and `data/processed/amazon*`. Split repeats 1–2 live in `outputs/amazon_repeat1/2`, which `pack_artifacts.py` does not pack (it packs repeat 0 only).
- **Group bundle:** `fraud_artifacts_amazon_20261019T220559Z.tgz` (148 MB, split 0), on OneDrive.
- **CFPB-era files** moved out of the clone to `~/projects/fraud-detect-gnn-cfpb-archive/`.
- **Every Amazon comparison is reported both ways** (all features and `_no36`; decided 2026-10-19). GNN embeddings run on the CPU (`gnn_embed.device: cpu`).

## Datasets

- **Source:** the CARE-GNN repository (Dou et al., CIKM 2020; Apache-2.0), commit `a64ff7523e187a24251f7ca88435d2c9d8f7dcd9`, folder `data/`: `YelpChi.zip` and `Amazon.zip`, each holding one `.mat` file. The SHA-256 checksums of the zips and the `.mat` files are in plan §2 and `configs/default.yaml`.
- Get the data only with the `download` stage (`uv run run.py --dataset <name> --only download`, or `./download_dataset.sh`), which writes to `data/raw/care_gnn/` and verifies every checksum. Do not fetch other datasets, other commits, or DGL/PyG copies of these files without asking. **Do not add DGL as a dependency;** read the files with `scipy.io.loadmat`.
- Do not modify files in `data/raw/`. Write derived data to `data/processed/<dataset>/` and outputs to `outputs/<dataset>/`.
- **Check the counts on load** against plan §2 (nodes, labelled nodes, positives, features, edges per relation). If they differ, stop and report rather than working around it.
- **Amazon's first 3,305 nodes are unlabelled but stored with label 0.** They are never labelled examples (never positives, never negatives); they stay in the node table as `is_labelled = False`, as graph context only.
- **YelpChi labels are user-level.** A user is a connected component of `net_rur`. All of a user's reviews stay on one side of every split.
- **Amazon has exact-duplicate feature rows** among labelled nodes; each set of identical rows is one split group.
- Features are anonymized: `f00`, `f01`, … in the `.mat` column order, prefixed `own__` in feature tables.

## Git workflow

This is a course project, not public software: **commits do not need a pull request, a review or passing checks.** The GitHub repository is publicly visible, though, so the [never-commit list](#never-commit) still applies in full.

- Commit directly to `main`, or to a branch (`feat/`, `fix/`, `exp/`, `docs/<short-name>`) when the work is experimental or long-running; either is fine. Pull before you start and push when you are done. If the user or the active plan says not to push yet, commit locally and wait.
- Pull requests are optional (for example, to ask a teammate to look at something). If you open one, its author decides when it is merged.
- Keep commits small and focused. The message says what changed and why; for a change to data, features or models, also which [leakage invariants](#leakage-checklist-check-before-committing-anything-that-touches-data-features-or-models) it touches and which platforms it ran on.
- Running the [checks](#checks-recommended-before-committing) before committing is recommended, not required. If you know a commit breaks a test, say so in its message.
- Never force-push, never rewrite published history, never delete remote branches, and never create tags or releases: teammates' clones depend on the shared history.
- Before every commit, run `git status` and `git diff --staged` and confirm that nothing from the [never-commit list](#never-commit) is staged.

## Results database

Every run whose numbers anyone might quote is recorded in **`results/runs/`**, one folder per run (`meta.json` + `results.csv`), committed to git so the group can see everyone's runs. Format, fields and examples: [`results/README.md`](results/README.md). Always use the script, never hand-written files:

- **After a pipeline run** (through `df_analyze_report`):
  `uv run scripts/results_db.py add --from-run outputs/<dataset>[_repeat<k>]/<feature_set> --run-log outputs/<dataset>[_repeat<k>]/runs/<stamp> --device <machine label> --description "<one line>"`.
  This fills in the dataset, feature set, split repeat, start and finish time (UTC), command, git commit, settings, Model A's headline numbers, and `results.csv` (`model_a_metrics.csv`: every tuned model's shared-CV and test metrics). `<stamp>` is the `run.py` invocation that ran `df_analyze`.
- **After anything else** (a probe, a comparison, `repeat_summary.csv`, a check): `add --kind probe|compare|check|other --csv <summary.csv> --started-utc ...Z --finished-utc ...Z` plus `--dataset` and `--feature-set` when they apply.
- Then `uv run scripts/results_db.py check` and commit the new folder (`git add results/runs/<run_id>`).
- **`--device`** is a short label for the machine (e.g. `rtx4060-laptop`, `macbook-m2`), the same for every run on it; OS, CPU and GPU are detected automatically. No host or person names.
- **Aggregate numbers only.** `results.csv` holds one row per model, contrast, fold or setting, never per node. The script refuses ID, label and prediction columns and tables over 1,000 rows; the test suite runs `check` on the committed folder.
- **Append-only:** correct an entry by adding a new one with `--supersedes <run_id>` and `--notes`; never edit or delete an existing one. Record failed and partial runs too (`--status failed|partial`); tag pilots `--tag pilot`.
- Reading: `list` (filters `--dataset`, `--device`, `--feature-set`, `--tag`), `show <run_id prefix>`, and `export` (writes `outputs/results_db/index.csv` and `all_results.csv`, never committed).
- Recording is part of finishing a run: report the new `run_id` with the result. Do not backfill earlier runs unless the user asks.

## Never commit

- Anything under `data/` or `outputs/` (raw `.zip`/`.mat` files, node tables, edge lists, feature blocks, df-analyze results, reports, checkpoints)
- Row-level data from any dataset (node features, labels, IDs with their labels), including in tests, fixtures, notebooks, logs, commit messages, PR descriptions, or the results database. Tests use small synthetic data built in memory (see [Required tests](#required-tests)).
- Model weights and saved models (`*.pt`, `*.pth`, `*.ckpt`, `*.joblib`, `*.cbm`), `*.mat`, `*.parquet`, `*.npy`, `*.npz`, `*.pkl`, or large CSVs
- `.env` files, API keys, tokens, or credentials
- Notebook outputs (clear outputs before committing)
- Any file larger than about 1 MB without asking first
- Output bundles from `pack_artifacts.py` (`fraud_artifacts_*.tgz`). They are shared outside git (see "Sharing outputs with the group" in the README).
- `PR_DESCRIPTION.md` (a draft the user pastes into GitHub)

If `.gitignore` does not already cover one of these, add the pattern in the same commit.

## Environments

There are two separate environments. Do not try to merge them.

| Environment | Location | Python | Used for |
|---|---|---|---|
| Project | this repo, managed by `uv` | 3.13 pinned in `.python-version`; `requires-python >=3.11` | download, loading, splits, features, audit, graph feature blocks, reports; GNN embeddings with the optional `gnn` extra (`uv sync --extra gnn`: torch 2.9, PyG; plain PyG, no compiled scatter extensions) |
| df-analyze | separate clone at the commit in `configs/default.yaml` (`df_analyze.commit`; path in `DF_ANALYZE_DIR`, default `../df-analyze`) | **3.13 only** (`>=3.13.11,<3.14`; its locked catboost has no 3.14 wheels), via df-analyze's own `uv sync --locked` | `df-analyze.py` and `scripts/dfa/*.py` only (`cv_select.py`, `export_model.py`, `predict.py`, `cuda_info.py`); loading any saved model |

- uv 0.9.16 or newer is required (older releases cannot download Python 3.13.11 for df-analyze); `fraud_detect.external` enforces this before calling df-analyze.
- Run project code with `uv run ...`. Add dependencies with `uv add`, never with plain `pip install`. Commit `uv.lock`.
- Keep the project compatible with Python 3.11–3.13 so it runs on every group member's laptop; run the tests on each version before changing the supported range.
- Call df-analyze through a subprocess using its own environment (`uv run --directory "$DF_ANALYZE_DIR" --python '>=3.13.11,<3.14' python df-analyze.py ...`; `fraud_detect.external.run_df_analyze_script` does this). Never import df-analyze into the project environment.
- **The project must run on Linux, macOS, and Windows**, because group members use all three. **On Windows, it runs inside WSL2** (e.g. Debian or Ubuntu), with the repo cloned into the WSL filesystem (`~/...`, not `/mnt/c/...`, which is much slower). Native Windows is not a target. The pipeline, `pack_artifacts.py`, and `unpack_artifacts.py` must work on each platform. In practice:
  - Use `pathlib`, and open text files with an explicit `encoding="utf-8"`.
  - Pass subprocess arguments as a list, and never use `shell=True`.
  - Do not depend on GNU-only command-line flags: macOS ships BSD versions of `sed`, `tar`, and similar tools. Inside Python code, use the standard library (`tarfile`, `zipfile`, `shutil`, `urllib`) instead of shelling out.
  - Never hard-code `cuda`. WSL2 and Linux laptops may have an NVIDIA GPU; Macs never do.
  - Say in the commit message which platforms a change was tested on.
- The main development machine is a Linux laptop (or Windows with WSL2) with an NVIDIA GPU. Always check `torch.cuda.is_available()` and fall back to CPU; never hard-code `cuda`. In df-analyze, CatBoost and GANDALF use the GPU; `gpu.require` decides whether a missing GPU stops the run.
- Hardware is limited. Ask before starting any job you expect to run longer than about 1 hour. Do not enable df-analyze's wrapper feature selection unless asked.

## Leakage checklist (check before committing anything that touches data, features, or models)

These are hard invariants. If a change would violate one, stop and ask.

1. **Labels enter only through the training loss.** No feature, aggregate, node attribute or graph input uses any node's label. This includes neighbour fraud rates and target encoding, even computed on training data only. (A later, documented exception may permit supervised out-of-fold GNN features for Model 6; that is an open decision, not a current rule.)
2. **Grouped splits.** No YelpChi user (`net_rur` connected component) and no Amazon duplicate-feature group appears in more than one of train/test, or in more than one CV fold.
3. **Amazon nodes 0–3,304 are never labelled examples.** They are unlabelled graph context only.
4. **The test set is frozen.** Every model on a dataset is evaluated on exactly the same test node IDs. Thresholds, feature choices, hyperparameters, model choice, early stopping and (later) graph settings are chosen on training data or its saved CV folds, never on the test set.
5. **Graph context rule.** Graph-derived features may use the *features* (never labels) of every node in the graph, including test and unlabelled nodes, because unlabelled reviews and users exist at deployment time. Graph feature builders take `(features, edges)` and never receive the label vector; a test with permuted labels enforces this (`tests/test_label_blindness.py`).
6. **Unsupervised transforms are fit on training nodes only:** PCA, scalers, and any feature-type decision based on data (`column_spec.json`).
7. **One feature schema per dataset.** Model 1's columns are exactly the `own__` block. No node IDs, group IDs, split names or relation degrees enter Model 1, and no feature block contains `label`, `node_id`, `group_id` or `split` columns.
8. **Choose the best df-analyze model by the shared, grouped cross-validation on the training set** (`select_model` stage, `model_selection_cv.csv`: every tuned combination refit on the saved `cv_folds.csv` folds and scored by PR-AUC from probabilities), never by holdout results. **Never by df-analyze's own tuning score** (`tuned_models_*.csv`): it is not comparable across models. df-analyze swaps AUROC for balanced accuracy on hard predictions for most models, and scores GANDALF on one validation split through a separate code path. Its internal CV is also random, not grouped, so its scores are optimistic. df-analyze's `5-fold` results table refits models on test-set folds and must never be reported as a test result.
9. **Every model in a comparison is scored with the same procedure and metric.** If a model cannot be scored that way, stop; do not compare it on a different basis. Before trusting a score produced by an external tool, check in its source code how that score is computed.

## Pipeline stages

`run.py` is the single entry point and runs the stages in order for one dataset: `uv run run.py --dataset yelpchi` (or `dataset:` in the config); `uv run run.py --list` shows the stages. Stages are defined in `src/fraud_detect/pipeline.py` (`STAGES`), and each also has a thin wrapper `scripts/NN_<stage>.py`. Each stage reads from and writes to `data/` or `outputs/` under the dataset's subfolder and must not depend on in-memory state from a previous stage. **Add every new stage to `STAGES` (and a matching script) so `run.py` stays the throughline.** Stages 8–13 run for one feature set (`--set feature_set=m1_own`, the default) and write under `outputs/<dataset>/<feature_set>/`. Stages 6–7 build a graph feature block only for feature sets that use it; their outputs are only the blocks the active feature set uses (none for `m1_own`), so adding a block stage never makes a finished run or a bundle rerun anything.

**Every stage must be safe to stop at any moment** (Ctrl+C, crash, shutdown), so runs can continue with `uv run run.py --dataset <name> --resume` (see `src/fraud_detect/runstate.py`; the run state is per dataset):
- **Write outputs atomically:** write to a temporary file, then rename, so a half-written file never counts as a finished stage. A stage's declared outputs must appear only when it has fully succeeded.
- **Split long stages into units.** A stage that takes more than a few minutes and consists of independent pieces (one fit per model, seed or fold set) must save each piece as it finishes with `ctx.unit_store("<stage>")`. Skip units where `store.done(unit)` is true, and build the stage's final outputs from `store.all()`. The store is emptied automatically on a fresh run and kept on `--resume`. The GNN stages must work this way, with one unit per (model or ablation, seed).
- **Units must be independent and deterministic given their inputs,** so resuming gives the same result as an uninterrupted run. Never split a computation into units whose results could differ between separate runs for reasons unrelated to the unit itself. That is why df-analyze runs all classifiers in one process: its feature selection is not seeded.
- **Test stop-and-resume behaviour** for any stage that uses units.

Stages (plan §7):

1. **`download`:** fetch `YelpChi.zip` / `Amazon.zip` from the pinned CARE-GNN commit into `data/raw/care_gnn/`, verify both checksums, unzip. Never overwrites a file with a different checksum without `--force`.
2. **`load`:** read the `.mat`; write `nodes.parquet` (node_id, label (null when unlabelled), is_labelled, group_id), one undirected edge list per relation plus `homo` under `graph/` (`src < dst`, no self-loops, deduplicated, `int32`) and `graph/manifest.json`; check every count against the config.
3. **`split`:** grouped, stratified 60/40 train/test split and 5 grouped CV folds on train (`StratifiedGroupKFold`, seeded); `train_ids.csv`, `test_ids.csv`, `cv_folds.csv`, `split_summary.json`; the split and fold are added to `nodes.parquet`.
4. **`features`:** the `own` feature block (`features/own.parquet`) and `column_spec.json` (binary / ordinal / continuous, decided on training nodes only).
5. **`audit`:** counts and statistics only: counts against plan §2, per-feature statistics and univariate AUROC (≥ 0.85 flagged as a possible shortcut), duplicate rows within and across splits, group sizes, edge counts, degrees and training-only edge homophily (`outputs/<dataset>/reports/audit.md`).
6. **`neighbour_agg`:** the `nbr` block (Model 2) and `features/nbr.spec.json`, for feature sets that use it.
7. **`gnn_embed`:** the `gnnssl` block (Model 5), `features/gnnssl.spec.json` and `outputs/<dataset>/gnn_embed/`, for feature sets that use it.
8. **`df_analyze_input`:** train/test tables for the active feature set, joining its blocks on `node_id` in the order of the saved IDs, with only feature columns and `target`. Column types come from `column_spec.json` (`own`) and each other block's `<block>.spec.json`.
9. **`df_analyze`:** run df-analyze with `--df-train` / `--df-tests`, then **verify** that its exported `X_train`/`X_test`/`y_*` match our saved split row for row, and that its inferred column types match `column_spec.json` with no feature dropped. If either check fails, stop and ask.
10. **`select_model`:** in df-analyze's environment (`scripts/dfa/cv_select.py`), refit every tuned combination on the saved grouped folds and score the held-out folds from probabilities (PR-AUC, AUROC, balanced accuracy, Brier), with tuned and default settings; save out-of-fold probabilities and the tuning budget. Stops if any tuned model cannot be scored.
11. **`df_analyze_report`:** choose Model A by shared-CV PR-AUC only; its decision threshold from its out-of-fold training predictions (max F1 by default); test metrics for every tuned combination at that threshold and at 0.5; group-bootstrap 95% intervals for Model A; a warning if Model A falls more than 0.05 PR-AUC below the sanity band (plan §2).
12. **`web_report`:** `outputs/<dataset>/<feature_set>/report/index.html`, a self-contained static page (no scripts, no external requests) built by `src/fraud_detect/report/web.py`. Its confidence cards show only **test** nodes (ID, label, score, top features by value). Every value inserted into the page must be HTML-escaped.
13. **`export_model`:** in df-analyze's environment (`scripts/dfa/export_model.py`), decode every tuned model df-analyze saved (`eval_htune_results_jsonpickle_00.json`, each refit on the full training set) and record in `models_check.csv` whether it is a fitted model and reproduces df-analyze's saved test probabilities. Save Model A (as chosen by `df_analyze_report`) in its native format with `model_a.json` under `outputs/<dataset>/<feature_set>/model/`; if it did not decode, refit it with `refit_tuned` and its tuned parameters, and stop and ask if it still does not reproduce its probabilities. Never refit any other combination. Decode only with `scripts/dfa/model_io.decode_results`, never plain `jsonpickle.decode`: GANDALF's saved state cannot be rebuilt, and after the first failure jsonpickle's references drift silently, so that decoder turns untrusted references into placeholders and a combination counts as usable only if it is intact and reproduces df-analyze's saved test probabilities and labels. **It stays last in `STAGES`**: `run.py` reruns every stage after the first one with missing outputs, so an earlier position would rerun df-analyze. `scripts/dfa/predict.py` loads the export (or any combination that passed the check) without retraining.

A clone that only unpacked a bundle has no raw data; the `download` stage counts as done when the `load` stage's outputs exist, so such a clone skips every stage.

**Comparison page (not a stage; it spans two feature sets).** `uv run scripts/compare_report.py --dataset <name> --a <feature_set> --b <feature_set>` (`src/fraud_detect/report/compare.py`) writes `outputs/<dataset>/compare/<a>__vs__<b>/compare.html` and `compare_metrics.json` from what the pipeline already saved: each side's Model A, its out-of-fold threshold and its test probabilities. It adds no outputs to any stage (that would make existing bundles rerun) and stops if the two sides were not scored on the same test nodes. **Exception to the `web_report` rule:** this page may contain inline JavaScript (clicking a point shows that node's card), but it must still make **no external requests** (no CDN, fonts, images or other linked resources; charts are inline SVG drawn in Python, no Plotly). Every inserted value is HTML-escaped. It contains row-level test data, shows a group-only banner, and is never committed.

Still to do (later branches; tracked as GitHub issues):

- **Done in phase 2 (merged to `main`, PR #37) and its validation (2026-10-20):** Model 2 (`nbr` block) and Model 5 (`gnnssl` block), one block per kind covering every relation, their `_no36` variants, the comparison page with paired group-bootstrap contrasts, 3-split pilot validation with pre-registered verdicts, and the #36 checks (a)–(b). See "Current state" above.
- **Phase 3 (ask before starting; [`docs/PHASE3_FEEDBACK_PLAN.md`](docs/PHASE3_FEEDBACK_PLAN.md)):** the professor's feedback first (embedding size and reduction, untrained-GNN control and GraphSAGE, the 4-feature variant, new datasets); then embedding settings chosen on training CV only; full-budget runs; `split.n_repeats: 3` for every compared rung; YelpChi; ablations.
- **Models 3 and 4:** GraphSAGE with all edges removed and heterogeneous GraphSAGE over all relations, the same code path, at least 5 seeds, one unit per (model, seed), early stopping on the saved CV folds.
- **Model 6:** supervised out-of-fold GNN features, only if the group amends invariant 1.
- **Explain:** SHAP on the tabular models, GNN explanations and relation ablations.

## Open decisions: do not decide these unilaterally

Ask before implementing anything that commits to one of these (plan §10):

- **Model 6:** amending invariant 1 to allow supervised out-of-fold GNN features.
- **Which dataset is the main one.** Plan v2 calls YelpChi the main dataset. On 2026-10-08 YelpChi was described as being for pipeline validation, so whether Amazon takes the main role in the ladder's contrasts is for the group to confirm.
- **Features flagged by the audit as possible shortcuts:** for Amazon ([#36](https://github.com/caelenm/fraud-detect-gnn/issues/36)) the user decided on 2026-10-19 to **report both ways** (see the decision log); keep #36 open until the group has seen both. A literature check (`docs/RESEARCH_PLAN.md` §11, 2026-10-19) suggests, without documentation, that `f19` is a helpfulness-vote count (leakage) and `f13`/`f15`/`f18` are rating statistics. Which variant is the headline, and whether to add a variant without every vote-derived column, is for the group after the local checks in §12. Never drop a feature without asking.
- **The earlier pilot's place in the final report** (the state at `main@3336d87`, see the decision log): omit it, or mention it as the motivating pilot.
- Adding any architecture beyond GraphSAGE and its graph-free control, any dataset beyond YelpChi and Amazon, or any text data. The professor asked for more datasets on 2026-10-21; which ones is still the group's decision (plan D4).
- **The phase 3 questions D1–D5** in [`docs/PHASE3_FEEDBACK_PLAN.md`](docs/PHASE3_FEEDBACK_PLAN.md) (what "DeepTune to reduce", "training the GNN" and "4 important features" mean, which datasets, backfilling the results database).

Also ask before changing the metrics, the split proportions or grouping rule, the classifiers, or the tuning settings.

**Decision log** (ask before changing any of these):

- **2026-10-07:** CFPB and IBM AMLworld are dropped in favour of YelpChi (main) and Amazon (replication), from CARE-GNN's preprocessed `.mat` files at commit `a64ff75`. Last CFPB state: `main@3336d87`. Reason: the professor's feedback asked for graph + tabular datasets usable by both df-analyze and a GNN, and for combining the two (GNN representations as df-analyze features).
- **2026-10-07:** YelpChi is split by user (`net_rur` connected components), and Amazon by exact-duplicate feature groups. Amazon nodes 0–3,304 are unlabelled context only.
- **2026-10-07:** For graph-derived features, the context rule is that features (never labels) of all nodes may be used, including test nodes. This replaces the earlier rule that test nodes attach only to training nodes.
- **2026-10-07:** This branch implements Model 1 (df-analyze on own features) robustly and prepares the node, graph and feature-block interfaces. GNN code is deferred.
- **2026-10-08:**
  - **YelpChi** uses its **full training set** (≈ 27.6k nodes). It validates the pipeline, so run it only a few times, not after every change.
  - **Amazon** runs the full Model 1 configuration (100 trials).
  - The **decision threshold** is **max F1 on out-of-fold training predictions** (`report.threshold_rule: max_f1`).
  - **`split.n_repeats` stays 1 for Model 1.** Set it to 3 for Amazon once Model 2 exists, so compared rungs share the same splits.
  - Long runs happen on the group's local GPU machine (plan §12).
- **2026-10-13 (phase 1, [`docs/PHASE1_AMAZON_PLAN.md`](docs/PHASE1_AMAZON_PLAN.md)):**
  - Phase 1 is **Amazon only**, on this branch: the full Model 1 run, then a shareable bundle. YelpChi code stays but is not run in phase 1.
  - The **split stays as it is** (seeded, stratified, grouped by duplicate rows; 60/40; 5 grouped CV folds), with `split.n_repeats: 1`.
  - **All 25 Amazon features are kept** pending issue [#36](https://github.com/caelenm/fraud-detect-gnn/issues/36) (`own__f13`, `own__f15`, `own__f18`, `own__f19` reach a univariate AUROC of 0.86–0.88). Do not drop or change a feature without the group.
  - **Bundles keep every tuned model** (the latest df-analyze run, with its jsonpickle results) plus the exported Model A, and **drop caches and temporary files** (`__JOBLIB_CACHE__`, `lightning_logs`, `__pycache__`, `*.tmp`, `*.part`).
  - The **`export_model` stage and `scripts/dfa/predict.py`** were added, so teammates load and use the models in df-analyze's environment without retraining.
  - The branch **goes to `main` through a PR** (this overrides plan §0's "do not push"); never merge it.
- **2026-10-18 (phase 2, [`docs/PHASE2_GNN_PLAN.md`](docs/PHASE2_GNN_PLAN.md)):**
  - Work on `feat/gnn-embeddings`, committed locally; **do not push or open a PR** until the user says so.
  - Model 5's embeddings are **self-supervised only** (Deep Graph Infomax); no label reaches any feature. Model 6 is out of scope.
  - **One block per kind covering all relations**: `nbr` (1-hop mean of every own feature plus log1p(degree) per relation) and `gnnssl` (DGI per relation, 2-layer GCN encoder, hidden/output 64, PReLU, row-shuffle corruption, Adam 1e-3, ≤ 300 epochs, patience 20, PCA to 8 per relation).
  - The scaler that feeds DGI and the PCA are **fit on training nodes only**; the DGI loss covers **non-test nodes** (train and unlabelled), and test nodes take part in message passing only.
  - **Pilot without touching phase 1**: `pilot_` feature sets with `df_analyze.classifiers=[lgbm,lr,catboost]` and `htune_trials=10`, all on the same machine.
  - The **comparison page** is a standalone script that may use inline JavaScript but makes no external requests (see "Comparison page" above).
- **2026-10-19 (user decisions):**
  - **Issue #36: report both.** Every Amazon rung is run with all 25 features and without `own__f13`, `own__f15`, `own__f18`, `own__f19` (`*_no36` feature sets; the config's `feature_variants.no36`). The four are also left out of the neighbour-aggregate and GNN inputs (`nbr_no36`, `gnnssl_no36`).
  - **GNN embeddings on the CPU** (`gnn_embed.device: cpu`). CUDA's sparse products differ by ~1e-7 between runs; the CPU is bit-identical.
  - **torch stays at 2.9** in the `gnn` extra: newer PyPI torch is built for CUDA 13, which the group laptop's driver (CUDA 12.8) cannot run.
- **2026-10-21 (user decisions):**
  - **No PR requirement.** Commits may go straight to `main`; unchecked commits are fine (it is a course project, not public software). The never-commit list, and never force-pushing or rewriting history, still apply.
  - **Results database** in `results/runs/` (one folder per run, written by `scripts/results_db.py`); every agent records the runs it finishes. Created empty; earlier runs are not backfilled unless the user asks.
  - **Professor's feedback** (embedding size, "potentially use DeepTune to reduce"; see if training the GNN helps; run on just the 4 important features; more datasets) is addressed by [`docs/PHASE3_FEEDBACK_PLAN.md`](docs/PHASE3_FEEDBACK_PLAN.md).
- **Kept from earlier work:** Model A uses df-analyze classifiers **`lgbm`, `lr`, `catboost`, `gandalf`, `rf`, `knn`** (plus the automatic dummy; `mlp` is left out because it is CPU-only in df-analyze and adds about 4 h per run), 100 tuning trials, tuned on **balanced accuracy** for every model (`htune_cls_metric: bal-acc`; not `acc`, which suits imbalanced classes poorly; not `auroc`, which df-analyze applies inconsistently across models), with `--filter-pred-classify auroc` for the prediction-based feature filter; Model A is **chosen by shared 5-fold CV PR-AUC on the training set** (`select_model`), not by df-analyze's tuning score; CatBoost and GANDALF run on the GPU; `--df-tests-method` is never passed (it hits an enum bug in df-analyze 4.1.0).

## Code conventions

- Python 3.11+ syntax (the project pins 3.13) with type hints on all public functions.
- Use `pathlib` for paths. No hard-coded absolute paths; configuration goes in `configs/` (YAML) or CLI arguments.
- Set and log random seeds for `random`, `numpy`, and `torch`. Log package versions and config for every run in the run's output directory.
- Lint and format with `ruff`; test with `pytest`.
- Notebooks are for exploration only. Anything the results depend on must live in `src/` or `scripts/`.
- Prefer clear, boring code over clever code. This is a course project that other group members will read and present.

## Required tests

Tests use small synthetic data only. **All synthetic rows live in `tests/synthetic.py`** (clearly marked, obviously fake values) and are built in memory; tests that need files (including tiny `.mat` files made with `scipy.io.savemat`) write them to pytest's `tmp_path`. Never commit data files of any kind, real or synthetic, and never write synthetic data into `data/` or `outputs/`. Maintain tests that assert:

- The Amazon unlabelled prefix never becomes a labelled example.
- Edge lists are symmetric-deduplicated (`src < dst`), with no self-loops, and a count mismatch on load fails loudly.
- No group crosses train/test or CV folds, the split is deterministic for a given seed, and the saved test IDs match df-analyze's exported test set.
- No feature table contains `label`, `node_id`, `group_id` or `split` columns, or unprefixed columns.
- Graph feature builders give identical output when the labels are permuted (label blindness).
- Feature-type decisions (`column_spec.json`) use training nodes only.
- The decision threshold is chosen without test data, and the bootstrap resamples whole groups.
- No node feature correlates perfectly with the label on synthetic data (a leakage smoke test).
- Every committed results-database entry passes `scripts/results_db.py check` (no row-level columns, at most 1,000 rows, valid UTC times).

## Checks (recommended before committing)

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
git status          # required: confirm nothing from the never-commit list is staged
```

Running ruff and pytest before a commit is recommended but not required. If you commit with a failing check, say which one in the commit message. The `git status` / `git diff --staged` look is always required.

## Writing and documentation

- Update `README.md` whenever a stage is implemented or a design decision changes.
- Do not add author names to any file.
- Do not overstate results. Report numbers with their seeds and variance, and describe limitations plainly.
