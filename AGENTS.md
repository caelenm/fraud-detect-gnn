# AGENTS.md

Instructions for AI coding agents (primarily Claude Code) working in this repository. Read this file fully before making changes. `README.md` describes the research design; this file describes how to work on it. **[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) is the working research plan:** the model ladder, verified dataset facts, splits, interfaces, milestones with acceptance criteria, and the open decisions. Read it before picking up any work item. If it conflicts with this file, stop and ask.

## Project in one paragraph

We ask: **does using a graph improve fraud classification over the same kind of model without one, and can a GNN's representations improve a tabular AutoML pipeline (df-analyze)?** The datasets are two graph fraud benchmarks with tabular node features: **YelpChi** (main; nodes are hotel and restaurant reviews, labelled spam by Yelp's filter, three review–review relations) and **Amazon** (replication; nodes are users, labelled fraudulent or not, three user–user relations). Both come from the preprocessed `.mat` files of the CARE-GNN repository. The model ladder (plan §1) is: (1) df-analyze on each node's own features (Model A); (2) df-analyze plus label-free neighbour aggregates per relation; (3) GraphSAGE with all edges removed; (4) heterogeneous GraphSAGE over all relations; (5) df-analyze plus self-supervised GNN embeddings; (6) df-analyze plus supervised, out-of-fold GNN features (needs a rule change). Every model on a dataset is trained on the same split and scored on the same frozen, grouped test set. The headline metric is PR-AUC, always reported next to the no-skill baseline (the test positive rate). Model 1 and the interfaces the later rungs need (node table, relation edge lists, feature blocks, feature sets) are implemented first; GNN code comes later.

## Datasets

- **Source:** the CARE-GNN repository (Dou et al., CIKM 2020; Apache-2.0), commit `a64ff7523e187a24251f7ca88435d2c9d8f7dcd9`, folder `data/`: `YelpChi.zip` and `Amazon.zip`, each holding one `.mat` file. The SHA-256 checksums of the zips and the `.mat` files are in plan §2 and `configs/default.yaml`.
- Get the data only with the `download` stage (`uv run run.py --dataset <name> --only download`, or `./download_dataset.sh`), which writes to `data/raw/care_gnn/` and verifies every checksum. Do not fetch other datasets, other commits, or DGL/PyG copies of these files without asking. **Do not add DGL as a dependency;** read the files with `scipy.io.loadmat`.
- Do not modify files in `data/raw/`. Write derived data to `data/processed/<dataset>/` and outputs to `outputs/<dataset>/`.
- **Check the counts on load** against plan §2 (nodes, labelled nodes, positives, features, edges per relation). If they differ, stop and report rather than working around it.
- **Amazon's first 3,305 nodes are unlabelled but stored with label 0.** They are never labelled examples (never positives, never negatives); they stay in the node table as `is_labelled = False`, as graph context only.
- **YelpChi labels are user-level.** A user is a connected component of `net_rur`. All of a user's reviews stay on one side of every split.
- **Amazon has exact-duplicate feature rows** among labelled nodes; each set of identical rows is one split group.
- Features are anonymized: `f00`, `f01`, … in the `.mat` column order, prefixed `own__` in feature tables.

## Git workflow (mandatory)

This is a **public** repository. Everything pushed is visible to anyone.

- **Never commit to `main`.** Never push to `main`.
- For every task, create a branch from an up-to-date `main`, named `feat/<short-name>`, `fix/<short-name>`, `exp/<short-name>`, or `docs/<short-name>`.
- Push the branch and open a pull request with `gh pr create`. **All changes reach `main` through a PR.** If the user or the active plan says not to push yet (plan §0 does for `feat/review-fraud-pivot`), commit locally and wait until the user says so.
- **Never merge PRs** (including your own), never force-push, never rewrite published history, never delete remote branches, and never create tags or releases.
- Keep PRs small and focused on one stage or concern.
- The PR description must include: what changed, why, how it was tested, and the [leakage checklist](#leakage-checklist-include-in-every-pr-that-touches-data-features-or-models) if the PR touches data, features, or models.
- Before every commit, run `git status` and `git diff --staged` and confirm that nothing from the [never-commit list](#never-commit) is staged.

## Never commit

- Anything under `data/` or `outputs/` (raw `.zip`/`.mat` files, node tables, edge lists, feature blocks, df-analyze results, reports, checkpoints)
- Row-level data from any dataset (node features, labels, IDs with their labels), including in tests, fixtures, notebooks, logs, or PR descriptions. Tests use small synthetic data built in memory (see [Required tests](#required-tests)).
- Model weights (`*.pt`, `*.pth`, `*.ckpt`), `*.mat`, `*.parquet`, `*.npy`, `*.npz`, `*.pkl`, or large CSVs
- `.env` files, API keys, tokens, or credentials
- Notebook outputs (clear outputs before committing)
- Any file larger than about 1 MB without asking first
- Output bundles from `pack_artifacts.py` (`fraud_artifacts_*.tgz`). They are shared outside git (see "Sharing outputs with the group" in the README).
- `PR_DESCRIPTION.md` (a draft the user pastes into GitHub)

If `.gitignore` does not already cover one of these, add the pattern in the same PR.

## Environments

There are two separate environments. Do not try to merge them.

| Environment | Location | Python | Used for |
|---|---|---|---|
| Project | this repo, managed by `uv` | 3.13 pinned in `.python-version`; `requires-python >=3.11` | download, loading, splits, features, audit, reports (later: graph features and GNNs) |
| df-analyze | separate clone at the commit in `configs/default.yaml` (`df_analyze.commit`; path in `DF_ANALYZE_DIR`, default `../df-analyze`) | **3.13 only** (`>=3.13.11,<3.14`; its locked catboost has no 3.14 wheels), via df-analyze's own `uv sync --locked` | `df-analyze.py` and `scripts/dfa/cv_select.py` only |

- uv 0.9.16 or newer is required (older releases cannot download Python 3.13.11 for df-analyze); `fraud_detect.external` enforces this before calling df-analyze.
- Run project code with `uv run ...`. Add dependencies with `uv add`, never with plain `pip install`. Commit `uv.lock`.
- Keep the project compatible with Python 3.11–3.13 so it runs on every group member's laptop; run the tests on each version before changing the supported range.
- Call df-analyze through a subprocess using its own environment (`uv run --directory "$DF_ANALYZE_DIR" --python '>=3.13.11,<3.14' python df-analyze.py ...`; `fraud_detect.external.run_df_analyze_script` does this). Never import df-analyze into the project environment.
- **The project must run on Linux, macOS, and Windows**, because group members use all three. **On Windows, it runs inside WSL2** (e.g. Debian or Ubuntu), with the repo cloned into the WSL filesystem (`~/...`, not `/mnt/c/...`, which is much slower). Native Windows is not a target. The pipeline, `pack_artifacts.py`, and `unpack_artifacts.py` must work on each platform. In practice:
  - Use `pathlib`, and open text files with an explicit `encoding="utf-8"`.
  - Pass subprocess arguments as a list, and never use `shell=True`.
  - Do not depend on GNU-only command-line flags: macOS ships BSD versions of `sed`, `tar`, and similar tools. Inside Python code, use the standard library (`tarfile`, `zipfile`, `shutil`, `urllib`) instead of shelling out.
  - Never hard-code `cuda`. WSL2 and Linux laptops may have an NVIDIA GPU; Macs never do.
  - Say in the PR which platforms a change was tested on.
- The main development machine is a Linux laptop (or Windows with WSL2) with an NVIDIA GPU. Always check `torch.cuda.is_available()` and fall back to CPU; never hard-code `cuda`. In df-analyze, CatBoost and GANDALF use the GPU; `gpu.require` decides whether a missing GPU stops the run.
- Hardware is limited. Ask before starting any job you expect to run longer than about 1 hour. Do not enable df-analyze's wrapper feature selection unless asked.

## Leakage checklist (include in every PR that touches data, features, or models)

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

`run.py` is the single entry point and runs the stages in order for one dataset: `uv run run.py --dataset yelpchi` (or `dataset:` in the config); `uv run run.py --list` shows the stages. Stages are defined in `src/fraud_detect/pipeline.py` (`STAGES`), and each also has a thin wrapper `scripts/NN_<stage>.py`. Each stage reads from and writes to `data/` or `outputs/` under the dataset's subfolder and must not depend on in-memory state from a previous stage. **Add every new stage to `STAGES` (and a matching script) so `run.py` stays the throughline.** Stages 6–10 run for one feature set (`--set feature_set=m1_own`, the default) and write under `outputs/<dataset>/<feature_set>/`.

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
6. **`df_analyze_input`:** train/test tables for the active feature set, joining its blocks on `node_id` in the order of the saved IDs, with only feature columns and `target`.
7. **`df_analyze`:** run df-analyze with `--df-train` / `--df-tests`, then **verify** that its exported `X_train`/`X_test`/`y_*` match our saved split row for row, and that its inferred column types match `column_spec.json` with no feature dropped. If either check fails, stop and ask.
8. **`select_model`:** in df-analyze's environment (`scripts/dfa/cv_select.py`), refit every tuned combination on the saved grouped folds and score the held-out folds from probabilities (PR-AUC, AUROC, balanced accuracy, Brier), with tuned and default settings; save out-of-fold probabilities and the tuning budget. Stops if any tuned model cannot be scored.
9. **`df_analyze_report`:** choose Model A by shared-CV PR-AUC only; its decision threshold from its out-of-fold training predictions (max F1 by default); test metrics for every tuned combination at that threshold and at 0.5; group-bootstrap 95% intervals for Model A; a warning if Model A falls more than 0.05 PR-AUC below the sanity band (plan §2).
10. **`web_report`:** `outputs/<dataset>/<feature_set>/report/index.html`, a self-contained static page (no scripts, no external requests) built by `src/fraud_detect/report/web.py`. Its confidence cards show only **test** nodes (ID, label, score, top features by value). Every value inserted into the page must be HTML-escaped.

Still to do (later branches; tracked as GitHub issues):

- **Model 2, neighbour features:** label-free aggregates of each node's neighbours' own features, one block per relation (`nbr_rur`, …), run through the same df-analyze pipeline as a new feature set.
- **Models 3 and 4:** GraphSAGE with all edges removed and heterogeneous GraphSAGE over all relations, the same code path, at least 5 seeds, one unit per (model, seed), early stopping on the saved CV folds.
- **Model 5:** self-supervised GNN embeddings per relation, PCA-compressed (fit on training nodes), as feature blocks (`gnnssl_rur`, …).
- **Model 6:** supervised out-of-fold GNN features, only if the group amends invariant 1.
- **Evaluate and explain:** paired group-bootstrap contrasts (5 − 1, 2 − 1, 4 − 3), SHAP on the tabular models, GNN explanations and relation ablations.

## Open decisions: do not decide these unilaterally

Ask before implementing anything that commits to one of these (plan §10):

- **Model 6:** amending invariant 1 to allow supervised out-of-fold GNN features.
- **Which dataset is the main one.** Plan v2 calls YelpChi the main dataset. On 2026-10-08 YelpChi was described as being for pipeline validation, so whether Amazon takes the main role in the ladder's contrasts is for the group to confirm.
- **Features flagged by the audit as possible shortcuts:** keep, drop, or report results both ways. Never drop a feature without asking.
- **The earlier pilot's place in the final report** (the state at `main@3336d87`, see the decision log): omit it, or mention it as the motivating pilot.
- Adding any architecture beyond GraphSAGE and its graph-free control, any dataset beyond YelpChi and Amazon, or any text data.

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

## Before opening a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
git status          # confirm nothing from the never-commit list is staged
```

All checks must pass. If a check fails and you cannot fix it, open the PR as a draft and explain the failure in the description.

## Writing and documentation

- Update `README.md` whenever a stage is implemented or a design decision changes.
- Do not add author names to any file.
- Do not overstate results. Report numbers with their seeds and variance, and describe limitations plainly.
