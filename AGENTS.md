# AGENTS.md

Instructions for AI coding agents (primarily Claude Code) working in this repository. Read this file fully before making changes. `README.md` describes the research design; this file describes how to work on it.

## Project in one paragraph

We classify CFPB consumer complaints as fraud/scam-related (1) or not (0) and compare two models trained on identical features and an identical train/test split: the best df-analyze (tabular AutoML) model and a heterogeneous GraphSAGE GNN in PyTorch Geometric. A graph-free control (the same network with no edges) isolates the effect of the graph. Both models are explained, and the comparison is made at the feature-group level (text, company, product, region). The headline metric is PR-AUC.

## Dataset

- The project uses **CFPB Narratives Archive files 2, 3, and 4** (complaints received May 2018 through August 2023). This range is a fixed design decision; see the README for the reasons.
- Get the data only with `./download_dataset.sh`, which writes to `data/raw/`. Do not download other archive files, change the date range, or fetch data from the live CFPB database or API without asking.
- Do not modify files in `data/raw/`. Write derived data to `data/interim/` or `data/processed/`.
- If a file lacks a populated complaint narrative column, stop and report it rather than working around it.

## Git workflow (mandatory)

This is a **public** repository. Everything pushed is visible to anyone.

- **Never commit to `main`.** Never push to `main`.
- For every task, create a branch from an up-to-date `main`, named `feat/<short-name>`, `fix/<short-name>`, `exp/<short-name>`, or `docs/<short-name>`.
- Push the branch and open a pull request with `gh pr create`. **All changes reach `main` through a PR.**
- **Never merge PRs** (including your own), never force-push, never rewrite published history, never delete remote branches, and never create tags or releases.
- Keep PRs small and focused on one stage or concern.
- The PR description must include: what changed, why, how it was tested, and the [leakage checklist](#leakage-checklist-include-in-every-pr-that-touches-data-features-or-models) if the PR touches data, features, or models.
- Before every commit, run `git status` and `git diff --staged` and confirm that nothing from the [never-commit list](#never-commit) is staged.

## Never commit

- Anything under `data/` or `outputs/` (raw archive files, samples, embeddings, PCA outputs, df-analyze results, graphs, checkpoints)
- Complaint narratives or any row-level CFPB data, including in tests, fixtures, notebooks, logs, or PR descriptions. Use small synthetic fixtures instead.
- Model weights (`*.pt`, `*.pth`, `*.ckpt`), `*.parquet`, `*.npy`, `*.npz`, `*.pkl`, or large CSVs
- `.env` files, API keys, tokens, or credentials
- Notebook outputs (clear outputs before committing)
- Any file larger than about 1 MB without asking first
- Output bundles from `pack_artifacts.py` (`fraud_artifacts_*.tgz`). They contain narratives and are shared outside git (see "Sharing outputs with the group" in the README).

If `.gitignore` does not already cover one of these, add the pattern in the same PR.

## Environments

There are two separate environments. Do not try to merge them.

| Environment | Location | Python | Used for |
|---|---|---|---|
| Project | this repo, managed by `uv` | 3.13 pinned in `.python-version`; `requires-python >=3.11` | data prep, PCA, graph construction, GNN, SHAP, evaluation |
| df-analyze | separate clone at the commit in `df_analyze.commit` (path in `DF_ANALYZE_DIR`, default `../df-analyze`) | **3.13 only** (`>=3.13.11,<3.14`; its locked catboost has no 3.14 wheels), via df-analyze's own `uv sync --locked` | `df-embed.py` and `df-analyze.py` only |

- uv 0.9.16 or newer is required (older releases cannot download Python 3.13.11 for df-analyze); `fraud_detect.external` enforces this before calling df-analyze.
- Run project code with `uv run ...`. Add dependencies with `uv add`, never with plain `pip install`. Commit `uv.lock`.
- Keep the project compatible with Python 3.11–3.13 so it runs on every group member's laptop; run the tests on each version before changing the supported range.
- Call df-analyze through a subprocess using its own environment (`uv run --directory "$DF_ANALYZE_DIR" --python '>=3.13.11,<3.14' python df-analyze.py ...`; `fraud_detect.external.run_df_analyze_script` does this). Never import df-analyze into the project environment.
- **The project must run on Linux, macOS, and Windows**, because group members use all three. **On Windows, it runs inside WSL2** (e.g. Debian or Ubuntu), with the repo cloned into the WSL filesystem (`~/...`, not `/mnt/c/...`, which is much slower). Native Windows is not a target. The pipeline, `pack_artifacts.py`, and `unpack_artifacts.py` must work on each platform. In practice:
  - Use `pathlib`, and open text files with an explicit `encoding="utf-8"`.
  - Pass subprocess arguments as a list, and never use `shell=True`.
  - Do not depend on GNU-only command-line flags: macOS ships BSD versions of `sed`, `tar`, and similar tools. Inside Python code, use the standard library (`tarfile`, `zipfile`, `shutil`) instead of shelling out.
  - Never hard-code `cuda`. WSL2 and Linux laptops may have an NVIDIA GPU; Macs never do.
  - Say in the PR which platforms a change was tested on.
- The main development machine is a Linux laptop (or Windows with WSL2) with an NVIDIA GPU. Always check `torch.cuda.is_available()` and fall back to CPU; never hard-code `cuda`.
- Hardware is limited. Ask before starting any job you expect to run longer than about 1 hour. For df-analyze on 30k rows, do not enable wrapper feature selection unless asked.

## Leakage checklist (include in every PR that touches data, features, or models)

These are hard invariants. If a change would violate one, stop and ask.

1. **Issue and Sub-issue never appear in any feature matrix, node feature, or edge.** They are used only to build the label.
2. **No feature anywhere is derived from labels.** This includes company-level fraud rates and any other target encoding, even when computed on training data only.
3. **Company-level statistics** (complaint counts, response-type rates, timely-response rate) **are computed from training complaints only**, and per-complaint company features are leave-one-out for training complaints (a complaint never sees its own outcome).
4. **The test set is frozen.** Both models are evaluated on exactly the same test Complaint IDs. Never tune hyperparameters, choose thresholds, select features, early-stop, or pick graph settings using the test set. Use a validation split carved from training data.
5. **Near-duplicate narratives are removed before splitting**, so that the same or nearly the same text cannot appear in both train and test.
6. **Similarity edges from test complaints point only to training complaints.**
7. **GNN evaluation is inductive.** Test complaint nodes and their edges are excluded from the training graph and added only at evaluation time.
8. **PCA on text embeddings is unsupervised, fit on training complaints only, and shared by both models.** Do not refit it per model.
9. **Choose the best df-analyze model by its internal-CV tuning score** (`tuned_models_*.csv`), never by holdout results. df-analyze's `5-fold` results table refits models on test-set folds and must never be reported as a test result.

## Pipeline stages

`run.py` is the single entry point and runs the stages in order; `uv run run.py --list` shows them. Stages are defined in `src/fraud_detect/pipeline.py` (`STAGES`), and each also has a thin wrapper `scripts/NN_<stage>.py`. Each stage reads from and writes to `data/` or `outputs/` and must not depend on in-memory state from a previous stage. **Add every new stage to `STAGES` (and a matching script) so `run.py` stays the throughline.**

Implemented (tested on synthetic data only so far):

0. **Download:** `./download_dataset.sh`.
1. **`load`:** read the three archive files from `data/raw/`, keep rows with a non-empty narrative received May 2018–Aug 2023, and report every Product/Issue/Sub-issue combination with counts (`outputs/reports/category_values.csv`).
2. **`label`:** apply the reviewed allow-list in `configs/categories.yaml` (see the label rule in the README), keep the target products, and drop Issue/Sub-issue. Refuses to run until the file is confirmed and consistent with the data.
3. **`sample`:** remove near-duplicate narratives (MinHash LSH), then draw a stratified 30k sample at the natural fraud rate.
4. **`split`:** stratified 60/40 train/test split; save `train_ids.csv` / `test_ids.csv`. We own the split and pass it to df-analyze.
5. **`embed`:** write the `text`/`label` parquet that `df-embed.py` expects and embed it in the df-analyze environment: by default with `scripts/dfa/embed_on_device.py` (df-embed's code on the GPU, verified against its CPU path), or `df-embed.py` itself (`embed.runner: df-embed`).
6. **`pca`:** 30 components, fit on training complaints only.
7. **`features`:** tabular and company features. Company statistics come from training complaints only and are leave-one-out for training complaints; categorical levels with <20 training complaints (or unseen in training) are merged using training counts only.
8. **`df_analyze_input`:** train/test tables without identifiers, in the order of the saved IDs.
9. **`df_analyze`:** run df-analyze with `--df-train` / `--df-tests`, then **verify** that its exported `X_train`/`X_test`/`y_*` match our saved split row for row. If verification fails, stop and ask.
10. **`df_analyze_report`:** compute test metrics (PR-AUC headline, AUROC, fraud-class F1/precision/recall) for every tuned combination from df-analyze's saved test probabilities, and choose Model A by internal-CV tuning score only (invariant 9); the dummy is never Model A.

Still to do (tracked as GitHub issues):

11. **Graph:** build a PyG `HeteroData` graph with `complaint`, `company`, `product`, and `region` (state) nodes, reverse edges, and optional complaint kNN edges built with approximate nearest-neighbour search and a cap on edges per node.
12. **Model B:** train heterogeneous GraphSAGE with neighbour sampling, class-weighted BCE, and early stopping on validation. Train the graph-free control with the same code path and edges removed. Use at least 5 seeds.
13. **Evaluate:** compute PR-AUC (headline), F1, recall, and AUROC on the frozen test IDs, reported as mean ± standard deviation across seeds.
14. **Explain:** SHAP on the best df-analyze model; GNNExplainer, group permutation importance, and edge-type ablations for the GNN; a feature-group comparison table across both models.

## Open decisions: do not decide these unilaterally

Ask before implementing anything that commits to one of these:

- Train/test split strategy: random stratified (implemented, the main comparison) vs. company-holdout
- Whether complaint–complaint similarity edges are included, and the value of k
- The product filter (`configs/categories.yaml`); the date range is already fixed at May 2018 to August 2023
- The target variable itself: binary fraud (current plan) vs. company-response relief vs. grouped multiclass. Switching targets changes the leakage rules, so ask first.
- Adding R-GCN or any architecture beyond heterogeneous GraphSAGE and the graph-free control

Also ask before changing the label definition, the metrics, the embedding model, or the number of PCA components.

Decided (ask before changing): region is **state only**; the sample is **30,000 complaints at the natural fraud rate**; the label is a **reviewed allow-list** (`configs/categories.yaml`, confirmed; ambiguous credit-report categories and lost/stolen instruments are negative, money-transfer "Unauthorized transactions or other transaction problem" is positive); target products include the 2023 renames "Credit card" and "Prepaid card" but not "Debt or credit management"; PCA is **fit on training complaints only**; Model A uses df-analyze classifiers **`lgbm`, `lr`, `catboost`, `gandalf`, `rf`, `knn`** (plus the automatic dummy; `mlp` is dropped for now because it is CPU-only in df-analyze and adds about 4 h per run) tuned on **AUROC** (`htune_cls_metric: auroc`, plus `--filter-pred-classify auroc`; df-analyze's accuracy default suits the ~16% positive rate poorly, and it offers no PR-AUC); CatBoost and GANDALF run on the GPU; embeddings are computed on the GPU with df-embed's own code (`scripts/dfa/embed_on_device.py`), checked against its CPU path.

## Code conventions

- Python 3.11+ syntax (the project pins 3.13) with type hints on all public functions.
- Use `pathlib` for paths. No hard-coded absolute paths; configuration goes in `configs/` (YAML) or CLI arguments.
- Set and log random seeds for `random`, `numpy`, and `torch`. Log package versions and config for every run in the run's output directory.
- Lint and format with `ruff`; test with `pytest`.
- Notebooks are for exploration only. Anything the results depend on must live in `src/` or `scripts/`.
- Prefer clear, boring code over clever code. This is a course project that other group members will read and present.

## Required tests

Tests use small synthetic data only. **All synthetic rows live in `tests/synthetic.py`** (clearly marked, obviously fake values) and are built in memory; tests that need files write them to pytest's `tmp_path`. Never commit data files of any kind, real or synthetic, and never write synthetic data into `data/` or `outputs/`. Maintain tests that assert:

- The label rule produces the expected labels on hand-written examples.
- No Issue/Sub-issue column, or column derived from them, exists in any feature output.
- Train and test Complaint IDs are disjoint, and the saved test IDs match df-analyze's exported test set.
- Company statistics are unchanged when test rows are modified or removed.
- The training graph contains no test complaint nodes, and similarity edges never point from training to test.
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
