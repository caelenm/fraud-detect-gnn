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

If `.gitignore` does not already cover one of these, add the pattern in the same PR.

## Environments

There are two separate environments. Do not try to merge them.

| Environment | Location | Python | Used for |
|---|---|---|---|
| Project | this repo, managed by `uv` | 3.14 | data prep, PCA, graph construction, GNN, SHAP, evaluation |
| df-analyze | separate clone (path in `DF_ANALYZE_DIR`, default `../df-analyze`) | managed by df-analyze's own `uv sync` | `df-embed.py` and `df-analyze.py` only |

- Run project code with `uv run ...`. Add dependencies with `uv add`, never with plain `pip install`.
- Call df-analyze through a subprocess using its own environment (`uv run --directory "$DF_ANALYZE_DIR" python df-analyze.py ...`). Never import df-analyze into the project environment.
- The machine is a Linux laptop with an NVIDIA GPU. Always check `torch.cuda.is_available()` and fall back to CPU; never hard-code `cuda`.
- Hardware is limited. Ask before starting any job you expect to run longer than about 1 hour. For df-analyze on 20k–30k rows, do not enable wrapper feature selection unless asked.

## Leakage checklist (include in every PR that touches data, features, or models)

These are hard invariants. If a change would violate one, stop and ask.

1. **Issue and Sub-issue never appear in any feature matrix, node feature, or edge.** They are used only to build the label.
2. **No feature anywhere is derived from labels.** This includes company-level fraud rates and any other target encoding, even when computed on training data only.
3. **Company-level statistics** (complaint counts, response-type rates, timely-response rate) **are computed from training complaints only.**
4. **The test set is frozen.** Both models are evaluated on exactly the same test Complaint IDs. Never tune hyperparameters, choose thresholds, select features, early-stop, or pick graph settings using the test set. Use a validation split carved from training data.
5. **Near-duplicate narratives are removed before splitting**, so that the same or nearly the same text cannot appear in both train and test.
6. **Similarity edges from test complaints point only to training complaints.**
7. **GNN evaluation is inductive.** Test complaint nodes and their edges are excluded from the training graph and added only at evaluation time.
8. **PCA on text embeddings is unsupervised and shared by both models.** Do not refit it per model.

## Pipeline stages

Implement stages as separate, rerunnable scripts under `scripts/`, with logic in `src/fraud_detect/`. Each stage reads from and writes to `data/` or `outputs/` and must not depend on in-memory state from a previous stage.

0. **Download:** `./download_dataset.sh` (already implemented).
1. **Load and label:** read the three archive files from `data/raw/`, keep rows with a non-empty narrative, apply the label rule, and drop Issue/Sub-issue from features. Before relying on the label rule, report the actual Issue and Sub-issue values and counts in this date range.
2. **Deduplicate and sample:** remove near-duplicate narratives, filter to the target products, and draw a stratified sample of 20k–30k complaints.
3. **Embed:** write the parquet with `text` and `label` columns that `df-embed.py` expects, run it in the df-analyze environment, then reduce to about 30 PCA components.
4. **Model A:** build the df-analyze input table and run df-analyze. Save its exported split.
5. **Split mapping:** map df-analyze's exported `X_train.csv` / `X_test.csv` rows back to Complaint IDs and save `train_ids` / `test_ids`. df-analyze drops identifier columns and re-encodes features, so **verify row alignment explicitly and add a test for it before any GNN work depends on it.** If alignment cannot be verified, stop and ask.
6. **Graph:** build a PyG `HeteroData` graph with `complaint`, `company`, `product`, and `region` nodes, reverse edges, and optional complaint kNN edges built with approximate nearest-neighbour search and a cap on edges per node.
7. **Model B:** train heterogeneous GraphSAGE with neighbour sampling, class-weighted BCE, and early stopping on validation. Train the graph-free control with the same code path and edges removed. Use at least 5 seeds.
8. **Evaluate:** compute PR-AUC (headline), F1, recall, and AUROC on the frozen test IDs, reported as mean ± standard deviation across seeds.
9. **Explain:** SHAP on the best df-analyze model; GNNExplainer, group permutation importance, and edge-type ablations for the GNN; a feature-group comparison table across both models.

## Open decisions: do not decide these unilaterally

Ask before implementing anything that commits to one of these:

- Train/test split strategy: random stratified (current default) vs. company-holdout
- Whether complaint–complaint similarity edges are included, and the value of k
- Region granularity (ZIP-3, state, or both)
- Final sample size and product filter (the date range is already fixed at May 2018 to August 2023)
- The target variable itself: binary fraud (current plan) vs. company-response relief vs. grouped multiclass. Switching targets changes the leakage rules, so ask first.
- Adding R-GCN or any architecture beyond heterogeneous GraphSAGE and the graph-free control

Also ask before changing the label definition, the metrics, the embedding model, or the number of PCA components.

## Code conventions

- Python 3.14 with type hints on all public functions.
- Use `pathlib` for paths. No hard-coded absolute paths; configuration goes in `configs/` (YAML) or CLI arguments.
- Set and log random seeds for `random`, `numpy`, and `torch`. Log package versions and config for every run in the run's output directory.
- Lint and format with `ruff`; test with `pytest`.
- Notebooks are for exploration only. Anything the results depend on must live in `src/` or `scripts/`.
- Prefer clear, boring code over clever code. This is a course project that other group members will read and present.

## Required tests

Tests use small synthetic data only. Maintain tests that assert:

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
