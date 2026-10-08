# fraud-detect-gnn

**Does a graph help? Tabular AutoML vs. graph neural networks for fraud detection on two graph fraud benchmarks (YelpChi and Amazon)**

Course project for CS555: Data Mining and Machine Learning, St. Francis Xavier University (StFX).

> **Status:** Model 1 (df-analyze on each node's own features) is implemented end to end for both datasets: download, load, grouped split, features, audit, df-analyze, grouped shared-CV model choice, test report with bootstrap intervals, and an HTML report. The node table, relation edge lists and feature-block interfaces that the graph models need are in place; the graph models themselves (Models 2–6) are not built yet. The working plan is [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md); its §12 lists the next steps (the full Amazon run, then one full YelpChi run, on a local GPU machine). An earlier version of this project used CFPB complaint narratives; that state is preserved at commit `3336d87` on `main`.

---

## Overview

We ask: **does using a graph improve fraud classification over the same kind of model without one, and can a GNN's representations improve a tabular AutoML pipeline?**

- **Tabular models.** [df-analyze](https://github.com/stfxecutables/df-analyze) treats each node as an independent row and searches over classifiers, feature selection methods and hyperparameters. Its best model is Model A.
- **Graph models.** Later rungs add the graph: hand-made neighbour aggregates, GraphSAGE in [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), and GNN embeddings fed back into df-analyze.

Both datasets come with precomputed, anonymous node features *and* several relations between nodes, so every rung of the ladder can be run on exactly the same nodes, split and metric.

## The model ladder

| # | Model | Graph information | Status |
|---|---|---|---|
| 1 | df-analyze on each node's own features (Model A) | none | **implemented** |
| 2 | df-analyze on own features + label-free neighbour aggregates, per relation | hand-made | interfaces ready |
| 3 | GraphSAGE with all edges removed (an MLP) | none | planned |
| 4 | Heterogeneous GraphSAGE over all relations | learned | planned |
| 5 | df-analyze on own features + self-supervised GNN embeddings, per relation, PCA-compressed | learned, no labels | interfaces ready |
| 6 | df-analyze on own features + supervised, out-of-fold GNN embeddings or score | learned from labels | needs a rule change (open decision) |

The key contrasts: **5 vs 1** (does the GNN help df-analyze?), **5 vs 2** (learned embeddings vs hand-made aggregates) and **4 vs 3** (message passing with the architecture held fixed). The headline metric is **PR-AUC** (average precision) on a frozen, grouped test set, always next to the no-skill baseline (the test positive rate).

## Datasets

Both are the preprocessed `.mat` files from the CARE-GNN repository ([Dou et al., CIKM 2020](https://arxiv.org/abs/2008.08692), Apache-2.0) at commit `a64ff75`, the files GADBench and DGL's `FraudYelpDataset` / `FraudAmazonDataset` are built from. The `download` stage verifies their SHA-256 checksums (`configs/default.yaml`).

| | YelpChi (main) | Amazon (replication) |
|---|---|---|
| Node | a hotel or restaurant review | a user reviewing musical instruments |
| Label | spam (1) if Yelp's filter removed the review, else recommended (0) ([Rayana & Akoglu, KDD 2015](https://doi.org/10.1145/2783258.2783370)) | fraudulent (1) if fewer than 20% of the user's helpful votes are positive, benign (0) if more than 80% (CARE-GNN; DGL's documentation) |
| Nodes | 45,954, all labelled | 11,944, of which 8,639 labelled (nodes 0–3,304 have no label) |
| Positives | 6,677 (14.5%) | 821 (9.5% of labelled) |
| Features | 32 handcrafted review and reviewer features (Rayana & Akoglu), scaled to [0, 1] | 25 handcrafted user features (Zhang et al., SIGIR 2020) |
| Relations (undirected edges) | R-U-R same user (49,315); R-T-R same product, same month (573,616); R-S-R same product, same star rating (3,402,743) | U-P-U reviewed a common product (175,608); U-S-U same star rating within a week (3,566,479); U-V-U top-5% review-text similarity (1,036,737) |

What the loader handles:

- **Amazon's first 3,305 nodes are unlabelled but stored with label 0.** They become `label = null` and are never used as negatives; they stay in the graph as context.
- **YelpChi labels are per user.** Reviews by one user (a connected component of R-U-R; 29,431 users) almost always share a label (99.6% within multi-review users in training), so a random split would let the model recognise users. The split keeps each user's reviews together.
- **Amazon has 293 exact-duplicate feature rows** among labelled users. Each set of identical rows is kept on one side of the split.
- **Features are anonymous**, named `own__f00`, `own__f01`, … in `.mat` column order.

The `audit` stage reports, from training nodes only: per-feature statistics and univariate AUROC, duplicate rows within and across the split, group sizes, and each relation's degree and label homophily. Our training-only "positive homophily" (the share of a fraud node's edges that lead to another fraud node) matches CARE-GNN's published label similarity: YelpChi 0.902 / 0.176 / 0.185 / 0.184 (R-U-R / R-T-R / R-S-R / all) vs. their 0.909 / 0.176 / 0.186 / 0.184. Four Amazon features reach a univariate AUROC of at least 0.85 (up to 0.88), which may mean a vote-derived feature partly encodes the label; whether to keep, drop, or report both ways is an open decision.

## Splits

One split per dataset, made by us, saved, and passed to df-analyze. The test set is touched only by the final report stages.

- **Grouped and stratified:** `StratifiedGroupKFold` over the labelled nodes, with 2 of 5 folds as the test set (about 60/40). Groups are YelpChi users and Amazon duplicate sets.
  - YelpChi: 27,571 train / 18,383 test.
  - Amazon: 5,183 / 3,456.
- **Shared CV folds:** the training nodes are split again into 5 grouped, stratified folds (`cv_folds.csv`). Every model choice uses these folds, and so will any later out-of-fold feature.
- **Split repeats:** `split.n_repeats` (default 1) runs the whole pipeline on several independently seeded splits and reports Model A's mean ± SD. Each repeat is a full df-analyze run. Decided: 1 for Model 1; 3 for Amazon once Model 2 exists.

The leakage rules (labels only through the training loss, grouped splits, frozen test set, label-free graph features, train-only transforms) are in [`AGENTS.md`](AGENTS.md).

## Model A: df-analyze

df-analyze runs six classifiers: LightGBM, logistic regression, CatBoost, GANDALF, a LightGBM random forest, and k-nearest neighbours, plus a dummy baseline. Each is tuned with 100 Optuna trials on each of df-analyze's feature selections. Lessons from earlier runs are built in:

- **Model A is chosen by our shared CV, not by df-analyze's tuning scores.** df-analyze tunes most models on balanced accuracy of hard predictions over 5 random folds, but GANDALF on one validation split through a different code path, so its scores are not comparable across models. Its internal folds are also not grouped, which makes them optimistic here. The `select_model` stage therefore refits every tuned (classifier, feature selection) combination on the saved grouped folds and scores it from predicted probabilities. Model A has the highest mean CV PR-AUC. Each combination is also refit with df-analyze's default hyperparameters, to show what tuning added.
- **Tuning metric:** balanced accuracy (`bal-acc`) for every model. Not accuracy, which always-"not fraud" would win at 85–90%. Not AUROC, which df-analyze silently replaces with balanced accuracy for most models.
- **Column types are explicit.** The `features` stage types each column on training nodes only: binary (2 values), ordinal (integer-valued, more than 2 values), or continuous. Ordinals are passed to df-analyze. After the run, the pipeline reads df-analyze's `inferred_types.csv` and stops if any column was typed differently or dropped. On the real tables df-analyze's inference matches ours exactly.
- **Column names:** df-analyze renames `own__f00` to `own_f00` (it collapses repeated underscores), so the df-analyze input already uses its names. A mapping back to the block names is saved in `feature_set.json`.
- **The split is ours.** After each run the pipeline checks df-analyze's exported train/test tables against our split row by row. df-analyze's `5-fold` results refit on test folds and are never reported. `--df-tests-method` is never passed (it hits an enum bug in df-analyze 4.1.0). The MLP is left out (CPU-only in df-analyze; about 4 h per run).

## Evaluation

- **PR-AUC** is the headline metric; AUROC is reported alongside. Both use the fraud probability and need no threshold.
- **Decision threshold:** F1, precision, recall, balanced accuracy and the confusion matrix use each model's threshold chosen to maximise F1 on its **out-of-fold training predictions** (`report.threshold_rule`). They are also reported at 0.5. The threshold is never chosen on test data.
- **95% intervals** for Model A's test PR-AUC and AUROC come from 1,000 bootstrap resamples of **whole groups** (users or duplicate sets), seeded. The same function computes paired differences between two models on the same test nodes, which the ladder's contrasts need later. The intervals cover test-set sampling, not variation across splits (use `split.n_repeats` for that).
- **Sanity band:** an untuned gradient-boosting model reaches a test PR-AUC of about 0.80 on YelpChi and 0.91 on Amazon. If Model A lands more than 0.05 below this, the report prints a prominent warning: assume a pipeline bug before reporting anything.

## Prepared for the graph models

Nothing on this branch trains a GNN, but the data model is ready for one:

- **Node table** `data/processed/<dataset>/nodes.parquet`: `node_id`, `label` (null when unlabelled), `is_labelled`, `group_id`, `split` (train/test/unlabelled), `cv_fold`.
- **Relations** `data/processed/<dataset>/graph/<relation>.npz` (plus `homo`): undirected int32 edge lists with `src < dst`, no self-loops, deduplicated; `graph/manifest.json` records counts and the source checksum.
- **Feature blocks** `data/processed/<dataset>/features/<block>.parquet` keyed by `node_id`, with columns prefixed `<block>__`. Only `own` exists so far; later blocks are e.g. `nbr_rur` and `gnnssl_rur`.
- **Feature sets** in the config (`feature_sets:`; for example `m1_own: [own]`) are the ladder's rungs. `--set feature_set=<name>` runs stages 6–10 for one set, writing to `outputs/<dataset>/<feature_set>/`.
- **Label blindness:** graph feature builders take `(features, edges)` and never labels. [`tests/test_label_blindness.py`](tests/test_label_blindness.py) holds the check every builder must pass: identical output after permuting all labels. Graph features may use the features (never labels) of all nodes, test and unlabelled ones included.

## Setup

Two separate Python environments are needed, because df-analyze manages its own pinned dependencies. `uv` installs the right Python version for each automatically.

**Requirements**
- **Operating system:** Linux, macOS, or Windows through **WSL2**.
  - On Windows, run every command in the WSL terminal.
  - Clone into the WSL home directory (`~/...`), not `/mnt/c/...`, which is much slower.
- **[uv](https://docs.astral.sh/uv/) 0.9.16 or newer.** Older releases cannot download Python 3.13.11, which df-analyze needs.
  - Check with `uv --version`; upgrade with `uv self update`.
  - If uv came from a system package manager, run `uv tool install 'uv>=0.9.16'` and put `~/.local/bin` first on `PATH`.
- **Git.**
- **An NVIDIA GPU** is used by df-analyze for CatBoost and GANDALF. Without one, set `gpu.require: false`.

| Environment | Python | Used for |
|---|---|---|
| This project | 3.13 (pinned in `.python-version`; 3.11+ supported) | everything except df-analyze |
| df-analyze (separate clone) | 3.13.11+ (not 3.14) | `df-analyze.py` and `scripts/dfa/*.py`, called as subprocesses |

```bash
# 1. This project
git clone https://github.com/caelenm/fraud-detect-gnn.git
cd fraud-detect-gnn
uv sync

# 2. df-analyze, next to this repository, at the tested commit, on Python 3.13
uv python install '>=3.13.11,<3.14'
git clone https://github.com/stfxecutables/df-analyze.git ../df-analyze
git -C ../df-analyze checkout 199e5638620693c267dac715784f1fd0e33fa796
uv sync --locked --python '>=3.13.11,<3.14' --directory ../df-analyze

# 3. The data (checksum-verified; safe to rerun)
./download_dataset.sh          # both datasets into data/raw/care_gnn/
```

Some details on the df-analyze clone:
- **Pinned commit.** The commit is recorded as `df_analyze.commit` in `configs/default.yaml`. The stages that call df-analyze stop if the clone is at a different commit.
- **Python 3.13, not 3.14.** df-analyze must run on 3.13 because its locked `catboost` publishes 3.13 wheels only. The pipeline passes `--python '>=3.13.11,<3.14'` itself.
- **Location.** If df-analyze is not at `../df-analyze`, set `DF_ANALYZE_DIR` or `df_analyze.dir`.

## Running the pipeline

`run.py` runs the stages in order for one dataset. Stages whose outputs already exist are skipped. Once a stage runs, every later stage in the same invocation reruns too, because its inputs changed. After changing the config, rerun the affected stages with `--force`.

```bash
uv run run.py --list                                   # show the stages
uv run run.py --dataset amazon --to audit              # data stages only (minutes)
uv run run.py --dataset amazon                         # everything
uv run run.py --dataset amazon --from df_analyze --force --set df_analyze.htune_trials=10  # pilot
uv run run.py --dataset yelpchi --only audit           # one stage
```

`--set section.key=value` overrides one config value for a single invocation (repeatable; the key must already exist). The overridden config is saved in the run log.

| # | Stage | What it does | Main outputs |
|---|---|---|---|
| 1 | `download` | Fetches the dataset's zip from the pinned CARE-GNN commit, verifies both checksums, unzips | `data/raw/care_gnn/*.mat` |
| 2 | `load` | Node table, one edge list per relation, count checks against the config | `nodes.parquet`, `graph/*.npz`, `graph/manifest.json` |
| 3 | `split` | Grouped, stratified 60/40 split and 5 grouped CV folds | `train_ids.csv`, `test_ids.csv`, `cv_folds.csv`, `split_summary.json` |
| 4 | `features` | The `own` feature block and column types from training nodes | `features/own.parquet`, `column_spec.json` |
| 5 | `audit` | Counts, feature statistics and shortcut flags, duplicates, groups, homophily | `outputs/<dataset>/reports/audit.md` + CSVs |
| 6 | `df_analyze_input` | Train/test tables for the active feature set | `data/processed/<dataset>/df_analyze/<feature_set>/` |
| 7 | `df_analyze` | Runs df-analyze; verifies its split and column types | `outputs/<dataset>/<feature_set>/df_analyze/<time>/`, `split_check.json`, `type_check.json` |
| 8 | `select_model` | Refits every tuned combination on the grouped CV folds; out-of-fold probabilities; tuning budget | `model_selection_cv.csv`, `oof_predictions.parquet`, `tuning_budget.csv` |
| 9 | `df_analyze_report` | Model A, thresholds, test metrics, bootstrap intervals, sanity band | `model_a_report.md`, `model_a_metrics.csv`, `model_a.json` |
| 10 | `web_report` | One self-contained HTML page | `outputs/<dataset>/<feature_set>/report/index.html` |

Paths in the table are relative to `data/processed/<dataset>/` or `outputs/<dataset>/<feature_set>/reports/` unless shown in full. Each stage can also be run on its own with `uv run scripts/NN_<stage>.py --dataset <name>`. Every invocation writes its config, seed, git commit and package versions to `outputs/<dataset>/runs/<timestamp>/`.

**The web report** needs no server: open the file in a browser (from WSL: `explorer.exe outputs/amazon/m1_own/report/index.html`). It shows Model A's configuration, tuned hyperparameters, test metrics with intervals, the confusion matrix at the chosen threshold, every tuned model, the tuning budget and the run configuration. It also has two scrollable cards of the most and least confident test nodes, each with its highest-ranked features (value and training percentile). It contains row-level test data: share it only within the group, never commit or post it.

### Stopping and resuming

A run can be stopped at any time (Ctrl+C, closing the terminal, shutting down). Continue it with `uv run run.py --dataset <name> --resume`.

- **Finished stages are kept.** The stages still to run are recorded in `outputs/<dataset>/pipeline_state.json`, so a resumed YelpChi run never touches Amazon. While an interrupted run is pending, a plain run refuses to start; use `--resume`, or `--force` to start over.
- **`select_model` saves every finished fit,** so a resume redoes only the interrupted one.
- **`df_analyze` restarts from its beginning:** it runs all classifiers in one process, because its feature selection is not seeded and separate runs could give different models different feature sets.
- **A resume refuses a changed config.**
- **Watchdog:** if one configuration in `select_model` takes longer than `select_model.config_timeout_s` (30 min), the stage dumps every thread's stack to `model_selection.log` and fails instead of hanging. On WSL this has come from a stuck GPU link: run `wsl --shutdown`, reopen WSL, then `--resume`.

### Runtime

df-analyze's runtime grows with the number of classifiers, the trials and the number of rows. Each model is tuned separately on each feature selection, and df-analyze stops each search at a time limit (15–60 min per model). In practice most searches end earlier: after 50 trials, a search stops if the last 15 found nothing better.
- **Earlier runs:** the 18k-row CFPB run took about 4 h for df-analyze on an RTX 4060 laptop.
- **YelpChi** (about 27.6k training rows) will take longer.
- **Amazon** (about 5.2k) far less.
- **Amazon pilot (10 trials per model, CPU only):** df-analyze 20 min (GANDALF alone 9 min), `select_model` 4.5 min. Details are in the decision log in [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).

Ask the group before any run expected to take more than about an hour.

### Sharing outputs with the group

df-analyze takes hours, so one person can run it and share the results outside git:

```bash
uv run pack_artifacts.py --dataset amazon        # fraud_artifacts_amazon_<time>.tgz
# group member, with the .tgz in the repo root:
uv run unpack_artifacts.py --dataset amazon      # verifies checksums; run.py then skips finished stages
```

A bundle holds `data/processed/<dataset>/` and `outputs/<dataset>/` (only the latest df-analyze run of each feature set), never the raw downloads or the per-invocation logs. Every file is checked against a SHA-256 manifest. If a file already exists with different contents, nothing is written unless you pass `--force`. Share bundles through OneDrive or Teams, never through the repository.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests use only small synthetic data, all defined in [`tests/synthetic.py`](tests/synthetic.py) (including tiny fake `.mat` files written with `scipy.io.savemat`). Tests that need files write them to pytest's temporary directory. No data files, real or synthetic, are committed.

## Repository layout

```
fraud-detect-gnn/
├── README.md
├── AGENTS.md            # rules for AI coding agents working in this repo
├── CLAUDE.md            # points Claude Code at AGENTS.md
├── docs/RESEARCH_PLAN.md
├── download_dataset.sh  # runs the download stage for both datasets
├── run.py               # runs the pipeline stages in order
├── pack_artifacts.py / unpack_artifacts.py
├── pyproject.toml / uv.lock / .python-version
├── configs/default.yaml # datasets, feature sets, split, df-analyze, report settings
├── src/fraud_detect/
│   ├── pipeline.py      # stage definitions (add new stages here)
│   ├── cli.py           # command line shared by run.py and scripts/
│   ├── config.py        # config loading and per-dataset paths
│   ├── columns.py       # node-table column names
│   ├── data/            # download, .mat loading, grouped split, audit
│   ├── features/        # own block, column types, feature blocks and sets
│   ├── models/          # df-analyze (Model A), model choice, evaluation, repeats
│   ├── report/          # static HTML report (web.py, report.css)
│   ├── external.py      # runs df-analyze scripts in their own environment
│   ├── artifacts.py     # packing and unpacking output bundles
│   ├── runstate.py      # stop/resume, atomic writes
│   └── runlog.py        # seeds and run metadata
├── scripts/             # NN_<stage>.py: run one stage; dfa/: scripts run in df-analyze's env
├── tests/
├── data/                # local only, git-ignored
└── outputs/             # local only, git-ignored
```

## Limitations

- **The features are precomputed and anonymous.** We cannot inspect or change how they were built. Some relations (same product and star rating, same month, text similarity) are themselves derived from review attributes, so part of the graph is constructed rather than observed.
- **Possible label shortcut on Amazon.** Labels come from helpfulness votes, and several features reach a univariate AUROC near 0.88. If a feature is derived from the same votes, Amazon results overstate what can be detected from behaviour. The audit flags these features; the decision on them is open.
- **df-analyze's internal CV is not grouped.** Its tuning and feature-selection folds are random, so a YelpChi user's reviews can sit on both sides of them. This makes its tuning scores optimistic, which is one more reason Model A is chosen by our grouped CV. The test set is grouped, so final scores are not affected; the tuned hyperparameters may still be slightly over-fitted to user-level leakage.
- **df-analyze scales continuous features over train and test together,** and infers types on both. No labels are used, but test rows influence the clip points. Our own type decisions use training nodes only.
- **Single split by default.** Bootstrap intervals cover test-set sampling only. Amazon is small, and its scores vary across splits; `split.n_repeats` measures this at the cost of one df-analyze run per repeat.
- **Labels are proxies:** Yelp's filter is not ground truth for spam, and helpfulness votes are not ground truth for fraud.

## References

- Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters* (CARE-GNN). CIKM. https://arxiv.org/abs/2008.08692. Data: https://github.com/YingtongDou/CARE-GNN
- Rayana, S., & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* KDD. https://doi.org/10.1145/2783258.2783370
- Zhang, S., Yin, H., Chen, T., Hung, Q. V. N., Huang, Z., & Cui, L. (2020). *GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection.* SIGIR. https://arxiv.org/abs/2005.10150 (the Amazon features)
- Tang, J., Hua, F., Gao, Z., Zhao, P., & Li, J. (2023). *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection.* NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2306.12251
- Hamilton, W., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS.
- df-analyze: https://github.com/stfxecutables/df-analyze
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric

## License

GPL-3.0. See [LICENSE](LICENSE).

The datasets are not redistributed in this repository; the download stage fetches them from the CARE-GNN repository (Apache-2.0).
