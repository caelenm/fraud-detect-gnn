# fraud-detect-gnn

**Does a graph help? Tabular AutoML vs. graph neural networks for fraud detection on two graph fraud benchmarks (YelpChi and Amazon)**

Course project for CS555: Data Mining and Machine Learning, St. Francis Xavier University (StFX).

> **Status:** Model 1 (df-analyze on each node's own features) is implemented end to end for both datasets, and **phase 1 (Amazon) is done** ([Results](#results-phase-1-amazon)). **Phase 2** ([`docs/PHASE2_GNN_PLAN.md`](docs/PHASE2_GNN_PLAN.md) is the plan; branch `feat/gnn-embeddings`) adds two graph rungs as feature blocks for df-analyze: **Model 2** (`nbr`, label-free neighbour means) and **Model 5** (`gnnssl`, self-supervised GNN embeddings), each also without the four features of issue [#36](https://github.com/caelenm/fraud-detect-gnn/issues/36), and a [comparison page](#comparing-two-feature-sets) that shows two rungs' Model A test node by test node. Amazon pilot results are in [Results (phase 2 pilot)](#results-phase-2-pilot-amazon). GraphSAGE (Models 3–4) and Model 6 are not built. **Phase 3** ([`docs/PHASE3_FEEDBACK_PLAN.md`](docs/PHASE3_FEEDBACK_PLAN.md)) answers the professor's feedback of 2026-10-21: the size of the GNN embeddings and how to reduce them, whether training the GNN helps, the four important features alone, and more datasets. Every run is recorded in the [results database](#results-database). The working plan is [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md). An earlier version of this project used CFPB complaint narratives; that state is preserved at commit `3336d87` on `main`.

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
| 2 | df-analyze on own features + label-free neighbour aggregates, per relation (`m2_own_nbr`) | hand-made | **implemented**; Amazon pilot |
| 3 | GraphSAGE with all edges removed (an MLP) | none | planned |
| 4 | Heterogeneous GraphSAGE over all relations | learned | planned |
| 5 | df-analyze on own features + self-supervised GNN embeddings, per relation, PCA-compressed (`m5_own_gnnssl`) | learned, no labels | **implemented**; Amazon pilot |
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

## Results (phase 1, Amazon)

One full Model 1 run (2026-10-13; seed 555, 100 tuning trials per model, one grouped split; details and timings in [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) §11).

| | Test PR-AUC [95% CI] | Test AUROC [95% CI] |
|---|---|---|
| **Model A: LightGBM, all 25 features** | **0.919** [0.890, 0.942] | **0.977** [0.967, 0.986] |
| Model A without the four #36 features (21 features; LightGBM, embedded selection; run 2026-10-19 with the same configuration and split) | 0.518 [0.465, 0.576] | 0.889 [0.868, 0.907] |
| No-skill (test fraud rate) | 0.095 | 0.500 |
| df-analyze's dummy baseline | 0.096 | 0.504 |

- **Model A was chosen by shared, grouped 5-fold CV on the training set** (PR-AUC 0.925 ± 0.029), never by test results. CatBoost with embedded feature selection came second (0.921 ± 0.033), within one standard deviation, so the choice between the two is not decisive.
- At the decision threshold chosen on out-of-fold training predictions (0.430), Model A flags 299 of 3,456 test users: precision 0.913, recall 0.832, F1 0.871.
- The intervals resample whole test groups and cover test-set sampling only, not other splits or seeds. Amazon is small (3,456 test users, 328 fraudulent), and this is one split.
- **Read these numbers with [#36](https://github.com/caelenm/fraud-detect-gnn/issues/36) in mind:** four features alone reach a univariate AUROC of 0.86–0.88 and may encode the helpfulness votes the labels come from.
  - The same full Model A without them scores 0.518, a paired difference of **−0.401** [−0.450, −0.347] on the same test users.
  - On training users only, `own__f19` alone separates the classes with 0.89 balanced accuracy (PR-AUC 0.80).
  - Results are reported both ways. A literature check (2026-10-19; [Limitations](#limitations)) found no public column list. The best-supported reading: `f19` is a helpfulness-vote count, which would leak part of the label, while `f13`, `f15`, `f18` are star-rating statistics. So 0.919 is likely inflated by leakage, and 0.518 is a conservative interim figure.
- The result is inside the sanity band (≈ 0.91 for untuned gradient boosting), and the 10-trial CPU pilot gave almost the same test PR-AUC (0.920).

## Results (phase 2 pilot, Amazon)

**Pilot, not final:** 10 tuning trials, 3 classifiers (LightGBM, logistic regression, CatBoost) plus the dummy, seed 555, all on the same RTX 4060 laptop. The tables below are split 0 (2026-10-19); the 3-split validation follows them. Each Model A was chosen by grouped CV on the training set only. Test PR-AUC with 95% group-bootstrap intervals (1,000 resamples, from each run's report; the comparison pages use 2,000, so their per-side intervals differ in the third decimal); no-skill PR-AUC 0.095.

| Feature set | Features | Model A | Test PR-AUC [95% CI] | Test AUROC |
|---|---|---|---|---|
| m1: own | 25 | LightGBM, embedded selection | 0.917 [0.888, 0.941] | 0.979 |
| m2: own + nbr | 103 | CatBoost, embedded selection (24 kept: 11 own, 13 nbr) | 0.926 [0.901, 0.946] | 0.984 |
| m5: own + gnnssl | 49 | LightGBM, embedded selection (all 49 kept) | 0.929 [0.905, 0.949] | 0.981 |
| m1 without #36 | 21 | LightGBM, association filter | 0.524 [0.470, 0.583] | 0.893 |
| m2 without #36 | 87 | CatBoost, association filter (all kept) | 0.710 [0.663, 0.754] | 0.935 |
| m5 without #36 | 45 | LightGBM, embedded selection (all kept) | 0.677 [0.626, 0.726] | 0.927 |

Paired differences on the same test nodes (B − A, 2,000 resamples of whole test groups; "B ahead" is the share of resamples, not a p-value):

| Contrast | ΔPR-AUC [95% CI] | B ahead | ΔAUROC [95% CI] |
|---|---|---|---|
| m2 − m1 | +0.009 [−0.001, +0.020] | 96% | +0.005 [−0.000, +0.011] |
| m5 − m1 | +0.012 [+0.000, +0.025] | 98% | +0.002 [−0.003, +0.006] |
| m5 − m2 | +0.003 [−0.006, +0.012] | 71% | −0.003 [−0.009, +0.001] |
| m2 − m1, without #36 | **+0.186** [+0.142, +0.225] | 100% | +0.042 [+0.031, +0.055] |
| m5 − m1, without #36 | **+0.153** [+0.104, +0.196] | 100% | +0.035 [+0.023, +0.047] |
| m5 − m2, without #36 | −0.033 [−0.062, −0.004] | 1% | −0.008 [−0.015, −0.000] |

**Reading:**
- **With all 25 features, the graph adds about one point** (m2 +0.009, m5 +0.012). That is in the 0–1.5 point range GADBench suggested. m2's interval includes 0; m5's lower bound is +0.0001, too close to 0 to call it a gain on one split at a pilot budget. Learned embeddings and hand-made neighbour means tie.
- **Without the four #36 features, the graph helps a lot:** own features alone drop to 0.524. Neighbour means add 0.19 and the GNN embeddings 0.15, with intervals well above 0. Here the untuned DGI embeddings are *worse* than simple neighbour means (−0.033, interval below 0).
- **The four #36 features carry much of the score:** 0.917 with them, 0.524 without. With them the task is close to saturated, which leaves the graph little room; this is the caveat of [#36](https://github.com/caelenm/fraud-detect-gnn/issues/36).
- **At a fixed threshold the picture is mixed:** with all features, m5 rescues 6 frauds that m1 misses but misses 12 that m1 catches, and removes 12 of m1's false alarms (each side at its own out-of-fold threshold). The gain is in the ranking, not in more frauds caught at the threshold.
- **Random-graph control** (training CV only, untuned gradient boosting; `scripts/progress_checks.py`): with every relation rewired at random but degrees kept, the all-features gain barely changes (+ nbr 0.935 real vs 0.932 random). So the small gain there is not evidence of real graph structure. Without #36, about half the gain needs the real edges (+ nbr 0.704 real vs 0.613 random; own alone 0.500).
- The `_no36` runs print the sanity-band warning (0.91 is the band for all features); that is expected, not a bug.
- **What these numbers do not show:** variation across splits and seeds, or the effect of tuning the embeddings or the full df-analyze budget. That is phase 3.

**Validation over 3 splits (2026-10-20).** All six pilot rungs were rerun on two more independently seeded grouped splits, under a decision rule committed before the runs:
1. the gain is positive on all 3 splits;
2. its mean exceeds its SD across splits;
3. a Nadeau–Bengio corrected paired t-test over the 15 grouped CV folds (training data only) gives p < 0.05.

All 18 runs passed the split and type checks, and no search hit a time limit.

| Variant | Contrast | Δ test PR-AUC, splits 0 / 1 / 2 | Mean ± SD | CV p | Verdict |
|---|---|---|---|---|---|
| All features | m2 − m1 | +0.009 / +0.019 / +0.003 | +0.010 ± 0.008 | 0.013 | supported |
| All features | m5 − m1 | +0.012 / +0.012 / +0.009 | +0.011 ± 0.002 | 0.090 | not distinguishable from zero |
| All features | m5 − m2 | +0.003 / −0.007 / +0.005 | +0.000 ± 0.007 | 0.55 | not distinguishable from zero |
| Without #36 | m2 − m1 | +0.186 / +0.166 / +0.134 | +0.162 ± 0.026 | 6e-7 | supported |
| Without #36 | m5 − m1 | +0.153 / +0.143 / +0.090 | +0.129 ± 0.034 | 6e-6 | supported |
| Without #36 | m5 − m2 | −0.033 / −0.024 / −0.044 | −0.033 ± 0.010 | 0.002 | worse |

- **Model 1 across splits:** test PR-AUC 0.917 / 0.919 / 0.919 with all features, and 0.524 / 0.545 / 0.574 without #36.
- **With all features,** neighbour means give a small (about +0.01) but consistent gain that passes the rule. The random-graph control (below) shows that most of that gain survives random rewiring with the same degrees, so it supports "neighbour-aggregate columns help", not "real graph structure helps". The GNN's +0.011 is just as consistent but not significant on the CV folds.
- **Without the #36 features,** both graph rungs clearly help: +0.16 for neighbour means and +0.13 for GNN embeddings. Neighbour means beat the untuned GNN embeddings on every split.

Each pair has a comparison page per split (`outputs/amazon*/compare/`), and the verdict table is `outputs/amazon/compare/repeat_summary.md`. Run times: 10.6–16.0 min per feature set, 85 min for all six; the CPU embeddings took 34 s (all features) and 22 s (without #36).

## Graph feature blocks (Models 2 and 5)

The graph reaches df-analyze as extra columns per node, joined next to the own features; df-analyze then treats each node as a table row exactly as in Model 1. Every rung uses the same split, folds, tuning and test scoring; only the input columns differ.

- **Node table** `data/processed/<dataset>/nodes.parquet`: `node_id`, `label` (null when unlabelled), `is_labelled`, `group_id`, `split` (train/test/unlabelled), `cv_fold`.
- **Relations** `data/processed/<dataset>/graph/<relation>.npz` (plus `homo`): undirected int32 edge lists with `src < dst`, no self-loops, deduplicated; `graph/manifest.json` records counts and the source checksum.
- **Feature blocks** `data/processed/<dataset>/features/<block>.parquet` keyed by `node_id`, with columns prefixed `<block>__`. One block per kind covers every relation, so the names are the same on both datasets:
  - **`nbr`** (stage `neighbour_agg`, Model 2): for each relation (not `homo`), the 1-hop mean of every own feature over a node's neighbours (0 when isolated) and log1p(degree). Amazon: 3 × 26 = 78 columns, e.g. `nbr__upu_mean_f13`. A sparse, row-normalised adjacency times the feature matrix; no learning.
  - **`gnnssl`** (stage `gnn_embed`, Model 5): for each relation, [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (PyG `DeepGraphInfomax`) trains a 2-layer GCN encoder (hidden 64, output 64, PReLU) over the sparse, GCN-normalised adjacency **without labels**: the encoder must tell the real graph from one with shuffled feature rows. Adam (lr 1e-3), at most 300 epochs, early stopping on the loss (patience 20), seeded, on the CPU. Each 64-d embedding is compressed by a PCA to 8 components: 3 × 8 = 24 columns, e.g. `gnnssl__upu_pc01`. The raw embeddings and the training log (`meta.json`: loss curves, best epoch, PCA variance, timing, versions) stay in `outputs/<dataset>/gnn_embed/`.
  - Each block's column types are decided on training nodes (`<block>.spec.json`), like `column_spec.json` for `own`.
- **Variants (issue #36):** a block named `<kind>_<variant>` is built from the own features without the columns `feature_variants.<variant>` lists. `no36` leaves out `own__f13`, `own__f15`, `own__f18`, `own__f19`: `own_no36` is the own block without them, and `nbr_no36` / `gnnssl_no36` are built from that reduced input (`outputs/<dataset>/gnn_embed_no36/`), so the GNN never sees them either.
- **Feature sets** in the config are the ladder's rungs: `m1_own: [own]`, `m2_own_nbr: [own, nbr]`, `m5_own_gnnssl: [own, gnnssl]`, their `_no36` versions (`[own_no36, gnnssl_no36]`, …), and `pilot_` copies of all six for reduced-budget runs. `--set feature_set=<name>` runs stages 8–13 for one set, writing to `outputs/<dataset>/<feature_set>/`; the block stages build only the blocks the set uses.
- **Leakage rules:**
  - Graph feature builders take features and edges and never labels; tests permute every label and expect identical output ([`tests/test_label_blindness.py`](tests/test_label_blindness.py), `test_graph_blocks.py`, `test_gnn_ssl.py`).
  - Graph features use the features (never labels) of all nodes, test and unlabelled ones included.
  - The scaler that feeds the GNN and each PCA are fit on training nodes only.
  - The DGI loss covers non-test nodes (train and unlabelled); test nodes only pass messages.
- **Probe:** `uv run scripts/probe_blocks.py --dataset amazon` fits one fixed, untuned gradient-boosting model per saved grouped CV fold (training nodes only) for `own`, `own+nbr`, `own+gnnssl` and their `_no36` versions, and writes `outputs/<dataset>/probe/probe_cv.csv`. It is a quick signal before spending df-analyze runs, not a result; nothing is chosen from it.

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
# 1. This project (with the optional `gnn` extra: torch and PyTorch Geometric,
#    needed only by the gnn_embed stage)
git clone https://github.com/caelenm/fraud-detect-gnn.git
cd fraud-detect-gnn
uv sync --extra gnn            # or plain `uv sync` without the GNN stage

# 2. df-analyze, next to this repository, at the tested commit, on Python 3.13
uv python install '>=3.13.11,<3.14'
git clone https://github.com/stfxecutables/df-analyze.git ../df-analyze
git -C ../df-analyze checkout 199e5638620693c267dac715784f1fd0e33fa796
uv sync --locked --python '>=3.13.11,<3.14' --directory ../df-analyze

# 3. The data (checksum-verified; safe to rerun)
./download_dataset.sh          # both datasets into data/raw/care_gnn/
```

About the `gnn` extra:
- It installs torch 2.9 and PyTorch Geometric into this project's environment only, never df-analyze's. Plain PyG; no compiled `torch_scatter`/`torch_sparse`.
- torch stays at 2.9, whose Linux wheels on PyPI are built for CUDA 12.8 (like df-analyze's torch). The newest ones are built for CUDA 13, which needs a newer NVIDIA driver than the group laptop has.
- The GNN embeddings run on the CPU by default (`gnn_embed.device: cpu`, about 35 s on Amazon), so no GPU is needed for them.
- Plain `uv sync` removes the extra again; use `uv sync --extra gnn`. Without it, the GNN tests are skipped and the `gnn_embed` stage stops with an install hint.

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
| 6 | `neighbour_agg` | The `nbr` block(s) the feature set uses (Model 2) | `features/nbr.parquet`, `nbr.spec.json` (and `nbr_no36.*`) |
| 7 | `gnn_embed` | The `gnnssl` block(s) the feature set uses (Model 5); one resumable unit per relation | `features/gnnssl.parquet`, `gnnssl.spec.json`, `outputs/<dataset>/gnn_embed/` (and `_no36`) |
| 8 | `df_analyze_input` | Train/test tables for the active feature set | `data/processed/<dataset>/df_analyze/<feature_set>/` |
| 9 | `df_analyze` | Runs df-analyze; verifies its split and column types | `outputs/<dataset>/<feature_set>/df_analyze/<time>/`, `split_check.json`, `type_check.json` |
| 10 | `select_model` | Refits every tuned combination on the grouped CV folds; out-of-fold probabilities; tuning budget | `model_selection_cv.csv`, `oof_predictions.parquet`, `tuning_budget.csv` |
| 11 | `df_analyze_report` | Model A, thresholds, test metrics, bootstrap intervals, sanity band | `model_a_report.md`, `model_a_metrics.csv`, `model_a.json` |
| 12 | `web_report` | One self-contained HTML page | `outputs/<dataset>/<feature_set>/report/index.html` |
| 13 | `export_model` | Checks that every tuned model df-analyze saved loads and reproduces its test probabilities; saves Model A in its native format | `outputs/<dataset>/<feature_set>/model/`: `model_a.joblib` (or `.cbm`), `model_a.json`, `models_check.csv` |

Paths in the table are relative to `data/processed/<dataset>/` or `outputs/<dataset>/<feature_set>/reports/` unless shown in full. `export_model` is last on purpose: `run.py` reruns every stage after the first one with missing outputs, so adding it earlier would rerun df-analyze. For the same reason the block stages count as done when the active feature set does not use their block, so an `m1_own` run or bundle never reruns anything. Each stage can also be run on its own with `uv run scripts/NN_<stage>.py --dataset <name>`. Every invocation writes its config, seed, git commit and package versions to `outputs/<dataset>/runs/<timestamp>/`.

**The web report** needs no server: open the file in a browser (from WSL: `explorer.exe outputs/amazon/m1_own/report/index.html`). It shows Model A's configuration, tuned hyperparameters, test metrics with intervals, the confusion matrix at the chosen threshold, every tuned model, the tuning budget and the run configuration. It also has two scrollable cards of the most and least confident test nodes, each with its highest-ranked features (value and training percentile). It contains row-level test data: share it only within the group, never commit or post it.

### Comparing two feature sets

Once two feature sets have finished the pipeline (through `df_analyze_report`):

```bash
uv run scripts/compare_report.py --dataset amazon --a pilot_m1_own --b pilot_m5_own_gnnssl
```

This writes `outputs/<dataset>/compare/<a>__vs__<b>/compare.html` and `compare_metrics.json`. It is not a stage, because it spans two feature sets, and it stops if they were not scored on the same test nodes. It reads only what the pipeline saved: each side's Model A, its threshold (chosen on out-of-fold training predictions) and its test probabilities.

- **Metrics:** each side's test PR-AUC and AUROC with group-bootstrap 95% intervals, Recall@K (K = the number of test frauds) and F1/precision/recall at its own threshold. The headline is the **paired ΔPR-AUC (B − A)**: both models are scored on the same 2,000 resamples of whole test groups, giving its 95% interval and the share of resamples with B ahead.
- **Scatter:** P(fraud | A) against P(fraud | B) for every test node, with both thresholds and the four outcomes (both catch, B rescues, B breaks, both miss).
- **Galleries:** the 20 fraud nodes B rescues and the 20 it breaks, by how far apart the two scores are.
- **Node cards** (click a point): features ranked by training percentile, and one neighbourhood per relation (at most 12 neighbours, seeded; only training neighbours coloured by label).
- **Embedding view:** a 2-D PCA (fit on training nodes) of a DGI embedding next to the own features.

The page is a single offline file: inline CSS, inline SVG and a little inline JavaScript, with no external requests. It contains row-level test data, so share it within the group only.

### What to expect (literature)

- **GADBench** ([Tang et al., NeurIPS 2023](https://arxiv.org/abs/2306.12251), tuned models, random splits), Amazon AUPRC:
  - plain XGBoost 91.88;
  - XGBoost with simple neighbour aggregation 93.33;
  - the best specialised GNN, BWGNN, 91.48.

  Model A (0.919) is already at the "trees without a graph" level, so the realistic gain on Amazon is 0 to about 1.5 points, which is inside its test interval (about ±2.6 points).
- **Frozen self-supervised embeddings** ([Cheng et al., arXiv 2410.18487](https://arxiv.org/abs/2410.18487)): DGI with an MLP reaches 88.92 AUPRC on Amazon. Such embeddings help most on sparse graphs with few labels; Amazon is dense and has over 5,000 training labels, so Model 5 may tie Model 1 or lose to Model 2. That is a valid finding: graph context may add little beyond strong tabular features.
- **YelpChi** is where graph information clearly helped trees in GADBench (84.0 → 91.1 AUPRC).
- **Issue #36:** if the four flagged features encode the label, they may saturate the task and hide any graph benefit. That is why every rung is also run without them.

These benchmarks use random splits and ours are grouped, so their numbers set expectations; they are not targets.

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
- **Amazon full run (100 trials, RTX 4060 laptop GPU, WSL2):** 2 h 05 min in total: df-analyze 1 h 57 min (CatBoost 47 min and GANDALF 46 min of tuning; every other model under 8 min), `select_model` 8 min, `export_model` 15 s.
- **Amazon pilot (10 trials per model, CPU only):** df-analyze 20 min (GANDALF alone 9 min), `select_model` 4.5 min. Details are in the decision log in [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).

Ask the group before any run expected to take more than about an hour.

### Results database

Everyone's runs are recorded in [`results/runs/`](results/README.md), one committed folder per run: the device, start and finish time (UTC), a one-line description, the settings and git commit, the headline numbers, and the run's results table (aggregate metrics only, never row-level data). One folder per run means two people recording at once never conflict.

```bash
uv run scripts/results_db.py add --from-run outputs/amazon/pilot_m1_own \
    --run-log outputs/amazon/runs/<stamp> --device rtx4060-laptop --description "..."
uv run scripts/results_db.py list            # or: show <run_id>, export, check
git add results/runs/<run_id> && git commit -m "Results: ..."
```

See [`results/README.md`](results/README.md) for every field and for recording probes, comparisons and checks.

### Sharing outputs with the group

df-analyze takes hours, so one person can run it and share the results outside git:

```bash
uv run pack_artifacts.py --dataset amazon        # fraud_artifacts_amazon_<time>.tgz
# group member, with the .tgz in the repo root:
uv run unpack_artifacts.py --dataset amazon      # verifies checksums; run.py then skips finished stages
```

A bundle holds `data/processed/<dataset>/` and `outputs/<dataset>/` (only the latest df-analyze run of each feature set), never the raw downloads, the per-invocation logs, caches (`__JOBLIB_CACHE__`, `lightning_logs`) or temporary files. Every file is checked against a SHA-256 manifest. If a file already exists with different contents, nothing is written unless you pass `--force`. Share bundles through OneDrive or Teams, never through the repository: they contain row-level data.

From phase 2 on, a bundle also holds every feature set run so far (including the `pilot_` ones), the graph blocks (`features/nbr*.parquet`, `gnnssl*.parquet`), the raw embeddings and training logs (`gnn_embed*/`), the probe table and the comparison pages (`compare/`).

**Nothing has to be retrained.** After unpacking, `uv run run.py --dataset amazon` skips every stage, `download` included (the bundle has no raw data, and none is needed). The bundle contains:
- **every tuned model** df-analyze saved, each refit on the full training set: `eval_htune_results_jsonpickle_00.json` in the df-analyze run's `results/test00/` folder, with df-analyze's prepared feature tables (`X_train_00.csv`, `X_test_00.csv`) next to it;
- **Model A, exported and verified** by the `export_model` stage: `outputs/<dataset>/m1_own/model/` holds the model in its library's own format (`model_a.joblib`, or `model_a.cbm` for CatBoost), `model_a.json` (input columns in order, tuned parameters, decision threshold, seed, df-analyze commit, package versions), and `models_check.csv`, which says for every tuned combination whether it loads and reproduces the test probabilities df-analyze saved.

**Which saved models load.** df-analyze saves its models with jsonpickle, which cannot rebuild GANDALF's training state, and as written the whole file fails to load. `scripts/dfa/model_io.py` reads it anyway: anything it cannot rebuild, or cannot trust after the first failure, becomes an explicit placeholder, never a silently wrong object. `models_check.csv` then lists, for every tuned combination, whether it loaded intact and reproduces the test probabilities and labels df-analyze saved. On the Amazon run, 17 of 28 do (all CatBoost, kNN, LightGBM and LightGBM random forest combinations, and logistic regression without feature selection). GANDALF (cannot be rebuilt), the dummy (random strategy) and three logistic regressions (one attribute cannot be rebuilt) do not, and were not refit. Model A loads intact.

Loading the models needs **df-analyze's environment** (Setup, step 2), not this project's: they were saved with its package versions. No GPU is needed: weights saved on a GPU are mapped to the CPU when there is none. `scripts/dfa/predict.py` predicts without retraining. Run it from this repository's root:

```bash
uv run --project ../df-analyze --python '>=3.13.11,<3.14' python scripts/dfa/predict.py \
    --dataset amazon --input <run>/train/<hash>/results/test00/X_test_00.csv --out preds.csv
# any other tuned combination that models_check.csv marks ok:
uv run --project ../df-analyze --python '>=3.13.11,<3.14' python scripts/dfa/predict.py \
    --dataset amazon --input <same X_test_00.csv> --out preds_rf.csv --model rf --selection pred
```

Here `<run>` is `outputs/amazon/m1_own/df_analyze/<time>`, the run named in `df_analyze/latest.txt`. `--project` uses df-analyze's environment without changing directory, so relative paths work. The input must be df-analyze's **prepared** features (its `X_train_00.csv`/`X_test_00.csv` exports, or a table with exactly Model A's input columns in order): df-analyze rescales continuous columns and encodes binary ones before modelling, so our raw `train.parquet`/`test.parquet` are not valid input. Column names and order are checked, and a mismatch stops the script. The output has `prob_fraud` and `label` (at the saved threshold, chosen on out-of-fold training predictions). The script warns if your package versions differ from those the model was saved with.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Commits go straight to `main` (or to a branch); no pull request or review is needed, and running the checks first is recommended rather than required. See `AGENTS.md`, "Git workflow".

Tests use only small synthetic data, all defined in [`tests/synthetic.py`](tests/synthetic.py) (including tiny fake `.mat` files written with `scipy.io.savemat`). Tests that need files write them to pytest's temporary directory. No data files, real or synthetic, are committed.

## Repository layout

```
fraud-detect-gnn/
├── README.md
├── AGENTS.md            # rules for AI coding agents working in this repo
├── CLAUDE.md            # points Claude Code at AGENTS.md
├── docs/RESEARCH_PLAN.md
├── docs/PHASE1_AMAZON_PLAN.md  # phase 1: Amazon Model A end to end, shareable bundle
├── docs/PHASE2_GNN_PLAN.md     # phase 2: Models 2 and 5 as df-analyze blocks, comparison page
├── docs/PHASE3_FEEDBACK_PLAN.md # phase 3: the professor's feedback (2026-10-21)
├── results/             # results database: one committed folder per run (results/README.md)
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
│   ├── features/        # own block, column types, feature blocks and sets, nbr block
│   ├── graph/           # self-supervised GNN embeddings (ssl.py; needs the gnn extra)
│   ├── models/          # df-analyze (Model A), model choice, evaluation, repeats, export, probe
│   ├── report/          # HTML report (web.py, report.css), comparison page (compare.py)
│   ├── external.py      # runs df-analyze scripts in their own environment
│   ├── artifacts.py     # packing and unpacking output bundles
│   ├── runstate.py      # stop/resume, atomic writes
│   ├── results_db.py    # the results database (scripts/results_db.py is its command line)
│   └── runlog.py        # seeds and run metadata
├── scripts/             # NN_<stage>.py: run one stage; compare_report.py, probe_blocks.py, results_db.py
│   └── dfa/             # run in df-analyze's env: cv_select, export_model, predict
├── tests/
├── data/                # local only, git-ignored
└── outputs/             # local only, git-ignored
```

## Limitations

- **The features are precomputed and anonymous.** We cannot inspect or change how they were built. Some relations (same product and star rating, same month, text similarity) are themselves derived from review attributes, so part of the graph is constructed rather than observed.
- **Possible label shortcut on Amazon** ([#36](https://github.com/caelenm/fraud-detect-gnn/issues/36)). Labels come from helpfulness votes, and four features (`own__f13`, `own__f15`, `own__f18`, `own__f19`) reach a univariate AUROC of 0.86–0.88 on their own. If they are derived from the same votes, Amazon results overstate what can be detected from behaviour. Phase 1 used all 25. From phase 2 on, every rung is reported both ways (decided 2026-10-19): with all 25 features, and without the four (`_no36`, also left out of the GNN input).
  - **What the four probably are** (literature check, 2026-10-19). Neither CARE-GNN (Dou et al. 2020), its repository and issues, DGL nor GADBench documents the column order of `Amazon.mat`. The features come from Zhang et al. (SIGIR 2020), Table 1, which lists feature *groups*, including 12 helpful/unhelpful-vote statistics, and expands to more than 25 scalars. CARE-GNN recomputed them on Musical Instruments themselves (Zhang et al. used Movies & TV). Inferred from the distributions, **not documented**:
    - `f19`: a vote count, most likely total unhelpful votes. It is part of the ratio that defines the label, so it would be leakage by construction.
    - `f15`: median star rating (9 values = 1, 1.5, …, 5).
    - `f18`: mean star rating.
    - `f13`: share of 4–5 star ratings.

    The three rating features are not computed from votes. Rating extremity predicts downvotes, which is what "fraud" means here.
  - **Consequences:**
    - 0.919 is likely inflated by leakage; so are published Amazon results that use all 25 features.
    - 0.518 drops three probably legitimate features, and may keep other vote-derived columns with weaker individual signal.
    - The defensible headline is a run without every column shown to come from votes. Local checks that settle this, starting with "is some column c such that c / (c + f19) reproduces the label rule?", are listed in `docs/RESEARCH_PLAN.md` §12.
- **GNN embedding settings are untuned** (64-d, up to 300 epochs, 8 PCA components per relation, the plan's defaults). Some relations were still improving at 300 epochs. Choosing them, on training CV only, is phase 3.
- **The graph comparison uses a reduced budget** (10 trials, 3 classifiers) on 3 seeded splits with one tuning seed; see the pilot section. The full-budget runs (6 classifiers, 100 trials) exist only for Model 1, on split 0.
- **df-analyze's internal CV is not grouped.** Its tuning and feature-selection folds are random, so a YelpChi user's reviews can sit on both sides of them. This makes its tuning scores optimistic, which is one more reason Model A is chosen by our grouped CV. The test set is grouped, so final scores are not affected; the tuned hyperparameters may still be slightly over-fitted to user-level leakage.
- **df-analyze scales continuous features over train and test together,** and infers types on both. No labels are used, but test rows influence the clip points. Our own type decisions use training nodes only.
- **Single split by default.** Bootstrap intervals cover test-set sampling only. Amazon is small, and its scores vary across splits; `split.n_repeats` measures this at the cost of one df-analyze run per repeat.
- **Labels are proxies:** Yelp's filter is not ground truth for spam, and helpfulness votes are not ground truth for fraud.

## References

- Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters* (CARE-GNN). CIKM. https://arxiv.org/abs/2008.08692. Data: https://github.com/YingtongDou/CARE-GNN
- Rayana, S., & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* KDD. https://doi.org/10.1145/2783258.2783370
- Zhang, S., Yin, H., Chen, T., Hung, Q. V. N., Huang, Z., & Cui, L. (2020). *GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection.* SIGIR. https://arxiv.org/abs/2005.10150 (the Amazon features)
- Tang, J., Hua, F., Gao, Z., Zhao, P., & Li, J. (2023). *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection.* NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2306.12251
- Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). *Deep Graph Infomax.* ICLR. https://arxiv.org/abs/1809.10341
- Cheng et al. (2024). arXiv:2410.18487 (frozen self-supervised GNN embeddings for graph anomaly detection; cited from the phase 2 plan, check the details before quoting)
- Hamilton, W., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS.
- df-analyze: https://github.com/stfxecutables/df-analyze
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric

## License

GPL-3.0. See [LICENSE](LICENSE).

The datasets are not redistributed in this repository; the download stage fetches them from the CARE-GNN repository (Apache-2.0).
