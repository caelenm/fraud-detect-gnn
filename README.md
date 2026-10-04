# fraud-detect-gnn

**Does a graph help? Tabular models vs. graph neural networks for fraud classification, on a constructed graph (CFPB complaints) and a real one (Elliptic Bitcoin transactions)**

Course project for CS555: Data Mining and Machine Learning, St. Francis Xavier University (StFX).

> **Status:** Stages 1–12 (data loading through the df-analyze baseline, fair model selection, and the web report) are implemented and have been run on the real archive. Results under label rule version 1 are superseded by version 2 (see [`docs/LABEL_RULE.md`](docs/LABEL_RULE.md)). Stage 13 (a label shortcut audit) is implemented. The plan now compares four models on two datasets (see [Comparison design](#comparison-design)); the neighbour-feature model, the graphs, the GNNs, Elliptic, evaluation and explanation are still planned.

---

## Overview

This project asks one question: **does using a graph improve fraud classification over the same kind of model without one?** We answer it with the same four models (see [Comparison design](#comparison-design)) on two datasets:

- **CFPB consumer complaints**, where there is no observed network (consumers are anonymous), so the graph has to be *constructed* from each complaint's own attributes: its company, product, state, and optionally text similarity.
- **The Elliptic Bitcoin transaction graph**, where the edges are *real* payment flows between transactions.

The interesting result is the contrast: whether the graph helps where the relationships are real but not where they are built from features the tabular model already sees. Either outcome is reported as it is.

On CFPB the task is **classifying fraud-related complaints**, not detecting fraud. The label is the category the consumer chose when filing, and the strongest features are narrative embeddings, so a model mostly learns whether a complaint is *about* fraud (for example, "I am a victim of identity theft…"). That is a legitimate text-classification task, and results are described that way.

- **Tabular models.** [df-analyze](https://github.com/stfxecutables/df-analyze) treats each example as an independent row and searches over models, feature selection methods, and hyperparameters. Its best model is Model A.
- **Graph neural networks.** A GraphSAGE network in [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) lets information flow along the graph's edges before classifying. The same network with its edges removed is the graph-free control.

The models are explained with feature-importance methods so we can compare *what* each one relies on, not just how well it scores.

## Research questions

1. Does a graph improve on the same model without one? If it does, is message passing needed, or do neighbour-aggregated features in a tree model capture the same information?
2. Does the answer differ between a constructed graph (CFPB) and a real one (Elliptic)?
3. Which feature groups (on CFPB: complaint text, company, product, region) drive each model's predictions, and do the models agree?

## Comparison design

Each dataset gets the same four models, trained on the same split and scored on the same frozen test set with the same metrics:

| # | Model | Uses the graph's information? | Message passing? |
|---|---|---|---|
| 1 | **Tabular:** df-analyze's best model on each node's own features (Model A) | no | no |
| 2 | **Tabular + neighbour features:** the same pipeline, with each node's own features plus aggregates of its neighbours' features (never their labels), following GADBench | yes | no |
| 3 | **Graph-free control:** the GNN below with all edges removed (an MLP on each node's own features) | no | no |
| 4 | **GNN:** GraphSAGE over the graph (heterogeneous on CFPB, homogeneous on Elliptic) | yes | yes |

- **4 vs. 3:** does message passing over the graph help, with the architecture held fixed?
- **2 vs. 1:** does the graph's information help a tree model?
- **4 vs. 2:** given the same graph information, does message passing beat simple aggregation? GADBench (NeurIPS 2023) found that tree ensembles with neighbourhood aggregation often match or beat GNNs built for fraud and anomaly detection.

On CFPB the edges are built from features Models 1 and 3 already see, so the expected result is GNN ≈ graph-free control. That is a finding, not a failure, and the Elliptic contrast is what makes it informative.

## Data

**Source:** [CFPB Consumer Complaint Database, Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/).

On August 14, 2026 the CFPB stopped publishing complaint narratives in the live database and moved previously published narratives to its FOIA Reading Room as public-domain bulk files. The archive is therefore a fixed snapshot, which makes this project reproducible. Identifying information about consumers was scrubbed by the CFPB before publication.

**Time range used: complaints received May 2018 through August 2023** (archive files 2, 3, and 4 of 21). The full archive covers December 2011 to August 14, 2026, but we restrict it for two reasons:

- **Consistent categories.** The CFPB revised its product and issue categories around 2017. Earlier files mix old and new category names, which would complicate the label rule. Narratives were also not published before 2015, so the earliest years have no text.
- **Less templated content.** Complaint volume surged after 2023, and the CFPB has attributed part of that surge to duplicate and AI-generated complaints and to credit-repair-firm filings. Excluding that period reduces templated, near-identical narratives.

Our 30,000-complaint sample needs only a small fraction of this range.

**Data is not included in this repository.** Run `./download_dataset.sh` to fetch the three archive files into `data/raw/` (see [Setup](#setup)).

### Second dataset: Elliptic (planned)

The [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (Weber et al., 2019) is a real transaction graph at course scale: 203,769 transactions (nodes), 234,355 payment flows (directed edges), and 49 time steps. About 4.5K transactions are labelled illicit, 42K licit, and the rest unknown. Each node has 166 features: the time step plus 93 local features describing the transaction itself, and 72 features that aggregate its one-hop neighbourhood. PyTorch Geometric ships a loader (`EllipticBitcoinDataset`). Like CFPB, it is downloaded locally and never committed.

It suits the comparison for three reasons. Its edges are observed, not constructed. Its 72 pre-aggregated neighbour features map exactly onto Model 2 (local features only vs. local plus aggregated). And the original paper found that a random forest on all features beat a graph convolutional network, so the tabular arms are a strong, published baseline.

| Field | Type | Use in this project |
|---|---|---|
| Consumer complaint narrative | Free text | Main text feature (embedded) |
| Product / Sub-product | Categorical | Feature and graph node |
| Issue / Sub-issue | Categorical | **Label only, never a feature** |
| Company | Categorical | Graph node |
| State | Categorical | Feature and region graph node (ZIP code is not used) |
| Date received | Date | Month/year features, sampling |
| Submitted via | Categorical | Feature |
| Tags (Older American, Servicemember) | Categorical | Feature |
| Company response, Timely response | Categorical / binary | Company-level features (training data only) |
| Complaint ID | Identifier | Joins only |

### Label

A complaint is **positive (1)** if the Issue or Sub-issue the consumer chose describes fraud, a scam, identity theft, or an unauthorized transaction, and **negative (0)** otherwise. Categories too ambiguous to call either way are **excluded**: their complaints are removed before sampling. Issue and Sub-issue are removed from all feature sets to prevent label leakage. A random sample of labels is hand-checked to estimate how noisy this rule is.

**[`docs/LABEL_RULE.md`](docs/LABEL_RULE.md) lists how every category is treated, with the reasons.**

The rule is an **explicit, reviewed allow-list** of (Product, Issue, Sub-issue) values in [`configs/categories.yaml`](configs/categories.yaml), not a keyword match. Keyword matching would mislabel categories such as "Credit monitoring or identity theft protection services" or "Problem with fraud alerts or security freezes", which contain the keywords but are not fraud events. The `load` stage writes every category combination with its count and flags keyword candidates. The `label` stage refuses to run until:
- the file is marked `confirmed: true`;
- every entry matches at least one complaint;
- no category is in two lists;
- every keyword candidate in the target products is listed as positive, reviewed-negative or excluded.

**Rule version 2** (2026-09-29) comes from a category-by-category review, decided on the meaning of each category only, before looking at any model errors:
- **Positive:** debt collection "Debt was result of identity theft", **"Debt is not yours"** and **"Impersonated attorney, law enforcement, or government official"**; money transfer "Fraud or scam" and "Unauthorized transactions or other transaction problem"; card charges the consumer did not make, and cards opened through identity theft, fraud, or without consent; checking/savings unauthorized transactions, and accounts opened through fraud or without consent.
- **Excluded:** the card company not resolving a purchase dispute; cards sent that were never applied for; "Information belongs to someone else"; credit inquiries the consumer doesn't recognize; lost or stolen checks and money orders (lost is not fraud, stolen is, and the category does not separate them); credit monitoring and fraud-alert services.

Before deduplication, 98,723 of the 334,773 labelled target-product complaints with narratives (**29.5%**) are positive, and 20,266 complaints are excluded. Debt collection is now 39.6% fraud and supplies 62% of the positives.

Version 1, used for the first df-analyze results, labelled "Debt is not yours" and "Impersonated…" 0, and labelled the excluded categories 0 as well. Its fraud rate was 16.8% (59,663 of 355,039). Results from the two versions are not comparable.

### Sample

30,000 complaints with narratives from fraud-prone products: credit cards and prepaid cards (including their 2023 renames "Credit card" and "Prepaid card"), checking and savings accounts, money transfers, and debt collection. The exact product names are in `configs/categories.yaml`. The sample is drawn by stratified random sampling so that the natural fraud rate is preserved. This size keeps df-analyze runs tractable on a single machine.

Near-duplicate narratives are removed before sampling and splitting, so that copies of the same text cannot appear in both train and test. Narratives are normalised (lower-cased, CFPB `XXXX` redactions collapsed, punctuation removed) and compared with MinHash LSH on word 5-gram shingles. Pairs with estimated Jaccard similarity ≥ 0.85 are grouped, and each group keeps its earliest complaint. To keep this cheap, deduplication runs on a stratified pool 1.5× the sample size, and the final sample is drawn from the deduplicated pool.

## Approach

### Shared features

Narratives are embedded with df-analyze's embedding code (multilingual E5-large, 1024 dimensions), then reduced to 30 PCA components. `df-embed.py` itself only runs on the CPU. By default the pipeline therefore runs df-embed's own loader, dataset and embedding function unchanged inside df-analyze's environment (`scripts/dfa/embed_on_device.py`), with the model on the GPU. On every run it re-embeds the first 64 narratives with the unmodified CPU code path and stops if the two disagree beyond float32 tolerance (1e-3). PCA is fit on the training complaints only and then applied to all complaints. Both models receive the same text components and the same tabular features:

| Group | Features |
|---|---|
| Text | `text_pc01` … `text_pc30` |
| Product | Product, Sub-product |
| Region | State |
| Company | Number of training complaints, timely-response rate, and the rate of each company-response type, all from **training complaints only**. For a training complaint they are **leave-one-out** (the company's *other* training complaints), so no complaint sees its own company response. A test complaint uses all the company's training complaints. With no such complaints, the count is 0 and the rates are the overall training rates. |
| Complaint metadata | Submission channel, Older American / Servicemember tags, year and month received |

Categorical levels (product, sub-product, state, channel) with fewer than 20 **training** complaints, and levels that never occur in training, are merged into one `__rare_or_unseen__` level. Every remaining level therefore has at least 20 training rows, and df-analyze's own rare-level merging (which counts train and test together) has nothing left to merge. The merged levels are listed in `outputs/reports/rare_levels_merged.json`.

The company name itself is not a Model A feature; in the GNN it becomes the company node. A complaint's own company response and timely-response flag are not used as per-complaint features, because they are outcomes recorded after the complaint was filed. No feature is derived from labels.

### Model A: df-analyze

We pass our own train and test tables to df-analyze (`--df-train` / `--df-tests`), so both models use exactly the same held-out complaints. Classifiers are LightGBM, logistic regression, CatBoost, GANDALF, random forest and kNN, plus df-analyze's dummy baseline, which it adds automatically. CatBoost and GANDALF train on the GPU when df-analyze's environment can see CUDA; the others run on the CPU. df-analyze's MLP is left out for now. df-analyze pins it to the CPU, where it uses up its 60-minute tuning limit on each of the 4 feature sets and adds about 4 hours per run. GANDALF is still included as a neural network. df-analyze runs its filter and embedded feature selection (wrapper selection is off). It tunes hyperparameters with Optuna for 100 trials, using **balanced accuracy** (at the 0.5 cut-off) for every model. The prediction-based filter selection uses AUROC, computed from probabilities, and picks the same features for every model. Both settings are configurable in `configs/default.yaml`.

**Why balanced accuracy, not AUROC or accuracy.**
- **Not accuracy** (df-analyze's default): with about 30% positives (16% under label rule v1), always predicting "not fraud" already scores about 0.70 (0.84 under v1).
- **Not AUROC:** when asked to tune on AUROC, df-analyze silently scores most models on balanced accuracy of hard 0/1 predictions instead (`enumerables.py`, `ClassifierScorer.tuning_score`), but scores GANDALF on real AUROC through its own code path. Models would then be tuned toward different goals, and their tuning scores would not be comparable. An earlier run did exactly this: GANDALF's tuning "AUROC" of about 0.91 was a different quantity from CatBoost's 0.70, which was really balanced accuracy.
- **Balanced accuracy** is computed the same way for every model.

**How Model A is chosen (fairly).** df-analyze's tuning scores still differ in protocol: most models use 5-fold CV, while GANDALF uses one validation split. Model A is therefore **not** chosen by them. The `select_model` stage takes every tuned (model, feature set) combination with its tuned hyperparameters and refits it with df-analyze's own code on the **same 5 stratified folds of the training set**. It scores each held-out fold from predicted probabilities: PR-AUC, AUROC, balanced accuracy and Brier score. Model A is the combination with the highest mean cross-validated **PR-AUC**, the headline metric. The test set is never read. Every model is scored identically, and the stage stops if any model cannot be scored.

The time limits described under Runtime notes mean slower models may complete fewer tuning trials. The report records the trials completed per model and whether each run stopped at its time limit.

df-analyze drops identifiers and re-encodes features in its exported `X_train.csv` / `X_test.csv`. After every run, the pipeline therefore checks that the export matches our saved split row for row: row counts, labels, and each text component (which df-analyze clips and rescales, so the check allows for that). The run fails if they do not match.

**Reading df-analyze's results without touching the test set.** Hyperparameters are tuned with internal cross-validation on the training table only. The best Model A is chosen by the shared-CV PR-AUC in `outputs/reports/model_selection_cv.csv`. df-analyze's own tuning scores (`tuning/test00/tuned_models_00.csv`) are for information only, because they are not comparable across models (see above). In df-analyze's results tables:
- `holdout` rows: models fit on train, scored on our test set. This is the Model A test result.
- `trainset` rows: scored on the training data itself.
- `5-fold` rows: df-analyze refits the tuned models on folds **of the test set**. These are not a valid test result and must not be compared with the GNN.

### Model B: graph neural network

**Graph schema (CFPB), in one paragraph.** Every complaint is a `complaint` node whose features are exactly Model A's features (the 30 text components plus the tabular and company features). Each complaint has one edge to its `company` node, one to its `product` node, and one to its `region` (state) node, each with a reverse edge, so information can flow complaint → company → other complaints about that company. Company nodes carry the company statistics computed from training complaints only (complaint count, timely-response rate, response-type rates); companies with no training complaints share one fallback node holding the overall training rates. Product and region nodes have no input features and learn an embedding instead. Optionally (an open decision), each complaint also links to its k most similar *training* complaints by narrative embedding. Test complaints are attached inductively. The training graph contains only training complaints. At evaluation time, each test complaint is added with its company, product and region edges (and any similarity edges, which point only to training complaints), and is scored without its label entering the graph. Large companies become hubs with thousands of complaints; neighbour sampling caps how many neighbours each node aggregates.

**Graph schema (Elliptic).** Nodes are transactions with their local features; directed edges are payment flows. Edges never cross time steps, so with the standard temporal split (time steps 1–34 train, 35–49 test) the test transactions form their own subgraphs, and evaluation is inductive by construction. Unlabelled transactions stay in the graph as context, but never enter the loss or the metrics. The time step defines the split and is not a feature (test time steps never occur in training). Model 1 (tabular) uses only the 93 local features; the 72 provided neighbour aggregates belong to Model 2.

No node or edge feature on either dataset is derived from labels.

**Architecture choice: heterogeneous GraphSAGE (primary), with a graph-free control**

We use GraphSAGE layers with a separate weight set per edge type (PyG's heterogeneous `to_hetero` / `HeteroConv` pattern). The reasoning:

1. **It covers R-GCN's main idea already.** R-GCN's key feature is relation-specific weights. A heterogeneous GraphSAGE model has those too, while adding the advantages below. With only a handful of edge types here, a separate R-GCN model is unlikely to behave very differently, so it is an optional ablation rather than a second headline model.
2. **It is inductive.** GraphSAGE learns a function of a node's own features and its neighbourhood, so it can score complaints it never saw during training. That matches a held-out test set, and it keeps the door open to a stricter split that holds out whole companies.
3. **It keeps a complaint's own signal separate from its neighbours'.** GraphSAGE transforms a node's own features separately from the aggregated neighbour features. Fraud graphs are known to be *heterophilous*: a fraud complaint is often surrounded by mostly non-fraud complaints about the same large company. Plain neighbourhood averaging can wash out the complaint's own evidence; keeping the self-term separate reduces that risk. Fraud-specific architectures such as CARE-GNN were designed for this camouflage problem, but they add substantial complexity for a course project.
4. **Neighbour sampling handles hub nodes.** A large bank can have thousands of complaints. Sampled mini-batches cap how many neighbours each node aggregates, which keeps memory bounded on a laptop GPU.
5. **Fancier heterogeneous models rarely pay off.** A large reproduction study of heterogeneous GNNs (Lv et al., KDD 2021) found that well-tuned simple GNNs matched or beat most specialised heterogeneous architectures. We therefore do not use HGT or HAN.

**Required controls:** the same network with all edges removed (Model 3, an MLP on each node's own features), and the tree model with neighbour-aggregated features (Model 2). Comparing the GNN against the first isolates the effect of the graph with the architecture held fixed; comparing it with the second separates the graph's information from message passing. A GNN-vs-LightGBM comparison alone cannot do either, because it changes the model family and the information at once.

**Training:** class-weighted binary cross-entropy, early stopping on a validation split carved from the training set, and several random seeds with mean ± standard deviation reported.

### Train/test split

- **Option A: random stratified split (implemented).** 60% train / 40% test, stratified by label, with a fixed seed. We make the split ourselves, save the Complaint IDs (`data/processed/train_ids.csv`, `test_ids.csv`), and pass the two tables to df-analyze. The test set is frozen: nothing is tuned, selected, or early-stopped on it.
- **Option B: company-holdout split (open decision).** Whole companies are held out for testing. This is a stricter test of generalisation to unseen companies. Because df-analyze accepts a predefined split, it is possible for both models.

The main comparison uses Option A. Option B may be run later if time permits.

## Evaluation

All metrics are computed on each dataset's frozen test set, identically for all four models:

- **PR-AUC** (headline metric, since the classes are imbalanced)
- F1, recall, AUROC

The neural models (3 and 4) are trained with at least 5 seeds and reported as mean ± standard deviation.

## Explainability

**df-analyze model:** univariate feature–target associations, LightGBM and linear feature importances, selected feature sets, plus SHAP on the best model.

**GNN:**
- GNNExplainer for individual predictions (which neighbours and edges mattered)
- Permutation importance by feature group
- Edge-type ablation: remove company, product, region, or similarity edges and measure the drop in PR-AUC. This directly tests whether each relation helps.

**Cross-model comparison:** the two model families attribute importance to different kinds of objects (SHAP to input columns, GNNExplainer to edges and neighbours, and individual PCA components have no direct meaning on their own). The head-to-head comparison is therefore done at the **feature-group** level (text, company, product, region) using permutation importance. Finer-grained explanations are reported as illustrative case studies.

## Setup

Two separate Python environments are needed, because df-analyze manages its own pinned dependencies. `uv` installs the right Python version for each automatically.

**Requirements:** Linux, macOS, or Windows. On Windows, use **WSL2** (e.g. Debian or Ubuntu from the Microsoft Store): run every command below in the WSL terminal, and clone into the WSL home directory (`~/...`), not `/mnt/c/...`, which is much slower. From Windows, the files are at `\\wsl.localhost\<distro>\home\<user>\...`. A recent NVIDIA Windows driver makes the GPU visible inside WSL2 automatically; check with `nvidia-smi` in the WSL terminal. Stages 1–8 have been run on Windows 11 through WSL2 (Debian 13) with an NVIDIA GPU; macOS is untested. A fresh WSL Debian may lack the tools below: `sudo apt install git curl wget unzip`. Also needed: [uv](https://docs.astral.sh/uv/) **0.9.16 or newer** (older uv releases cannot download Python 3.13.11, which df-analyze needs; check with `uv --version` and upgrade with `uv self update`; if uv came from a system package manager such as `dnf` that has no newer version, run `uv tool install 'uv>=0.9.16'`, put `~/.local/bin` first on your `PATH` with `uv tool update-shell`, and open a new shell), Git, `wget` or `curl`, and `unzip`. An NVIDIA GPU is optional (used for the GNN later).

| Environment | Python | Used for |
|---|---|---|
| This project | 3.13 (pinned in `.python-version`; 3.11+ supported and tested) | everything except embedding and df-analyze |
| df-analyze (separate clone) | df-analyze's own (3.13) | `df-embed.py` and `df-analyze.py`, called as subprocesses |

```bash
# 1. This project
git clone https://github.com/caelenm/fraud-detect-gnn.git
cd fraud-detect-gnn
uv sync

# 2. df-analyze, next to this repository, at the tested commit, on Python 3.13
#    (not 3.14: df-analyze's locked catboost has no Python 3.14 wheels)
uv python install '>=3.13.11,<3.14'
git clone https://github.com/stfxecutables/df-analyze.git ../df-analyze
git -C ../df-analyze checkout 199e5638620693c267dac715784f1fd0e33fa796
uv sync --locked --python '>=3.13.11,<3.14' --directory ../df-analyze
uv run --python '>=3.13.11,<3.14' --directory ../df-analyze python -c "import pytorch_lightning"   # sanity check
uv run --python '>=3.13.11,<3.14' --directory ../df-analyze python df-embed.py --download --modality nlp   # one-time model download
```

The pinned commit is recorded as `df_analyze.commit` in `configs/default.yaml`, and the stages that call df-analyze stop if the clone is at a different commit. Older df-analyze checkouts (e.g. the old `master` branch) lack dependencies that `df-embed.py` imports, such as `pytorch_lightning`. df-analyze must run on **Python 3.13 (3.13.11 or newer), not 3.14**: its locked `catboost==1.2.8` publishes Python 3.13 wheels only, so on 3.14 uv tries to compile catboost from source and the build fails. Always pass `--python '>=3.13.11,<3.14'` as above; the pipeline does the same when it calls df-analyze (`df_analyze.python` in `configs/default.yaml`). If `uv python install` reports `No download found for request: cpython->=3.13.11, <3.14`, your uv is older than 0.9.16; upgrade it as described under Requirements and retry. The pipeline checks this too and stops with the same instructions. If a df-analyze `.venv` was already created on 3.14, delete it (`rm -rf ../df-analyze/.venv`) and rerun the `uv sync` line.

If df-analyze is somewhere other than `../df-analyze`, set `DF_ANALYZE_DIR=/path/to/df-analyze` or `df_analyze.dir` in `configs/default.yaml`.

Download the dataset (three archive files; safe to rerun, since completed files are skipped and interrupted downloads resume):

```bash
./download_dataset.sh              # downloads and extracts into data/raw/
./download_dataset.sh --no-extract # download the zip files only
```

## Running the pipeline

`run.py` runs the pipeline stages in order. Stages whose outputs already exist are skipped. Once a stage runs, every later stage in the same invocation reruns too, because its inputs changed. After changing the config, rerun the affected stages with `--force`.

```bash
uv run run.py --list              # show the stages
uv run run.py --only load         # stage 1, then review the category report
uv run run.py                     # everything else, once categories are confirmed
uv run run.py --to split          # stop after a given stage
uv run run.py --from pca --force  # rerun from a stage onwards
uv run run.py --from df_analyze --force --set df_analyze.htune_trials=10  # pilot run
```

`--set section.key=value` overrides one config value for a single invocation (repeatable; the key must already exist in the config). The overridden config is what gets saved in the run log.

### Stopping and resuming

A run can be stopped at any time with Ctrl+C, by closing the terminal, or by shutting the computer down. Continue it later with:

```bash
uv run run.py --resume
```

- **Every finished stage is kept.** When `run.py` starts, it records the stages it still has to run in `outputs/pipeline_state.json` and ticks each one off as it completes. `--resume` runs exactly the remaining stages. While an interrupted run is pending, a plain `uv run run.py` refuses to start, so stale outputs from an earlier run are never mistaken for finished work. Use `--resume` to continue, or `--force` to start over.
- **Long stages also save inside themselves.** `select_model` saves every finished fit, so a resume redoes only the fit that was interrupted. Stages added later (for example one GNN fit per model and seed) use the same per-unit checkpoints (`outputs/checkpoints/<stage>/`).
- **Some stages restart from their beginning:** `df_analyze` (about 4–5 h) runs all classifiers in one process that cannot be paused, and `embed` (about 30 min) is a single unit. Running df-analyze once per classifier would allow resuming between models. It was rejected because df-analyze's feature selection is not seeded, so separate runs could give different models different feature sets.
- **A resume refuses a changed config**, because the finished stages were computed with the old one. Start over with `--force` instead.
- **Watchdog.** If one configuration in `select_model` takes longer than `select_model.config_timeout_s` (30 min), the stage prints every thread's stack to `model_selection.log` and fails instead of hanging. On WSL a hang like this has come from the GPU link getting stuck (`nvidia-smi` stops responding). Run `wsl --shutdown` in Windows PowerShell, open WSL again, then `uv run run.py --resume`.

Each stage can also be run on its own with `uv run scripts/NN_<stage>.py`.

| # | Stage | What it does | Main outputs |
|---|---|---|---|
| 1 | `load` | Reads archives 2–4, keeps complaints with a narrative received May 2018–Aug 2023, and reports every Product/Issue/Sub-issue combination | `data/interim/complaints.parquet`, `outputs/reports/category_values.csv` |
| 2 | `label` | Checks the reviewed allow-list in `configs/categories.yaml`, keeps the target products, adds the label, and drops Issue/Sub-issue | `data/interim/labeled.parquet` |
| 3 | `sample` | Removes near-duplicate narratives, then draws a 30k stratified sample | `data/interim/sample.parquet` |
| 4 | `split` | 60/40 stratified split; saves Complaint IDs | `data/processed/train_ids.csv`, `test_ids.csv` |
| 5 | `embed` | Embeds the narratives with df-embed's code on the GPU (`embed.runner: gpu`), checked against its CPU path; `embed.runner: df-embed` runs `df-embed.py` itself (CPU only) | `data/processed/embeddings.parquet` |
| 6 | `pca` | 30 PCA components, fit on training complaints | `data/processed/text_pca.parquet` |
| 7 | `features` | Tabular and company features (training statistics only) | `data/processed/tabular_features.parquet` |
| 8 | `df_analyze_input` | Writes the df-analyze train/test tables (no identifiers) | `data/processed/df_analyze/{train,test}.parquet` |
| 9 | `df_analyze` | Runs df-analyze (Model A) and checks its exported split against ours | `outputs/df_analyze/<timestamp>/`, `outputs/reports/df_analyze_split_check.json` |
| 10 | `select_model` | Refits every tuned df-analyze combination on the same 5 folds of the training set and scores it from probabilities (PR-AUC, AUROC, balanced accuracy, Brier), with its tuned settings and with df-analyze's default settings (the before-tuning baseline); records each model's tuning budget | `outputs/reports/model_selection_cv.csv`, `tuning_budget.csv` |
| 11 | `df_analyze_report` | Test-set metrics for every tuned df-analyze model, computed from its saved probabilities (PR-AUC, AUROC, fraud-class F1/precision/recall, accuracy, balanced accuracy); picks Model A by shared-CV PR-AUC only | `outputs/reports/model_a_report.md`, `model_a_metrics.csv`, `model_a.json` |
| 12 | `web_report` | One self-contained HTML page: Model A's configuration, tuned hyperparameters, test metrics and confusion matrix; every tuned model; the tuning budget; the run configuration; and two scrollable cards of Model A's most and least confident test predictions. Placeholders are reserved for Model B and inter-model agreement | `outputs/report/index.html` |
| 13 | `label_audit` | Label shortcut check: fraud rate by product name × year (and sub-product × year) for all labelled complaints and for the sample, plus the months in which each product name occurs. Counts and rates only. It sits last so it never forces the long stages to rerun; run it with `uv run run.py --only label_audit` | `outputs/reports/label_audit.md`, `label_audit_*.csv` |

**Viewing the web report.** It needs no server. On Windows with WSL, run `explorer.exe outputs/report/index.html` from the repo folder, or open `\\wsl.localhost\<distro>\home\<user>\...\fraud-detect-gnn\outputs\report\index.html` in a browser. On Linux or macOS, open the file directly. The page contains complaint narratives: share it only with the group, the same way as the bundle, and never commit or post it. Confidence is the probability the model's output gives the class it predicted, so it runs from 0.5 to 1.0. Very small and very large probabilities are shown in scientific notation (e.g. `1 − 6.4e-07`) so they stay distinguishable.

**The label rule is already reviewed** (`configs/categories.yaml`, `confirmed: true`, rule version 2; see `docs/LABEL_RULE.md`). If the data or the rule changes, rerun `load`, check `outputs/reports/category_values.csv`, and make sure every keyword candidate in the target products is listed as `positive`, `reviewed_negative` or `excluded`. The `label` stage writes the excluded complaints per category to `outputs/reports/label_excluded.csv`. The `label` stage explains exactly what is missing if the file is not consistent with the data.

Every invocation writes its config, seed, git commit, and package versions to `outputs/runs/<timestamp>/`.

**GPU:** with `gpu.require: true` (the default), the `embed` and `df_analyze` stages first check that df-analyze's environment can see a CUDA GPU. They stop with troubleshooting steps if it can't, and record the GPU in `outputs/runs/<timestamp>/cuda_info_*.json`. Set `gpu.require: false` on machines without an NVIDIA GPU; embedding then runs on the CPU, which takes hours for 30,000 narratives.

**Runtime notes:** long narratives are truncated to the model's 512-token limit. df-analyze's runtime grows with the number of classifiers and `htune_trials` (each Optuna trial for GANDALF trains a neural network), so consider a pilot run with fewer trials first. Each model is tuned separately on each of df-analyze's 4 feature sets, and df-analyze stops each tuning run at a fixed time limit (15–60 minutes depending on the model). In practice most searches end earlier: once 50 settings have been tried, the search stops if the last 15 found nothing better. That means the settings search has settled; it does not mean the model failed to learn. The full run of 2026-09-28 on an RTX 4060 laptop (WSL2) took 4 h 07 min for df-analyze. `select_model` took 17 min scoring tuned settings only, and about twice that now that it also scores default settings.

**Effect of tuning.** `select_model` also refits every (model, feature set) with **df-analyze's default hyperparameters** on the same folds. The reports show cross-validated PR-AUC untuned vs. tuned, and the difference as the tuning gain. The true first try of the search cannot be reproduced, because df-analyze saves only the best settings. Tuning optimises balanced accuracy, not PR-AUC, so the gain can be negative. Only the tuned scores are used to choose Model A. The worst case, with every model hitting its time limit, is about 12–16 hours. Because of these time limits, the number of trials actually completed, and so the results, can depend on how fast the machine is.

### Sharing outputs with the group

Embedding and df-analyze take hours, so one person can run them and share the results. The results cannot go in git, because they contain complaint narratives and are too large for GitHub (see `AGENTS.md`).

```bash
# Person who ran the pipeline: bundle every stage output into one file
uv run pack_artifacts.py            # writes fraud_artifacts_<time>.tgz in the repo root
```

The bundle includes `data/interim/` (except the embedding scratch files), `data/processed/`, `outputs/reports/`, and the latest finished df-analyze run. It leaves out the raw archives and the per-invocation logs in `outputs/runs/`. It is roughly 1 GB. Share it through OneDrive or Teams, never through the repository.

```bash
# Group member: after the Setup steps, put the .tgz in the repo root, next to unpack_artifacts.py
uv run unpack_artifacts.py          # verifies checksums and puts every file where it belongs
uv run run.py                       # skips the finished stages
```

On Windows (WSL2), copy the downloaded bundle into the repo in File Explorer at `\\wsl.localhost\<distro>\home\<user>\...\fraud-detect-gnn`, or run `cp /mnt/c/Users/<you>/Downloads/fraud_artifacts_*.tgz .` in the WSL terminal.

Every file is checked against a SHA-256 manifest. If a file already exists with different contents, nothing is written until you pass `--force`. Files are written to a temporary name first, so an interrupted unpack never leaves a partial file that `run.py` would treat as finished. Both scripts print which stages `run.py` will skip. They also warn if the bundle was made at a different git commit or with a different `configs/default.yaml` or `configs/categories.yaml`; in that case, rerun the affected stages with `--force`. Group members who only use the bundle do not need to download the raw data or the embedding model. They still need df-analyze set up (see Setup) to run the df-analyze stage.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests use only small synthetic rows, all defined in [`tests/synthetic.py`](tests/synthetic.py). Tests that need files write them to pytest's temporary directory. No real or synthetic data files are committed.

## Repository layout

```
fraud-detect-gnn/
├── README.md
├── AGENTS.md            # rules for AI coding agents working in this repo
├── CLAUDE.md            # points Claude Code at AGENTS.md
├── LICENSE
├── download_dataset.sh  # fetches CFPB archive files 2–4 into data/raw/
├── run.py               # runs the pipeline stages in order
├── pack_artifacts.py    # bundles stage outputs into a .tgz to share (not via git)
├── unpack_artifacts.py  # restores a bundle so run.py skips finished stages
├── pyproject.toml / uv.lock / .python-version
├── configs/
│   ├── default.yaml     # paths, sampling, dedup, split, PCA, df-analyze settings
│   └── categories.yaml  # reviewed label rule and product filter
├── src/fraud_detect/
│   ├── pipeline.py      # stage definitions (add new stages here)
│   ├── cli.py           # command line shared by run.py and scripts/
│   ├── data/            # loading, labelling, deduplication, sampling
│   ├── features/        # embeddings, PCA, tabular features
│   ├── models/          # df-analyze (Model A), fair model selection; GNN later
│   ├── report/          # static HTML report (web.py, report.css)
│   ├── external.py      # runs df-analyze scripts in their own environment
│   ├── artifacts.py     # packing and unpacking output bundles
│   └── runlog.py        # seeds and run metadata
├── scripts/             # NN_<stage>.py: run a single stage
├── tests/
├── data/                # local only, git-ignored
└── outputs/             # local only, git-ignored
```

Planned: neighbour-aggregated features, `graph/`, GNN models, an Elliptic loader, `eval/`, and `explain/` modules as their stages land.

## Limitations

- **Labels are consumer-chosen categories, not verified fraud.** Consumers choose the Issue category when filing, so some disputes or billing errors are tagged as fraud, and some real fraud is filed under generic issues. The CFPB task is therefore classifying fraud-related complaints, not detecting fraud.
- **Taxonomy changes could create a shortcut.** The CFPB renamed products and sub-issues over time (for example the 2023 split of "Credit card or prepaid card"). If fraud and not-fraud categories came from different taxonomy versions, product name and date would predict the label without the text. The `label_audit` stage tabulates the fraud rate by product × year to check this; if it shows a shortcut, a time-based split is an open decision.
- **The CFPB graph is constructed, not observed.** Its edges come from attributes Model A already uses, so a GNN on it is expected to add little. The Elliptic comparison is there to show the contrast with real edges.
- **Complaints are unverified allegations.** The CFPB does not verify narratives.
- **Not a representative sample.** Narratives are published only when consumers opt in, and complaints reflect negative experiences by design.
- **Results cover May 2018 to August 2023 only.** Complaint patterns before and after this window (including the post-2023 surge) may differ, so results may not generalize to them.
- **Templated complaints still exist inside the window.** Credit-repair services and identical disputes filed against several companies predate 2024. Deduplication reduces but does not eliminate this.
- **df-analyze scales continuous features over train and test together.** Given separate train and test files, df-analyze concatenates them before its robust clip-and-rescale of continuous features. This uses no labels, but test rows influence the clip points and scaling. (Its rare-level merging, which also counts over both, is neutralised by our train-only merging; see Shared features.) PCA and company statistics, which we compute ourselves, use training complaints only.
- **Company features are leave-one-out for training complaints, but a GNN company node cannot be.** Model A's per-complaint company features exclude the complaint's own outcome. A company node in the planned GNN carries one aggregate over all of the company's training complaints, including each one's own response. This matters most for companies with few complaints, and should be kept in mind when comparing the models.
- **Results depend on the sampled subset** and on graph-construction choices (edge types, similarity threshold, region granularity).

## References

- df-analyze: https://github.com/stfxecutables/df-analyze
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric
- Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE).
- Schlichtkrull et al. (2018). *Modeling Relational Data with Graph Convolutional Networks* (R-GCN).
- Lv et al. (2021). *Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks.* KDD. https://arxiv.org/abs/2112.14936
- Tang et al. (2023). *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection.* NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2306.12251
- Weber et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD Workshop on Anomaly Detection in Finance. https://arxiv.org/abs/1908.02591
- Dou et al. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters* (CARE-GNN). https://arxiv.org/abs/2008.08692
- Ying et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.*
- CFPB, *The CFPB to Cease Discretionary Publication of Complaint Narratives and Visualizations* (August 14, 2026).

## License

GPL-3.0. See [LICENSE](LICENSE).

The CFPB complaint data is public domain and is not redistributed in this repository.
