# fraud-detect-gnn

**Explainable fraud complaint detection: tabular AutoML vs. graph neural networks**

Course project for CS555: Data Mining and Machine Learning, St. Francis Xavier University (StFX).

> **Status:** early development. Stages 1–9 (data loading through the df-analyze baseline) are implemented and tested on synthetic data; they have not yet been run on the real archive. The graph, GNN, evaluation, and explanation stages are still planned. No results yet.

---

## Overview

This project classifies U.S. consumer financial complaints as fraud/scam-related or not, and asks whether modelling the *relationships* between complaints, companies, products, and regions improves on a strong tabular baseline.

We compare two model families trained on the same features and the same train/test split:

- **Model A: tabular AutoML.** [df-analyze](https://github.com/stfxecutables/df-analyze) treats each complaint as an independent row and searches over models, feature selection methods, and hyperparameters.
- **Model B: graph neural network.** A heterogeneous GNN built in [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) connects each complaint to its company, product, and region (and optionally to similar complaints) and lets information flow across those links before classifying.

Both models are then explained with feature-importance methods so we can compare *what* each one relies on, not just how well it scores.

## Research questions

1. Does a GNN that uses relational structure outperform the best tuned tabular model?
2. Which feature groups (complaint text, company, product, region) drive each model's predictions, and do the two models agree?

## Data

**Source:** [CFPB Consumer Complaint Database, Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/).

On August 14, 2026 the CFPB stopped publishing complaint narratives in the live database and moved previously published narratives to its FOIA Reading Room as public-domain bulk files. The archive is therefore a fixed snapshot, which makes this project reproducible. Identifying information about consumers was scrubbed by the CFPB before publication.

**Time range used: complaints received May 2018 through August 2023** (archive files 2, 3, and 4 of 21). The full archive covers December 2011 to August 14, 2026, but we restrict it for two reasons:

- **Consistent categories.** The CFPB revised its product and issue categories around 2017. Earlier files mix old and new category names, which would complicate the label rule. Narratives were also not published before 2015, so the earliest years have no text.
- **Less templated content.** Complaint volume surged after 2023, and the CFPB has attributed part of that surge to duplicate and AI-generated complaints and to credit-repair-firm filings. Excluding that period reduces templated, near-identical narratives.

Our 30,000-complaint sample needs only a small fraction of this range.

**Data is not included in this repository.** Run `./download_dataset.sh` to fetch the three archive files into `data/raw/` (see [Setup](#setup)).

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

A complaint is **positive (1)** if its Issue or Sub-issue describes fraud, a scam, identity theft, or unauthorized transactions, and **negative (0)** otherwise. Issue and Sub-issue are removed from all feature sets to prevent label leakage. A random sample of labels is hand-checked to estimate how noisy this rule is.

The rule is an **explicit, reviewed allow-list** of (Product, Issue, Sub-issue) values in [`configs/categories.yaml`](configs/categories.yaml), not a keyword match. Keyword matching would mislabel categories such as "Identity theft protection or other monitoring services" or "Problem with fraud alerts or security freezes", which contain the keywords but are not fraud events. The `load` stage writes every category combination with its count and flags keyword candidates. The `label` stage refuses to run until the file is marked `confirmed: true`, every entry matches at least one complaint, and every keyword candidate in the target products has been classified as positive or reviewed-negative.

### Sample

30,000 complaints with narratives from fraud-prone products (credit cards, bank accounts, money transfers, debt collection; exact product names are in `configs/categories.yaml`), drawn by stratified random sampling so that the natural fraud rate is preserved. This size keeps df-analyze runs tractable on a single machine.

Near-duplicate narratives are removed before sampling and splitting, so that copies of the same text cannot appear in both train and test. Narratives are normalised (lower-cased, CFPB `XXXX` redactions collapsed, punctuation removed) and compared with MinHash LSH on word 5-gram shingles. Pairs with estimated Jaccard similarity ≥ 0.85 are grouped, and each group keeps its earliest complaint. To keep this cheap, deduplication runs on a stratified pool 1.5× the sample size, and the final sample is drawn from the deduplicated pool.

## Approach

### Shared features

Narratives are embedded with df-analyze's `df-embed.py` (multilingual E5-large, 1024 dimensions), then reduced to 30 PCA components. PCA is fit on the training complaints only and then applied to all complaints. Both models receive the same text components and the same tabular features:

| Group | Features |
|---|---|
| Text | `text_pc01` … `text_pc30` |
| Product | Product, Sub-product |
| Region | State |
| Company | Number of training complaints, timely-response rate, and the rate of each company-response type. All are computed from **training complaints only**; companies with no training complaints get a count of 0 and the overall training rates. |
| Complaint metadata | Submission channel, Older American / Servicemember tags, year and month received |

The company name itself is not a Model A feature; in the GNN it becomes the company node. A complaint's own company response and timely-response flag are not used as per-complaint features, because they are outcomes recorded after the complaint was filed. No feature is derived from labels.

### Model A: df-analyze

We pass our own train and test tables to df-analyze (`--df-train` / `--df-tests`), so both models use exactly the same held-out complaints. Classifiers are LightGBM and logistic regression, plus df-analyze's dummy baseline, which it adds automatically. df-analyze runs its filter and embedded feature selection (wrapper selection is off) and tunes hyperparameters with Optuna, using its default tuning metric (accuracy) and 100 trials. Both are configurable in `configs/default.yaml`.

df-analyze drops identifiers and re-encodes features in its exported `X_train.csv` / `X_test.csv`. After every run, the pipeline therefore checks that the export matches our saved split row for row: row counts, labels, and each text component (which df-analyze clips and rescales, so the check allows for that). The run fails if they do not match.

### Model B: heterogeneous GNN

**Graph schema**

- Node types: `complaint`, `company`, `product`, `region`
- Edges: complaint–company, complaint–product, complaint–region, each with a reverse edge
- Optional: complaint–complaint k-nearest-neighbour edges from narrative-embedding similarity (built with approximate nearest-neighbour search). Test complaints link only to training complaints.
- Company node features are computed from training complaints only. No node feature anywhere is derived from labels.

**Architecture choice: heterogeneous GraphSAGE (primary), with a graph-free control**

We use GraphSAGE layers with a separate weight set per edge type (PyG's heterogeneous `to_hetero` / `HeteroConv` pattern). The reasoning:

1. **It covers R-GCN's main idea already.** R-GCN's key feature is relation-specific weights. A heterogeneous GraphSAGE model has those too, while adding the advantages below. With only a handful of edge types here, a separate R-GCN model is unlikely to behave very differently, so it is an optional ablation rather than a second headline model.
2. **It is inductive.** GraphSAGE learns a function of a node's own features and its neighbourhood, so it can score complaints it never saw during training. That matches a held-out test set, and it keeps the door open to a stricter split that holds out whole companies.
3. **It keeps a complaint's own signal separate from its neighbours'.** GraphSAGE transforms a node's own features separately from the aggregated neighbour features. Fraud graphs are known to be *heterophilous*: a fraud complaint is often surrounded by mostly non-fraud complaints about the same large company. Plain neighbourhood averaging can wash out the complaint's own evidence; keeping the self-term separate reduces that risk. Fraud-specific architectures such as CARE-GNN were designed for this camouflage problem, but they add substantial complexity for a course project.
4. **Neighbour sampling handles hub nodes.** A large bank can have thousands of complaints. Sampled mini-batches cap how many neighbours each node aggregates, which keeps memory bounded on a laptop GPU.
5. **Fancier heterogeneous models rarely pay off.** A large reproduction study of heterogeneous GNNs (Lv et al., KDD 2021) found that well-tuned simple GNNs matched or beat most specialised heterogeneous architectures. We therefore do not use HGT or HAN.

**Required control:** the same network with all edges removed (an MLP on the complaint features). Comparing the GNN against this control isolates the effect of the graph itself, which a GNN-vs-LightGBM comparison alone cannot do, because that comparison also changes the model family.

**Training:** class-weighted binary cross-entropy, early stopping on a validation split carved from the training set, and several random seeds with mean ± standard deviation reported.

### Train/test split

- **Option A: random stratified split (implemented).** 60% train / 40% test, stratified by label, with a fixed seed. We make the split ourselves, save the Complaint IDs (`data/processed/train_ids.csv`, `test_ids.csv`), and pass the two tables to df-analyze. The test set is frozen: nothing is tuned, selected, or early-stopped on it.
- **Option B: company-holdout split (open decision).** Whole companies are held out for testing. This is a stricter test of generalisation to unseen companies. Because df-analyze accepts a predefined split, it is possible for both models.

The main comparison uses Option A. Option B may be run later if time permits.

## Evaluation

All metrics are computed on the same held-out test complaints:

- **PR-AUC** (headline metric, since the classes are imbalanced)
- F1, recall, AUROC

## Explainability

**df-analyze model:** univariate feature–target associations, LightGBM and linear feature importances, selected feature sets, plus SHAP on the best model.

**GNN:**
- GNNExplainer for individual predictions (which neighbours and edges mattered)
- Permutation importance by feature group
- Edge-type ablation: remove company, product, region, or similarity edges and measure the drop in PR-AUC. This directly tests whether each relation helps.

**Cross-model comparison:** the two model families attribute importance to different kinds of objects (SHAP to input columns, GNNExplainer to edges and neighbours, and individual PCA components have no direct meaning on their own). The head-to-head comparison is therefore done at the **feature-group** level (text, company, product, region) using permutation importance. Finer-grained explanations are reported as illustrative case studies.

## Setup

Two separate Python environments are needed, because df-analyze manages its own pinned dependencies. `uv` installs the right Python version for each automatically.

**Requirements:** Linux (macOS should work but is untested), [uv](https://docs.astral.sh/uv/), Git, `wget` or `curl`, and `unzip`. An NVIDIA GPU is optional (used for the GNN later).

| Environment | Python | Used for |
|---|---|---|
| This project | 3.13 (pinned in `.python-version`; 3.11+ supported and tested) | everything except embedding and df-analyze |
| df-analyze (separate clone) | df-analyze's own (3.13) | `df-embed.py` and `df-analyze.py`, called as subprocesses |

```bash
# 1. This project
git clone https://github.com/caelenm/fraud-detect-gnn.git
cd fraud-detect-gnn
uv sync

# 2. df-analyze, next to this repository, at the tested commit, in its own environment
uv python install '>=3.13.11'  # df-analyze's minimum; needed if uv's Python downloads are off
git clone https://github.com/stfxecutables/df-analyze.git ../df-analyze
git -C ../df-analyze checkout 199e5638620693c267dac715784f1fd0e33fa796
uv sync --locked --directory ../df-analyze
uv run --directory ../df-analyze python -c "import pytorch_lightning"   # sanity check
uv run --directory ../df-analyze python df-embed.py --download --modality nlp   # one-time model download
```

The pinned commit is recorded as `df_analyze.commit` in `configs/default.yaml`, and the stages that call df-analyze stop if the clone is at a different commit. Older df-analyze checkouts (e.g. the old `master` branch) lack dependencies that `df-embed.py` imports, such as `pytorch_lightning`. df-analyze needs Python 3.13.11 or newer. If your system Python is older, `uv python install '>=3.13.11'` installs a uv-managed copy (uv does this automatically unless Python downloads are set to `manual`).

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
```

Each stage can also be run on its own with `uv run scripts/NN_<stage>.py`.

| # | Stage | What it does | Main outputs |
|---|---|---|---|
| 1 | `load` | Reads archives 2–4, keeps complaints with a narrative received May 2018–Aug 2023, and reports every Product/Issue/Sub-issue combination | `data/interim/complaints.parquet`, `outputs/reports/category_values.csv` |
| 2 | `label` | Checks the reviewed allow-list in `configs/categories.yaml`, keeps the target products, adds the label, and drops Issue/Sub-issue | `data/interim/labeled.parquet` |
| 3 | `sample` | Removes near-duplicate narratives, then draws a 30k stratified sample | `data/interim/sample.parquet` |
| 4 | `split` | 60/40 stratified split; saves Complaint IDs | `data/processed/train_ids.csv`, `test_ids.csv` |
| 5 | `embed` | Runs df-analyze's `df-embed.py` on the narratives (CPU; slow) | `data/processed/embeddings.parquet` |
| 6 | `pca` | 30 PCA components, fit on training complaints | `data/processed/text_pca.parquet` |
| 7 | `features` | Tabular and company features (training statistics only) | `data/processed/tabular_features.parquet` |
| 8 | `df_analyze_input` | Writes the df-analyze train/test tables (no identifiers) | `data/processed/df_analyze/{train,test}.parquet` |
| 9 | `df_analyze` | Runs df-analyze (Model A) and checks its exported split against ours | `outputs/df_analyze/<timestamp>/`, `outputs/reports/df_analyze_split_check.json` |

**Before the `label` stage:** open `outputs/reports/category_values.csv`, edit `configs/categories.yaml` so that every keyword candidate in the target products is listed as `positive` or `reviewed_negative`, and set `confirmed: true`. The `label` stage explains exactly what is missing if the file is not ready.

Every invocation writes its config, seed, git commit, and package versions to `outputs/runs/<timestamp>/`.

**Runtime notes:** `df-embed.py` runs on CPU, so embedding 30,000 narratives on a laptop can take hours. Long narratives are truncated to the model's 512-token limit. df-analyze's runtime grows with the number of classifiers and `htune_trials`.

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
├── pyproject.toml / uv.lock / .python-version
├── configs/
│   ├── default.yaml     # paths, sampling, dedup, split, PCA, df-analyze settings
│   └── categories.yaml  # reviewed label rule and product filter
├── src/fraud_detect/
│   ├── pipeline.py      # stage definitions (add new stages here)
│   ├── cli.py           # command line shared by run.py and scripts/
│   ├── data/            # loading, labelling, deduplication, sampling
│   ├── features/        # embeddings, PCA, tabular features
│   ├── models/          # df-analyze (Model A); GNN later
│   ├── external.py      # runs df-analyze scripts in their own environment
│   └── runlog.py        # seeds and run metadata
├── scripts/             # NN_<stage>.py: run a single stage
├── tests/
├── data/                # local only, git-ignored
└── outputs/             # local only, git-ignored
```

Planned: `graph/`, GNN models, `eval/`, and `explain/` modules as their stages land.

## Limitations

- **Labels are consumer-chosen categories, not verified fraud.** Consumers choose the Issue category when filing, so some disputes or billing errors are tagged as fraud, and some real fraud is filed under generic issues.
- **Complaints are unverified allegations.** The CFPB does not verify narratives.
- **Not a representative sample.** Narratives are published only when consumers opt in, and complaints reflect negative experiences by design.
- **Results cover May 2018 to August 2023 only.** Complaint patterns before and after this window (including the post-2023 surge) may differ, so results may not generalize to them.
- **Templated complaints still exist inside the window.** Credit-repair services and identical disputes filed against several companies predate 2024. Deduplication reduces but does not eliminate this.
- **df-analyze preprocesses train and test together.** Given separate train and test files, df-analyze concatenates them before normalising continuous features and merging rare categorical levels (fewer than 20 rows). This uses no labels, but test rows influence the scaling and level merging. PCA and company statistics, which we compute ourselves, use training complaints only.
- **Company statistics include each training complaint's own outcome.** A training complaint's company response contributes to its own company's rates, while a test complaint's does not. This is the same aggregate the GNN's company node will carry, and it is not label information, but the effect is largest for companies with few complaints.
- **Results depend on the sampled subset** and on graph-construction choices (edge types, similarity threshold, region granularity).

## References

- df-analyze: https://github.com/stfxecutables/df-analyze
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric
- Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE).
- Schlichtkrull et al. (2018). *Modeling Relational Data with Graph Convolutional Networks* (R-GCN).
- Lv et al. (2021). *Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks.* KDD. https://arxiv.org/abs/2112.14936
- Dou et al. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters* (CARE-GNN). https://arxiv.org/abs/2008.08692
- Ying et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.*
- CFPB, *The CFPB to Cease Discretionary Publication of Complaint Narratives and Visualizations* (August 14, 2026).

## License

GPL-3.0. See [LICENSE](LICENSE).

The CFPB complaint data is public domain and is not redistributed in this repository.
