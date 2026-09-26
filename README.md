# fraud-detect-gnn

**Explainable fraud complaint detection: tabular AutoML vs. graph neural networks**

Course project for CS555: Data Mining and Machine Learning, St. Francis Xavier University (StFX).

> **Status:** early development. The pipeline described below is the plan; commands and results will be filled in as each stage lands.

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

Our 20,000–30,000 complaint sample needs only a small fraction of this range.

**Data is not included in this repository.** Run `./download_dataset.sh` to fetch the three archive files into `data/raw/` (see [Setup](#setup)).

| Field | Type | Use in this project |
|---|---|---|
| Consumer complaint narrative | Free text | Main text feature (embedded) |
| Product / Sub-product | Categorical | Feature and graph node |
| Issue / Sub-issue | Categorical | **Label only, never a feature** |
| Company | Categorical | Graph node |
| State, ZIP code (first 3 digits) | Categorical | Graph node or feature |
| Date received | Date | Month/year features, sampling |
| Submitted via | Categorical | Feature |
| Tags (Older American, Servicemember) | Categorical | Feature |
| Company response, Timely response | Categorical / binary | Company-level features (training data only) |
| Complaint ID | Identifier | Joins only |

### Label

A complaint is **positive (1)** if its Issue or Sub-issue mentions fraud, scam, identity theft, or unauthorized transactions, and **negative (0)** otherwise. Issue and Sub-issue are removed from all feature sets to prevent label leakage. A random sample of labels is hand-checked to estimate how noisy this rule is.

### Sample

About 20,000 to 30,000 complaints with narratives from fraud-prone products (credit cards, bank accounts, money transfers, debt collection), stratified by label. This size keeps df-analyze runs tractable on a single machine. Near-duplicate narratives are removed before splitting so that copies of the same text cannot appear in both train and test.

## Approach

### Shared features

Narratives are embedded with df-analyze's `df-embed.py` (multilingual E5-large, 1024 dimensions), then reduced to about 30 PCA components. Both models receive the same text components and the same tabular features.

### Model A: df-analyze

Classifiers: LightGBM, random forest, logistic regression, kNN, and MLP, with df-analyze's built-in feature selection and Bayesian hyperparameter tuning. The best model on the internal validation is the tabular baseline. df-analyze exports its train/test split (`X_train.csv`, `X_test.csv`), which the GNN reuses so that both models are scored on exactly the same held-out complaints.

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

### Train/test split (open decision)

- **Option A: random stratified split** (df-analyze's default, 40% test). This is the simplest option and guarantees identical test sets for both models.
- **Option B: company-holdout split.** Whole companies are held out for testing. This is a stricter test of generalisation to unseen companies, but it depends on df-analyze accepting a predefined split.

The current plan is to use Option A for the main comparison and to run Option B for the GNN and the graph-free control if time permits.

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

Two separate Python environments are needed, because df-analyze manages its own pinned dependencies.

**Requirements:** Linux, [uv](https://docs.astral.sh/uv/), Git, and an NVIDIA GPU (optional; used for the GNN).

```bash
# 1. This project (Python 3.14)
git clone https://github.com/caelenm/fraud-detect-gnn.git
cd fraud-detect-gnn
uv sync

# 2. df-analyze, in its own directory and environment
git clone https://github.com/stfxecutables/df-analyze.git ../df-analyze
cd ../df-analyze
uv sync
uv run python df-embed.py --download --modality nlp   # one-time model download
```

Then, from the root of this repository, download the dataset:

```bash
./download_dataset.sh              # downloads and extracts into data/raw/
./download_dataset.sh --no-extract # download the zip files only
```

The script needs `wget` or `curl`, plus `unzip`. It is safe to rerun: completed files are skipped and interrupted downloads resume.

Pipeline commands will be documented here as each stage is implemented.

**Note:** `df-embed.py` runs on CPU only, so embedding tens of thousands of narratives on a laptop can take a while. Long narratives are truncated to the embedding model's maximum input length.

## Repository layout (planned)

```
fraud-detect-gnn/
├── README.md
├── AGENTS.md            # rules for AI coding agents working in this repo
├── CLAUDE.md            # points Claude Code at AGENTS.md
├── LICENSE
├── .gitignore
├── download_dataset.sh  # fetches CFPB archive files 2–4 into data/raw/
├── pyproject.toml
├── configs/             # sampling, graph, and training configs
├── src/fraud_detect/
│   ├── data/            # loading, labelling, deduplication, sampling
│   ├── features/        # embedding, PCA, tabular features
│   ├── graph/           # heterogeneous graph construction
│   ├── models/          # GNN and graph-free control
│   ├── explain/         # SHAP, GNNExplainer, permutation, ablation
│   └── eval/            # metrics and comparison tables
├── scripts/             # numbered pipeline entry points
├── tests/
├── notebooks/           # exploration only
├── data/                # local only, git-ignored
└── outputs/             # local only, git-ignored
```

## Limitations

- **Labels are consumer-chosen categories, not verified fraud.** Consumers choose the Issue category when filing, so some disputes or billing errors are tagged as fraud, and some real fraud is filed under generic issues.
- **Complaints are unverified allegations.** The CFPB does not verify narratives.
- **Not a representative sample.** Narratives are published only when consumers opt in, and complaints reflect negative experiences by design.
- **Results cover May 2018 to August 2023 only.** Complaint patterns before and after this window (including the post-2023 surge) may differ, so results may not generalize to them.
- **Templated complaints still exist inside the window.** Credit-repair services and identical disputes filed against several companies predate 2024. Deduplication reduces but does not eliminate this.
- **Text components are fit on the full sample.** PCA on the embeddings is unsupervised and does not use labels, but it is fit before df-analyze splits the data. The same components are used by both models, so the comparison is fair, but absolute scores may be slightly optimistic.
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
