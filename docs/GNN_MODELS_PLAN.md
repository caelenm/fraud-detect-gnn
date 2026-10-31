# GNN models plan: end-to-end GNNs against df-analyze, then the best GNN combined with neighbour means

**Status:** approved by Caelen on 2026-10-31 (decisions in §7); running. The pre-registration is RESEARCH_PLAN §11, 2026-10-31 (fifth amendment). This plan extends `docs/ANALYSIS_PLAN.md` §8 (GraphSAGE; still a draft in the dev worktree) and uses its framework: the axes, the justification rules J1–J6, the strength levels and the verdict rule. If it conflicts with `AGENTS.md`, `AGENTS.md` wins: stop and ask.

**Where the proposal came from.** A suggested list of three models: heterogeneous GraphSAGE, R-GCN and HGT. Its wording ("test complaints", relations "company, product, state, text-similarity") describes the CFPB complaint graph that was dropped on 2026-10-07. This plan rereads each suggestion for the graphs we actually use:
- **Amazon:** users; U-P-U, U-S-U and U-V-U.
- **YelpChi:** reviews; R-U-R, R-T-R and R-S-R.

**Feature variant for this round (Caelen, 2026-10-31).** The main comparison uses every feature, as published studies do: all 25 on Amazon (with `f19`) and all 32 on YelpChi. The large effect of `f19` (RQ1: B falls from 0.918 to 0.539 without it) is reported as an observation, not as a second variant. No with/without-leak variants are run for these rungs unless the group asks. YelpChi is the validation dataset.

## 1. How useful each suggestion is here

**1.1 Facts about our setting that decide it.**
- **One node type, three edge types.** Both graphs are multi-relational but have a single node type. "Heterogeneous" here only means per-relation weights. HGT's main idea, separate projections per node type and per meta-relation, has nothing to act on.
- **Transductive by design.** Invariant 5 puts every node in the graph, test and unlabelled nodes included (features only). The suggestion's reason for GraphSAGE ("handles test complaints that were not in the training graph") does not apply. Other reasons to use GraphSAGE still hold:
  - a separate root weight keeps a node's own features apart from its neighbours'. That matters under the camouflage and low homophily of these fraud graphs;
  - mean aggregation is exactly what the strong E1 block computes, as a learned version.
- **Dense relations.** U-S-U has about 7.1M directed edges for 11,944 nodes, and YelpChi's R-S-R has 6.8M. `graph/ssl.py` avoids per-edge messages for this reason: it uses one sparse product per layer.
- **What we already know (Amazon, pilot budget).**
  - With all 25 features: B 0.918, E1 0.929, E2 0.929, E3 0.933. E1 − B is +0.010, supported on 3 splits, so the graph has little headroom to add.
  - Without `f19`: every graph setup beats B by +0.11 to +0.16. 1-hop means (E1 0.702) do at least as well as GCN embeddings (E2 0.648–0.693, E3 0.662–0.702, sizes 2 to raw 64). DGI training adds nothing over random weights.

  So the open question is not which label-free encoder to use. It is whether training on the labels, end to end, adds anything: the professor's "see if training GNN helps", reading (b).
- **Dependencies.** The `gnn` extra is plain PyG with no compiled extensions (pyg-lib, torch-sparse). PyG's `NeighborLoader` needs one of them.

**1.2 Verdict per model.**

| Suggestion | Decision | Why |
|---|---|---|
| **1. Heterogeneous GraphSAGE** (2 layers, mean aggregation, root weight; `to_hetero` / `HeteroConv` over `SAGEConv`) | **Built: E5, the main model,** with its edge-free twin E6 and the rewired-graph control N3 | It is the ladder's Model 4, and E6 is Model 3. E5 − E6 answers Q1 for a GNN: the architecture is fixed and only the edges are removed. |
| **2. R-GCN** (`RGCNConv`) | **Built as E5g: the same code with the R-GCN-style symmetric normalisation** (G1 yes) | With one node type and mean normalisation, an R-GCN layer is h′ = W₀h + Σᵣ Wᵣ·mean₍ᵣ₎(h). That is exactly E5's layer: PyG's `RGCNConv(aggr="mean")` and `HeteroConv(SAGEConv, aggr="sum")` both compute it, and a unit test checks this. The variant worth running changes the normalisation to D^-½AD^-½ per relation. Basis decomposition is pointless with 3 relations. |
| **3. HGT** (`HGTConv`) | **Not run** (G2 no) | It needs per-edge attention, which does not fit full-batch on U-S-U/R-S-R in 8 GB. Mini-batching needs a compiled PyG extension, and thinning U-S-U would change the graph. |

## 2. Part A: end-to-end GNNs against df-analyze

**2.1 Setups** (feature-set names keep the `pilot_` prefix: they run beside the pilot-budget df-analyze rungs):

| ID | Feature set | Model | Graph |
|---|---|---|---|
| **E5** | `pilot_g5_sage` | heterogeneous GraphSAGE: 2 layers, each a root weight plus one weight per relation on the 1-hop **mean**, summed; ReLU, dropout; a linear head | all relations |
| **E5g** | `pilot_g5g_rgcn` | E5 with symmetric normalisation D^-½AD^-½ per relation (R-GCN-style) | all relations |
| **E6** | `pilot_g6_mlp` | the same network with every edge removed: the root path only, an MLP of the same width and depth | none |
| **N3** | `pilot_g5n3_sage_rw` | E5 on N2's degree-preserving rewired graph (`graph_rewired/seed<split seed>/`, built or reused) | rewired |

Inputs are the own block, all features, standardised with a scaler fit on training nodes (invariant 6). They are the same columns B sees.

**2.2 Training protocol** (invariants 1–4 and 9):
- **Loss:** binary cross-entropy on training-node labels, with `pos_weight` = negatives/positives of the fitted nodes. Test and unlabelled nodes (Amazon 0–3,304) take part in message passing only.
- **Out-of-fold predictions on the saved grouped folds** (`cv_folds.csv`). For each fold k, the model is fit on the other 4 folds and predicts fold k.
- **Early stopping inside each fit.** One grouped, stratified fifth of the fit's own training nodes is held out as inner validation. Training stops after 20 epochs without a better validation PR-AUC (≤ 300 epochs), and the best epoch is kept. Never fold k, never the test set.
- **Test scores:** the mean of the 5 fold models' probabilities (cross-fitting).
- **Seeds:** 5 per rung (AGENTS.md), each one unit (`ctx.unit_store`) holding its 5 fold fits. The rung's scores are the probabilities averaged over seeds. Per-seed test PR-AUC is reported beside them.
- **Tuning grid, on training CV only, per rung and per split:**
  - hidden {64, 128} × dropout {0, 0.3} × learning rate {1e-3, 5e-3}; Adam with weight decay 5e-4;
  - seed 0, the 5 saved folds; the setting with the highest mean fold PR-AUC wins, ties to the earlier setting;
  - E5, E5g and E6 each choose from the grid. **N3 uses E5's chosen setting** on the same split: it is E5's null-graph control, so only the edges differ.
  - If the timing run shows a fit over 60 s on Amazon, the grid drops hidden 128 (4 settings). That is decided on time only, before any result.
- **Device:** CPU, as for the embeddings (bit-identical for a given seed).
- **Scoring with the project's own code** (invariant 9):
  - the max-F1 threshold on the out-of-fold predictions;
  - test PR-AUC, AUROC, F1, precision and recall; group-bootstrap 95% intervals;
  - outputs in the Model A layout: `reports/model_a.json` (`"source": "gnn"`), `model_a_metrics.csv`, plus `gnn_oof.parquet` and `gnn_test.parquet`.
  - `compare_report.py`, `repeat_summary.py` and `results_db.py` read a GNN side like a df-analyze side.

**2.3 Implementation:**
- **`src/fraud_detect/graph/supervised.py`:** the `RelSAGE` layer h′ = W_root·h + b + Σᵣ Wᵣ·(Âᵣh), with Âᵣ = Dᵣ⁻¹Aᵣ (mean) or Dᵣ^-½AᵣDᵣ^-½ (sym).
  - It uses sparse products with an explicit transpose for the backward pass, so there are no per-edge messages.
  - Layer 1's propagation of the fixed input is computed once per fit.
  - An empty relation list gives E6.
- **A unit test** checks the layer against PyG's `HeteroConv({r: SAGEConv(aggr="mean")}, aggr="sum")` and `RGCNConv(aggr="mean")` on a tiny synthetic graph.
- **Stage `gnn_classify`,** last in `STAGES` (so no finished run reruns). It has outputs only for feature sets listed in `gnn_models`, and for those sets the df-analyze stages have none.
- **Tests:**
  - stop and resume;
  - label blindness: permuting test labels changes nothing, and permuting fold k's labels leaves fold k's out-of-fold scores unchanged;
  - test labels are never read.

**2.4 Contrasts** (verdict rule unchanged; Amazon splits 0–4, YelpChi splits 0–2):

| Contrast | Isolates | Status |
|---|---|---|
| **E5 − E6** | message passing, architecture fixed (Model 4 vs Model 3) | primary (both datasets) |
| **E5 − N3** | real vs rewired neighbours for a GNN | primary (Amazon) |
| E5 − B, E6 − B, E5 − E1 | GNN vs df-analyze | secondary; different model families, read as "which family", not as a graph effect |
| E5g − E5 | normalisation | secondary |

**2.5 "The best GNN", per split, on training data only** (invariant 4):
- **Candidates:** E5 and E5g.
- **Rule:** on each split, take the per-fold difference in out-of-fold PR-AUC (E5g − E5) over the 5 saved folds. Choose E5g only if the mean difference exceeds its standard error (SD/√5); otherwise choose E5.
- Each split uses only its own training folds, because another split's training nodes overlap this split's test nodes.
- The test set never enters the choice.

## 3. Part B: the best GNN's representation plus neighbour means in df-analyze

**3.1 Label-free combination (G4 yes; existing blocks):**

| ID | Feature set | Blocks |
|---|---|---|
| **E7** | `pilot_m7_own_nbr_gnnssl` | own + nbr + gnnssl (d = 8): 25 + 78 + 24 = 127 columns on Amazon |
| E7r | `pilot_m7r_own_nbr_gnnrand` | the same with gnnrand (the untrained twin; Amazon only) |

**Contrasts:**
- **E7 − E1:** do label-free GNN embeddings add anything to neighbour means? This is primary on both datasets.
- Secondary: E7 − E2, and E7 − E7r (training, in combination).

**3.2 Supervised combination: Model 6 (G3 yes; invariant 1 amended 2026-10-31).**
- **Feature:** block `gnnsup`, one column `gnnsup__score`. It is the logit of the best GNN's (§2.5) out-of-fold probability, averaged over 5 seeds, with that GNN's setting chosen in Part A.
  - It is not the hidden-layer embedding. Each out-of-fold row comes from a different fold model, and hidden units of separately trained models are not aligned. A score is comparable across models.
  - Test nodes get the logit of the mean probability of the 5 outer fold models (× seeds).
- **Nested cross-fitting for the shared CV:**
  - For each outer fold k, the training rows of the other folds get inner out-of-fold scores from GNNs fit without fold k (4 inner fits), saved as `gnnsup.nested.parquet`.
  - `select_model` replaces the `gnnsup` column of the rows it fits on with those values when fold k is held out. df-analyze's export may rescale the column, so the values are first mapped through the exact affine map between the export and our table; the stage stops if that map is not exact.
  - The held-out rows keep their outer score. So Model A's choice, its threshold and the CV p-values never use a score built with the held-out labels.
  - Cost: 25 fits per seed per split.
- **Leakage guard:**
  - only the `gnn_oof` stage may read training labels to build a feature, and it never reads test labels;
  - every other builder stays label-blind;
  - tests: fold k's labels never reach fold k's outer score or any nested score for outer fold k, and permuting test labels changes nothing.
- **Setups:**
  - **E8** `pilot_m8_own_nbr_gnnsup` = own + nbr + gnnsup;
  - E8s `pilot_m8s_own_gnnsup` = own + gnnsup (Amazon only).
- **Contrasts:**
  - **E8 − E1** (primary, both datasets);
  - secondary: E8 − E7, E8 − E5best (stacking vs the GNN alone; different families), E8s − B, E8 − E8s.

## 4. Compute

**Amazon (splits 0–4):**
- df-analyze pilots:
  - E7 and E7r × 5 splits;
  - B, E1, E2 and E3 on splits 3–4 (splits 0–2 exist);
  - E8 and E8s × 5 splits.
  - In all, 28 runs of 10–25 min each.
- GNN fits: E5, E5g, E6 and N3 × 5 splits, plus gnnsup × 5 splits.

**YelpChi (splits 0–2):**
- data preparation and audit;
- df-analyze: B, E1, E2, E7, E8 × 3 splits (15 runs; 40–90 min each by extrapolation);
- GNN: E5, E5g, E6 × 3, then gnnsup × 3.

The queue runs continuously from 2026-10-31 (Caelen: "start everything now, do not stop"), Amazon first.

## 5. Order of work

| WP | What |
|---|---|
| WP-B1 | E7, E7r pilots on Amazon (no new code): started first |
| WP-A2 | Pre-registration in RESEARCH_PLAN §11 before any run of this round (done with this commit) |
| WP-Y0 | YelpChi: download, load, split (3 repeats), features, audit; then B, E1, E2, E7 pilots |
| WP-A1 | `graph/supervised.py`, `gnn_classify`, tests; the compare/summary/results-DB path checked on one run |
| WP-A0 | Timing run (one E5 fit, split 0); fixes the grid (§2.2) |
| WP-A3 | E5, E5g, E6, N3 on Amazon splits 0–4; E5, E5g, E6 on YelpChi splits 0–2 |
| WP-B2 | `gnn_oof` with nested cross-fitting; E8 (and E8s on Amazon) |
| WP-S | Summaries (5-split Amazon, 3-split YelpChi), Holm within each family, results database, RESEARCH_PLAN §11 results |

## 6. Threats specific to this plan

1. **The GNN is tuned less than df-analyze.** The grid has 8 settings, against 10 trials over 3 classifiers. E5 − B is therefore a family comparison at a matched, small budget, not a ceiling.
2. **Cross-fitted test scores** average 5 models trained on 80% of the training set. That is the same for E5, E5g, E6 and N3.
3. **Full-batch transductive training:** the results do not cover inductive deployment.
4. **N3 is one rewiring per split,** as for N2. E5 − N3 is a lower bound.
5. **Model 6 is stacking.** Its gain may come from adding a strong second model rather than from the graph. E8s and E8 − E5best are there to show that.
6. **With `f19` the Amazon headroom is small** (B 0.918). A null on Amazon may mean "no room left", so the YelpChi verdicts carry more weight for the graph question.
7. **The best-GNN choice and its setting use all of a split's training labels** before gnnsup's out-of-fold scores are built. That is a second-order selection effect, standard for stacking.

## 7. Decisions (Caelen, 2026-10-31)

| # | Decision | Answer |
|---|---|---|
| G0 | Build GraphSAGE E5/E6 (+ N3) | **yes** |
| G1 | The R-GCN-style normalisation variant (E5g) | **yes** |
| G2 | HGT | **no** |
| G3 | Amend invariant 1 for Model 6 (gnnsup, nested cross-fitting) | **yes** |
| G4 | E7/E7r now | **yes** |
| — | Feature variant | all features, as other studies; `f19`'s effect is an observation; no leak variants unless asked |
| — | YelpChi | validation dataset for this round |
| — | Scheduling | start everything now and keep the queue running |
