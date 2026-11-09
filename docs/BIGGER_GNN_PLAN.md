# Larger GNNs plan: a deeper, wider GraphSAGE and SIGN

**Status: DRAFT, branch `exp/bigger-gnn`, not approved by the group, not pre-registered, not run.** The code, config and tests are in place so the group can review them. Nothing here changes a rung that has already run. Before any run, the group needs to approve the setups (AGENTS.md: "Adding any architecture beyond GraphSAGE and its graph-free control" and changes to tuning settings need the group), and the contrasts in §4 need to go into RESEARCH_PLAN §11 as a pre-registration amendment.

**Why.** The professor asked us to try bigger GNN architectures. The GNNs of the 2026-10-31 round are small: 2 layers, hidden 64 or 128, 23k–82k parameters (§2). GraphSAGE (E5) trails df-analyze on own features (Amazon −0.023, YelpChi −0.071 test PR-AUC) and df-analyze with neighbour means by more (−0.033, −0.104; `docs/GNN_RESULTS.md`). This plan asks whether size closes that gap, in two ways.

## 1. What is built

Both options reuse the GNN round's code path (`gnn_classify`, `src/fraud_detect/graph/supervised.py`): the same inputs (the own block, scaler fit on training nodes), the same full-batch sparse products, the same grid-then-5-seeds protocol, cross-fitting on the saved grouped folds, early stopping on a grouped fifth of each fit's own training nodes, and Model A scoring (`docs/GNN_MODELS_PLAN.md` §2.2). Labels enter only through the training loss (invariant 1).

A rung can now carry its own `train:` block in `gnn_models`. It overrides `gnn_classify` for that rung only. The rungs that have run (E5, E5g, E6, N3) have no such block, and with the new options off the network is the 2026-10-31 one parameter for parameter (`test_default_network_is_the_2026_10_31_network`).

| ID | Feature set | Model | Graph |
|---|---|---|---|
| **E5w** | `pilot_g5w_sage_big` | **Option 1:** E5's layer (root weight + one weight per relation on the 1-hop mean), **3 layers**, **hidden 256 or 512**, **residual** connections from layer 2, **LayerNorm** before each ReLU, and a **jumping-knowledge** head that reads all three layers' outputs | all relations |
| **E6w** | `pilot_g6w_mlp_big` | E5w with every edge removed (its edge-free twin) | none |
| **E9** | `pilot_g9_sign` | **Option 2: SIGN** (Frasca et al., 2020). Fixed inputs x and Â_r^k x for k = 1, 2, 3 per relation (10 inputs on both datasets; Â_r = D⁻¹A_r, the mean). One linear branch of width 256 or 512 per input, concatenated, then 2 hidden layers and a linear head; LayerNorm, ReLU and dropout throughout | all relations, through the fixed inputs |
| **N9** | `pilot_g9n3_sign_rw` | E9 on N2's degree-preserving rewired graph, with E9's chosen setting (as N3 is to E5) | rewired |

**Why these two.** Option 1 changes only the size of the model we already have, so E5w − E5 isolates size. Option 2 follows our own finding that the graph's benefit comes from propagating features along edges (untrained encoders and 1-hop means do as well as trained ones): SIGN propagates up to 3 hops without labels, as E1 does for 1 hop, and puts a large trained network on top. Its graph inputs are computed once, so a large model costs little more than an MLP.

**Not built.** HGT (declined 2026-10-31: per-edge attention does not fit full-batch on U-S-U or R-S-R in 8 GB, and our graphs have one node type), attention models in general (the same memory limit), and a larger DGI encoder (the size sweep found 8 per relation enough and DGI training adds nothing measurable).

## 2. Sizes

Parameters, counted from the model classes (3 relations; Amazon 25 features, YelpChi 32):

| Network | Amazon | YelpChi |
|---|---|---|
| E5 today, hidden 64 / 128 | 22,977 / 78,721 | 24,769 / 82,305 |
| E5w, hidden 256 / 512 | 552,961 / 2,154,497 | 560,129 / 2,168,833 |
| E6w, hidden 256 / 512 | 140,545 / 543,233 | 142,337 / 546,817 |
| E9 (SIGN), hidden 256 / 512 | 794,369 / 3,030,529 | 812,289 / 3,066,369 |

So the larger rungs are 7–39× the size of E5. Each fit's log now records its parameter count (`n_parameters`).

## 3. Settings (proposed; the group decides)

- **Grid per rung and split, on training CV only:** hidden {256, 512} × dropout {0.3, 0.5} × learning rate {1e-3, 5e-3}; Adam, weight decay 5e-4, ≤ 300 epochs, patience 20. Dropout moves up from {0, 0.3} because the networks are larger (about 5,200 labelled Amazon training users, roughly 500 of them fraud).
- **Fixed, not tuned:** 3 layers; SIGN's 3 hops; residual, LayerNorm and jumping knowledge on for E5w and E6w; LayerNorm on for E9.
- **E6w and N9 share their partner's network** through YAML anchors in `configs/default.yaml`, and `config.py` refuses a `setting_from` control whose `train:` block differs from its source's.
- **Timing first.** Before any queue: one fit of E5w and E9 at hidden 512 on Amazon split 0. If a fit takes more than 5 minutes on the CPU, drop hidden 512 from the grid (decided on time only, before any result, as the GNN round did for 128). Cost per rung and split: 8 grid settings + 4 more seeds = 12 cross-fits × 5 folds = 60 fits. The sparse products scale with edges × width, so U-S-U (about 7.1M directed edges) at width 512 is the slow part; SIGN does those products once per graph.
- **Device:** CPU, as before (bit-identical for a given seed). `gnn_classify.device: cuda` works but CUDA's sparse products differ by about 1e-7 between runs.

## 4. Proposed contrasts (to pre-register before the runs)

Same datasets and splits as the GNN round (Amazon 5 splits, all 25 features; YelpChi 3 splits, all 32), same verdict rule: Δ test PR-AUC > 0 on every split, mean > SD, Nadeau–Bengio corrected p < 0.05 over the training CV folds.

**Primary (Holm within the family):**
1. **E5w − E5:** does a larger GraphSAGE beat the current one? (the professor's question)
2. **E9 − E5:** does SIGN beat the current GraphSAGE?
3. **Best of E5w/E9 − E1:** does a larger GNN catch df-analyze with neighbour means, the best simple setup? The best is chosen by mean CV PR-AUC on each split, never test.

**Secondary:**
- **E5w − E6w:** do the edges help the larger network? (E5 − E6 for the larger size)
- **E9 − N9:** does SIGN need the real graph, or only degree and extra inputs?
- **E5w − B, E9 − B:** against df-analyze on own features.

A null on 1 and 2 is itself an answer to the professor: more capacity does not help on these graphs.

## 5. Decisions for the group

| # | Decision | Proposed |
|---|---|---|
| L1 | Build and run E5w and E6w (option 1) | yes |
| L2 | Build and run E9 and N9 (option 2, SIGN) | yes |
| L3 | The grid in §3 (hidden 256/512, dropout 0.3/0.5) | yes |
| L4 | The contrasts in §4, added to RESEARCH_PLAN §11 before any run | yes |
| L5 | Amazon (5 splits) and YelpChi (3 splits), all features, as in the GNN round | yes |
| L6 | A larger E5w at 4 layers, or E5g's symmetric normalisation at the larger size | not now; only if E5w helps |
| L7 | Model 6 with a larger GNN as a gnnsup candidate | not now |

## 6. Tests

`tests/test_gnn_supervised.py` (synthetic data, CPU) checks that:
- with the new options off, the network is the 2026-10-31 one, weight for weight and output for output;
- each option changes only what it says, and the parameter counts match the formula (and the Amazon numbers in §2);
- SIGN's inputs are Â_r^k x per relation (against dense matrix powers, both normalisations), computed once per graph, and only x without edges;
- SIGN cross-fitting is deterministic for a seed and fold k's labels never reach fold k's out-of-fold scores;
- a rung's `train:` block overrides only that rung, and bad values fail;
- all four rungs run end to end on a fake dataset, E6w and N9 share their partner's network, and N9 takes E9's setting.

`tests/test_cli.py` checks the config rules (unknown keys, unknown `train:` settings, a control with a different network); `tests/test_dashboard_registry.py` maps the four new feature sets to setups.
