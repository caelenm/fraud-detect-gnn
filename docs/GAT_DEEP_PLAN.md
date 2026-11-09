# GAT plan: attention as an extra GNN rung (E10, N10)

**Status:** code and tests done on branch `exp/gat-deep`, built on `exp/bigger-gnn` (2026-11-08). Timing done (§4). **No full run has started:** that waits for the user's approval. The user approved adding a GAT rung on 2026-11-08, an exception to AGENTS.md's "ask before adding any architecture beyond GraphSAGE". Every other AGENTS.md rule still applies.

**Where this comes from.** The professor's feedback after the 2026-11-03 presentation:
- try adding GAT;
- deep learner;
- "convolutional GNN, what is it";
- bigger GNN.

**Who does what:**
- **This plan:** GAT and the explanation for the slide (§1).
- **[`BIGGER_GNN_PLAN.md`](BIGGER_GNN_PLAN.md) (draft on `exp/bigger-gnn`):** the bigger GNN, a deeper and wider GraphSAGE (E5w) and SIGN (E9). Its per-rung `train:` block, `model:` key, residual connections and LayerNorm are what GAT uses here, so there is one mechanism rather than two.
- **Teammates:** "find other GNNs" and SHAP.

**These rungs are additions.** E10 and N10 are new feature sets with their own output folders, next to E5, E5g, E6 and N3 and the draft E5w, E6w, E9 and N9. Neither is a `gnn_oof` candidate, so E8 (Model 6) is unchanged. Adding one needs the user's OK.

The existing rungs have no `train:` block and are unchanged. A check on synthetic data shows E5-style fits bit-identical to `main`'s: 12 configurations covering both normalisations, with and without edges, and 1 to 3 layers.

## 1. What a "convolutional GNN" is (slide note)

A convolutional GNN updates each node from its neighbours with weights it shares across the whole graph, the way an image convolution reuses one filter at every pixel. There are two lineages:
- **Spectral:** filters defined through the graph Laplacian's eigenvectors.
- **Spatial (message passing):** each node combines its own vector with a weighted sum of its neighbours' vectors.

The **GCN** (Kipf & Welling) bridges them. It is a cheap spectral filter that works as a spatial layer, with fixed neighbour weights `D^-½(A+I)D^-½`.
- **GraphSAGE** keeps the node's own vector separate and averages its neighbours.
- **R-GCN** gives each relation its own weights.
- **GAT** learns the neighbour weights: a softmax-normalised attention score per edge replaces the fixed ones.

**In our code:**
- **GCN:** inside DGI (E2).
- **GraphSAGE:** E5.
- **R-GCN-style:** E5g.
- **GAT:** E10.

## 2. What was added

The changes are in `src/fraud_detect/graph/supervised.py`. The existing `gnn_classify` stage runs them, with the same grid, seeds, saved folds, early stopping and scoring as E5.

**`model: gat` (layer `RelGAT`).**
- `h' = W_root h + b + Σ_r GAT_r(h)`. Each relation has `heads` attention heads, each `hidden / heads` wide, concatenated.
- The attention score is `e_ij = LeakyReLU_0.2(a_dstᵀ W h_i + a_srcᵀ W h_j)`, softmax-normalised over node i's incoming edges. Isolated nodes get only the root term.
- Attention dropout uses the setting's dropout.
- Each relation's term equals PyG's `GATConv(heads, concat=True, add_self_loops=False, bias=False)`, which a test checks.
- `heads` is a new `train:` key. GAT shares `GNNClassifier` with the sage layers, so `residual`, `layer_norm` and `jk` from `exp/bigger-gnn` work for it too. E10 leaves them off: it is E5's size with attention, so E10 − E5 isolates attention.

**Memory and speed.**
- R-S-R has about 6.8M directed edges and U-S-U about 7.1M, so no edges × hidden tensor is built. The weighted sum is a custom autograd function made of per-head sparse products:
  - forward `A_α V` and `grad_V = A_αᵀ G`, as CSR × dense products;
  - `grad_α`, a sampled dense product at the edges (`torch.sparse.sampled_addmm`).
- This replaces the chunked `index_add_` the brief proposed. That version was correct, but its single-threaded scatter made an epoch on YelpChi take 15.6 s. The sparse version takes 3.4 s, gives the same forward values (max difference 0.0 on a 6.8M-edge test), and is deterministic on the CPU.
- Gradients are checked with `gradcheck` and against the materialised messages.
- For GAT the graph input holds directed edge lists, and the `Â` matrices are not built.

**Rungs** (`configs/default.yaml`; dashboard IDs in `dashboard/registry.py`).

| Feature set | ID | What |
|---|---|---|
| `pilot_g10_gat` | E10 | GAT: 2 layers, 4 heads per relation, hidden 64/128, real graph |
| `pilot_g10n3_gat_rw` | N10 | E10 on N2's rewired graph, with E10's chosen setting (null-graph control, as N3 is to E5) |

**Leakage.**
- Invariant 1 is unchanged: labels enter only through each fit's training loss and inner validation.
- E10 and N10 reuse `cross_fit` and the `gnn_classify` scoring, so invariants 2–5 hold as for E5.
- All 526 tests pass, including the leakage, label-blindness and fold tests.

## 3. Contrasts to pre-register (into RESEARCH_PLAN §11 before any run)

These use the usual rule: Δ test PR-AUC > 0 on every split, mean > SD, and Nadeau–Bengio p < 0.05 over the saved CV folds. They run on Amazon (5 splits, all 25 features) and YelpChi (3 splits, all 32 features), as in the 2026-10-31 round.

1. **E10 − E5:** does attention beat mean aggregation?
2. **E10 − N10:** does GAT use real structure, against the degree-preserving rewired graph?
3. **Best of {E10, E5w} − E1:** does a better end-to-end GNN close the gap to df-analyze with neighbour means? "Best" is chosen by mean shared-CV PR-AUC, never by test. This contrast only applies if the group approves E5w.

The brief's "deep − E5" is BIGGER_GNN_PLAN's E5w − E5.

## 4. Timing and cost (split 0, this machine: `rtx4060-desktop`)

The machine is a Ryzen 7 7700 with 8 CPU threads in WSL2, 15 GB RAM and an RTX 4060 (8 GB). The fits use hidden 128, dropout 0.3 and lr 0.005, on one outer fold's training rows. Peak RAM is the process's maximum RSS.

| | YelpChi E5 | YelpChi E10 | Amazon E5 | Amazon E10 |
|---|---|---|---|---|
| CPU, seconds per epoch | 0.39 | 3.4 | 0.17 | 3.4 |
| CPU, one fit to early stopping | 50 s (129 epochs) | 622 s (182 epochs) | 12 s (75 epochs) | 423 s (125 epochs) |
| CPU peak RAM | 1.6 GB | 3.4 GB | 1.2 GB | 3.4 GB |
| CUDA, seconds per epoch | – | 1.1 | – | 0.7 |
| GPU peak memory | – | 1.9 GB | – | 1.9 GB |

As a calibration, this method estimates E5 on YelpChi at 50 minutes per split. It took 39 minutes on the laptop.

**Cost.**
- **E10:** 8 grid settings × 5 folds, plus 4 more seeds × 5 folds, is 60 fits per split.
- **N10:** 5 seeds × 5 folds is 25 fits, with no grid.
- **Lower bound:** each fit stops as early as the measured one. Settings at lr 0.001 will likely train for longer; at the 300-epoch cap a fit costs up to 1.6× (YelpChi) or 2.4× (Amazon) the measured one.

| Hours, CPU | per split | Amazon (5 splits) | YelpChi (3 splits) |
|---|---|---|---|
| E10 | Amazon 7–17, YelpChi 10–17 | 35–85 | 31–51 |
| N10 | Amazon 3–7, YelpChi 4–7 | 15–35 | 13–21 |
| **Total** | | **50–120** | **44–72** |

That is **about 4 to 8 days on the CPU**, run one at a time: each fit already uses all 8 threads. On CUDA, at 3× (YelpChi) and 5× (Amazon) faster per epoch, it is about **25–45 hours**.

**Device.** AGENTS.md's 2026-10-19 decision puts GNNs on the CPU, because CUDA's sparse products differ by about 1e-7 between runs. CUDA runs fit easily in 8 GB, but they would not be bit-reproducible, and the user would need to approve them as an exception.

**Ways to cut the cost, for the user to decide** (each changes the design or the tuning settings, so none is applied here):
- **(a)** Run split 0 of each dataset first, as a pilot: about 17–34 h on the CPU, or 5–8 h on CUDA.
- **(b)** Fix lr at 0.005, the faster one, for GAT. The grid halves, and E10 drops to 40 fits per split (−33%).
- **(c)** Allow CUDA for E10 and N10 only, and mark them as not bit-reproducible.
- **(d)** Use 3 Amazon splits instead of 5 (−40% on Amazon). The contrasts would then cover fewer splits than E5's.

## 5. "Deep learner": another reading for the group

"Deep learner" may mean a deep **tabular** model rather than a deeper GNN. df-analyze already has two:
- **GANDALF**, which is already in our full-budget runs (6 classifiers);
- **an MLP**, left out because df-analyze runs it on the CPU only and it adds about 4 h per run.

Adding the MLP to the classifier list, or reporting GANDALF's tuned score separately, would answer that reading. This plan does not run it: the group should decide which reading the professor meant.
