# GAT and bigger-GNN results (2026-11-09/11)

Results of the round pre-registered in RESEARCH_PLAN §11 (sixth amendment, 2026-11-09; extended to Amazon splits 3–4 and, as the seventh amendment, to every GNN classifier without `f19`, both on 2026-11-10). Plans: [`GAT_DEEP_PLAN.md`](GAT_DEEP_PLAN.md) (E10 GAT, N10) and [`BIGGER_GNN_PLAN.md`](BIGGER_GNN_PLAN.md) (E5w bigger GraphSAGE, E6w, E9 SIGN, N9). All numbers come from the results database (`results/runs/`, tags `gat-deep` and `gnn-no19`).

- **All features** (Amazon all 25, YelpChi all 32): 48 jobs, Amazon on 5 splits and YelpChi on 3, from 2026-11-09 03:57 to 2026-11-10 20:01 UTC.
- **Without `f19`** (Amazon, 24 features): every GNN classifier, 30 jobs on splits 0–2, from 2026-11-10 20:01 to 2026-11-11 01:10 UTC.

**Machines and devices:**
- Everything new ran on `rtx4060-desktop`.
- E5, E5g, E6, N3, B and E1 (all features), and B and E1 without `f19`, ran on `rtx4060-laptop` in earlier rounds, on the same splits and test nodes.
- Amazon E5w (splits 0–2) and Amazon E10 split 0 ran on the CPU. Every later job ran on the GPU, from 2026-11-09 21:04 UTC (§11).
- GAT fits and GPU fits are not bit-identical between runs (§11).

**Verdict rule:** Δ test PR-AUC > 0 on every split, mean > SD, and Nadeau–Bengio p < 0.05 over the training CV folds of every split.
- **Contrasts between two rungs run here** have full verdicts.
- **Contrasts against a laptop rung** (E5, E1 or B with all features; B or E1 without `f19`) still need that rung's per-fold scores. Their p-value is pending, and the tables say where the first two conditions already decide the verdict.

## Test PR-AUC per split, all features (mean ± SD)

| Setup | What | Amazon, splits 0 / 1 / 2 / 3 / 4 | YelpChi, splits 0 / 1 / 2 |
|---|---|---|---|
| **E10** | GAT: 4-head attention per relation, 2 layers | 0.919 / 0.899 / 0.909 / 0.925 / 0.908, **0.912 ± 0.010** | 0.782 / 0.782 / 0.795, **0.786 ± 0.007** |
| **E9** | SIGN: x and 1–3-hop means per relation, a branch per input, 3 layers | 0.919 / 0.907 / 0.923 / 0.922 / 0.904, **0.915 ± 0.009** | 0.764 / 0.754 / 0.770, 0.763 ± 0.008 |
| **E5w** | bigger GraphSAGE: 3 layers, hidden 256/512, residual, LayerNorm, JK | 0.911 / 0.885 / 0.904 / 0.907 / 0.884, 0.898 ± 0.013 | 0.758 / 0.764 / 0.775, 0.766 ± 0.008 |
| E5 | GraphSAGE (2026-10-31) | 0.913 / 0.878 / 0.895 / 0.911 / 0.902, 0.900 ± 0.014 | 0.771 / 0.759 / 0.770, 0.767 ± 0.007 |
| E5g | GraphSAGE, R-GCN-style normalisation (2026-10-31) | 0.917 / 0.894 / 0.910 / 0.923 / 0.912, 0.911 ± 0.011 | (GNN_RESULTS.md) |
| **E6w** | E5w without edges | 0.893 / 0.894 / 0.906 / 0.914 / 0.889, 0.899 ± 0.010 | 0.631 / 0.606 / 0.641, 0.626 ± 0.018 |
| E6 | E5 without edges (2026-10-31) | 0.907 / 0.886 / 0.901 / 0.914 / 0.884, 0.898 ± 0.013 | 0.639 / 0.607 / 0.645, 0.630 ± 0.020 |
| **N10** | E10 on N2's rewired graph (same degrees) | 0.899 / 0.891 / 0.894 / 0.916 / 0.895, 0.899 ± 0.010 | 0.629 / 0.622 / 0.645, 0.632 ± 0.011 |
| **N9** | E9 on the rewired graph | 0.894 / 0.895 / 0.904 / 0.921 / 0.904, 0.904 ± 0.011 | 0.606 / 0.601 / 0.617, 0.608 ± 0.008 |
| N3 | E5 on the rewired graph (2026-10-31, Amazon only) | 0.884 / 0.873 / 0.893 / 0.907 / 0.878, 0.887 ± 0.014 | – |
| B | df-analyze on own features (pilot) | 0.917 / 0.919 / 0.919 / 0.935 / 0.921, 0.922 ± 0.007 | 0.840 / 0.830 / 0.843, 0.838 ± 0.007 |
| E1 | df-analyze + neighbour means (pilot) | 0.926 / 0.938 / 0.923 / 0.942 / 0.936, **0.933 ± 0.008** | 0.871 / 0.884 / 0.857, **0.871 ± 0.013** |

The no-skill baseline (test positive rate) is 0.095 on Amazon and 0.145 on YelpChi. New rungs are in bold.

## Pre-registered primary contrasts, all features (B − A as written)

| Contrast | Dataset | Δ per split | Mean ± SD | p | Expected | Verdict |
|---|---|---|---|---|---|---|
| E5w − E5 | Amazon (5) | −0.003 / +0.007 / +0.009 / −0.003 / −0.018 | −0.002 ± 0.011 | pending | not supported | **not supported** (Δ < 0 on 3 splits) |
| E5w − E5 | YelpChi (3) | −0.013 / +0.005 / +0.004 | −0.001 ± 0.010 | pending | not supported | **not supported** (Δ < 0 on split 0) |
| E10 − E5 | Amazon (5) | +0.006 / +0.020 / +0.014 / +0.014 / +0.005 | +0.012 ± 0.006 | pending | not supported | positive on every split, mean > SD |
| E10 − E5 | YelpChi (3) | +0.011 / +0.023 / +0.025 | +0.020 ± 0.007 | pending | not supported | positive on every split, mean > SD |
| E10 − N10 | Amazon (5) | +0.020 / +0.008 / +0.016 / +0.008 / +0.013 | +0.013 ± 0.005 | 0.026 | not supported | **supported** (3 splits: p 0.067, not distinguishable) |
| E10 − N10 | YelpChi (3) | +0.153 / +0.160 / +0.150 | +0.154 ± 0.005 | 9.4e-13 | supported | **supported** |
| best of {E5w, E10} − E1 (E10 on every split, by CV) | Amazon (5) | −0.007 / −0.039 / −0.013 / −0.018 / −0.028 | −0.021 ± 0.013 | pending | not supported | **not supported** (worse on every split) |
| best of {E5w, E10} − E1 | YelpChi (3) | −0.089 / −0.102 / −0.063 | −0.085 ± 0.020 | pending | not supported | **not supported** (worse on every split) |

Holm adjustment within each dataset's family waits for the pending p-values. The Amazon 5-split verdicts replace the 3-split ones (§11 extension); the 3-split tables stay in the database (tag `splits3`).

## Secondary contrasts, all features

| Contrast | Dataset | Δ per split | Mean ± SD | p | Verdict |
|---|---|---|---|---|---|
| E5w − E6w (edges, bigger GraphSAGE) | Amazon (5) | +0.017 / −0.009 / −0.002 / −0.007 / −0.005 | −0.001 ± 0.011 | 0.97 | worse (no gain from edges) |
| E5w − E6w | YelpChi (3) | +0.128 / +0.158 / +0.134 | +0.140 ± 0.016 | 1.3e-09 | **supported** |
| E9 − N9 (SIGN, real vs rewired) | Amazon (5) | +0.025 / +0.012 / +0.019 / +0.001 / −0.000 | +0.011 ± 0.011 | 0.45 | not distinguishable from zero (see the N9 note) |
| E9 − N9 | YelpChi (3) | +0.159 / +0.153 / +0.153 | +0.155 ± 0.003 | 5.2e-12 | **supported** |
| E9 − E5 | Amazon (5) | +0.006 / +0.029 / +0.028 / +0.012 / +0.002 | +0.015 ± 0.013 | pending | positive on every split, mean > SD |
| E9 − E5 | YelpChi (3) | −0.007 / −0.004 / +0.000 | −0.004 ± 0.003 | pending | not supported |
| E9 − E1 | Amazon (5) | −0.007 / −0.031 / +0.000 / −0.020 / −0.032 | −0.018 ± 0.014 | pending | not supported |
| E9 − E1 | YelpChi (3) | −0.107 / −0.130 / −0.087 | −0.108 ± 0.021 | pending | not supported |
| E10 − B | Amazon (5) / YelpChi (3) | +0.002 / −0.020 / −0.010 / −0.011 / −0.014 and −0.058 / −0.048 / −0.048 | −0.011 and −0.051 | pending | not supported |
| E9 − B | Amazon (5) / YelpChi (3) | +0.002 / −0.012 / +0.004 / −0.013 / −0.017 and −0.076 / −0.075 / −0.073 | −0.007 and −0.075 | pending | not supported |

**N9 on Amazon (all features):** SIGN on the rewired graph often stopped at epoch 1–2: in 14, 7, 16 and 17 of 25 fits on splits 1–4. This did not happen on split 0, nor anywhere on YelpChi, nor (much) without `f19` (below); E9 itself stopped that early in at most 4 of 60 fits per split. It is an early-stopping failure that makes N9's CV folds erratic (CV PR-AUC 0.83–0.84 on splits 3–4 against test 0.90–0.92). So the Amazon E9 − N9 p is not meaningful. Whether to fix it is for the group.

## Without `f19` (Amazon, splits 0–2; seventh amendment)

`f19` is the column RQ1 found to leak the label (§11, 2026-10-26). Here it is left out of every node's input, and so out of every neighbour message, hop mean and attention score. B and E1 without `f19` are the df-analyze rungs from 2026-10-27 (`rtx4060-laptop`).

| Setup | Splits 0 / 1 / 2 | Mean | With all features (splits 0–2) |
|---|---|---|---|
| **E10** GAT | 0.685 / 0.691 / 0.701 | **0.692** | 0.909 |
| **E5g** GraphSAGE, R-GCN-style | 0.701 / 0.686 / 0.689 | **0.692** | 0.907 |
| **E9** SIGN | 0.658 / 0.690 / 0.701 | 0.683 | 0.916 |
| **E5w** bigger GraphSAGE | 0.668 / 0.683 / 0.699 | 0.683 | 0.900 |
| **E5** GraphSAGE | 0.669 / 0.678 / 0.683 | 0.677 | 0.895 |
| **N9** SIGN, rewired graph | 0.537 / 0.560 / 0.555 | 0.551 | 0.898 |
| **N10** GAT, rewired graph | 0.544 / 0.554 / 0.536 | 0.545 | 0.894 |
| **E6** E5 without edges | 0.508 / 0.522 / 0.550 | 0.527 | 0.898 |
| **N3** GraphSAGE, rewired graph | 0.525 / 0.508 / 0.535 | 0.523 | 0.884 |
| **E6w** E5w without edges | 0.498 / 0.518 / 0.545 | 0.520 | 0.898 |
| B df-analyze, own features | 0.516 / 0.535 / 0.565 | 0.539 | 0.918 |
| E1 df-analyze + neighbour means | 0.703 / 0.699 / 0.706 | **0.702** | 0.929 |

**Primary contrasts (Holm over these 4):**

| Contrast | Δ 0 / 1 / 2 | Mean ± SD | p | Expected | Verdict |
|---|---|---|---|---|---|
| E5 − E6 (graph vs no edges) | +0.161 / +0.156 / +0.133 | +0.150 ± 0.015 | 1.3e-06 | supported | **supported** |
| E5 − N3 (real vs rewired graph) | +0.143 / +0.170 / +0.148 | +0.154 ± 0.015 | 5.1e-09 | supported | **supported** |
| best GNN − E1 (E10 on every split, by CV) | −0.019 / −0.008 / −0.005 | −0.011 ± 0.007 | pending | not supported | **not supported** (worse on every split) |
| best GNN − B | +0.168 / +0.155 / +0.135 | +0.153 ± 0.017 | pending | supported | positive on every split, mean > SD |

Holm: the two p-values computed here (1.3e-06, 5.1e-09) stay below 0.05 after adjustment for 4 tests.

**Secondary:**

| Contrast | Mean ± SD | p | Verdict |
|---|---|---|---|
| E10 − N10 (GAT, real vs rewired) | +0.148 ± 0.015 | 8.9e-05 | **supported** |
| E9 − N9 (SIGN, real vs rewired) | +0.133 ± 0.012 | 6.7e-04 | **supported** |
| E5w − E6w (bigger GraphSAGE, edges vs none) | +0.163 ± 0.008 | 1.3e-05 | **supported** |
| E10 − E5 (attention vs mean) | +0.015 ± 0.003 | 0.50 | not distinguishable (positive on every split) |
| E9 − E5 | +0.007 ± 0.015 | 0.99 | not distinguishable |
| E5w − E5 (bigger vs E5) | +0.007 ± 0.009 | 0.94 | not distinguishable |
| E5g − E5 (normalisation) | +0.015 ± 0.015 | 0.23 | not distinguishable |

Without `f19`, N9 stopped at epoch 1–2 in only 5, 1 and 3 of 25 fits, so this E9 − N9 is not affected the way the all-feature one is.

**All features − without `f19`** (descriptive, splits 0–2): the real-graph GNNs lose 0.22–0.23 without `f19`; the edge-free and rewired ones lose 0.35–0.38; B loses 0.38 and E1 0.23.

## Reading

- **Bigger does not help.** E5w has 7–25× E5's parameters and scores the same on both datasets and without `f19` (−0.002, −0.001, +0.007). Model capacity is not what holds GraphSAGE back.
- **Attention helps a little and consistently.** E10 beats E5 on all 11 splits it was compared on (+0.012 Amazon, +0.020 YelpChi, +0.015 without `f19`). The contrasts computed here cannot separate it from zero (without `f19`, p 0.50); the all-feature p-values are pending.
- **`f19` hides the graph's value on Amazon.**
  - With every feature, Amazon's real graph adds little to a GNN: edges − no edges about 0, and real − rewired +0.011 to +0.013 (supported only for GAT, over 5 splits).
  - Without `f19`, every GNN gains 0.13–0.16 from the real graph, all supported, and a rewired graph with the same degrees adds almost nothing (N3, N9, N10 are within about 0.03 of the edge-free models). So without the leak, the GNNs' graph gain is real structure, not degree.
  - This is a cleaner answer to RQ2 than df-analyze gave (degree +0.112 and structure +0.076 of E1's gain, 2026-10-27).
- **On YelpChi the real graph is essential.** Every real-graph GNN beats its rewired or edge-free twin by 0.13–0.16 (supported every time). YelpChi's labels are per user, and the GNNs lean on the same-user relation (attention below).
- **No end-to-end GNN beats df-analyze with neighbour means.**
  - With every feature, the best GNN trails E1 by 0.018–0.021 (Amazon) and 0.085 (YelpChi); on YelpChi every GNN is also below B.
  - Without `f19` the gap nearly closes: GAT and E5g reach 0.692 against E1's 0.702, and GAT trails by 0.005–0.019 on each split. Every real-graph GNN beats B without `f19` by 0.14–0.15.
  - The full-budget ladders (README) move B and E1 by at most a few hundredths, which does not change this order.

## Where GAT puts its attention (`scripts/gat_attention.py`, split 0, all features)

The analysis refits E10's fold-0 model with the run's setting and seed. The refit matches the run within float noise: held-out fold PR-AUC 0.9514 vs 0.9506 on Amazon (CPU run), and 0.7370 vs 0.7330 on YelpChi (GPU run, refit on the CPU). Both are within the 0.005 tolerance.

"Focus" is attention entropy over log(degree): 1 means equal weight on every neighbour, which is what E5's mean does.

| | Amazon | YelpChi |
|---|---|---|
| Focus, layer 1 | 0.92 (U-S-U, ~600 neighbours) to 0.97 | 0.93 (R-U-R, ~6 neighbours) to 0.97 |
| Focus, layer 2 | 0.94–0.99 | R-T-R and R-S-R 1.00 (pure averaging); R-U-R 0.90 |
| Relation term vs own-features term | layer 1 U-S-U 2.9×; U-P-U 0.8×; U-V-U 0.5× | layer 2 R-U-R 2.1×; R-T-R and R-S-R 0.5–0.7× |
| Attention on fraud neighbours vs their equal share | above, for every target group (layer 1 U-S-U: fraud targets 0.26 vs 0.16, benign 0.074 vs 0.039) | about equal (within ±0.02) |

On Amazon, GAT learned to up-weight fraud neighbours, mostly through U-S-U. On YelpChi it mostly reduced to averaging over R-T-R and R-S-R and leaned on the same-user relation, the relation whose loss to rewiring costs 0.15. So the attention weights alone do not explain its gain over E5 there.

Training labels were used only to describe the model, never as an input. Attention weights describe where the model puts weight, not why it decides. Per-node attention statistics stay local (row-level).

## Not done yet

- **p-values for every contrast against a laptop rung** (E5, E1, B with all features; B and E1 without `f19`), and the Holm-adjusted families. They need the laptop's `pilot_g5_sage`, `pilot_m2_own_nbr`, `pilot_m1_own` (splits 0–4 Amazon, 0–2 YelpChi), `pilot_m1_own_no19` and `pilot_m2_own_nbr_no19` (splits 0–2) outputs.
- **Whether N9's early stopping on Amazon (all features) needs a fix**, for example a minimum number of epochs, is for the group. Changing it now would be a protocol change after the results.
- **Without `f19` on splits 3–4 and on YelpChi** were not run (YelpChi has no `f19`).
