# GAT and bigger-GNN results (2026-11-09/10)

Results of the round pre-registered in RESEARCH_PLAN §11 (sixth amendment, 2026-11-09). Plans: [`GAT_DEEP_PLAN.md`](GAT_DEEP_PLAN.md) (E10 GAT, N10) and [`BIGGER_GNN_PLAN.md`](BIGGER_GNN_PLAN.md) (E5w bigger GraphSAGE, E6w, E9 SIGN, N9). All numbers come from the results database (`results/runs/`, tag `gat-deep`). Every feature is used: Amazon all 25, YelpChi all 32. All 36 jobs finished between 2026-11-09 03:57 and 2026-11-10 10:41 UTC.

**Machines and devices:**
- Everything new ran on `rtx4060-desktop`.
- E5, E6, N3, B and E1 ran on `rtx4060-laptop` in earlier rounds, on the same splits and test nodes.
- Amazon E5w (splits 0–2) and Amazon E10 split 0 ran on the CPU. Every later job ran on the GPU, from 2026-11-09 21:04 UTC (§11).
- GAT fits and GPU fits are not bit-identical between runs (§11).

**Verdict rule:** Δ test PR-AUC > 0 on every split, mean > SD, and Nadeau–Bengio p < 0.05 over the 3 × 5 training CV folds.
- **Contrasts between two new rungs** (E10 − N10, E9 − N9, E5w − E6w) have full verdicts.
- **Contrasts against E5, E1 or B** still need those rungs' per-fold scores from the laptop. Their p-value is pending, and the table says where the first two conditions already decide the verdict.

## Test PR-AUC per split (splits 0 / 1 / 2, mean ± SD)

| Setup | What | Amazon | YelpChi |
|---|---|---|---|
| **E10** | GAT: 4-head attention per relation, 2 layers | 0.919 / 0.899 / 0.909, **0.909 ± 0.010** | 0.782 / 0.782 / 0.795, **0.786 ± 0.007** |
| **E9** | SIGN: x and 1–3-hop means per relation, a branch per input, 3 layers | 0.919 / 0.907 / 0.923, **0.916 ± 0.008** | 0.764 / 0.754 / 0.770, 0.763 ± 0.008 |
| **E5w** | bigger GraphSAGE: 3 layers, hidden 256/512, residual, LayerNorm, JK | 0.911 / 0.885 / 0.904, 0.900 ± 0.013 | 0.758 / 0.764 / 0.775, 0.766 ± 0.008 |
| E5 | GraphSAGE (2026-10-31) | 0.913 / 0.878 / 0.895, 0.895 ± 0.018 | 0.771 / 0.759 / 0.770, 0.767 ± 0.007 |
| **E6w** | E5w without edges | 0.893 / 0.894 / 0.906, 0.898 ± 0.007 | 0.631 / 0.606 / 0.641, 0.626 ± 0.018 |
| E6 | E5 without edges (2026-10-31) | 0.907 / 0.886 / 0.901, 0.898 ± 0.011 | 0.639 / 0.607 / 0.645, 0.630 ± 0.020 |
| **N10** | E10 on N2's rewired graph (same degrees) | 0.899 / 0.891 / 0.894, 0.894 ± 0.004 | 0.629 / 0.622 / 0.645, 0.632 ± 0.011 |
| **N9** | E9 on the rewired graph | 0.894 / 0.895 / 0.904, 0.898 ± 0.005 | 0.606 / 0.601 / 0.617, 0.608 ± 0.008 |
| N3 | E5 on the rewired graph (2026-10-31, Amazon only) | 0.884 / 0.873 / 0.893, 0.884 ± 0.010 | – |
| B | df-analyze on own features (pilot) | 0.917 / 0.919 / 0.919, 0.918 ± 0.001 | 0.840 / 0.830 / 0.843, 0.838 ± 0.007 |
| E1 | df-analyze + neighbour means (pilot) | 0.926 / 0.938 / 0.923, **0.929 ± 0.008** | 0.871 / 0.884 / 0.857, **0.871 ± 0.013** |

The no-skill baseline (test positive rate) is 0.095 on Amazon and 0.145 on YelpChi. New rungs are in bold.

## Pre-registered primary contrasts (B − A as written)

| Contrast | Dataset | Δ 0 / 1 / 2 | Mean ± SD | p | Expected | Verdict |
|---|---|---|---|---|---|---|
| E5w − E5 | Amazon | −0.003 / +0.007 / +0.009 | +0.004 ± 0.006 | pending | not supported | **not supported** (Δ < 0 on split 0) |
| E5w − E5 | YelpChi | −0.013 / +0.005 / +0.004 | −0.001 ± 0.010 | pending | not supported | **not supported** (Δ < 0 on split 0) |
| E10 − E5 | Amazon | +0.006 / +0.020 / +0.014 | +0.013 ± 0.007 | pending | not supported | positive on every split, mean > SD |
| E10 − E5 | YelpChi | +0.011 / +0.023 / +0.025 | +0.020 ± 0.007 | pending | not supported | positive on every split, mean > SD |
| E10 − N10 | Amazon | +0.020 / +0.008 / +0.016 | +0.014 ± 0.006 | 0.067 | not supported | **not distinguishable from zero** |
| E10 − N10 | YelpChi | +0.153 / +0.160 / +0.150 | +0.154 ± 0.005 | 9.4e-13 | supported | **supported** |
| best of {E5w, E10} − E1 (E10 on every split, by CV) | Amazon | −0.007 / −0.039 / −0.013 | −0.020 ± 0.017 | pending | not supported | **not supported** (worse on every split) |
| best of {E5w, E10} − E1 | YelpChi | −0.089 / −0.102 / −0.063 | −0.085 ± 0.020 | pending | not supported | **not supported** (worse on every split) |

Holm adjustment within each dataset's family waits for the pending p-values.

## Secondary contrasts

| Contrast | Dataset | Δ 0 / 1 / 2 | Mean ± SD | p | Verdict |
|---|---|---|---|---|---|
| E5w − E6w (edges, bigger GraphSAGE) | Amazon | +0.017 / −0.009 / −0.002 | +0.002 ± 0.014 | 0.75 | not distinguishable from zero |
| E5w − E6w | YelpChi | +0.128 / +0.158 / +0.134 | +0.140 ± 0.016 | 1.3e-09 | **supported** |
| E9 − N9 (SIGN, real vs rewired) | Amazon | +0.025 / +0.012 / +0.019 | +0.019 ± 0.006 | 0.57 | not distinguishable from zero (see the N9 note) |
| E9 − N9 | YelpChi | +0.159 / +0.153 / +0.153 | +0.155 ± 0.003 | 5.2e-12 | **supported** |
| E9 − E5 | Amazon | +0.006 / +0.029 / +0.028 | +0.021 ± 0.013 | pending | positive on every split, mean > SD |
| E9 − E5 | YelpChi | −0.007 / −0.004 / +0.000 | −0.004 ± 0.003 | pending | not supported |
| E9 − E1 | Amazon | −0.007 / −0.031 / +0.000 | −0.012 ± 0.016 | pending | not supported |
| E9 − E1 | YelpChi | −0.107 / −0.130 / −0.087 | −0.108 ± 0.021 | pending | not supported |
| E10 − B | Amazon / YelpChi | +0.002 / −0.020 / −0.010 and −0.058 / −0.048 / −0.048 | −0.010 and −0.051 | pending | not supported |
| E9 − B | Amazon / YelpChi | +0.002 / −0.012 / +0.004 and −0.076 / −0.075 / −0.073 | −0.002 and −0.075 | pending | not supported |

**N9 on Amazon:** SIGN on the rewired graph often stopped at epoch 1–2, in 14 of 25 fits on split 1 and 7 of 25 on split 2. This did not happen on split 0, nor anywhere on YelpChi; E9 itself stopped that early in at most 4 of 60 fits per split. It is an early-stopping failure that lowers N9 and makes its CV folds erratic (one fold at 0.515). So the Amazon E9 − N9 Δ overstates the real-structure gain, and its p is not meaningful.

## Reading

- **Bigger does not help.** E5w has 7–25× E5's parameters and scores the same on both datasets (+0.004, −0.001). Model capacity is not what holds GraphSAGE back.
- **Attention helps a little and consistently.** E10 beats E5 on all six splits (+0.013 on Amazon, +0.020 on YelpChi), against the expectation. It is the only new model that beats E5 on both datasets. The p-value is pending.
- **SIGN is the best GNN on Amazon, not on YelpChi.** On Amazon (0.916) it beats E5 on every split and is level with B. On YelpChi it is level with E5. Multi-hop label-free feature averaging works where the graph adds little (Amazon), which is also where df-analyze on neighbour means is strongest.
- **On YelpChi the real graph is essential; on Amazon it adds little.**
  - On YelpChi, every real-graph GNN beats its rewired or edge-free twin by 0.13–0.16 (supported every time). Without the real connections, the networks fall to about 0.61–0.63, the level of no edges at all. YelpChi's labels are per user, and the GNNs lean on the same-user relation (attention below).
  - On Amazon, real − rewired is +0.014 to +0.019 and edges − no edges is +0.002, none distinguishable from zero. This matches the RQ2 finding of 2026-10-27: most of Amazon's graph gain is degree and feature averaging.
- **No end-to-end GNN reaches df-analyze with neighbour means.** The best GNN trails E1 by 0.012 (SIGN, Amazon) and 0.085 (GAT, YelpChi). On YelpChi every GNN is also well below B. Tree models on simple neighbour aggregates remain the strongest setup, as in the 2026-10-31 round. The full-budget ladders (README) move B and E1 by at most a few hundredths, which does not change this order.

## Where GAT puts its attention (`scripts/gat_attention.py`, split 0)

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

- **p-values for every contrast against E5, E1 or B,** and the Holm-adjusted primary family, from `scripts/repeat_summary.py`. This needs the laptop's `pilot_g5_sage`, `pilot_m2_own_nbr` and `pilot_m1_own` outputs for splits 0–2 on both datasets.
- **Whether N9's early stopping on Amazon needs a fix** (for example, a minimum number of epochs) is for the group. Changing it now would be a protocol change after the results.
