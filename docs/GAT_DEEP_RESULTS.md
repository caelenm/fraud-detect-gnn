# GAT and bigger-GNN results (2026-11-09/10)

Results of the round pre-registered in RESEARCH_PLAN §11 (sixth amendment, 2026-11-09). Plans: [`GAT_DEEP_PLAN.md`](GAT_DEEP_PLAN.md) (E10, GAT) and [`BIGGER_GNN_PLAN.md`](BIGGER_GNN_PLAN.md) (E5w, bigger GraphSAGE). All numbers come from the results database (`results/runs/`, tag `gat-deep`). Every feature is used: Amazon all 25, YelpChi all 32.

**Status (2026-11-10 05:30 UTC):**
- **Done:** E5w and E10 on splits 0–2 of both datasets.
- **Still running:** the controls N10 and E6w, SIGN (E9) and its control N9.

**What is not final yet:** the third condition of each verdict, Nadeau–Bengio p < 0.05 over the training CV folds. It needs E5's and E1's per-fold scores, and those outputs are on `rtx4060-laptop`. The table shows where the first two conditions (Δ > 0 on every split, mean > SD) already settle a verdict.

**Machines and devices:**
- E5, E1 and B ran on `rtx4060-laptop` (CPU, 2026-10-19 to 2026-11-02).
- E5w and E10 ran on `rtx4060-desktop`, on the same splits and test nodes.
- Amazon E5w (splits 0–2) and Amazon E10 split 0 ran on the CPU. Every later job ran on the GPU, from 2026-11-09 21:04 UTC (§11).

## Test PR-AUC per split

| Setup | What | Amazon 0 / 1 / 2 | Amazon mean ± SD | YelpChi 0 / 1 / 2 | YelpChi mean ± SD |
|---|---|---|---|---|---|
| **E10** | GAT: 4-head attention per relation, 2 layers | 0.919 / 0.899 / 0.909 | **0.909 ± 0.010** | 0.782 / 0.782 / 0.795 | **0.786 ± 0.007** |
| **E5w** | bigger GraphSAGE: 3 layers, hidden 256/512, residual, LayerNorm, JK | 0.911 / 0.885 / 0.904 | 0.900 ± 0.013 | 0.758 / 0.764 / 0.775 | 0.766 ± 0.008 |
| E5 | GraphSAGE (2026-10-31 round) | 0.913 / 0.878 / 0.895 | 0.895 ± 0.018 | 0.771 / 0.759 / 0.770 | 0.767 ± 0.007 |
| B | df-analyze on own features (pilot) | 0.917 / 0.919 / 0.919 | 0.918 ± 0.001 | 0.840 / 0.830 / 0.843 | 0.838 ± 0.007 |
| E1 | df-analyze + neighbour means (pilot) | 0.926 / 0.938 / 0.923 | 0.929 ± 0.008 | 0.871 / 0.884 / 0.857 | 0.871 ± 0.013 |

The no-skill baseline (test positive rate) is 0.095 on Amazon and 0.145 on YelpChi. Every split chose E10 over E5w by mean CV PR-AUC. E10's mean CV PR-AUC was 0.918 on Amazon and 0.779 on YelpChi; E5w's was 0.905 and 0.761.

## Pre-registered primary contrasts (B − A as written)

| Contrast | Dataset | Δ 0 / 1 / 2 | Mean ± SD | Expected | Status |
|---|---|---|---|---|---|
| E5w − E5 | Amazon | −0.003 / +0.007 / +0.009 | +0.004 ± 0.006 | not supported | **not supported** (Δ < 0 on split 0) |
| E5w − E5 | YelpChi | −0.013 / +0.005 / +0.004 | −0.001 ± 0.010 | not supported | **not supported** (Δ < 0 on split 0) |
| E10 − E5 | Amazon | +0.006 / +0.020 / +0.014 | +0.013 ± 0.007 | not supported | positive on every split, mean > SD; **p pending** |
| E10 − E5 | YelpChi | +0.011 / +0.023 / +0.025 | +0.020 ± 0.007 | not supported | positive on every split, mean > SD; **p pending** |
| best of {E5w, E10} − E1 (E10 on every split) | Amazon | −0.007 / −0.039 / −0.013 | −0.020 ± 0.017 | not supported | **not supported** (worse on every split) |
| best of {E5w, E10} − E1 | YelpChi | −0.089 / −0.102 / −0.063 | −0.085 ± 0.020 | not supported | **not supported** (worse on every split) |
| E10 − N10 | both | – | – | Amazon not supported, YelpChi supported | N10 running |

## Reading

- **Bigger does not help.** E5w has 7–25× E5's parameters and scores the same: +0.004 on Amazon and −0.001 on YelpChi. That answers the professor's "bigger GNN": on these graphs, model capacity is not what holds GraphSAGE back.
- **Attention helps a little and consistently.** E10 beats E5 on all six splits, by +0.013 on Amazon and +0.020 on YelpChi. This goes against the pre-registered expectation, and the p-value is still to come. GAT is the best end-to-end GNN of the round.
- **No GNN reaches df-analyze with neighbour means.** E10 still trails E1 by 0.020 on Amazon and 0.085 on YelpChi, and it is below B (df-analyze on own features alone) on both. Tree models on simple neighbour aggregates remain the strongest setup, as in the 2026-10-31 round.
- **Budget:** the GNNs and df-analyze are compared at the pilot budget. The full-budget ladders (README) moved E1 and B by at most a few hundredths, which does not change this order.

## Where GAT puts its attention (`scripts/gat_attention.py`, split 0)

The analysis refits E10's fold-0 model with the run's setting and seed. The refit matches the run within float noise: held-out fold PR-AUC 0.9514 vs 0.9506 on Amazon (CPU run), and 0.7370 vs 0.7330 on YelpChi (GPU run, refit on the CPU). Both are within the 0.005 tolerance. GAT fits are not bit-identical between runs (§11).

"Focus" is attention entropy over log(degree): 1 means equal weight on every neighbour, which is what E5's mean does.

| | Amazon | YelpChi |
|---|---|---|
| Focus, layer 1 | 0.92 (U-S-U, ~600 neighbours) to 0.97 | 0.93 (R-U-R, ~6 neighbours) to 0.97 |
| Focus, layer 2 | 0.94–0.99 | R-T-R and R-S-R 1.00 (pure averaging); R-U-R 0.90 |
| Relation term vs own-features term | layer 1 U-S-U 2.9×; U-P-U 0.8×; U-V-U 0.5× | layer 2 R-U-R 2.1×; R-T-R and R-S-R 0.5–0.7× |
| Attention on fraud neighbours vs their equal share | above, for every target group (layer 1 U-S-U: fraud targets 0.26 vs 0.16, benign 0.074 vs 0.039) | about equal (within ±0.02) |

On Amazon, GAT learned to up-weight fraud neighbours, mostly through U-S-U. On YelpChi it mostly reduced to averaging over R-T-R and R-S-R and leaned on the same-user relation, yet still scored above E5. So the attention weights alone do not explain its gain there.

Training labels were used only to describe the model, never as an input. Attention weights describe where the model puts weight, not why it decides. Per-node attention statistics stay local (row-level).

## Not done yet

- **p-values for E10 − E5 and E5w − E5,** and the Holm-adjusted family, from `scripts/repeat_summary.py`. This needs the laptop's `pilot_g5_sage`, `pilot_m2_own_nbr` and `pilot_m1_own` outputs for splits 0–2 on both datasets.
- **The controls and SIGN:** N10 (E10 − N10, the real-structure contrast), E6w, E9 and N9. They are running; this page will be extended when they finish.
