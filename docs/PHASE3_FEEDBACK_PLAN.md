# Phase 3: the professor's feedback (2026-10-21)

Plan for Claude Code. Written 2026-10-21. Before starting, read `AGENTS.md` (the invariants always win), `docs/RESEARCH_PLAN.md` §11–12 (every result so far) and `results/README.md` (how to record runs). This plan wins over `RESEARCH_PLAN.md` §12 only on order: the feedback comes first. For anything else that is unclear, stop and ask.

## The feedback

Verbatim, from the professor:

> Find out size of gnn embeddings > Potentially use DEEPTUNE to reduce
> * See if training GNN helps
> * Run on just 4 important features
> * More datasets

| # | Feedback | Reading used in this plan | Tasks |
|---|---|---|---|
| F1 | Size of the GNN embeddings; maybe DeepTune to reduce them | Report every size, from the raw DGI output to the columns df-analyze receives; sweep the size on training CV; compare reduction methods, DeepTune among them | 1, 2 |
| F2 | See if training the GNN helps | (a) Does *training* the encoder beat the same encoder with random, untrained weights? (b) Does a GNN *trained on the labels* (GraphSAGE, Models 3–4) beat df-analyze? | 3, 4 |
| F3 | Run on just the 4 important features | The four #36 features (`own__f13`, `own__f15`, `own__f18`, `own__f19`) alone, with and without the graph; checked against a training-only top-4 ranking | 5 |
| F4 | More datasets | Add one or two graph fraud datasets with tabular node features, chosen by the group from the shortlist in Task 6 | 6 |

The readings are ours, not the professor's; **questions D1–D3 below ask the group to confirm them.** Tasks 1, 3a and the probe in Task 5 need no decision (they are training-CV checks that take minutes), so they can start right away.

## Questions for the group (ask before the tasks that depend on them)

- **D1 (F1, before Task 2b–c):** what did the professor mean by "DeepTune to reduce"? What we found (below) is that DeepTune has no label-free tool that reduces tabular or graph embeddings. Ask the professor, or pick one of Task 2's options.
- **D2 (F2, before Task 4):** reading (a), (b) or both? This plan does (a) first because it is cheap, then (b), which was already on the ladder.
- **D3 (F3, before the Task 5 runs):** "4 important features" = the four #36 features, or the top 4 by a training-only importance ranking? On Amazon these are probably the same four (Task 5 checks). For YelpChi only the ranking applies.
- **D4 (F4, before Task 6):** which datasets, and which split rule for each (Task 6). AGENTS.md requires the group's approval for any dataset beyond YelpChi and Amazon.
- **D5:** should the runs already done (RESEARCH_PLAN §11) be backfilled into the results database? It was created empty on purpose.

## What we found about DeepTune (2026-10-21)

From its repository ([moayadeldin/deeptune](https://github.com/moayadeldin/deeptune), commit `80e6d74`, 2026-10-07) and its documentation:

- It is an AutoML command-line tool for deep learning from the same StFX Medical Imaging Bioinformatics lab as df-analyze (Levman). It covers images (ResNet, ViT, SigLIP, …), text (GPT-2, multilingual BERT), tabular data (GANDALF, TabPFN) and time series (DeepAR). It does **not** cover graphs.
- **Its "embeddings" are extracted from a model it trained or fine-tuned**, then passed to df-analyze as features (its docs show `df-analyze.py --df <embeddings>.parquet`). For tabular data the models are GANDALF and TabPFN, **trained on the label**. Features made that way break invariant 1 unless they are made strictly out-of-fold, which is open decision Model 6.
- **Its autoencoder is for images only** (a convolutional encoder–decoder, marked experimental). Nothing in it reduces a tabular or embedding matrix.
- It pins `torch==2.14.0` (CUDA 13 wheels). The group laptop's driver supports CUDA 12.8, which is why the `gnn` extra stays on torch 2.9. So DeepTune would need its own environment, like df-analyze, called through a subprocess, and it may not see the GPU.
- No license file was found in the repository. Use it as an external tool only; never copy its code into this repository.

So "reduce" is possible in three ways (Task 2): we already reduce with PCA (64 → 8 dimensions per relation, fit on training nodes); a label-free autoencoder bottleneck written here would be the nearest match to DeepTune's idea; and DeepTune itself would mean supervised GANDALF or TabPFN embeddings, which needs D1 and the Model 6 amendment.

## Ground rules

- **Git:** commit directly to `main` or to a branch (AGENTS.md, "Git workflow"); no PR needed. Never commit anything on the never-commit list.
- **Record every run in the results database** right after it finishes (`uv run scripts/results_db.py add ...`, see `results/README.md`), including probes, failed runs and pilots (`--tag pilot`). Put the `run_id` next to the result in RESEARCH_PLAN §11.
- **Invariants:** no label in any feature (invariant 1); every choice (size, method, top-4 features, hyperparameters) made on training data or the saved CV folds, never the test set (invariant 4); scalers, PCA and autoencoders fit on training nodes only (invariant 6). Graph blocks get `(features, edges)` and never the labels; each new block gets the label-blindness test.
- **Reporting:** Amazon is always reported both ways (all features and `_no36`). Every new contrast uses the pre-registered verdict rule of 2026-10-19 (RESEARCH_PLAN §11): test ΔPR-AUC > 0 on all 3 splits, mean Δ > its SD, and Nadeau–Bengio corrected p < 0.05 over the 15 CV folds. Never compare a pilot rung with a full-budget one.
- **Compute:** give an estimate and wait for "go" before anything expected to take over about an hour. Run every rung of a comparison on the same machine.
- **Phase 1–2 outputs** (`m1_own`, the `pilot_*` runs and their repeats) are never rerun or overwritten. New rungs get new feature-set names.

## Tasks

### Task 1: How big are the GNN embeddings? (F1; minutes; no df-analyze)

What is known already (RESEARCH_PLAN §11): DGI produces 64 dimensions per relation (3 × 64 = 192 on Amazon), PCA keeps 8 per relation (24 `gnnssl` columns), and those 8 components hold 93.6% / 99.7% / 98.9% of the variance (U-P-U / U-S-U / U-V-U). df-analyze receives 25 (m1), 103 (m2) and 49 (m5) columns, or 21 / 87 / 45 without #36.

1. **`scripts/embedding_sizes.py --dataset <name>`** reads `gnn_embed*/meta.json`, the raw embeddings (`raw_<relation>.npy`) and the feature blocks, and writes `outputs/<dataset>/gnn_embed/sizes.md` and `sizes.csv`:
   - per block and relation: raw shape, epochs, the PCA's cumulative explained variance at k = 1, 2, 4, 8, 16, 32, 64 (fit on training nodes), and the k needed for 90/95/99%;
   - per feature set: the column count per block and in total, and how many columns each Model A kept after df-analyze's feature selection (from `selected_cols` in its saved results);
   - the effective rank of each raw embedding (how many of the 64 dimensions carry anything).
2. **Size sweep** with the probe (`scripts/probe_blocks.py`, fixed HistGradientBoosting, saved grouped folds, training nodes only): `own + gnnssl` with k ∈ {1, 2, 4, 8, 16, 32} per relation and the raw 64 (no PCA), and DGI output sizes 16 / 32 / 64, both variants. This overlaps RESEARCH_PLAN §12 step 1; do it once, here.
3. **Pre-registered choice:** the smallest k whose mean CV PR-AUC is within one standard error (SD / √5) of the best k. That k, the curve, and whether 8 was already enough go into §11.

**Done when:** `sizes.md` exists, the sweep is in the results database (`--kind probe`), and §11 says how big the embeddings are and how small they can be without losing CV PR-AUC.

### Task 2: Reducing the embeddings (F1; ask D1 before 2b–c)

- **2a (no decision needed):** PCA is the current method; Task 1 measures it.
- **2b, a label-free autoencoder** (the nearest match to "DeepTune to reduce" that keeps invariant 1): a small MLP autoencoder (64 → 32 → k → 32 → 64), one per relation, trained on the training nodes' raw embeddings only, seeded, CPU, in the `gnn` extra. Its bottleneck replaces PCA as an option `gnn_embed.reduce: pca | autoencoder` (default `pca`, so existing blocks and their fingerprints do not change), giving a block such as `gnnssl` built with `reduce: autoencoder` under a new block name. Compare it with PCA at the same k in the probe. Tests: label-blind, deterministic, fit only on training rows.
- **2c, DeepTune itself:** only if D1 says so. Its tabular embeddings are supervised, so it also needs the Model 6 amendment of invariant 1, done strictly out-of-fold on the saved CV folds. Set it up in its own environment like df-analyze (`DEEPTUNE_DIR`, a pinned commit, subprocess calls), and check first that it can use the GPU under the laptop's CUDA 12.8 driver.

**Done when:** the probe compares PCA and the autoencoder at Task 1's k (and DeepTune if chosen). A pilot rung is run on 3 splits only if a method beats PCA on CV by more than one standard error.

### Task 3: Does training the GNN help? (a) trained vs untrained encoder (F2; probe in minutes)

Random, untrained graph encoders are a known strong baseline: much of what a GNN embedding carries can come from propagating features over the graph, not from training. Our own results point the same way: untrained neighbour means (`nbr`) beat DGI without #36.

1. **A `gnnrand` block:** the same encoder as `gnnssl` (2 propagation layers, 64 dimensions, PReLU), the same feature scaler and PCA fit on training nodes, but **no training step**: weights from the seeded initialisation. Add `gnnrand` to `BLOCK_KINDS`, build it in the `gnn_embed` stage (one unit per relation), and allow variants (`gnnrand_no36`). Tests: label-blind, deterministic, the same shapes and names as `gnnssl`.
2. **Probe first:** `own + gnnrand` against `own + gnnssl` and `own + nbr`, both variants; for `gnnrand`, 5 initialisation seeds to show how much random weights vary.
3. **Then the pilot rung on 3 splits** (ask first; about 15 min per run, so 6 runs ≈ 1.5 h): `pilot_m5r_own_gnnrand` and `pilot_m5r_own_gnnrand_no36`, with the same settings as the other pilot rungs. Contrasts: m5 − m5r (does DGI training add anything over random weights?) and m5r − m1 (does untrained propagation alone help?), with the verdict rule.

**Done when:** both contrasts have a verdict in §11 and a results-database entry.

### Task 4: Does training the GNN help? (b) GraphSAGE, Models 3–4 (F2; ask D2)

This is the ladder's existing next build (AGENTS.md, "Models 3 and 4"). It answers whether a GNN trained end to end on the labels beats df-analyze.

- **Models:** Model 3 is GraphSAGE with every edge removed (the graph-free control: the same network on the node's own features only). Model 4 is heterogeneous GraphSAGE over all relations (PyG `HeteroConv` of mean-`SAGEConv` per relation, summed). Plain PyG, full-batch, CPU by default, seeded. Fixed settings, chosen before any run: 2 layers, hidden 64, dropout 0.5, Adam lr 0.01, weight decay 5e-4, at most 200 epochs, patience 20, class-weighted loss (weight from the training class ratio).
- **Labels and graph:** the loss uses training labels only; message passing uses every node's features, test and unlabelled included (invariant 5); Amazon nodes 0–3,304 are never in the loss.
- **Scored like Model A** (invariant 9): for each seed, 5 fold-fits on the saved grouped CV folds give out-of-fold training probabilities (each fold-fit stops early on another training fold, never on the fold it predicts). The final fit on all training nodes runs for the median best epoch. The threshold is max F1 on the out-of-fold predictions. The test score is PR-AUC from probabilities with the group bootstrap. With 5 seeds, the model's probability is the mean over seeds, and the per-seed spread is reported too.
- **Resumable:** one unit per (model, seed, fold) with `ctx.unit_store`; stop-and-resume test.
- **Output:** a new stage `gnn_train` (added to `STAGES` before `export_model` with outputs only for GNN feature sets, so existing bundles skip it), or a script if the group prefers; it writes the same `model_a.json` / `model_a_metrics.csv` layout so `compare_report.py`, `repeat_summary.py` and the results database read it unchanged.
- **Contrasts:** 4 − 3 (message passing with the architecture held fixed), 4 − m1 (a trained GNN against df-analyze), and 4 − m5 (end-to-end against embeddings in df-analyze), both variants, 3 splits.

**Done when:** Models 3–4 have test PR-AUC on 3 splits for both variants, the contrasts have verdicts, and every run is in the results database. Give a time estimate after the first fold-fit; ask if the total is over an hour.

### Task 5: Just the 4 important features (F3; probe in minutes; ask D3 before the runs)

1. **Training-only ranking:** per CV fold, permutation importance of a fixed HistGradientBoosting on the training folds (scored on the held-out training fold), averaged over folds; and the univariate AUROC from the audit. Write `outputs/<dataset>/probe/top_features.csv`. Check whether the top 4 are the four #36 features. Never rank on test data.
2. **A "keep only" variant:** today a `feature_variants` entry lists the own columns to leave out. Allow `{keep: [...]}` as well (its complement is left out), so the existing `no36` entry and every fingerprint stay as they are:
   ```yaml
   feature_variants:
     no36: [own__f13, own__f15, own__f18, own__f19]
     only36: {keep: [own__f13, own__f15, own__f18, own__f19]}
   ```
   The blocks `own_only36`, `nbr_only36` (3 × (4 + 1) = 15 columns) and `gnnssl_only36` (24 columns) follow from the existing variant code: the GNN sees only the four as well. If D3 picks the training-ranked top 4 and they differ, the variant is `top4` with those columns.
3. **Probe:** `own_only36`, `+ nbr_only36` and `+ gnnssl_only36`, next to the existing all-features and `_no36` rows.
4. **Pilot rungs on 3 splits** (ask first; few features, so about 10 min per run, ≈ 1.5 h for 9 runs): `pilot_m1_own_only36`, `pilot_m2_own_nbr_only36`, `pilot_m5_own_gnnssl_only36`. Report m1_only36 against m1 (how much of the 25 features' result do the four carry?) and m2/m5 against m1 within the variant (does the graph help when only the four are known?), with the verdict rule.

The three variants together bracket the question: all 25 features, the 21 without the four, and the four alone.

**Done when:** §11 has the ranking, the probe and the three rungs' verdicts, and the README's results section shows all three variants side by side.

### Task 6: More datasets (F4; ask D4 before downloading anything)

Each new dataset must have a binary fraud or anomaly label, tabular node features, a graph, a license that allows course use, and a size df-analyze can handle (it is meant for up to about 200,000 samples; DeepTune's documentation describes it as best under about 50–100 features). Published facts below are approximate and **must be verified on download**: the `load` stage checks every count against the config, as for Amazon.

| Dataset | Nodes / edges | Features | Positives | Source | Notes |
|---|---|---|---|---|---|
| **Elliptic** (Weber et al., 2019) | 203,769 Bitcoin transactions / 234,355 payment flows | 166: 94 local (including the time step) and 72 pre-aggregated over neighbours | 4,545 illicit and 42,019 licit labelled (≈ 10% of labelled); the rest unlabelled | Kaggle (account needed) | Real financial fraud. The 72 aggregated columns are the data provider's own neighbour features, so Model 1 must use the 94 local ones only, and the 72 make a ready-made Model 2 check. 49 time steps: the standard split is temporal (train on early steps, test on late ones), not our stratified random split. |
| **T-Finance** (Tang et al., ICML 2022) | 39,357 accounts / about 21.2M transactions | 10 | 4.6% | BWGNN authors; also in GADBench | Financial fraud; published results show a large graph benefit, the opposite of Amazon. Many edges: check memory for `nbr` and DGI. |
| **Tolokers** (Platonov et al., ICLR 2023) | 11,758 workers / 519,000 shared-task links | 10 | 21.8% banned | `yandex-research/heterophilous-graphs` (`.npz`) | Small and quick; "banned worker" rather than fraud. |
| Questions, Reddit, Weibo | 10k–49k nodes | 64–400, mostly text embeddings | 3–10% | GADBench, PyGOD | Features come from text, which AGENTS.md excludes without approval. Low priority. |
| DGraph-Fin, T-Social | 3.7M and 5.8M nodes | 17 and 10 | 1.3% and 3% | registration; GADBench | Too large for df-analyze without subsampling. Not recommended. |

**Recommendation:** Elliptic (real financial fraud, a well-known benchmark, and its pre-aggregated columns test Model 2 directly), then T-Finance (a dataset where the graph is reported to matter). Tolokers is a cheap third if time allows.

**For each chosen dataset**, in order:

1. Record the source, license, pinned URL and SHA-256 in RESEARCH_PLAN §2 and the config; the `download` stage verifies them. Never commit the data.
2. Decide its split rule with the group (temporal for Elliptic; grouped stratified otherwise) and its split groups (none, or connected components of a same-entity relation, if there is one).
3. Add a loader next to `care_gnn.py` (`datasets.<name>.format: care_gnn_mat | elliptic_csv | npz`) that writes the same `nodes.parquet` and `graph/` layout, with the expected counts in the config. Tests use tiny synthetic files in `tmp_path`.
4. Run `--to audit` and check the shortcut flags (univariate AUROC ≥ 0.85), as #36 did for Amazon, before any df-analyze run.
5. Then the pilot ladder on it (m1, m2, m5, and m3–m4 once they exist) on 3 splits, with estimates first. Elliptic's labelled set (≈ 46.6k) is larger than YelpChi's training set, so a full df-analyze run is likely 5–8 h or more.

**Done when:** each new dataset passes `load`, `split` and `audit` with verified counts, and its pilot ladder is in §11 and the results database.

### Task 7: Answer the professor

A one-page summary in the README ("Results", a new "Feedback of 2026-10-21" subsection), with numbers from §11 and their `run_id`s:

- how big the embeddings are and the smallest size that loses nothing on CV (F1), and what DeepTune can and cannot do here;
- trained vs untrained encoder, and GraphSAGE against df-analyze (F2);
- the four features alone against all 25 and against the 21 (F3);
- which datasets were added and their first ladder (F4).

## Order and compute

| Step | Tasks | Needs | Compute |
|---|---|---|---|
| 1 | 1, 3a probe, 5.1 and 5.3 probes | nothing | minutes each, CPU |
| 2 | ask D1–D5 with the probe results in hand | the group | – |
| 3 | 3a pilot rungs, 5.4 pilot rungs | "go" | ≈ 3 h on the GPU laptop |
| 4 | 4 (GraphSAGE) | D2 | build first; runs estimated after one fold-fit |
| 5 | 2b–c | D1 | probe minutes; DeepTune setup unknown |
| 6 | 6 | D4 | downloads plus about 1 day of runs per dataset |
| 7 | 7 | the results | – |

Full-budget runs (RESEARCH_PLAN §12 step 2) come after this, once the rungs worth running at full budget are known.
