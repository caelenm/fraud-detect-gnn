# Phase 2: df-analyze vs df-analyze + GNN embeddings, with a comparison page

Plan for Claude Code. Written 2026-10-18. Before starting, read `AGENTS.md`, `docs/RESEARCH_PLAN.md` and `docs/PHASE1_AMAZON_PLAN.md` **on the `feat/review-fraud-pivot` branch**. The invariants in `AGENTS.md` always win. This plan wins over `RESEARCH_PLAN.md` only on the points under "Decisions". For anything else that is unclear, stop and ask.

> **Update 2026-10-19 (user decisions, after the plan was implemented through Task 4):**
> 1. **Issue #36: report both.** Every rung is run twice on the same frozen test set: with all 25 features, and without `own__f13`, `own__f15`, `own__f18`, `own__f19` (feature sets `*_no36`). The four are also left out of the neighbour-aggregate and GNN inputs (`nbr_no36`, `gnnssl_no36`). This replaces Decision 7's "a `_no36` variant comes after the group decides" for the pilot: the pilot is six runs.
> 2. **GNN embeddings on the CPU** (`gnn_embed.device: cpu`) instead of `auto`: CUDA's sparse products are not bit-reproducible (~1e-7 between runs); the CPU is and takes about 35 s on Amazon.
> 3. **torch is pinned to 2.9** in the `gnn` extra: the newest PyPI torch is built for CUDA 13, which the group laptop's driver (CUDA 12.8) cannot run.

## Goal

On Amazon, add two rungs of the model ladder next to the existing Model 1 and compare them on the same split, folds and metric:

| Feature set | Blocks | What it adds | Status |
|---|---|---|---|
| `m1_own` | `own` | nothing (Model A, test PR-AUC 0.919) | done, in the bundle |
| `m2_own_nbr` | `own`, `nbr` | hand-made, label-free neighbour means per relation | **build** (cheap baseline and demo fallback) |
| `m5_own_gnnssl` | `own`, `gnnssl` | self-supervised DGI embeddings per relation, PCA-compressed | **build** (the main experiment) |

Then a self-contained HTML **comparison page** that shows two feature sets' Model A side by side, test node by test node.

- **Primary contrast:** `m5_own_gnnssl` vs `m1_own` (does the GNN help df-analyze?).
- **Secondary contrasts:** `m2` vs `m1` (does any graph information help?) and `m5` vs `m2` (learned vs hand-made).
- **First demo: Tuesday 2026-10-20.** For the demo, all three rungs are run at the same reduced "pilot" budget (Task 5). Full-budget runs come after the demo (Phase 3, at the end).

## Decisions (2026-10-18, from the user)

1. **Branch.** Create `feat/gnn-embeddings` from `feat/review-fraud-pivot`. Commit locally in small commits. **Do not push and do not open a PR** unless the user says so in chat. This overrides phase 1's push-and-PR end state.
2. **Labels.** Embeddings are **self-supervised only** (Model 5). Invariant 1 is unchanged: no label reaches any feature. Model 6 (supervised out-of-fold embeddings) is out of scope.
3. **One block per kind, covering all relations.** Use the blocks `nbr` and `gnnssl`, not `nbr_upu`, `nbr_usu`, …. Columns are named `nbr__upu_mean_f00` or `gnnssl__upu_pc01`. This keeps the feature-set names identical for YelpChi (`rur`/`rtr`/`rsr`). Check that `df_analyze.df_analyze_names` maps these names without collisions.
4. **`nbr` block (Model 2).** For each relation in the dataset's config (not `homo`):
   - the 1-hop mean of the 25 `own` features;
   - `log1p(degree)`.

   On Amazon that is 3 × 26 = 78 columns, 103 with `own`. The 1-hop max and 2-hop means are later ablations: adding them now would push df-analyze past its recommended feature count.
5. **`gnnssl` block (Model 5).**
   - **Model:** Deep Graph Infomax (PyG `DeepGraphInfomax`), one model per relation.
   - **Encoder:** 2 layers of GCN normalisation (or mean-SAGE) over a sparse adjacency, hidden 64, output 64, PReLU.
   - **Corruption:** row-shuffle of the node features.
   - **Training:** Adam, lr 1e-3, at most 300 epochs, early stopping on the DGI loss with patience 20, seeded.
   - **Compression:** PCA to 8 components per relation, giving 24 columns (49 with `own`).
6. **Train-only fitting (invariant 6), and the DGI loss mask:**
   - The feature standardisation that feeds DGI and the PCA are **fit on training nodes only**, then applied to every node.
   - The DGI loss is computed over **non-test nodes** (train and unlabelled). Test nodes take part in message passing only, which invariant 5 allows, and contribute no gradient.
   - Record this choice in `RESEARCH_PLAN.md` §11.
7. **#36 is unchanged.** All 25 `own` features stay in, including as GNN input. A `_no36` variant comes after the group decides; its GNN input must drop the same four features, not only its df-analyze input.
8. **Dependencies.**
   - torch and torch_geometric go into **this project's environment only**, as an optional extra `gnn` in `pyproject.toml`. Never touch df-analyze's environment.
   - Use plain PyG; no compiled `torch_scatter`/`torch_sparse` extensions.
   - Check `torch.cuda.is_available()` and fall back to CPU. Never hard-code `cuda`.
9. **Pilot without overwriting phase 1.**
   - The 100-trial `outputs/amazon/m1_own/` is never rerun or overwritten.
   - The pilot uses separate feature-set names with the same blocks: `pilot_m1_own`, `pilot_m2_own_nbr`, `pilot_m5_own_gnnssl`.
   - All three are run with `df_analyze.classifiers=[lgbm,lr,catboost]` and `df_analyze.htune_trials=10`, **on the same machine**. A slower machine hits df-analyze's time limits sooner, so the comparison would not be fair.
10. **The comparison page.**
    - A static, self-contained HTML file built by a standalone script, `scripts/compare_report.py`. It is **not** a pipeline stage, because it spans two feature sets.
    - Unlike `web_report`, it may contain **inline** JavaScript (click a point to show its node card), but it must make **no external requests**: no CDN, fonts or images.
    - Draw charts as inline SVG from Python; no Plotly.
    - Add this exception to `AGENTS.md` in the same commit.
    - The page contains row-level test data. Show a group-only banner on it and never commit it.

## Ground rules

- **Git:**
  - Small, focused commits. Before each one: `git status`, `git diff --staged`, `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
  - Never commit to `main` or `feat/review-fraud-pivot`. Never merge, force-push or tag.
- **What never gets committed:**
  - anything under `data/` or `outputs/`;
  - bundles;
  - `*.pt`, `*.npy`, `*.npz` or other model and embedding files. Add `.gitignore` patterns in the same commit that introduces them.
- **Tests** use only synthetic data (`tests/synthetic.py`, `tests/helpers.py`) and must run on CPU. GPU scatter is non-deterministic, so the determinism and label-blindness tests force CPU.
- **Timing.** Give a time estimate and wait for "go" before any run expected to take more than about an hour (AGENTS.md). Record the wall-clock time of every real run.
- **Phase 1 bundle.** After this work, an unpacked phase-1 bundle must still skip every stage. **Nothing in phase 1 may be retrained.**

## Tasks

### Task 0: Branch and baseline check

- Check out the branch and set up the extra:
  ```
  git checkout feat/review-fraud-pivot && git pull --ff-only
  git checkout -b feat/gnn-embeddings
  uv sync
  uv sync --extra gnn
  ```
  `uv sync --extra gnn` only works after Task 1 adds the extra; until then, plain `uv sync`.
- If `outputs/amazon/m1_own/` is missing, ask the user for the phase-1 bundle. Then run `uv run unpack_artifacts.py --dataset amazon`.
- `uv run run.py --dataset amazon` must report every stage as skipped. **If it wants to run anything, stop and report.**
- The tests and ruff must pass before any change.

**Done when:** the baseline is green and phase 1's outputs are present.

### Task 1: Plumbing for new feature blocks

1. **The `gnn` extra.** Add the optional extra to `pyproject.toml`:
   - `torch` and `torch-geometric`, compatible with Python 3.13; the Linux PyPI torch wheels already include CUDA;
   - a `gnn` test marker, so `pytest` still passes without the extra by skipping those tests.

   Check: `uv run python -c "import torch, torch_geometric; print(torch.cuda.is_available())"`.
2. **Column types for new blocks.** `run_df_analyze_input` requires every column to be in `column_spec.json`, which the `features` stage writes for `own` only.
   - Each new block stage writes `features/<block>.spec.json`, typed on **training nodes** with the same typing function the `features` stage uses.
   - `df_analyze_input` merges `column_spec.json` with the spec of each block in the active feature set.
   - Leave `column_spec.json` itself unchanged, so existing bundles stay valid.
3. **New stages must not trigger reruns.**
   - `plan_stages` reruns everything from the first stage whose outputs are missing.
   - New block stages therefore go right after `audit`, and their outputs must be **only the blocks the active feature set uses**: an empty list for `m1_own`, so the stage counts as done. Today `Stage.outputs` receives `Paths` only. Extend `Paths` (or the `Stage` signature) so a stage can see the active feature set's blocks.
   - Extend `tests/test_bundle_skips_stages.py`: an `m1_own` bundle still has no stage to run. Add a matching test that with `feature_set=m5_own_gnnssl` only `gnn_embed` and the stages from `df_analyze_input` on would run.
4. **Feature sets.** In `configs/default.yaml`, replace the commented "later:" lines with real entries:
   ```yaml
   m2_own_nbr: [own, nbr]
   m5_own_gnnssl: [own, gnnssl]
   pilot_m1_own: [own]
   pilot_m2_own_nbr: [own, nbr]
   pilot_m5_own_gnnssl: [own, gnnssl]
   ```
   Also add `neighbour_agg:` and `gnn_embed:` config sections holding the values from Decisions 4–6, plus `device: auto` and `usu_edge_fraction: 1.0` (see Task 3).

**Done when:** the tests above pass and the baseline check from Task 0 still reports everything skipped.

### Task 2: `neighbour_agg` stage → `features/nbr.parquet`

- **Builder:** `build_nbr_block(features, edges_by_relation) -> DataFrame` in `src/fraud_detect/features/graph.py`.
  - It takes the `own` block and edge lists, never `nodes.parquet` or labels.
  - Compute neighbour means with a scipy sparse row-normalised adjacency; no torch needed. Isolated nodes get 0.
  - It uses the features of **all** nodes, test and unlabelled included (invariant 5).
- **Stage:** `neighbour_agg`. It writes `features/nbr.parquet` and `features/nbr.spec.json`, and the block passes `blocks.validate_block`.
- **Tests:**
  - `assert_label_blind` from `tests/test_label_blindness.py`; move the helper to `tests/helpers.py` if it needs to be shared.
  - A hand-checked mean on a 5-node graph.
  - No nulls, one row per node.

**Done when:** on Amazon the block has 11,944 rows and 78 columns, with no nulls, and the run time is recorded.

### Task 3: `gnn_embed` stage → `features/gnnssl.parquet`

- **Code:** `src/fraud_detect/graph/ssl.py` holds the sparse adjacency builder, the encoder, and the DGI training loop with early stopping.
  - The public entry point takes `(own_features, edges, fit_mask, loss_mask, config)`.
  - `fit_mask` is the training nodes; `loss_mask` is the non-test nodes. Both are derived from the `split` column **before** the call, so the function never sees `nodes.parquet`.
- **Full-batch training** with a sparse adjacency for every relation.
  - U-S-U has about 7.1M directed edges, so never materialise per-edge 64-d messages.
  - If CUDA runs out of memory on `usu`, retry on CPU, or set `gnn_embed.usu_edge_fraction` below 1 to sample edges once with a fixed seed. Log which happened.
- **Outputs:**
  - `features/gnnssl.parquet` (24 columns: `gnnssl__upu_pc01` … `gnnssl__uvu_pc08`) and `features/gnnssl.spec.json`;
  - `outputs/<dataset>/gnn_embed/raw_<relation>.npy` (11,944 × 64), used by the comparison page;
  - `outputs/<dataset>/gnn_embed/meta.json`: per relation, the loss curve, best epoch, PCA explained variance, device, run time, torch and PyG versions, and a config hash.
- **Tests** (synthetic, CPU, `gnn` marker):
  - **Label-blind:** identical embeddings when labels are permuted.
  - **Determinism:** same seed gives identical output.
  - **Train-only fitting:** the scaler and PCA parameters depend only on `fit_mask` rows. Changing non-train rows' inputs to the scaler and PCA fit leaves the fitted parameters unchanged.
  - **Shape:** shapes and names, no nulls.
- **Before Task 5, a label-free sanity probe** (no test data). Add `scripts/probe_blocks.py`:
  - It fits a fixed-parameter `HistGradientBoostingClassifier` on the saved grouped `cv_folds` (training nodes only) for `own`, `own+nbr` and `own+gnnssl`.
  - It writes the mean ± SD CV PR-AUC to `outputs/<dataset>/probe/probe_cv.csv`. It runs in minutes and gives an early signal for the demo.
  - It never touches test rows. Never choose or change anything because of test results.

**Done when:** all three relations train on the GPU machine, the block validates, `meta.json` is written, and the probe table exists.

### Task 4: The comparison page

- **Command:**
  ```
  uv run scripts/compare_report.py --dataset amazon --a <feature_set> --b <feature_set>
  ```
- **Writes** `outputs/<dataset>/compare/<a>__vs__<b>/compare.html` and `compare_metrics.json`.
- **Inputs.** For each side, read that feature set's Model A, its threshold (from out-of-fold training predictions, `model_a.json`) and its test probabilities with the existing helpers (`df_analyze_report.load_run`, `choose_model_a`, `web.find_entry`).
  - Do **not** add outputs to existing stages: that would make phase-1 bundles rerun.
  - Stop if the two sides' test node IDs differ.
- **Metrics** (into `compare_metrics.json` and the page header): for each side:
  - test PR-AUC and AUROC with group-bootstrap CIs;
  - Recall@K, with K = the number of test frauds;
  - F1, precision and recall at its own out-of-fold threshold.

  Plus the **paired** difference ΔPR-AUC (B − A): `evaluation.group_bootstrap(..., score_a=B, score_b=A)` with 2,000 resamples, its 95% CI, and the share of resamples with Δ > 0.
- **Page sections:**
  1. **Header:** the metrics table, Δ with its CI, the run settings (trials, classifiers) of each side, and a banner: *"Contains row-level test data. Group only: do not upload or commit."*
  2. **Scatter (SVG):** P(fraud | A) on x and P(fraud | B) on y, for every test node.
     - Colour by true label; draw each side's threshold as a line; label the four quadrants (both catch, B rescues, B breaks, both miss).
     - Clicking a point opens its node card (inline JS).
  3. **Disagreement gallery:** the top 20 "B rescues" (A false negative, B true positive) and the top 20 "B breaks", ranked by |pB − pA|.
  4. **Node card:**
     - node ID, label, pA and pB, and the decision under each threshold;
     - top features by training percentile (reuse `web.py`'s helper), with `nbr`/`gnnssl` columns labelled by relation;
     - **one ego network per relation:** at most 12 neighbours, sampled with a fixed seed, drawn as inline SVG with a simple circular or spring layout.
       - Colour only **training** neighbours by label; test and unlabelled neighbours are grey.
       - Show each node's true degree and the fraction of its training neighbours that are fraud.
       - Precompute cards only for gallery nodes plus about 200 seeded random test nodes, to keep the file small.
  5. **Embedding view:** a 2-D PCA of `raw_upu.npy` fit on training nodes, with training nodes filled and test nodes hollow, coloured by label. Show it next to the same projection of `own`.
- **Safety and tests:**
  - HTML-escape every inserted value.
  - Tests check: the page builds from synthetic runs; it contains no `http://`/`https://` resource references; a constructed B that beats A gives Δ > 0; a test-ID mismatch stops the script.
- **Bundles:** check that `pack_artifacts.py` includes `outputs/<dataset>/gnn_embed/`, the new feature blocks and `compare/`. Update it if not.

**Done when:** the page opens offline in a browser from WSL (`explorer.exe`), and you have looked at it and checked it before telling the user.

### Task 5: Pilot runs (for the demo)

Give the estimate and wait for "go". The 10-trial CPU pilot with six classifiers took about 20 minutes, so expect roughly 20–40 minutes per feature set and 1–2 hours in total. Run all three on the same machine:

```bash
for FS in pilot_m1_own pilot_m2_own_nbr pilot_m5_own_gnnssl; do
  uv run run.py --dataset amazon --set feature_set=$FS \
    --set "df_analyze.classifiers=[lgbm,lr,catboost]" --set df_analyze.htune_trials=10
done
uv run scripts/compare_report.py --dataset amazon --a pilot_m1_own --b pilot_m5_own_gnnssl
uv run scripts/compare_report.py --dataset amazon --a pilot_m1_own --b pilot_m2_own_nbr
uv run scripts/compare_report.py --dataset amazon --a pilot_m2_own_nbr --b pilot_m5_own_gnnssl
```

- Check every run's `split_check.json` and `type_check.json`.
- Note which `nbr`/`gnnssl` columns df-analyze's feature selection kept. That is a result in its own right.
- Never change anything because of a test result.

**Done when:** three verified runs and three comparison pages exist.

### Task 6: Docs and hand-off

- **README:**
  - ladder status (Models 2 and 5 implemented);
  - the new stages in the stage table;
  - how to install the `gnn` extra;
  - how to build the comparison page.
- **`RESEARCH_PLAN.md`:**
  - §11 decision log: Decisions 3–6 and 9–10;
  - the pilot results table, clearly labelled **pilot** (10 trials, 3 classifiers, one split), with Δ and paired CIs and the run times and GPU;
  - an update to §12.
- **Bundle:** `uv run pack_artifacts.py --dataset amazon`, then tell the user its path, to share on OneDrive.
- **Report to the user** in chat: the results table, the honest reading (below), and anything left undone.

## Honest expectations (put this in the README and the demo)

- **Literature (GADBench, Tang et al., NeurIPS 2023, tuned):**
  - plain XGBoost reaches 91.88 AUPRC on Amazon;
  - XGBoost with simple neighbour aggregation reaches 93.33;
  - the best specialised GNN, BWGNN, reaches 91.48.

  Model A (0.919) is already at the "trees without graph" level, so the realistic gain on Amazon is 0 to about 1.5 points. That is inside the current test CI (±2.6 points).
- **Self-supervised embeddings** (Cheng et al., arXiv 2410.18487): frozen DGI embeddings with an MLP reach 88.92 AUPRC on Amazon. These embeddings help most on sparse graphs with few labels; Amazon is dense and has more than 5,000 training labels. So **M5 may tie M1 or lose to M2**. That is a valid, reportable finding: "does graph context add anything beyond strong tabular features?"
- **Where to look next:** YelpChi is where graph information clearly helped trees in GADBench (84.0 → 91.1 AUPRC).
- **#36 caveat:** if the four flagged features encode the label, they may saturate the task and hide any graph benefit. The `_no36` runs are where the graph has room to show value.

Those benchmarks use random splits; ours are grouped. Their numbers set expectations; they are not targets.

## Demo fallback

If Task 3 slips, demo `pilot_m1_own` vs `pilot_m2_own_nbr` with the same comparison page. It still answers "does graph information help df-analyze?", and it is the baseline GADBench recommends.

## Phase 3 (after the demo; not in this plan's scope; ask before starting)

1. **Embedding settings, chosen on training data only:** a small grid (dim 16/32/64, epochs 100/300, PCA k 4/8/16), scored by the probe on the grouped CV folds.
2. **Full-budget runs:** `m1_own` (exists), `m2_own_nbr` and `m5_own_gnnssl`, all on one machine. About 2 hours each on the RTX 4060 laptop.
3. **Repeats:** `split.n_repeats: 3` for all compared rungs. Refit the scaler and PCA per split. Report the per-repeat Δ, plus the Nadeau–Bengio corrected t-test over the 15 paired CV folds as supporting evidence.
4. **#36:** after the group decides, `m1_own_no36` and `m5_own_gnnssl_no36`, with the GNN input also excluding the four features.
5. **YelpChi:** the same ladder.
6. **Ablations:** GraphMAE, or a homogeneous ("homo") embedding.
