# Research plan v2: review-fraud graphs (YelpChi, Amazon)

**Version 2, 2026-10-07.** This replaces plan v1 (CFPB + IBM AMLworld), which exists only on the unmerged branch `claude/cool-euler-o58nwf`. Commit this file as `docs/RESEARCH_PLAN.md` in the first milestone.

**Instructions for Claude Code.** Read `AGENTS.md`, then this whole plan, before changing anything.
- Work through the milestones in order (§8).
- Each milestone has deliverables and acceptance criteria; meet them before moving on.
- Anything marked **[group decides]** is an open decision. Do not settle it yourself. Stop and ask (§10).
- This plan amends some rules in `AGENTS.md`. M1 writes those amendments into `AGENTS.md`. Until then, if this plan and `AGENTS.md` disagree, stop and ask.

---

## 0. Scope of this branch

**In scope:**
- Switch the repository from CFPB/AMLworld to two graph fraud datasets: **YelpChi** (main) and **Amazon** (replication).
- Make the df-analyze pipeline (Model 1) robust on both, end to end: load, audit, split, features, df-analyze, shared-CV model choice, test report.
- Prepare clean interfaces so graph-derived feature blocks and GNNs can be added later without reworking the pipeline (§6).

**Out of scope for this branch:**
- Writing any GNN or graph-learning code, or adding PyTorch Geometric or DGL.
- Neighbour-aggregated features (Model 2).
- Text data of any kind.

The data-model changes in this plan exist so those later pieces fit. Do not implement them.

**Git (overrides `AGENTS.md`'s "push and open a PR" step for this plan).**
- Create one local branch from an up-to-date `main`: `feat/review-fraud-pivot`.
- Make small commits, one or more per milestone, each with a clear message.
- **Do not push the branch or open a PR until the user explicitly says so.** The repository is public. **Update 2026-10-13:** the user decided to push the branch and open its PR to `main` at the end of phase 1 ([`PHASE1_AMAZON_PLAN.md`](PHASE1_AMAZON_PLAN.md), decision 7). The branch has now been pushed and its PR opened; it is never merged by an agent.
- Everything else in `AGENTS.md`'s git section still applies: never commit to `main`, never merge, never force-push, never tag.
- At the end, write the PR description (what, why, how tested, leakage checklist) to `PR_DESCRIPTION.md` in the repo root. Do **not** commit that file; the user will use it when they open the PR.

---

## 1. Question and design (what the pipeline must eventually support)

**Main question:** does using a graph improve fraud classification over the same kind of model without one, and can a GNN's representations improve a tabular AutoML pipeline (df-analyze)?

The full model ladder. This branch builds Model 1 only; the others shape the interfaces.

| # | Model | Graph information | Built on this branch? |
|---|---|---|---|
| 1 | df-analyze on each node's own features (Model A) | none | **yes** |
| 2 | df-analyze on own features + label-free neighbour aggregates, per relation | yes, hand-made | no (interface only) |
| 3 | GraphSAGE with all edges removed (an MLP) | none | no |
| 4 | Heterogeneous GraphSAGE over all relations | yes, learned | no |
| 5 | df-analyze on own features + **self-supervised** GNN embeddings, per relation, PCA-compressed | yes, learned, no labels | no (interface only) |
| 6 | df-analyze on own features + **supervised, out-of-fold** GNN embeddings or score | yes, learned from labels | no; needs a rule change **[group decides]** |

The key contrasts later will be 5 vs 1 (does the GNN help df-analyze?), 5 vs 2 (learned embeddings vs hand-made aggregates), and 4 vs 3 (message passing with the architecture held fixed).

**Headline metric:** PR-AUC (average precision) on the frozen test set, always reported next to the no-skill baseline (the test positive rate).

---

## 2. Datasets: verified facts

**Source.** The preprocessed `.mat` files from the CARE-GNN repository (Dou et al., CIKM 2020), Apache-2.0 licensed: `https://github.com/YingtongDou/CARE-GNN`, branch `master`, commit `a64ff7523e187a24251f7ca88435d2c9d8f7dcd9`, folder `data/`. These are the files that GADBench and the DGL `FraudYelpDataset` / `FraudAmazonDataset` loaders are based on. Load them with `scipy.io.loadmat`. **Do not add DGL as a dependency.**

**SHA-256 checksums (verified 2026-10-07):**

| File | SHA-256 |
|---|---|
| `YelpChi.zip` | `3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3` |
| `Amazon.zip` | `c1b7a1f8cd8621467d5121b690fef61a7d398e4f582232bb3e8a154424f7a5b9` |
| `YelpChi.mat` (unzipped) | `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42` |
| `Amazon.mat` (unzipped) | `4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63` |

**Keys in each `.mat`:** `features` (sparse CSC), `label`, `homo`, and three relation matrices. Adjacency matrices are symmetric, so the number of stored entries (nnz) is twice the number of undirected edges.

| | YelpChi (main) | Amazon (replication) |
|---|---|---|
| Node | a review | a user |
| Nodes | 45,954 | 11,944 |
| **Labelled nodes** | 45,954 | **8,639** (nodes 3,305 onwards; see below) |
| Positives | 6,677 (14.53%) | 821 (9.50% of labelled) |
| Features | 32, already scaled to [0, 1], no integer-valued columns; columns 2–5 are binary | 25; columns 0–6, 16, 17, 19, 20, 22, 24 are integer-valued; column 22 is binary |
| Relation keys (nnz) | `net_rur` same user (98,630); `net_rtr` same product, same month (1,147,232); `net_rsr` same product, same star rating (6,805,486) [corrected] | `net_upu` reviewed a common product (351,216); `net_usu` same star rating within one week (7,132,958); `net_uvu` top-5% review-text similarity (2,073,474) |
| Exact duplicate feature rows (labelled nodes) | 0 | 293 |
| Labels come from | Yelp's review filter (Rayana & Akoglu, KDD 2015) | Helpfulness votes: users with more than 80% helpful votes are benign, fewer than 20% fraudulent (CARE-GNN, Dou et al. 2020; DGL `FraudAmazonDataset`; features from Zhang et al. 2020) [verified] |

**Corrections made during implementation (2026-10-07).** [corrected] The draft of this plan described `net_rtr` as same product and same star rating and `net_rsr` as same product and same month. It is the other way round (R-T-R: same month; R-S-R: same star rating), per CARE-GNN and the DGL loader's documentation; the keys and edge counts were right. [verified] The Amazon label source is confirmed above.

**Dataset gotchas (the loader must handle all of them):**

1. **Amazon's first 3,305 nodes are unlabelled but stored as label 0.** CARE-GNN's own training code uses only `range(3305, n)`. Treating them as negatives is a silent bug. They are excluded from all labelled sets but kept in the node table as `is_labelled = False`, for later graph context.
2. **YelpChi labels are user-level.** The connected components of `net_rur` are users: 29,431 components, 22,123 of them singletons, largest 47 reviews. Labels are about 99.7% pure within a multi-review component. **The split must keep each user's reviews together** (§4).
3. **Amazon has 293 exact-duplicate feature rows** among labelled nodes. Treat each set of identical rows as one group in the split, so copies cannot fall on both sides. This is the analogue of the CFPB near-duplicate rule.
4. **Features are anonymized.** Name them `f00`…`f31` (YelpChi) and `f00`…`f24` (Amazon), with the prefix `own__` in feature tables (§6).
5. **Possible label shortcut on Amazon.** One feature alone reaches a univariate AUROC of about 0.88. If labels come from helpfulness votes, a vote-derived feature would be a shortcut. The audit stage reports this (M4); it does not resolve it.

**Rough expected performance (sanity bands, not results).** These come from one untuned `HistGradientBoostingClassifier` on own features, with a 60/40 split, seed 0.

| | Test PR-AUC | Test AUROC | No-skill PR-AUC |
|---|---|---|---|
| YelpChi, user-grouped split | ≈ 0.80 | ≈ 0.94 | 0.145 |
| Amazon, stratified split | ≈ 0.91 | ≈ 0.98 | 0.095 |

If Model A lands more than about 0.05 PR-AUC **below** these, assume a pipeline bug (mis-typed features, a wrong label vector, a misaligned split) and investigate before reporting anything.

**For later work (do not implement now):** label-free neighbour means computed over the whole graph raised YelpChi PR-AUC from about 0.81 to about 0.88 under the user-grouped split. Restricting neighbours to training nodes instead made it *worse* (about 0.72), because under a grouped split test reviews have no training same-user neighbours, so train and test see different feature distributions. This is why the graph-context rule in §5 differs from the old CFPB rule.

---

## 3. What happens to the old code

- The last CFPB state is `main` at `3336d87`. Record this commit in the decision log (§11), so CFPB results stay reproducible from git history.
- **Remove** CFPB-specific code, configs, docs and tests:
  - stages `load`, `label`, `sample`, `embed`, `pca`, `label_audit`;
  - the data modules `load`, `label`, `dedup`, `sample`, `audit`;
  - `features/text.py`;
  - `configs/categories.yaml`, `docs/LABEL_RULE.md`, `download_dataset.sh`'s CFPB logic;
  - the CFPB columns in `columns.py`;
  - their tests.
- Remove the planned AMLworld text from the README and `AGENTS.md`.
- **Keep and generalize** (dataset-agnostic):
  - `run.py`, `cli.py`, `config.py`, `runstate.py`, `runlog.py`, `external.py`;
  - `models/df_analyze.py` (`--df-train`/`--df-tests`, split verification);
  - `scripts/dfa/cv_select.py` and `models/selection.py` (shared-CV model choice, tuning budget);
  - `models/df_analyze_report.py` and `report/web.py`;
  - `artifacts.py`, `pack_artifacts.py`, `unpack_artifacts.py`;
  - the watchdog, resume and atomic-write machinery;
  - the df-analyze pin (commit, Python 3.13 range).
- Keep the hard-won df-analyze lessons documented in the README and config comments. **Do not re-derive them.**
  - Tune on `bal-acc`: df-analyze silently swaps AUROC for balanced accuracy on most models, and scores GANDALF through a different path.
  - Choose Model A by shared-CV PR-AUC, never by df-analyze's tuning scores.
  - Never report df-analyze's `5-fold` rows; they refit on test folds.
  - Do not pass `--df-tests-method`; it hits an enum bug in df-analyze 4.1.0.
  - Leave the MLP out: CPU-only, about 4 h.
  - Check every score a tool produces against its source code.
- The web report currently shows complaint narratives. Replace those cards with **most and least confident test nodes, showing node ID, label, score and the top features by value**. There is no text.

---

## 4. Splits (frozen test set, grouped where needed)

**One split per dataset, made by us, saved, and passed to df-analyze.** The test set is touched only by the final report stages.

| | YelpChi | Amazon |
|---|---|---|
| Labelled pool | all 45,954 reviews | the 8,639 labelled users (exclude nodes 0–3,304) |
| Group ID | connected component of `net_rur` (= user) | identical-feature-row group (singletons otherwise) |
| Train / test | 60 / 40, `StratifiedGroupKFold` (5 folds; take 2 as test), seeded | 60 / 40, same procedure with the duplicate groups |
| Approximate sizes | ≈ 27.6k train / 18.4k test | ≈ 5.2k train / 3.5k test |

**Shared CV folds for model selection.** Assign each *training* node to one of 5 folds with `StratifiedGroupKFold` using the same group IDs. Save the assignment (`cv_folds.csv`: node_id, fold).
- `select_model` (`scripts/dfa/cv_select.py`) must use these saved folds instead of making its own stratified folds.
- Later, any out-of-fold GNN feature (Model 6) must use the same folds.

**Saved split artifacts**, under `data/processed/<dataset>/`:
- `train_ids.csv` and `test_ids.csv`, with columns `node_id, label, group_id`;
- `cv_folds.csv`;
- `split_summary.json`, with sizes, positive rates and group counts per part, plus checks that no group crosses train/test or folds.

**Known, unfixable limitation (document it in the README).** df-analyze's *internal* tuning CV and feature-selection splits are random, not grouped. Same-user reviews can therefore sit on both sides of its internal folds, which makes its tuning scores optimistic. That is one more reason Model A is chosen by our grouped shared CV, never by df-analyze's scores. Our test set is grouped, so the final scores are not affected.

**[group decides] YelpChi training size.** About 27.6k training rows is within df-analyze's guidance (under 30k), but runtime will be longer than the CFPB run (18k rows, about 4 h on an RTX 4060 laptop). Default: use the full training set. Alternative: a seeded, group-preserving subsample of the training set only, never the test set. Ask before the full run (M8).

**Amazon split variance.** Amazon is small, so its scores depend noticeably on the split. Implement `split.n_repeats` (default 1). With 3, the pipeline runs df-analyze on 3 independently seeded grouped splits and reports mean ± SD. Only enable this after asking, because it triples the runtime.

---

## 5. Leakage invariants for the new datasets

These replace the CFPB and AMLworld-specific invariants in `AGENTS.md` (M1). The general invariants (4, 9 and 10: frozen test set, choose Model A by shared CV, every model scored the same way) stay.

1. **Labels enter only through the training loss.** No feature, aggregate, node attribute or graph input uses any node's label. This includes neighbour fraud rates and target encoding, even on training data only. (A later, documented exception may permit supervised *out-of-fold* GNN features for Model 6; see §10.)
2. **Grouped splits.** No YelpChi user (`net_rur` component) and no Amazon duplicate-feature group appears in more than one of train/test, or in more than one CV fold.
3. **Amazon nodes 0–3,304 are never labelled examples.** They are unlabelled graph context only.
4. **The test set is frozen.** Thresholds, feature choices, hyperparameters, model choice and (later) graph settings are chosen on training data or its CV folds only.
5. **Graph context rule (for later stages; record it now).** Graph-derived features may use the *features* (never labels) of every node in the graph, including test nodes, because unlabelled reviews and users exist at deployment time. This replaces the CFPB rule that test nodes attach only to training nodes, which breaks grouped splits (§2). Every graph-derived feature is computed from features and edges only, and its code must never receive the label vector. Add a test that enforces this (§6).
6. **Unsupervised transforms are fit on training nodes only:** PCA, scalers, and any feature-type decisions based on data.
7. **One feature schema per dataset.** Model 1's columns are exactly the `own__` block. No node IDs, group IDs or relation degrees enter Model 1.

---

## 6. Interfaces that prepare for the GNN (implement the plumbing, register only `own`)

**Node table**, `data/processed/<dataset>/nodes.parquet`, one row per node in the graph, including Amazon's unlabelled nodes:
- `node_id` (int, the row index in the `.mat`);
- `label` (nullable int; null when unlabelled);
- `is_labelled`;
- `group_id`;
- `split` (`train` / `test` / `unlabelled`);
- `cv_fold` (nullable).

**Relations**, `data/processed/<dataset>/graph/<relation>.npz`, one per relation plus `homo`:
- an undirected COO edge list with `src < dst`, no self-loops, deduplicated, `int32`;
- `graph/manifest.json`, with each relation's name, edge count, node count, source key, and the SHA-256 of the `.mat` it came from.

Nothing on this branch consumes these files except tests and the audit's edge counts. They exist so the GNN and Model 2 stages can start from a stable format.

**Feature blocks.** Every feature source writes one parquet, `data/processed/<dataset>/features/<block>.parquet`, keyed by `node_id`, with all columns prefixed `<block>__`. On this branch the only block is `own`. Later blocks would be `nbr_rur`, `nbr_rtr`, …, `gnnssl_rur`, ….

**Feature sets (the model rungs) in config:**

```yaml
feature_sets:
  m1_own: [own]
  # later: m2_own_nbr: [own, nbr_rur, nbr_rtr, nbr_rsr]
  #        m5_own_gnnssl: [own, gnnssl_rur, gnnssl_rtr, gnnssl_rsr]
```

**How stages use them:**
- `df_analyze_input` builds train/test tables for one feature set by joining its blocks on `node_id` in split order.
- It fails if a block is missing, has unprefixed columns, has nulls, or contains `label`, `node_id`, `group_id` or `split` columns.
- Stages 9–12 run per feature set, with outputs under `outputs/<dataset>/<feature_set>/`.
- Make the active feature set selectable with `--set feature_set=m1_own`. Default to `m1_own`.

**Label-blindness guard for graph features.** Add a function signature convention and a test:
- Graph feature builders take `(features, edges)` and never `labels`.
- A test builds a block from synthetic data twice, with the labels permuted between the two calls, and asserts the outputs are identical.
- Ship this test now with a trivial example builder in `tests/`, not in `src/`, so the convention exists before any real graph block does.

---

## 7. Robust df-analyze on these datasets

**Explicit feature types; do not let df-analyze guess.**
- The `features` stage writes `data/processed/<dataset>/column_spec.json`.
- Every `own__` column is assigned `binary`, `ordinal` or `continuous`, **using training nodes only**:
  - `binary`: exactly 2 values;
  - `ordinal`: integer-valued with more than 2 values;
  - `continuous`: everything else.
- There are no categoricals in these datasets.
- `df_analyze_args` passes the ordinals explicitly. Check in df-analyze's source, at the pinned commit, how to pass *no* categoricals: the current code joins a list, and an empty string may be parsed as one column named "". Handle that case correctly and add a test.
- After each run, read df-analyze's `inspection/inferred_types.csv` and **fail** if any column's final type differs from `column_spec.json`, or if df-analyze dropped any feature. Also report its "destructive changes" (dropped constant, identifier-like or rare-level features).
- Known risks to look for:
  - YelpChi's high-cardinality continuous columns (some have over 29k unique values) being flagged as identifiers;
  - Amazon's low-cardinality integer columns (5–10 levels) being inferred as categorical.

**Split verification.** Keep and adapt the existing check that df-analyze's exported `X_train.csv`/`X_test.csv` match our split row for row: row counts, labels, and each column after df-analyze's clip-and-rescale, with the same tolerance logic as before.

**Shared-CV model choice** (stage `select_model`).
- Use the saved grouped folds (§4).
- Everything else stays as is: every tuned (classifier, feature-selection) combination refit on the same folds, scored from probabilities (PR-AUC, AUROC, balanced accuracy, Brier), defaults vs tuned, tuning budget, watchdog.
- Model A is the highest mean CV PR-AUC.
- Also save each combination's **out-of-fold probabilities** (`oof_predictions.parquet`), because the threshold below needs them.

**Decision threshold** (for F1, precision, recall, confusion matrix).
- Choose the threshold that maximizes F1 on Model A's out-of-fold training predictions.
- Report test metrics at that threshold *and* at 0.5.
- Never choose it on test.
- **[group decides]** confirm the max-F1 rule. Implement it as the default, behind a config key.

**Test report** (stage `df_analyze_report`).
- For every tuned model: test PR-AUC, AUROC, F1/precision/recall at both thresholds, balanced accuracy, and the no-skill PR-AUC.
- **Bootstrap 95% intervals** for Model A's test PR-AUC and AUROC: 1,000 resamples, **resampling whole groups** (users for YelpChi, duplicate groups for Amazon), seeded.
- Implement the bootstrap as a reusable function that accepts two prediction vectors (a paired difference), because the later contrasts (5 − 1, 2 − 1) need exactly that.
- Add the sanity-band check from §2: print a prominent warning, not an error, if Model A's test PR-AUC is more than 0.05 below the band.

**Audit stage** (`audit`, new; counts and statistics only, training nodes only unless stated). Writes `outputs/<dataset>/reports/audit.md` plus CSVs:
- dataset counts versus §2: fail if node, label, feature or relation counts differ;
- the labelled/unlabelled breakdown;
- per feature: unique count, min, max, NaN count, constant flag, and **univariate AUROC** (oriented to be at least 0.5), flagging AUROC ≥ 0.85 as a possible shortcut;
- exact duplicate rows within train, within test, and across train/test (the last must be 0 after the grouped split);
- group size distribution;
- per relation: edge count, mean degree of labelled nodes, and edge label homophily computed on **training–training edges only**. This is reported as a dataset statistic and never used as a feature. Compare it with CARE-GNN's published label similarities: YelpChi about 0.91 (R-U-R) vs about 0.18 (R-T-R, R-S-R); Amazon about 0.17 (U-P-U) vs about 0.05 (U-S-U, U-V-U).

**Classifiers and tuning.** Keep the current settings: `[lgbm, lr, catboost, gandalf, rf, knn]`, `htune_trials: 100`, `htune_cls_metric: bal-acc`, `--filter-pred-classify auroc`, wrapper selection off, MLP off.
- Watch kNN's runtime on YelpChi's ≈ 27.6k training rows, and record it in `tuning_budget.csv` as now.
- If one model dominates runtime with no chance of winning, propose dropping it rather than dropping it yourself.

**GPU.** Unchanged. `gpu.require` governs CatBoost and GANDALF. There is no embedding stage any more, so check whether the GPU requirement still applies anywhere else and simplify the config text accordingly.

**Multi-dataset plumbing.**
- Add `--dataset {yelpchi,amazon}` (required, or `dataset:` in config).
- Per-dataset sections hold paths, the `.mat` filename, relation keys, the unlabelled-prefix length (Amazon: 3305, YelpChi: 0), the grouping rule, and the expected counts.
- All data and outputs move under `<dataset>/` subfolders.
- `pipeline_state.json` is per dataset, so a resumed YelpChi run never touches Amazon state.
- `pack_artifacts.py` and `unpack_artifacts.py` take `--dataset`.

**New stage list:**

| # | Stage | Main outputs |
|---|---|---|
| 1 | `download` (also `./download_dataset.sh`) | `data/raw/care_gnn/*.zip` and `*.mat`, checksum-verified |
| 2 | `load` | `nodes.parquet` (labels, unlabelled flags), `graph/*.npz`, `manifest.json` |
| 3 | `split` | `train_ids.csv`, `test_ids.csv`, `cv_folds.csv`, `split_summary.json` |
| 4 | `features` | `features/own.parquet`, `column_spec.json` |
| 5 | `audit` | `reports/audit.md` and CSVs |
| 6 | `df_analyze_input` | per feature set: `train.parquet`, `test.parquet` |
| 7 | `df_analyze` | df-analyze run, split and type checks |
| 8 | `select_model` | grouped shared-CV scores, OOF predictions, tuning budget |
| 9 | `df_analyze_report` | test metrics, thresholds, bootstrap CIs, sanity-band check |
| 10 | `web_report` | `outputs/<dataset>/<feature_set>/report/index.html` |

**Download script.**
- Fetch the two zips from the pinned CARE-GNN commit, using the raw GitHub URL for commit `a64ff75…`, not `master`.
- Verify the SHA-256 values in §2, unzip, and verify the `.mat` checksums.
- Never overwrite a file with a different checksum without `--force`.
- Keep it safe to rerun.
- Write it in Python (`scripts/` or the `download` stage) rather than shell where reasonable, for macOS/BSD and WSL portability.
- `data/` stays git-ignored. Confirm `.gitignore` still covers `data/raw/care_gnn/`.

---

## 8. Milestones

Run `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` at the end of every milestone. Commit after each.

**M0: Orient (no code changes).**
- Read `AGENTS.md`, the README and this plan.
- Run the test suite on `main` and record the result.
- Create `feat/review-fraud-pivot`.
- **Acceptance:** the branch exists; you can summarize which modules §3 keeps and which it removes.

**M1: Rules and decisions first.**
- Add this plan as `docs/RESEARCH_PLAN.md`.
- Rewrite `AGENTS.md`'s project paragraph, dataset section and leakage invariants per §5.
- Add the decision log entries (§11).
- Remove the AMLworld and CFPB dataset rules.
- Keep the git workflow, the never-commit list (replace narrative-specific wording with "row-level data from any dataset"), the environments and the platform rules.
- **Acceptance:** `AGENTS.md` and the plan agree; nothing in `AGENTS.md` still refers to CFPB or AMLworld except the decision log's pointer to commit `3336d87`.

**M2: Remove CFPB code; add multi-dataset config.**
- Delete what §3 lists.
- Add the `dataset` selector and per-dataset config sections (§7).
- Make the stage list in `pipeline.py` match the new table, with not-yet-implemented stages raising a clear `StageError`.
- **Acceptance:** `uv run run.py --list --dataset yelpchi` shows the new stages; the tests pass (CFPB tests removed; generic tests for runstate, artifacts and the CLI kept and adapted).

**M3: Download and load.**
- Implement checksum-verified download and the `load` stage: node table, relation edge lists, manifest, count checks against §2, and Amazon's unlabelled prefix.
- **Tests:** use tiny synthetic `.mat` files made with `scipy.io.savemat` in pytest's temp directory, never the real data. Test:
  - that the unlabelled prefix is excluded from labels;
  - edge-list symmetry, deduplication and self-loop removal;
  - that a count mismatch fails loudly;
  - checksum failure handling.
- **Acceptance:** on the real files, both datasets load with exactly the §2 counts.

**M4: Split, features, audit.**
- Implement `split` (§4), `features` with `column_spec.json` (§7), and `audit` (§7).
- **Tests:**
  - no group crosses train/test or folds;
  - stratification is within tolerance;
  - the split is deterministic for a given seed;
  - Amazon duplicate groups are respected;
  - the column-spec rules work, using training nodes only;
  - the audit flags a planted shortcut feature in synthetic data.
- **Acceptance:** the real `split_summary.json` and `audit.md` for both datasets show zero cross-split groups and zero cross-split duplicate rows; homophily is close to CARE-GNN's published values. Report the univariate-AUROC flags to the user. **Do not drop any flagged feature without asking.**

**M5: Feature blocks and the df-analyze input.**
- Implement the block and feature-set plumbing and the validation rules (§6).
- Implement the label-blindness test convention.
- **Tests:**
  - joining a synthetic extra block works;
  - forbidden columns fail;
  - row order matches the split files;
  - the permuted-label test passes for the example builder.
- **Acceptance:** `df_analyze_input` writes `m1_own` tables for both datasets with exactly the `own__` columns plus `target`.

**M6: df-analyze run and checks.**
- Generalize `df_analyze_args` (explicit ordinals, the no-categoricals case) and the split verification.
- Add the inferred-type check against `column_spec.json`.
- Make `select_model` use the saved grouped folds and save OOF predictions.
- **Tests:** the argument construction, including the empty-categoricals case; the type-check failure path, using a fake `inferred_types.csv`; that `select_model` uses the saved folds (synthetic).
- **Acceptance:** all tests pass. **No real df-analyze run yet** (see M8).

**M7: Report.**
- Thresholds from OOF predictions, the group-bootstrap function (single and paired), the sanity-band warning, and web report cards without narratives.
- **Tests:**
  - the threshold is chosen without test data (the function must not accept test arrays);
  - the bootstrap resamples whole groups;
  - the paired bootstrap of identical predictions gives an interval around 0.
- **Acceptance:** the report renders from synthetic df-analyze outputs.

**M8: Pilot runs, then full runs (ask first).**
1. Pilot Amazon end to end with `--set df_analyze.htune_trials=10`. Expected to be well under 1 h; confirm the timing from the pilot.
2. Fix anything it reveals.
3. Show the user the audit, split summary, inferred-type check, Model A and test metrics with CIs, and the sanity-band comparison.
4. **Ask before** the full Amazon run, any YelpChi run longer than about 1 h, and `split.n_repeats > 1`. Give runtime estimates from the pilot.
- **Acceptance:** an Amazon pilot completes with every check passing, and the user has seen its results.

**M9: Documentation and hand-off.**
- Rewrite the README:
  - question and ladder (§1);
  - datasets with citations: CARE-GNN, Rayana & Akoglu 2015, and the Amazon label source;
  - splits and limitations (§4, including df-analyze's ungrouped internal CV);
  - setup and stages;
  - what is prepared for the GNN.
- Update the repository layout section.
- Write `PR_DESCRIPTION.md` (uncommitted) with the leakage checklist rewritten for §5.
- **Do not push; tell the user the branch is ready.**

---

## 9. Compute and runtime expectations

- **Amazon (≈ 5.2k training rows):** the full df-analyze run with 6 classifiers and 100 trials should take well under the CFPB run's 4 h. Measure it in the pilot.
- **YelpChi (≈ 27.6k training rows):** expect it to be longer than CFPB's 18k-row run, about 4 h for df-analyze plus about 35 min for `select_model` on an RTX 4060 laptop. kNN and GANDALF are the likely bottlenecks. Ask before running (§4, **[group decides]** subsample).
- Each later feature set (Models 2, 5, 6) adds one more df-analyze run per dataset. Plan the group's compute around roughly 3–4 runs per dataset in total.

---

## 10. Open decisions ([group decides])

1. ~~**YelpChi training size:** full ≈ 27.6k (default) or a group-preserving subsample.~~ **Decided 2026-10-08: the full training set** (see §11).
2. ~~**Threshold rule:** maximize F1 on out-of-fold predictions (implemented default) vs. another rule.~~ **Decided 2026-10-08: max F1 on out-of-fold training predictions.**
3. ~~**Amazon split repeats:** 1 (default) or 3 seeded grouped splits.~~ **Decided 2026-10-08: 1 for now; 3 once Model 2 exists,** so every rung of a comparison shares the same 3 splits.
4. **Model 6 (later):** amend invariant 1 to allow supervised GNN features produced strictly out-of-fold on the saved CV folds, with test-node features from a GNN trained on all training nodes. Not needed for this branch.
5. **Flagged shortcut features (after M4):** keep, drop, or report results both ways.
6. **The CFPB result's place in the final report:** omit it, or mention it as the motivating pilot (README wording only).
7. **Phase 3 (the professor's feedback, 2026-10-21):** questions D1–D5 in [`PHASE3_FEEDBACK_PLAN.md`](PHASE3_FEEDBACK_PLAN.md): what "DeepTune to reduce", "training the GNN" and "4 important features" mean; which new datasets and their split rules; backfilling the results database.

---

## 11. Decision log (add these in M1)

- **2026-10-07:** CFPB and IBM AMLworld are dropped in favour of YelpChi (main) and Amazon (replication), from CARE-GNN's preprocessed `.mat` files at commit `a64ff75`. Last CFPB state: `main@3336d87`. Reason: the professor's feedback asked for graph + tabular datasets usable by both df-analyze and a GNN, and for combining the two (GNN representations as df-analyze features).
- **2026-10-07:** YelpChi is split by user (`net_rur` connected components), and Amazon by exact-duplicate feature groups. Amazon nodes 0–3,304 are unlabelled context only.
- **2026-10-07:** For graph-derived features, the context rule is that features (never labels) of all nodes may be used, including test nodes. This replaces the CFPB rule that test nodes attach only to training nodes.
- **2026-10-07:** This branch implements Model 1 (df-analyze on own features) robustly and prepares the node, graph and feature-block interfaces. GNN code is deferred.
- **2026-10-08 (M8 pilot, Amazon, `htune_trials=10`, CPU only, no GPU):** every check passed (counts, grouped split, df-analyze's split and column types). df-analyze took 19 min 41 s, of which tuning was 15 min and GANDALF 9 min of that. `select_model` took 4 min 27 s; the whole pilot took 24 min. Model A was LightGBM with embedded (linear) feature selection: shared-CV PR-AUC 0.924 ± 0.030. Test PR-AUC 0.920 [95% group-bootstrap 0.893, 0.943] and AUROC 0.981 [0.973, 0.988], against a no-skill PR-AUC of 0.095. That is inside the sanity band (≈ 0.91). The full run (100 trials) and any YelpChi run still need the group's go-ahead (given the same day; see next entry).
- **2026-10-08 (decisions on §10.1–3, after the pilot):**
  - **Amazon:** run the full Model 1 configuration (100 trials) next.
  - **YelpChi:** use the full training set (≈ 27.6k nodes). YelpChi serves to validate the pipeline, so it should be run only a few times, not for every change. Plan v2 calls YelpChi the main dataset; whether Amazon takes that role in the ladder's contrasts is for the group to confirm.
  - **Threshold:** max F1 on out-of-fold training predictions.
  - **Split repeats:** stay at 1 for Model 1. Enable 3 for Amazon once Model 2 exists, so the compared rungs share the same splits.
  - **Compute:** the long runs happen locally, on the group's GPU machine, in a new session (§12).
- **2026-10-08 (YelpChi smoke test, CPU only):** a pipeline check, not a result: LightGBM and logistic regression only, 3 trials each.
  - Every stage passed on the full YelpChi split (27,571 train / 18,383 test).
  - df-analyze kept all 32 features with our types, so no high-cardinality column was taken for an identifier.
  - Timings: 10 min in total (df-analyze 7.8 min, `select_model` 2 min).
  - Model A (LightGBM): test PR-AUC 0.831 [0.817, 0.845], AUROC 0.950; no-skill 0.145. This is above the sanity band (≈ 0.80).
- **2026-10-13 (phase 1 decisions, [`PHASE1_AMAZON_PLAN.md`](PHASE1_AMAZON_PLAN.md)):** Amazon only; split unchanged (`n_repeats: 1`); all 25 features kept pending [#36](https://github.com/caelenm/fraud-detect-gnn/issues/36); bundles keep every tuned model and drop caches; an `export_model` stage and `scripts/dfa/predict.py` added; the branch goes to `main` through a PR (never merged by an agent).
- **2026-10-13/14 (full Amazon run, Model 1, phase 1).** WSL2 (Debian) on a Windows laptop with an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, driver 571.96); torch 2.9.1+cu128 in df-analyze's environment; df-analyze `199e563`; seed 555; 100 trials; classifiers `lgbm, lr, catboost, gandalf, rf, knn` + dummy; `bal-acc`; `--filter-pred-classify auroc`; no wrapper selection.
  - **Checks:** download checksums matched (`Amazon.zip` c1b7a1f8…a5b9, `Amazon.mat` 4b7e3f9c…9d63). Every §2 count matched (11,944 nodes, 8,639 labelled, 821 positive, 25 features, U-P-U 175,608 / U-S-U 3,566,479 / U-V-U 1,036,737 edges; nodes 0–3,304 unlabelled). Split 5,183 train / 3,456 test (5,008 / 3,338 groups); no group crosses train/test or folds; 293 duplicate rows (175 within train, 118 within test, 0 across). The four shortcut flags reappeared (#36). df-analyze's exported split matched ours row by row, and it kept all 25 features with our types (none dropped or coerced). `model_selection_cv.csv` scored all 28 tuned combinations (and their default settings). No sanity-band warning.
  - **Wall-clock:** load 19 s, split 1 s, features and audit < 1 s each; df-analyze **1 h 57 min** (tuning 111.6 min: CatBoost 47.0, GANDALF 46.1, LightGBM RF 7.4, LightGBM 5.0, logistic regression 4.5, kNN 1.5, dummy 0.2); `select_model` 8 min 15 s; report and web report 4 s; `export_model` 15 s. Whole run 2 h 05 min. CatBoost and GANDALF stopped early after 51–66 trials; no search hit its time limit.
  - **Model A: LightGBM, feature selection `none`** (all 25 features). Shared-CV PR-AUC **0.9254 ± 0.0286** (5 grouped folds). Test PR-AUC **0.9188** [95% group-bootstrap 0.8902, 0.9421], AUROC **0.9765** [0.9665, 0.9855]; no-skill PR-AUC 0.0949 (dummy 0.0957). At the out-of-fold max-F1 threshold 0.4299: F1 0.871, precision 0.913, recall 0.832 (TP 273, FP 26, FN 55, TN 3,102). The runner-up by shared CV, CatBoost with embedded selection (0.9214 ± 0.0332), is within one SD. The 10-trial CPU pilot had chosen LightGBM with embedded selection (test PR-AUC 0.920); Model A's test PR-AUC is essentially unchanged.
  - **Saved models (`export_model`).** df-analyze's `eval_htune_results_jsonpickle_00.json` does not decode with jsonpickle as written: GANDALF's training state cannot be rebuilt (a `torch.dtype` cannot be created; the torch optimizer and Lightning internals fail in `__setstate__`), which stops the whole decode. Skipping those failures makes jsonpickle's object numbering drift, so references in every model saved after GANDALF (kNN, LightGBM, LR, RF) point at wrong objects. `scripts/dfa/model_io.py` now decodes with a tolerant unpickler that turns such references into explicit placeholders, and restores LightGBM's two shared references from their first copies. **17 of 28 combinations** load intact and reproduce df-analyze's saved test probabilities and labels exactly (max difference 0): CatBoost ×4, kNN ×4, LightGBM ×4, LightGBM RF ×4, logistic regression (`none`). Not usable from the file: GANDALF ×4 (cannot be rebuilt), dummy ×4 (random strategy: its predictions differ on every call), logistic regression ×3 (`n_iter_` cannot be rebuilt). None of these was refit. Model A decoded intact and was saved as `model_a.joblib`; reloaded, it reproduces the test probabilities to 0 and the labels exactly.
  - **Bundle.** `pack_artifacts.py --dataset amazon`: 89 files, 54 MB, **25 MB** compressed. The largest files: `eval_htune_results_jsonpickle_00.json` 23 MB (every tuned model), `prediction_results_00.json` 8.7 MB, the graph edge lists 15 MB (`homo` 7.2 MB, `usu` 5.9 MB, `uvu` 1.9 MB, `upu` 0.3 MB), `oof_predictions.parquet` 1.9 MB, `df_analyze.log` 1.0 MB; everything else is under 1 MB (Model A's `model_a.joblib` 59 KB). df-analyze 4.1.0 wrote no `__JOBLIB_CACHE__` in this run (its use is commented out at the pinned commit), so leaving out caches and temporary files changed nothing here: the bundle was 25 MB before and after.
  - **Round trip as a teammate (Task 6).** A fresh clone in WSL2 that only unpacked the bundle: all 89 checksums verified; `uv run run.py --dataset amazon` skipped every stage, `download` included, with no raw data present; `predict.py` reproduced Model A's saved test probabilities (max difference 1.1e-16) and CatBoost (`embed_linear`) via `--model`, also with the GPU hidden (`CUDA_VISIBLE_DEVICES=''`).
- **2026-10-18 (phase 2 decisions, [`PHASE2_GNN_PLAN.md`](PHASE2_GNN_PLAN.md), decisions 3–6 and 9–10):**
  - **One block per kind, covering every relation** (`nbr`, `gnnssl`; columns `nbr__upu_mean_f00`, `gnnssl__upu_pc01`), so feature-set names are the same on YelpChi and Amazon. df-analyze's names (`nbr_upu_mean_f00`) have no collisions on either dataset (tested).
  - **`nbr` (Model 2):** per relation (not `homo`), the 1-hop mean of every own feature and log1p(degree): 78 columns on Amazon. 1-hop max and 2-hop means are later ablations.
  - **`gnnssl` (Model 5):** Deep Graph Infomax per relation. Encoder: two rounds of GCN-normalised propagation over a sparse A + I, hidden 64, output 64, PReLU. Corruption: row shuffle. Adam lr 1e-3, at most 300 epochs, early stopping on the DGI loss (patience 20), seeded. PCA to 8 per relation: 24 columns.
  - **Train-only fitting and the loss mask:** the scaler that feeds DGI and every PCA are fit on training nodes only. The DGI loss and its summary vector cover non-test nodes (train and unlabelled); test nodes take part in message passing only (invariant 5) and add no loss term. Both masks come from the `split` column, never the labels.
  - **Pilot without touching phase 1:** `pilot_` feature sets, `df_analyze.classifiers=[lgbm,lr,catboost]`, `htune_trials=10`, all on the same machine. The 100-trial `m1_own` run is never rerun.
  - **Comparison page:** a standalone script (`scripts/compare_report.py`), not a stage. It may use inline JavaScript but makes no external requests, and it has a group-only banner.
- **2026-10-19 (user decisions):**
  - **Issue #36, report both:** every Amazon rung also runs without `own__f13`, `own__f15`, `own__f18`, `own__f19` (`*_no36`). The four are left out of the df-analyze input and of the `nbr_no36` / `gnnssl_no36` inputs.
  - **GNN embeddings on the CPU:** on the GPU, two runs differed by ~1e-7 per training step (also with `torch.use_deterministic_algorithms`); CPU runs are bit-identical.
  - **torch 2.9** in the `gnn` extra: PyPI's newest torch (2.14.1) is built for CUDA 13 and could not see the GPU under the laptop's driver (CUDA 12.8).
- **2026-10-19 (phase 2 build facts, Amazon, WSL2, RTX 4060 laptop, torch 2.9.1+cu128, PyG 2.8.1):**
  - **`nbr`:** 11,944 × 78, no nulls, all continuous on the 5,183 training nodes; under 1 s. `nbr_no36`: 66 columns from 21 own features. U-P-U leaves 1,720 nodes isolated (mean 0); median degrees U-P-U 6, U-S-U 264, U-V-U 90.
  - **`gnnssl` on the CPU** (34 s for the stage): U-P-U 246 epochs (best 226, loss 0.775, 4.0 s), U-S-U 300 (best 285, 0.047, 20.6 s), U-V-U 263 (best 243, 0.054, 7.9 s). PCA variance kept by 8 components: 93.6%, 99.7%, 98.9%. U-S-U hit the 300-epoch cap. 24 continuous columns.
  - **`gnnssl_no36` on the CPU** (22 s): U-P-U 246 epochs (best 226), U-S-U 112 (best 92), U-V-U 300 (best 291, cap). PCA variance 94.6%, 99.8%, 98.2%.
  - An earlier GPU build (stage 9 s; U-P-U 1.5 s, U-S-U 3.6 s, U-V-U 2.8 s) was replaced by the CPU build.
  - **Probe** (`scripts/probe_blocks.py`; fixed untuned HistGradientBoosting, 5 saved grouped CV folds, training nodes only; a signal, not a result):

    | Blocks | Features | CV PR-AUC (mean ± SD) | CV AUROC |
    |---|---|---|---|
    | own | 25 | 0.907 ± 0.028 | 0.977 |
    | own + nbr | 103 | 0.935 ± 0.029 | 0.986 |
    | own + gnnssl | 49 | 0.934 ± 0.030 | 0.988 |
    | own_no36 | 21 | 0.500 ± 0.043 | 0.879 |
    | own_no36 + nbr_no36 | 87 | 0.704 ± 0.034 | 0.928 |
    | own_no36 + gnnssl_no36 | 45 | 0.672 ± 0.046 | 0.929 |

    Without the four #36 features, own-only PR-AUC falls by about 0.41. Both graph blocks recover about 0.17–0.20 of it, much more than the about 0.03 they add with all features.

- **2026-10-19 (phase 2 PILOT, Amazon).** This is a pilot, not a final result: 10 trials; classifiers `lgbm, lr, catboost` plus the dummy; one grouped split; seed 555; all six runs sequentially on the same machine (WSL2, RTX 4060 laptop GPU 8 GB, CatBoost on the GPU, embeddings on the CPU); df-analyze `199e563`.
  - **Checks:** for every run, `split_check.json` matched our split row for row (5,183 / 3,456; labels identical) and `type_check.json` had all column types matching, with nothing dropped or coerced. The `_no36` runs print the sanity-band warning, which is expected because the band describes all 25 features. No search hit its time limit (12 "all trials" and 4 "early stop" per run).
  - **Wall-clock:** `pilot_m1_own` 12.8 min, `_no36` 10.6; `pilot_m2_own_nbr` 16.0, `_no36` 15.3; `pilot_m5_own_gnnssl` 14.6, `_no36` 15.6; 85 min in total. `compare_report.py`: 22 s per page, 3.1–4.3 MB.

    | Feature set | Features | Model A (selection; columns kept) | Shared-CV PR-AUC | Test PR-AUC [95% CI] | Test AUROC [95% CI] |
    |---|---|---|---|---|---|
    | pilot_m1_own | 25 | LightGBM (embed linear; 25) | 0.921 ± 0.035 | 0.917 [0.888, 0.941] | 0.979 [0.970, 0.987] |
    | pilot_m2_own_nbr | 103 | CatBoost (embed linear; 24: 11 own, 13 nbr from all three relations) | 0.939 ± 0.029 | 0.926 [0.901, 0.946] | 0.984 [0.978, 0.990] |
    | pilot_m5_own_gnnssl | 49 | LightGBM (embed linear; 49) | 0.938 ± 0.030 | 0.929 [0.905, 0.949] | 0.981 [0.972, 0.989] |
    | pilot_m1_own_no36 | 21 | LightGBM (assoc; 21) | 0.567 ± 0.042 | 0.524 [0.470, 0.583] | 0.893 [0.873, 0.910] |
    | pilot_m2_own_nbr_no36 | 87 | CatBoost (assoc; 87) | 0.714 ± 0.045 | 0.710 [0.663, 0.754] | 0.935 [0.920, 0.948] |
    | pilot_m5_own_gnnssl_no36 | 45 | LightGBM (embed linear; 45) | 0.684 ± 0.045 | 0.677 [0.626, 0.726] | 0.927 [0.912, 0.941] |

  - **Paired contrasts** (`compare_metrics.json`; B − A on the same test nodes, 2,000 group-bootstrap resamples, seed 555):

    | Contrast | ΔPR-AUC [95% CI] | B ahead | ΔAUROC [95% CI] | Recall@K A → B | Fraud: B rescues / breaks |
    |---|---|---|---|---|---|
    | m2 − m1 | +0.0089 [−0.0009, +0.0201] | 96.1% | +0.0051 [−0.0001, +0.0111] | 0.872 → 0.857 | 9 / 7 |
    | m5 − m1 | +0.0115 [+0.0001, +0.0251] | 97.6% | +0.0018 [−0.0029, +0.0064] | 0.872 → 0.860 | 6 / 12 |
    | m5 − m2 | +0.0027 [−0.0063, +0.0121] | 70.7% | −0.0033 [−0.0085, +0.0011] | 0.857 → 0.860 | 2 / 10 |
    | m2 − m1, no36 | +0.1855 [+0.1424, +0.2254] | 100% | +0.0422 [+0.0307, +0.0547] | 0.558 → 0.646 | 18 / 28 |
    | m5 − m1, no36 | +0.1529 [+0.1037, +0.1956] | 100% | +0.0346 [+0.0232, +0.0474] | 0.558 → 0.625 | 25 / 30 |
    | m5 − m2, no36 | −0.0326 [−0.0621, −0.0043] | 1.0% | −0.0076 [−0.0154, −0.0001] | 0.646 → 0.625 | 23 / 18 |

  - **Reading:**
    - With all features, the graph rungs add about one PR-AUC point. That is borderline on one split at a pilot budget: m2's interval includes 0, and m5's lower bound is +0.0001.
    - Without the #36 features, own features fall to 0.524, and both graph rungs add 0.15–0.19 with intervals well above 0. There, untuned DGI embeddings are worse than neighbour means.
    - The #36 features account for about 0.39 of Model 1's test PR-AUC. At each side's own threshold the differences are small and mixed: gains come from ranking, not from more frauds caught.
    - The pilot `m1` (0.917) is close to the 100-trial `m1_own` (0.919). Never compare a pilot rung with the 100-trial run.
- **2026-10-19 (issue #36: the full Model A without the four features).** The issue asks for Model A rerun "on the same split" without the four features. This is that run: `m1_own_no36`, the same configuration as `m1_own` (6 classifiers, 100 trials, `bal-acc`, seed 555, split 0), on the same machine, run alone.
  - **Checks:** df-analyze's exported split matched ours row by row; it kept all 21 features with our types. No search hit its time limit (24 "early stop", 4 "all trials").
  - **Wall-clock:** df-analyze 1 h 02 min, `select_model` 6 min; 1 h 09 min in total. The 25-feature run took 2 h 05 min, mostly because CatBoost tuned faster here.
  - **Model A:** LightGBM, embedded (linear) selection. Shared-CV PR-AUC 0.570 ± 0.043.
  - **Test:** PR-AUC **0.518** [0.465, 0.576] and AUROC 0.889 [0.868, 0.907]. At the out-of-fold threshold 0.234: precision 0.471, recall 0.677, F1 0.556 (TP 222, FP 249, FN 106).
  - **Versus `m1_own` (0.919)** on the same test users: paired ΔPR-AUC **−0.401** [−0.450, −0.347]; 0% of 2,000 group resamples had the 21-feature model ahead (`compare/m1_own__vs__m1_own_no36/`).
  - The pilot-budget version gave 0.524, so the budget barely matters here.
  - The sanity-band warning is expected; the band describes all 25 features.
  - `export_model`: 17 of 28 combinations load and reproduce their test probabilities (the same pattern as `m1_own`: GANDALF, the dummy and three logistic regressions do not). Model A loads.
  - **Training-only single-feature check** (`probe/shortcut_check.csv`): best single-threshold balanced accuracy `f13` 0.835, `f15` 0.842, `f18` 0.844, `f19` 0.893. `f19` alone reaches PR-AUC 0.80; it is a 0–199 integer, higher means fraud.
  - **Still open for #36:** what the four features are (Zhang et al., SIGIR 2020; the CARE-GNN preprocessing).
- **2026-10-19 (issue #36: literature check of what the four features are).** A separate research session read the primary sources. **Sources state:**
  - the label rule (> 80% helpful votes = benign, < 20% = fraudulent; CARE-GNN §4.1.1);
  - that the 25 features come from Zhang et al. (SIGIR 2020, arXiv:2005.10150), whose Table 1 lists feature *groups*, among them 12 helpful/unhelpful-vote statistics (totals, ratios, means, median/min/max). Expanded literally the groups give 36 scalars, so CARE-GNN used a subset and the table cannot give the column order;
  - that Zhang et al. used Movies & TV (labels at 0.7 / 0.3, ≥ 20 votes), not Musical Instruments, so CARE-GNN recomputed the features itself;
  - that no column order is published: CARE-GNN's README and paper, issue #2 (asked, unanswered for Amazon), DGL and GADBench all say only "25 handcrafted features". The repository's `amazon_preprocess.py` could not be opened from that session.

  **Inferred from our distribution statistics** (confidence in brackets):
  - `f19`: a helpfulness-vote count, most likely total unhelpful votes (medium) → leakage by construction;
  - `f15`: median rating, since 9 distinct values = {1, 1.5, …, 5} (medium-high);
  - `f18`: mean rating (medium);
  - `f13`: share of 4–5 star ratings (medium-low);
  - other 21 columns: unidentified, and more vote-derived columns are likely among them.

  A direct helpful-ratio column would reach AUROC ≈ 1.0 on labelled users, so none of the four is the ratio itself.

  **Recommended reading** (not yet decided by the group): don't headline 0.919 (likely leakage-inflated); report 0.518 as a conservative interim figure; the headline should be a run without every column shown to be vote-derived, which keeps the rating features. The checks that settle it are in §12. Published Amazon results built on these 25 features would inherit the same issue; the research cites CARE-GNN's own AUCs (Dou et al. 2020, Table 3) as an example, not verified here.
- **2026-10-20 (overnight validation; only checks planned before any of their results).** Branch `exp/validation-repeats`. 26 steps, none failed, 187 min, one job at a time (01:40–04:48 UTC).
  - **Split repeats 1 and 2 of the six pilot rungs:** 11 runs, mean 16 min each. Every run (18 with split 0) passed the split check (5,183 / 3,456, identical labels) and the type check (nothing dropped or coerced); no search hit its time limit.
  - **Model A per split** (test PR-AUC):

    | Rung | Split 0 | Split 1 | Split 2 |
    |---|---|---|---|
    | m1 | 0.917 | 0.919 | 0.919 |
    | m2 | 0.926 | 0.938 | 0.923 |
    | m5 | 0.929 | 0.931 | 0.928 |
    | m1_no36 | 0.524 | 0.545 | 0.574 |
    | m2_no36 | 0.710 | 0.711 | 0.709 |
    | m5_no36 | 0.677 | 0.688 | 0.665 |

  - **Verdicts under the pre-registered rule** (`compare/repeat_summary.md`; Δ mean ± SD across splits; Nadeau–Bengio corrected paired t-test over 15 grouped CV folds, training data only):
    - all features: m2 − m1 +0.0104 ± 0.0079, p = 0.013, **supported**; m5 − m1 +0.0106 ± 0.0017, p = 0.090, not distinguishable from zero; m5 − m2 +0.0002 ± 0.0066, p = 0.55, not distinguishable;
    - without #36: m2 − m1 +0.162 ± 0.026, p = 6e-7, **supported**; m5 − m1 +0.129 ± 0.034, p = 6e-6, **supported**; m5 − m2 −0.033 ± 0.010, p = 0.002, **worse**.
  - **Caveats:**
    - Each Model A was chosen on the same CV folds the t-test uses, which makes both sides' CV scores slightly optimistic.
    - The all-features m2 gain is "supported" as a gain from the neighbour-aggregate columns. The random-graph control shows most of it survives rewiring, so it is not evidence of real graph structure.
  - **Issue #36 identity checks** (`scripts/feature_identity_checks.py`, §12 step 0 a–b; training rows; `probe/feature_identity.md`):
    - (a) **negative:** no column c makes c / (c + f19) reproduce the label rule (best AUROC 0.906, `f22`). The 25 features do not contain a helpful/unhelpful vote pair; f19 may still be a vote count whose partner column is absent.
    - (b) **consistent:** f15 takes exactly the 9 half-star values 1–5, f18 ∈ [1, 5], f13 ∈ [0, 1], Spearman(f15, f18) = 0.908. That fits median rating, mean rating and a share; it is not proof.
  - **Embedding sensitivity** (training CV probe; 3 sizes × 2 epoch caps × 3 PCA sizes; a check only, the runs keep the defaults):
    - all features: default 0.934, grid 0.927–0.940 (median 0.933);
    - without #36: default 0.672, grid 0.653–0.694 (median 0.679).

    The default sits mid-grid; tuning could add about 0.01–0.02 on CV, and choosing settings this way is a phase-3 step on training data.
- **2026-10-19 (issue #36: how published work handles these features; second research pass, not exhaustive).**
  - **All 25 features everywhere:** every paper found uses all 25 features (CARE-GNN, DIGNN, HA-GNN, a generative semi-supervised GAD paper; DGL and GADBench list 25). None drops columns or raises leakage. One noise-robustness paper (arXiv:2505.00946) lists Amazon with 24 dimensions; typo or dropped column, unverified.
  - **Fraud rate convention:** 6.87% = 821/11,944, counting unlabelled users as non-fraud; 9.5% = 821/8,639, labelled users only, which is our convention.
  - **Features vs graph:** BWGNN notes that an SVM beats many GNNs on Amazon, and another paper says Amazon's anomalies show mainly in raw features, with low correlation to the graph. Both fit a feature shortcut, though neither gives a reason. A CS224W blog post reports the opposite.
  - **What the label means:** GADBench describes the label as users paid to write fake reviews; CARE-GNN and DGL define it by helpfulness votes. So "fraud" here means "downvoted".
  - **Consequence:** published Amazon numbers (e.g. PC-GNN's AUROC 95.86) come from all 25 features and are comparable only to our all-features runs, not to the without-#36 ones.
  - **Recommended reporting:** the all-features result for comparability with the literature; a leak-controlled result as the one we trust, once the vote columns are identified (§12, step 0).
- **2026-10-19 (random-graph control, training CV only, split 0; `probe/random_graph_control.csv`).** Each relation's edges were rewired at random, keeping degrees approximately (85–92% of edges kept after dropping self-loops and repeats); nbr and gnnssl were rebuilt and probed.
  - **All features:** own 0.907; + nbr: real graph 0.935, random 0.932; + gnnssl: real 0.934, random 0.927. The small gain mostly does not need the real edges.
  - **Without #36:** own 0.500; + nbr: real 0.704, random 0.613; + gnnssl: real 0.672, random 0.590. About half the gain needs the real edges; the rest comes from degree and the extra columns alone.
  - The embedding sensitivity grid was stopped to free the CPU for the runs and was not completed. Split repeats 1–2 were paused after one finished run (`pilot_m1_own`, split 1) and remain "later"; they can be continued with `--resume`.
- **2026-10-19 (progress-report runs: decision rules written BEFORE the runs).** The user approved a plan of under 3 h for defensible initial results. Fixed in advance; nothing here is chosen from test results.
  - **Runs:** the six pilot rungs (`pilot_m1_own`, `pilot_m2_own_nbr`, `pilot_m5_own_gnnssl` and their `_no36` versions) on split repeats 1 and 2 (`split.n_repeats: 3`; split seed = 555 + 1000 × repeat), with the same pilot budget and settings as repeat 0. The two repeats run in parallel on the same machine; no tuning search came near df-analyze's time limit.
  - **Embedding settings stay at the plan's defaults,** which were fixed before any result. A settings grid is run only as a training-CV sensitivity check and changes nothing.
  - **Contrasts:** m2 − m1, m5 − m1 and m5 − m2, for both variants.
  - **Verdict rule:** a gain is **supported** if all three hold:
    1. test ΔPR-AUC > 0 on all 3 splits;
    2. the mean Δ across splits exceeds its SD across splits;
    3. the Nadeau–Bengio corrected paired t-test over the 15 CV folds (3 splits × 5 grouped folds; training data only; per-fold PR-AUC of each side's Model A from its out-of-fold predictions) gives two-sided p < 0.05.

    Otherwise a positive mean Δ is reported as **not distinguishable from zero**, and a negative one as **worse**.
  - **Controls (training CV only, the probe):**
    - **Random-graph control:** each relation's edges are rewired at random, keeping degrees approximately, and nbr/gnnssl are rebuilt from them. If the graph gain comes from real structure, it should mostly vanish.
    - **Single-feature #36 check:** how well each of `own__f13`, `own__f15`, `own__f18`, `own__f19` alone separates the training classes.
  - **Reporting:** both variants always side by side. Model A choice, thresholds and every setting come from training data only.
- **2026-10-21 (professor's feedback and user decisions).**
  - **Feedback:** "Find out size of gnn embeddings > Potentially use DEEPTUNE to reduce. See if training GNN helps. Run on just 4 important features. More datasets." The plan is [`PHASE3_FEEDBACK_PLAN.md`](PHASE3_FEEDBACK_PLAN.md); its readings of the feedback are for the group to confirm (D1–D3).
  - **DeepTune** (`moayadeldin/deeptune` at `80e6d74`, read 2026-10-21): the StFX MIB lab's deep-learning AutoML tool for images, text, tabular data (GANDALF, TabPFN) and time series; no graph support. Its embeddings come from models trained on the label, so as df-analyze features they need the Model 6 amendment; its autoencoder is image-only. It pins torch 2.14.0 (CUDA 13), so it would need its own environment. No license file found.
  - **Git:** no PR requirement; commits may go straight to `main` and need not pass the checks first. The never-commit list and the no-force-push rule stay.
  - **Results database:** `results/runs/<run_id>/` (`meta.json` + `results.csv`), written by `scripts/results_db.py`; every agent records the runs it finishes. Created empty: none of the runs above is in it yet (D5).
- **2026-10-23 (phase 3 overnight: training-only probes, then pilot rungs; rules written before any pilot result).** Branch `exp/phase3-feedback`, on top of the phase 3 plan's branch (`claude/awesome-wozniak-wm548a`). Both branches were merged into `main` with PR #39 (`f4175e3`). The user asked for "the next set of experiments" overnight, which is the plan's "go" for the pilot rungs of Tasks 3a and 5. Split 0's training nodes and saved grouped folds only, unless stated; all probes use the fixed HistGradientBoosting of `models/probe.py`.
  - **Task 5.1, which four features (`scripts/top_features.py`, `probe/top_features.csv`).** Permutation importance (drop in held-out-fold PR-AUC, 10 shuffles, mean over 5 folds): `f19` 0.759 ± 0.036, `f00` 0.074 ± 0.014, `f23` 0.010 ± 0.010, `f18` 0.007 ± 0.006, then `f12` 0.007 ± 0.005 (ranks 3–5 are within noise). `f13` and `f15` have about zero importance: once `f18` and `f19` are known they add nothing. By univariate AUROC the top 4 are the four #36 features. **So D3's two readings differ on Amazon.** Both are run as pilots, so the group can choose without another night: `only36` (the four #36 features) and `top4` (`f00`, `f18`, `f19`, `f23`, the permutation top 4, fixed now from this table).
  - **Task 5.3 probe (`probe/only36_probe.csv`):** own / + nbr / + gnnssl: all 25 features 0.907 / 0.935 / 0.934; without #36 0.500 / 0.704 / 0.672; only36 0.876 / 0.918 / 0.912; top4 0.910 / 0.938 / 0.931. Four features carry the 25 features' result.
  - **Task 3a probe (`probe/gnnrand_probe.csv`):** own + gnnrand (the gnnssl encoder with its seeded initial weights, no training) over 5 initialisation seeds: all features 0.933 ± 0.004 (gnnssl 0.934, nbr 0.935); without #36 0.677 ± 0.007 (gnnssl 0.672, nbr 0.704). On CV, DGI training adds nothing over random weights.
  - **Task 1.1 (`scripts/embedding_sizes.py`, `gnn_embed/sizes.md`):** DGI gives 64 dimensions per relation (192 on Amazon); none is constant on the training nodes, but the effective rank (entropy of the singular values) is 27 / 8 / 15 (U-P-U / U-S-U / U-V-U; without #36 26 / 6 / 17). 90% of the variance needs 6 / 2 / 3 components, 99% needs 27 / 5 / 9. df-analyze receives 24 `gnnssl` columns (PCA 8 per relation) and both pilot `m5` Model As kept all 24 after feature selection; pilot `m2`'s Model A kept 13 of 78 `nbr` columns (all 66 without #36).
  - **Runs (pilot settings: lgbm, lr, catboost, 10 trials, `bal-acc`, seed 555; splits 0, 1, 2; one at a time on the RTX 4060 laptop):** `pilot_m1_own_only36`, `pilot_m2_own_nbr_only36`, `pilot_m5_own_gnnssl_only36`; `pilot_m5r_own_gnnrand`, `pilot_m5r_own_gnnrand_no36`; `pilot_m1_own_top4`, `pilot_m2_own_nbr_top4`, `pilot_m5_own_gnnssl_top4`. 24 runs. Every graph block keeps the plan's default settings (DGI 64, ≤ 300 epochs, PCA 8); gnnrand uses the run seed, so it is exactly the encoder gnnssl's training starts from. The size sweep (Task 1.2) is a training-CV check and changes no run.
  - **Contrasts** (B − A; test ΔPR-AUC per split from `compare_report.py`, verdict by the rule of 2026-10-19: Δ > 0 on all 3 splits, mean Δ > SD, Nadeau–Bengio p < 0.05 over the 15 CV folds; `scripts/repeat_summary.py --pairs`):
    - Task 3a (`compare/gnnrand_summary.md`): m5 − m5r (does DGI training add anything over random weights?) and m5r − m1, each with all features and without #36.
    - Task 5 (`compare/only36_summary.md`, `compare/top4_summary.md`): m1_only36 − m1 and m1_top4 − m1 (how much of the 25 features' result do four carry?), and within each variant m2 − m1, m5 − m1, m5 − m2.
  - **Reading, fixed now:** "worse" for m1_only36 − m1 means the other 21 features add something measurable; "not distinguishable" means the four carry the result. For m5 − m5r, anything but "supported" means DGI training is not shown to help. No pilot result is compared with a full-budget run.
- **2026-10-23 (phase 3 overnight: results).** Queue 04:40–09:32 UTC, one run at a time, 24 runs of 8.3–16.0 min. Every run passed the split check (5,183 / 3,456) and the type check, and no tuning search hit its time limit. The first run (`pilot_m1_own_only36`, split 0) finished at 04:49, 9 minutes before the rules above were committed (`1f23671`); its result was not read before then. Run IDs are in the results database (`results/runs/20261023T04*` to `T09*`, tag `phase3`).
  - **Model A test PR-AUC, splits 0 / 1 / 2** (pilot budget; no-skill 0.095):

    | Rung | Features | Split 0 | Split 1 | Split 2 |
    |---|---|---|---|---|
    | m1 (phase 2) | 25 | 0.917 | 0.919 | 0.919 |
    | m1_only36 | 4 (#36) | 0.885 | 0.888 | 0.889 |
    | m2_only36 | 4 + 15 nbr | 0.909 | 0.917 | 0.917 |
    | m5_only36 | 4 + 24 gnnssl | 0.907 | 0.913 | 0.906 |
    | m1_top4 | 4 (top 4) | 0.924 | 0.919 | 0.922 |
    | m2_top4 | 4 + 15 nbr | 0.931 | 0.938 | 0.933 |
    | m5_top4 | 4 + 24 gnnssl | 0.944 | 0.941 | 0.943 |
    | m5 (phase 2) | 25 + 24 gnnssl | 0.929 | 0.931 | 0.928 |
    | m5r | 25 + 24 gnnrand | 0.936 | 0.932 | 0.930 |
    | m5_no36 (phase 2) | 21 + 24 gnnssl | 0.677 | 0.688 | 0.665 |
    | m5r_no36 | 21 + 24 gnnrand | 0.656 | 0.679 | 0.659 |

  - **Verdicts** (rule of 2026-10-19; `compare/only36_summary.md`, `gnnrand_summary.md`, `top4_summary.md`; results-DB entries `20261023T062119Z_amazon_11445b`, `20261023T074837Z_amazon_f1ed69`, `20261023T092759Z_amazon_f8430a`). The rule labels any negative mean "worse"; the p-value says whether it is also significant.

    | Contrast | Δ splits 0 / 1 / 2 | Mean ± SD | CV p | Verdict |
    |---|---|---|---|---|
    | m5 − m5r (DGI training), all features | −0.007 / −0.002 / −0.002 | −0.004 ± 0.003 | 0.69 | worse (not significant) |
    | m5 − m5r, without #36 | +0.021 / +0.009 / +0.005 | +0.012 ± 0.008 | 0.69 | not distinguishable from zero |
    | m5r − m1, all features | +0.019 / +0.013 / +0.010 | +0.014 ± 0.004 | 0.028 | supported |
    | m5r − m1, without #36 | +0.132 / +0.134 / +0.085 | +0.117 ± 0.028 | 5e-5 | supported |
    | m1_only36 − m1 | −0.032 / −0.031 / −0.030 | −0.031 ± 0.001 | 1e-4 | worse |
    | m2 − m1, only36 | +0.024 / +0.029 / +0.028 | +0.027 ± 0.003 | 3e-4 | supported |
    | m5 − m1, only36 | +0.022 / +0.025 / +0.017 | +0.021 ± 0.004 | 0.006 | supported |
    | m5 − m2, only36 | −0.002 / −0.004 / −0.011 | −0.006 ± 0.005 | 0.64 | worse (not significant) |
    | m1_top4 − m1 | +0.006 / +0.001 / +0.003 | +0.003 ± 0.003 | 0.67 | not distinguishable from zero |
    | m2 − m1, top4 | +0.007 / +0.019 / +0.011 | +0.012 ± 0.006 | 0.036 | supported |
    | m5 − m1, top4 | +0.020 / +0.022 / +0.021 | +0.021 ± 0.001 | 0.095 | not distinguishable from zero |
    | m5 − m2, top4 | +0.013 / +0.003 / +0.010 | +0.009 ± 0.005 | 0.89 | not distinguishable from zero |

  - **Reading:**
    - **F2 (a), does training the GNN help?** No, not measurably. Untrained random-weight embeddings do as well as DGI-trained ones with all features, and about 0.01 worse without #36, which is not significant. Untrained embeddings alone give the gain over m1 that passes the rule (+0.014 with all features, where DGI's +0.011 did not pass; +0.117 without #36). Most of what the GNN embeddings add comes from propagating the features over the graph, not from DGI's training.
    - **F3, four features.** The four #36 features alone lose 0.031 against all 25 on every split (0.887 vs 0.918). The permutation top 4 (`f00`, `f18`, `f19`, `f23`) match all 25 (+0.003, not distinguishable). So four features carry Amazon's result, and the right four are not the four flagged in #36. Within either four-feature variant the graph still adds 0.01–0.03; `m5_top4` (0.941–0.944) is the highest pilot score so far, but its contrast with `m1_top4` is not significant on CV (p = 0.095).
    - **F1, size** (Task 1, `gnn_embed/sizes.md`, size sweep `20261023T044927Z_amazon_43983f`): see the 2026-10-23 entry above. At DGI size 64, the pre-registered rule picks k = 1 per relation with all features and k = 16 without #36 (8 falls 0.002 short of the band). DGI sizes 16 and 32 do about as well as 64. Trained embeddings have a lower effective rank than untrained ones (8–27 vs 18–32 per relation): training compresses them without making them more useful to df-analyze.
  - **Caveats:**
    - Pilot budget, one seed per run. Each side's Model A was chosen on the same CV folds the t-test uses.
    - **`top4` was chosen on split 0's training folds.** Splits 1 and 2 put some of those nodes in their test sets, so their `top4` results are mildly optimistic. Re-ranked on their own training folds (`outputs/amazon_repeat{1,2}/probe/top_features.csv`), splits 1 and 2 keep `f19`, `f00` and `f23` but pick `f12` for the fourth place instead of `f18` (ranks 3–5 are within noise). Split 0's contrast (+0.006 for m1_top4 − m1) has no such issue. The same caveat applies, more weakly, to the #36 set, flagged by the split-0 audit.
- **2026-10-26 (decisions by Caelen; the group is to confirm the framing).** Branch `exp/rq-plan`, committed locally, not pushed.
  - **Research questions.** RQ1 (headline): how much of Amazon's published performance, and of the graph's apparent benefit, depends on the leaked vote column `f19`? RQ2: once leaks are removed, how much of the remaining graph gain is real relational structure, and how much is degree and extra columns? RQ3: does training the GNN encoder matter when its embeddings feed df-analyze, and how small can those embeddings be?
  - **Datasets:** Amazon main, YelpChi validation, nothing else (settles D4).
  - **Feature variants on Amazon going forward:** all 25, and `no19` (all except `own__f19`, also left out of every neighbour and GNN input) as the candidate headline. The `no36`, `only36` and `top4` runs stay in the report as answered questions.
  - **Experiment IDs.** B is the baseline, df-analyze on own features (Model 1), at the pilot budget on splits 0–2. Every experiment changes one thing relative to B:
    - **F (feature-set checks, no graph):** F1 without the #36 four, F2 the #36 four only, F3 top 4 (all done on 2026-10-19 and 2026-10-23); **F4** without `f19` (`B_no19 − B_all`); **F5** the top 4 of the 24 non-`f19` columns, ranked on each split's own training folds (`B_top4_no19 − B_no19`).
    - **E (graph experiments, variant fixed):** **E1** + neighbour means (`nbr`); **E2** + DGI embeddings (`gnnssl`); **E3** + untrained embeddings (`gnnrand`); E4 an autoencoder instead of PCA; E5 heterogeneous GraphSAGE (Model 4); E6 GraphSAGE without edges (Model 3).
    - **N (null-graph controls):** **N1** + log1p(degree) per relation only (`deg`); N2 + neighbour means on a degree-preserving rewired graph; N3 GraphSAGE on the rewired graph.
    - **Z1:** E2 and E3 at d ∈ {2, 4, 8, 16, 32, 64} and raw, with effective rank (training-CV probe).
    - Others (V1 rebuilding `f19` from the raw reviews, P1 GADBench, S1 an ungrouped YelpChi split, FB full budget) need a decision first.
  - **Out of scope until decided:** V1, `igraph` for N2, P1, S1, GraphSAGE (E5, E6, N3), E4, any YelpChi run, backfilling the results database.
- **2026-10-26 (issue #36 checks (c) and (d): the Amazon column map).**
  - **(d) CARE-GNN's script.** `amazon_preprocess.py` at `a64ff75` (Apache-2.0; SHA-256 `af235d9c…a7233`; read through the GitHub API, not cloned) builds 36 GraphRfi-style features per user (`build_features`, lines 230–271; `cal_votes`, lines 107–125). Columns 19–30 are vote statistics: 19–20 total helpful and unhelpful votes, 21–24 helpful and unhelpful ratio and mean, 25–30 the median, max and min of helpful votes, then of unhelpful votes. The saved file keeps:

    ```python
    # filter polluted features
    new_features = features[:, :19]
    new_features = np.hstack([new_features, features[:, 30:]])
    ```

    So the 25 columns are 0–18 and 30–35, and **`f19` is column 30, `min(unhelp)`: the smallest number of unhelpful votes on any one of the user's reviews.** The filter meant to drop the vote columns but is off by one. The label (lines 284–296) uses the same votes: users with ≥ 20 votes, helpful / total > 0.8 benign and < 0.2 fraud. One caveat: the script loads a saved `amz_features_36.npz` (line 316) instead of calling `build_features`, so the map is checked against our data below.
  - **(c) Identities on split 0's training users** (`scripts/feature_identity_checks.py`, `probe/column_map_check.csv`, results DB `20261026T144315Z_amazon_1d532a`): all hold for 5,183 / 5,183 users. f02 + … + f06 = f00 and f07 = f02 / f00 exactly; f07 + … + f11 = 1; f12 = f07 + f08 and f13 = f10 + f11 exactly; f14 = the natural-log entropy of f07–f11 within 1e-4 (stored to 4 decimals; log2 fails); f16 ≥ f15, f18 ≥ f17; f22 = 1 exactly when f20 = 0; f24 ∈ {−1, 0, 1}; f19 is a non-negative integer.
  - **Column map (confirmed by (c) and (d)):**

    | Column | Meaning (GraphRfi column) | Family |
    |---|---|---|
    | f00 | number of products the user rated (0) | activity |
    | f01 | length of the username (1) | profile |
    | f02–f06 | number of 1- to 5-star ratings (2–6) | rating |
    | f07–f11 | share of 1- to 5-star ratings (7–11) | rating |
    | f12, f13 | share of negative (1–2 star) and positive (4–5 star) ratings (12–13) | rating |
    | f14 | entropy of the rating shares (14) | rating |
    | f15–f18 | median, max, min and mean rating (15–18) | rating |
    | **f19** | **minimum unhelpful votes on any one review (30)** | **vote: the label is built from votes** |
    | f20–f22 | days between first and last review; time entropy; all reviews on one date (31–33) | time |
    | f23 | mean summary length (34) | text |
    | f24 | review sentiment, −1 / 0 / 1 (35) | text |

  - **Reading.** `f19` is label-derived by construction and is the only vote column left. `f13`, `f15` and `f18` are rating statistics: close in meaning to a "downvoted user" label, but not built from votes, so they are not leaks. `no19` is the leak-controlled variant; `no36` stays as the stricter one. What remains unproven without the raw reviews (V1): that `amz_features_36.npz` was built by this `build_features`; the identities make that very likely for columns 0–18 and 31–35, but none of them tests column 30 directly.
- **2026-10-26 (overnight queue: rules written BEFORE any of these runs).** Caelen's message of 2026-10-26 is the "go" for these runs only. One job at a time on the RTX 4060 laptop (`outputs/amazon/rq_logs/queue.txt`, `bin/runner.sh`); pilot settings as every earlier pilot (lgbm, lr, catboost; 10 trials; `bal-acc`; seed 555; splits 0, 1, 2 with `split.n_repeats: 3`).
  - **New inputs (branch `exp/rq-plan`):** the variant `no19` (`own__f19` left out of own, nbr and gnnssl/gnnrand inputs); the `deg` block (log1p(degree) per relation, 3 columns, built in the `neighbour_agg` stage; the same values as nbr's `log1p_degree` columns; label-blind and deterministic by test); the per-split variant `top4no19` (`keep_by_repeat`, resolved for the active split repeat).
  - **F5's four features, fixed now** (`scripts/top_features.py --variant no19 --repeat k`: the probe is fit on the 24 columns other than `f19`, permutation importance on each split's own training folds; `outputs/amazon*/probe/top_features_no19.csv`): split 0 `f07`, `f15`, `f18`, `f23`; split 1 `f15`, `f18`, `f23`, `f24`; split 2 `f13`, `f18`, `f23`, `f24`. Only `f18` and `f23` are in all three. Unlike 2026-10-23's `top4`, each split's list comes from its own training folds only.
  - **Z1 probe** (`scripts/feedback_probes.py --task z1 --variant no19`; split 0 training CV; recorded with `--kind probe`): own + gnnssl (E2, DGI size 64) and own + gnnrand (E3, 5 initialisation seeds) at d = 2, 4, 8, 16, 32, 64 PCA components per relation and raw, with each raw embedding's effective rank. The chosen d per block is the smallest d whose mean CV PR-AUC is within one standard error of the best d's (for gnnrand, on the mean over seeds). It changes no run tonight; the chosen d and raw 64 go to later pilot rungs (Step 2, item 5).
  - **Runs:** `pilot_m1_own_no19` (B), `pilot_m2_own_nbr_no19` (E1), `pilot_m5_own_gnnssl_no19` (E2), `pilot_n1_own_deg_no19` (N1) on splits 0–2 (12 runs); `pilot_m1_own_top4no19` (F5) on splits 0–2 (3 runs); then, only if nothing failed, `pilot_m5r_own_gnnrand_no19` (E3) on splits 0–2 (3 runs). B on all 25 features is phase 2's `pilot_m1_own` (same settings and machine).
  - **Primary contrasts** (test ΔPR-AUC per split, B − A as written; `repeat_summary.py --pairs`):
    - **F4** = B_no19 − B_all (`no19_summary`);
    - **E1 − B**, **E2 − B**, **N1 − B**, all on no19 (`no19_summary`);
    - **F5** = B_top4no19 − B_no19 (`f5_summary`);
    - if E3 runs: **E3 − B** on no19 (`e3_summary`).
  - **Secondary contrasts** (labelled secondary in every table): E1 − N1 on no19 (what neighbour means add beyond degree; not yet beyond a random graph, which needs N2); if E3 runs, E2 − E3 on no19.
  - **Verdict rule** (unchanged from 2026-10-19): **supported** if test Δ > 0 on all 3 splits, the mean Δ exceeds its SD across splits, and the Nadeau–Bengio corrected paired t-test over the 15 CV folds gives p < 0.05; otherwise a positive mean is **not distinguishable from zero** and a negative mean is **worse**.
  - **Readings, fixed now:**
    - F4 "worse" means `f19` carries signal the other 24 columns do not; its size is RQ1's headline number for df-analyze.
    - E1 − B or E2 − B "supported" on no19 means the graph helps once the leak is gone. Comparing that Δ with the all-features Δ (+0.010, +0.011) and the no36 Δ (+0.162, +0.129) says how far the leak hid the graph's benefit.
    - N1 − B "supported" means degree alone accounts for part of the graph gain. E1 − N1 "supported" means neighbour means add something beyond degree. Whether that something is real structure is N2's question, not tonight's.
    - F5 "not distinguishable" or positive means four non-vote columns carry the 24-column result. "Worse" means they do not.
  - **Not run tonight:** N2 (needs `igraph` or a decision on the rewiring tool), V1, GraphSAGE, E4, YelpChi, full-budget runs.
  - **Status:** Caelen put the queue on hold the same day. It was stopped 2 minutes into the Z1 probe, before any df-analyze run; nothing was recorded.
- **2026-10-26 (N2 probe, RQ2; training CV of split 0 only, no19; `scripts/null_graph_probe.py`; results DB `20261026T150141Z_amazon_dd4a8c`).**
  - **Rewiring.** `features/graph.py` `swap_rewire`: double-edge swaps (a-b, c-d → a-d, c-b), 10 per edge, rejecting self-loops, existing edges and repeats. It is NumPy only, so no `igraph` is needed. Every node keeps exactly its degree in every relation, which the script checks each time.
  - **How random the rewired graphs are.** Real edges still present after rewiring: U-P-U 22%, U-S-U 36%, U-V-U 16%. U-S-U is dense (3.57M edges on 11,944 nodes), so a random graph with the same degrees necessarily shares many of its edges. U-S-U takes about 55 s per rewiring on one CPU.
  - **Probe CV PR-AUC** (the fixed HistGradientBoosting probe on 5 grouped folds): B 0.498; N1 (+ deg) 0.630; N2 (+ nbr on the rewired graph) 0.627 ± 0.016 over 5 rewiring seeds (0.607–0.649); E1 (+ nbr on the real graph) 0.719.
  - **Split of E1 − B = +0.221:** degree (N1 − B) +0.132; neighbour columns on a random graph (N2 − N1) −0.004; real structure (E1 − N2) +0.092.
  - **Reading** (a training-CV check, not a result; the pre-registered pilots decide). Without `f19`, degree alone gives about 60% of the neighbour-mean gain. Neighbour means on a degree-matched random graph add nothing beyond degree. The real edges add about 0.09 on top. Because the rewired U-S-U keeps 36% of the real edges, E1 − N2 probably understates the real-structure part.
  - **Side finding:** own features without `f19` score 0.498 on CV, the same as without all four #36 features (0.500, 2026-10-19). That fits the column map: `f13`, `f15` and `f18` can be computed from the star counts `f02`–`f06`, so dropping them loses nothing, and `f19` alone carries the #36 signal.
  - **Not done:** N2 pilot runs need a pipeline block of neighbour means on the rewired graph, with the rewiring seed tied to the split repeat and the graph saved for N3. That block is not built, and the runs need a go. The question of which rewiring tool to use (`igraph`) is moot if the group accepts this implementation.
  - **Recording note:** the results-DB entry's `git_dirty` is true. The only uncommitted change at the time was README text; the probe code was committed (`b75f377`).
- **2026-10-26 (amendment to the overnight pre-registration: N2 added, written BEFORE any pilot run; the queue is still on hold).** Caelen asked for the N2 block and its pilots.
  - **Block:** `nbrrw` is the `nbr` block (1-hop means of every own feature plus log1p(degree), per relation) built on a rewired graph. Stage `neighbour_agg`, `features/graph.py` `swap_rewire`, 10 double-edge swaps per edge.
  - **Rewiring seed:** tied to the split repeat. It is the split seed plus the relation index: split 0 uses 555 / 556 / 557, split 1 uses 1555 / 1556 / 1557, split 2 uses 2555 / 2556 / 2557.
  - **Saved graph:** `data/processed/amazon*/graph_rewired/seed<split seed>/`, with a manifest, and reused while the settings match, so a later GraphSAGE control (N3) sees the same graph.
  - **Built and checked on split 0:** 99 columns, no `f19`, every degree unchanged; 22% / 36% / 16% of real edges kept. This is the same graph as the N2 probe's seed 555.
  - **Runs added:** `pilot_n2_own_nbrrw_no19` (N2) on splits 0, 1, 2, each right after that split's N1 run (3 runs, about 40 minutes).
  - **Contrasts (primary for RQ2, `n2_summary`; B − A as written):**
    - **E1 − N2:** real structure;
    - **N2 − N1:** neighbour columns on a degree-matched random graph;
    - **N2 − B.**

    These use the same verdict rule as the entry above.
  - **What was known when this was written:** only the training-CV probe of split 0 (N2 0.627 vs E1 0.719 and N1 0.630). No test-set number for any no19 rung exists yet. The probe chose nothing here: the swap count, the seed rule and the block are the plan's, fixed before the probe.
  - **Readings, fixed now:**
    - E1 − N2 "supported" means the real edges add something that a degree-matched random graph does not.
    - N2 − N1 "worse" or "not distinguishable" means that, on a random graph, neighbour means add nothing beyond degree.
    - Because the rewired U-S-U keeps 36% of the real edges, a positive E1 − N2 is a lower bound on the real-structure part.
- **2026-10-27 01:55 UTC (second amendment, written BEFORE these runs; the queue started at 01:41 UTC).** Caelen asked for the embedding-size rungs, then the graph on the four features, with the laptop free by 8 am Atlantic (11:00 UTC). A guard skips any of these jobs that would start after 10:30 UTC (summaries 10:45).
  - **Z1 result** (training CV, split 0, no19; results DB `20261027T014148Z_amazon_1552db`). CV PR-AUC by d:

    | Block | 2 | 4 | 8 | 16 | 32 | 64 | raw |
    |---|---|---|---|---|---|---|---|
    | gnnssl (E2) | 0.643 | 0.662 | 0.688 | 0.696 | 0.695 | 0.700 | 0.696 |
    | gnnrand (E3) | 0.634 | 0.653 | 0.673 | 0.697 | 0.700 | 0.701 | 0.702 |

    The one-standard-error rule picks d = 8 for gnnssl and d = 16 for gnnrand.
  - **Sizes run:** 8 (the default, already queued as E2 and E3), 16 (gnnrand's choice, run for both so E2 and E3 are matched), and raw 64 (the plan's contrasting size; 192 columns). The size is set by `gnn_embed_sizes` through variants `no19d16` and `no19raw`, which have the no19 columns.
  - **Runs:** `pilot_m5r_own_gnnrand_no19d16`, `pilot_m5_own_gnnssl_no19d16`, `pilot_m5_own_gnnssl_no19raw`, `pilot_m5r_own_gnnrand_no19raw` on splits 0–2 (12 runs). Then `pilot_m2_own_nbr_top4no19` and `pilot_m5_own_gnnssl_top4no19` on splits 0–2 (6 runs; the graph blocks see only each split's four columns).
  - **Primary contrasts** (B − A as written; the 2026-10-19 rule):
    - at each size, E2_d − B_no19 and E3_d − B_no19 (`size16_summary`, `sizeraw_summary`);
    - on the four features, E1 − B_top4no19 and E2 − B_top4no19 (`top4no19_summary`).
  - **Secondary contrasts:** at each size, E2_d − E3_d (does training help at that size?); E2_d − E2_8 and E3_d − E3_8 (does a larger embedding help over the default?); on the four features, E2 − E1.
  - **Readings, fixed now:**
    - For the professor's "size of GNN embeddings", the smallest size whose E − B is supported and whose E_d − E_8 is not "supported" is enough.
    - For "does training help", E2_d − E3_d must be supported at some size.
    - For "4 important features", E1/E2 − B_top4no19 "supported" means the graph still adds to the four non-vote columns.
- **2026-10-27 (overnight results, RQ1–RQ3; Amazon, pilot settings above, RTX 4060 laptop).** The queue ran from 01:41 to 10:45 UTC.
  - **Coverage:**
    - 32 of 39 queued pilot runs finished, all exit 0.
    - Raw-64 runs took 23–26 min each, against 13–16 min for the others, so the queue ran late.
    - The guard skipped `pilot_m5r_own_gnnrand_no19raw` on split 2: it would have started at 10:43, after the 10:30 cutoff.
    - The four-feature graph runs (`pilot_m2_own_nbr_top4no19`, `pilot_m5_own_gnnssl_top4no19`, 6 runs) and `top4no19_summary` never started.
    - `sizeraw_summary` then failed (exit 1) on the missing split-2 Model A and left no aggregate.
    - At 10:46 it was rerun with only the two DGI pairs (E2raw − B, E2raw − E2₈), which are complete on all 3 splits.
    - E3raw − B, E2raw − E3raw and E3raw − E3₈ have no verdict. E3raw's splits 0 and 1 are reported below as numbers only.
  - **Test PR-AUC per split (0 / 1 / 2),** no-skill 0.095:

    | Setup | Feature set | Test PR-AUC |
    |---|---|---|
    | B | `pilot_m1_own_no19` | 0.516 / 0.535 / 0.565 |
    | F5 | `pilot_m1_own_top4no19` | 0.504 / 0.531 / 0.553 |
    | E1 | `pilot_m2_own_nbr_no19` | 0.703 / 0.699 / 0.706 |
    | E2 (d 8) | `pilot_m5_own_gnnssl_no19` | 0.662 / 0.674 / 0.671 |
    | E2 (d 16) | `pilot_m5_own_gnnssl_no19d16` | 0.681 / 0.677 / 0.676 |
    | E2 (raw 64) | `pilot_m5_own_gnnssl_no19raw` | 0.672 / 0.699 / 0.677 |
    | E3 (d 8) | `pilot_m5r_own_gnnrand_no19` | 0.651 / 0.676 / 0.672 |
    | E3 (d 16) | `pilot_m5r_own_gnnrand_no19d16` | 0.647 / 0.646 / 0.687 |
    | E3 (raw 64) | `pilot_m5r_own_gnnrand_no19raw` | 0.693 / 0.694 / skipped |
    | N1 | `pilot_n1_own_deg_no19` | 0.650 / 0.652 / 0.652 |
    | N2 | `pilot_n2_own_nbrrw_no19` | 0.619 / 0.628 / 0.631 |

  - **Verdicts** (test Δ, B − A as written; mean ± SD over 3 splits; Nadeau–Bengio p, two-sided; "worse" = negative mean, significant only if p < 0.05):

    | Contrast | Role | Δ 0 / 1 / 2 | Mean ± SD | p | Verdict |
    |---|---|---|---|---|---|
    | F4: B_no19 − B_all | primary | −0.401 / −0.384 / −0.354 | −0.379 ± 0.024 | 3.4e-11 | worse (significant) |
    | E1 − B | primary | +0.187 / +0.163 / +0.140 | +0.163 ± 0.023 | 9.7e-06 | supported |
    | E2 − B | primary | +0.146 / +0.139 / +0.105 | +0.130 ± 0.022 | 0.0011 | supported |
    | N1 − B | primary | +0.134 / +0.116 / +0.086 | +0.112 ± 0.024 | 0.0003 | supported |
    | E1 − N1 | secondary | +0.053 / +0.047 / +0.054 | +0.051 ± 0.004 | 0.0009 | supported |
    | E1 − N2 | primary | +0.084 / +0.070 / +0.074 | +0.076 ± 0.007 | 0.00069 | supported |
    | N2 − N1 | primary | −0.031 / −0.023 / −0.020 | −0.025 ± 0.006 | 0.10 | worse (not significant) |
    | N2 − B | primary | +0.102 / +0.093 / +0.066 | +0.087 ± 0.019 | 4.5e-05 | supported |
    | F5: B_top4no19 − B_no19 | primary | −0.013 / −0.005 / −0.012 | −0.010 ± 0.004 | 0.60 | worse (not significant) |
    | E3 − B | primary | +0.135 / +0.141 / +0.107 | +0.127 ± 0.018 | 0.00033 | supported |
    | E2 − E3 | secondary | +0.011 / −0.002 / −0.001 | +0.003 ± 0.007 | 0.85 | not distinguishable |
    | E2₁₆ − B | primary | +0.165 / +0.142 / +0.111 | +0.139 ± 0.027 | 0.00038 | supported |
    | E3₁₆ − B | primary | +0.130 / +0.110 / +0.122 | +0.121 ± 0.010 | 8.7e-05 | supported |
    | E2₁₆ − E3₁₆ | secondary | +0.035 / +0.031 / −0.011 | +0.018 ± 0.025 | 0.76 | not distinguishable |
    | E2₁₆ − E2₈ | secondary | +0.019 / +0.003 / +0.005 | +0.009 ± 0.009 | 0.46 | not distinguishable |
    | E3₁₆ − E3₈ | secondary | −0.004 / −0.030 / +0.015 | −0.007 ± 0.023 | 0.54 | worse (not significant) |
    | E2raw − B | primary | +0.156 / +0.164 / +0.112 | +0.144 ± 0.028 | 7e-05 | supported |
    | E2raw − E2₈ | secondary | +0.010 / +0.025 / +0.006 | +0.014 ± 0.010 | 0.47 | not distinguishable |

  - **Readings** (by the rules fixed before the runs):
    - **RQ1.**
      - Dropping `f19` costs 0.38 PR-AUC (F4).
      - With `f19` the graph adds +0.010 (E1) and +0.011 (E2). Without it, it adds +0.163 and +0.130. The leak hides almost all of the graph's benefit.
      - Every no19 number is within 0.01 of its no36 counterpart (B 0.539 vs 0.548, E1 0.702 vs 0.710, E2 0.669 vs 0.676). This fits the column map: `f19` carries the #36 signal.
    - **RQ2** (without `f19`). E1's +0.163 splits into three parts:
      - degree alone: +0.112 (N1 − B, supported);
      - neighbour columns on a degree-matched random graph: −0.025 (N2 − N1, lower on all splits but not significant);
      - real structure: +0.076 (E1 − N2, supported).

      About two thirds of the graph gain is degree, and the real edges add the rest. Because the rewired U-S-U keeps 36% of the real edges, +0.076 is a lower bound. The test split matches the training-CV probe of 2026-10-26 (+0.132 / −0.004 / +0.092).
    - **RQ3 / professor, embedding size.**
      - At every size, both embeddings beat B (supported).
      - A larger size does not measurably beat 8 per relation: E2₁₆ − E2₈ and E2raw − E2₈ are not distinguishable, and E3₁₆ − E3₈ is negative.
      - By the size rule, **8 components per relation (24 columns) is enough** for both blocks.
      - E3 raw scored 0.693 and 0.694 on splits 0 and 1, the highest embedding numbers on those splits. With split 2 missing, it has no verdict.
    - **RQ3 / professor, training.**
      - E2 − E3 is not supported at any tested size (8: +0.003; 16: +0.018 ± 0.025).
      - Training the encoder with DGI does not measurably help over the same encoder with random weights. This repeats 2026-10-23's finding without `f19`.
      - The raw-64 pair has no verdict.
    - **Professor, four features.**
      - Each split's top four non-`f19` columns score 0.010 below the 24 (F5).
      - That is lower on all 3 splits, so the rule labels it "worse", but p = 0.60.
      - By the reading fixed beforehand, the four do not carry the 24-column result, although the gap is small.
      - Whether the graph still adds to these four is not known: those runs did not start.
    - **Not pre-registered, descriptive only:** neighbour means (E1, 0.702) score above every embedding rung at d 8 or 16 without `f19`, as without #36 on 2026-10-20.
  - **Results DB:**
    - Summaries:
      - `20261027T060614Z_amazon_01f812` (no19)
      - `20261027T061152Z_amazon_4f749a` (N2)
      - `20261027T061515Z_amazon_9aaac3` (F5)
      - `20261027T065806Z_amazon_cff161` (E3)
      - `20261027T083511Z_amazon_384e37` (size 16)
      - `20261027T104610Z_amazon_6ca8b3` (raw 64, DGI only)
    - Z1 probe: `20261027T014148Z_amazon_1552db`.
    - The 32 pilot runs are tagged `rq-plan` (`uv run scripts/results_db.py list --tag rq-plan`).
    - The one-page summary is `uv run scripts/rq_report.py --dataset amazon` → `outputs/amazon/compare/rq_results.md`, copied to [`RQ_RESULTS.md`](RQ_RESULTS.md).
  - **Still to run** (each needs a go): the four-feature graph runs and their summary (about 2 h); E3 raw on split 2 and its summary (about 25 min).
- **2026-10-29 ~01:40 UTC (third amendment, written BEFORE any of the runs it adds; the 3-split results above were known).** Caelen's go (2026-10-28 22:15 Atlantic): last night's leftovers, then splits 3–4, sizes 2 and 4, and full-budget runs. The laptop should be free by 8 am Atlantic (11:00 UTC), and "a bit over is OK". The queue restarted at 01:18 UTC with the leftovers, which are pre-registered in the second amendment; only their guard times moved to tonight. Order and guards:
  1. **Leftovers.** E1 and E2 on `top4no19` (splits 0–2) and `top4no19_summary`. Then E3 raw on split 2 (job `r2_pilot_m5r_own_gnnrand_no19raw_rerun`) and `sizeraw_e3_summary` (3-split: E3raw − B; secondary E2raw − E3raw and E3raw − E3₈).
  2. **Splits 3 and 4** (split seeds 3555, 4555; `--set split.n_repeats=5`; the split seed is `seed + 1000 × repeat`, so splits 0–2 are unchanged).
     - Runs: `pilot_m1_own_no19`, `pilot_m5_own_gnnssl_no19`, `pilot_m5r_own_gnnrand_no19`, and the same at 16 and at raw 64 (`…_no19d16`, `…_no19raw`). That is 7 rungs × 2 splits = 14 runs, pilot settings.
     - **Code change needed first:** `with_dataset` used to resolve every `keep_by_repeat` variant for the active split. `top4no19` lists splits 0–2 only, so every split-3/4 run would have failed. Now it fails only if the active feature set uses such a variant (test added).
  3. **5-split summaries** (`size8_5split`, `size16_5split`, `sizeraw_5split`; prefix `pilot_`, `--repeats 5`).
     - The contrasts are the 3-split ones extended to 5 splits:
       - primary: E2_d − B and E3_d − B;
       - secondary: E2_d − E3_d, E2_d − E2₈ and E3_d − E3₈;
       - for d = 8, 16 and raw.
     - Rule: **supported** if Δ > 0 on all 5 splits, the mean Δ exceeds its SD, and the Nadeau–Bengio corrected test over the 25 CV folds gives p < 0.05. Otherwise, as before, a positive mean is not distinguishable and a negative mean is worse.
     - **Caelen's decision: the 5-split verdict replaces the 3-split verdict** for these contrasts. The 3-split verdicts stay in this log for reference.
     - The readings are the second amendment's (size rule, "training helps" rule).
  4. **Sizes 2 and 4** (3 splits, 0–2).
     - Runs: `pilot_m5_own_gnnssl_no19d{2,4}` and `pilot_m5r_own_gnnrand_no19d{2,4}` (12 runs), then `size24_summary`.
     - Primary: E2_d − B and E3_d − B.
     - Secondary: E2₈ − E2_d and E3₈ − E3_d (does 8 beat the smaller size?), and E2_d − E3_d.
     - **Reading, fixed now:** a size d < 8 is enough if E_d − B is supported and E₈ − E_d is not supported. Otherwise 8 remains the answer.
     - What was known when this was written: on training CV (Z1), d = 2 and 4 scored below 8 (gnnssl 0.643 / 0.662 against 0.688), and nothing on the test set.
  5. **Full budget**, split 0 only, the full Model A settings (6 classifiers, 100 trials): `m1_own_no19` (B), then `m2_own_nbr_no19` (E1), then `m5_own_gnnssl_no19` (E2).
     - Each starts only if it can finish by about 12:30 UTC (B ≈ 2 h, so it starts no later than 10:30 UTC; E1 and E2 no later than 10:00 UTC). B is probably the only one that fits.
     - These are **descriptive only** (one split, no verdict):
       - each run's test PR-AUC with its group-bootstrap 95% interval;
       - the paired Δ with its interval from the comparison page for E1 − B and E2 − B (if both sides ran) and for B_full − B_pilot (the budget effect);
       - the 5-fold CV p next to it.
     - Summary `full_no19_summary` (`--repeats 1 --prefix ""`). The verdict column in that table is not the pre-registered rule.
  - **Guards:** no pilot run starts after 11:00 UTC and no summary after 11:30 UTC; full-budget runs as in step 5.
- **2026-10-29 (overnight results, third amendment; Amazon, pilot settings, RTX 4060 laptop).** The queue ran from 01:18 to 11:13 UTC (08:13 Atlantic).
  - **Coverage:**
    - All 33 pilot runs and all 7 summaries finished, all exit 0.
    - Full budget: `m1_own_no19` (B) started at 09:58 UTC and took 75 min. `m2_own_nbr_no19` (E1) and `m5_own_gnnssl_no19` (E2) were skipped by their 10:00 UTC guard, so `full_no19_summary` holds only B_full − B_pilot.
    - Every df-analyze run passed the split and column-type checks. The full B run raised the sanity-band warning (0.525 against 0.91). That is expected without `f19`; `m1_own_no36` raised it too.
  - **Test PR-AUC per split** (0 / 1 / 2 / 3 / 4; splits 0–2 of B, F5 and E2/E3 at 8, 16 and raw are from 2026-10-27), no-skill 0.095:

    | Setup | Feature set | Test PR-AUC |
    |---|---|---|
    | B | `pilot_m1_own_no19` | 0.516 / 0.535 / 0.565 / 0.558 / 0.542 |
    | E2 (d 2) | `pilot_m5_own_gnnssl_no19d2` | 0.641 / 0.647 / 0.656 |
    | E2 (d 4) | `pilot_m5_own_gnnssl_no19d4` | 0.650 / 0.663 / 0.639 |
    | E2 (d 8) | `pilot_m5_own_gnnssl_no19` | 0.662 / 0.674 / 0.671 / 0.702 / 0.675 |
    | E2 (d 16) | `pilot_m5_own_gnnssl_no19d16` | 0.681 / 0.677 / 0.676 / 0.731 / 0.701 |
    | E2 (raw 64) | `pilot_m5_own_gnnssl_no19raw` | 0.672 / 0.699 / 0.677 / 0.726 / 0.692 |
    | E3 (d 2) | `pilot_m5r_own_gnnrand_no19d2` | 0.646 / 0.680 / 0.661 |
    | E3 (d 4) | `pilot_m5r_own_gnnrand_no19d4` | 0.656 / 0.678 / 0.670 |
    | E3 (d 8) | `pilot_m5r_own_gnnrand_no19` | 0.651 / 0.676 / 0.672 / 0.685 / 0.675 |
    | E3 (d 16) | `pilot_m5r_own_gnnrand_no19d16` | 0.647 / 0.646 / 0.687 / 0.702 / 0.692 |
    | E3 (raw 64) | `pilot_m5r_own_gnnrand_no19raw` | 0.693 / 0.694 / 0.698 / 0.731 / 0.696 |
    | F5 | `pilot_m1_own_top4no19` | 0.504 / 0.531 / 0.553 |
    | E1 on F5's four | `pilot_m2_own_nbr_top4no19` | 0.671 / 0.667 / 0.681 |
    | E2 on F5's four | `pilot_m5_own_gnnssl_top4no19` | 0.651 / 0.657 / 0.643 |
    | B, full budget | `m1_own_no19` | split 0 only: 0.525 [0.471, 0.585] (CatBoost; AUROC 0.894 [0.874, 0.910]) |

  - **Verdicts** (test Δ, B − A as written; Nadeau–Bengio p, two-sided; "worse" = negative mean, significant only if p < 0.05). By Caelen's decision the 5-split rows replace the 3-split rows of 2026-10-27 and the 3-split `sizeraw_e3_summary` of tonight:

    | Contrast | Role | Splits | Δ per split | Mean ± SD | p | Verdict |
    |---|---|---|---|---|---|---|
    | E2₈ − B | primary | 5 | +0.146 / +0.139 / +0.105 / +0.144 / +0.133 | +0.133 ± 0.016 | 3.2e-05 | supported |
    | E3₈ − B | primary | 5 | +0.135 / +0.141 / +0.107 / +0.126 / +0.132 | +0.128 ± 0.013 | 1.6e-05 | supported |
    | E2₈ − E3₈ | secondary | 5 | +0.011 / −0.002 / −0.001 / +0.018 / +0.001 | +0.005 ± 0.009 | 0.50 | not distinguishable |
    | E2₁₆ − B | primary | 5 | +0.165 / +0.142 / +0.111 / +0.173 / +0.159 | +0.150 ± 0.025 | 5.4e-06 | supported |
    | E3₁₆ − B | primary | 5 | +0.130 / +0.110 / +0.122 / +0.144 / +0.150 | +0.131 ± 0.016 | 2.8e-07 | supported |
    | E2₁₆ − E3₁₆ | secondary | 5 | +0.035 / +0.031 / −0.011 / +0.029 / +0.010 | +0.019 ± 0.019 | 0.71 | not distinguishable |
    | E2₁₆ − E2₈ | secondary | 5 | +0.019 / +0.003 / +0.005 / +0.029 / +0.026 | +0.016 ± 0.012 | 0.37 | not distinguishable |
    | E3₁₆ − E3₈ | secondary | 5 | −0.004 / −0.030 / +0.015 / +0.018 / +0.017 | +0.003 ± 0.021 | 0.21 | not distinguishable |
    | E2raw − B | primary | 5 | +0.156 / +0.164 / +0.112 / +0.167 / +0.150 | +0.150 ± 0.022 | 9.3e-07 | supported |
    | E3raw − B | primary | 5 | +0.177 / +0.158 / +0.132 / +0.173 / +0.154 | +0.159 ± 0.018 | 7.1e-07 | supported |
    | E2raw − E3raw | secondary | 5 | −0.021 / +0.006 / −0.021 / −0.005 / −0.004 | −0.009 ± 0.012 | 0.51 | worse (not significant) |
    | E2raw − E2₈ | secondary | 5 | +0.010 / +0.025 / +0.006 / +0.023 / +0.017 | +0.016 ± 0.008 | 0.53 | not distinguishable |
    | E3raw − E3₈ | secondary | 5 | +0.042 / +0.018 / +0.026 / +0.047 / +0.022 | +0.031 ± 0.013 | 0.019 | supported |
    | E2₄ − B | primary | 3 | +0.133 / +0.128 / +0.074 | +0.112 ± 0.033 | 0.0002 | supported |
    | E3₄ − B | primary | 3 | +0.140 / +0.143 / +0.105 | +0.129 ± 0.021 | 0.00026 | supported |
    | E2₈ − E2₄ | secondary | 3 | +0.012 / +0.011 / +0.032 | +0.018 ± 0.012 | 0.89 | not distinguishable |
    | E3₈ − E3₄ | secondary | 3 | −0.005 / −0.002 / +0.001 | −0.002 ± 0.003 | 0.53 | worse (not significant) |
    | E2₄ − E3₄ | secondary | 3 | −0.006 / −0.015 / −0.031 | −0.018 ± 0.013 | 0.28 | worse (not significant) |
    | E2₂ − B | primary | 3 | +0.124 / +0.111 / +0.090 | +0.109 ± 0.017 | 0.00074 | supported |
    | E3₂ − B | primary | 3 | +0.130 / +0.145 / +0.096 | +0.123 ± 0.025 | 0.00027 | supported |
    | E2₈ − E2₂ | secondary | 3 | +0.021 / +0.027 / +0.015 | +0.021 ± 0.006 | 0.42 | not distinguishable |
    | E3₈ − E3₂ | secondary | 3 | +0.005 / −0.004 / +0.011 | +0.004 ± 0.008 | 0.46 | not distinguishable |
    | E2₂ − E3₂ | secondary | 3 | −0.005 / −0.033 / −0.005 | −0.015 ± 0.016 | 0.91 | worse (not significant) |
    | E1 − B_top4no19 | primary | 3 | +0.167 / +0.136 / +0.128 | +0.144 ± 0.021 | 3.3e-08 | supported |
    | E2 − B_top4no19 | primary | 3 | +0.147 / +0.127 / +0.090 | +0.121 ± 0.029 | 0.00022 | supported |
    | E2 − E1 (on top4no19) | secondary | 3 | −0.020 / −0.010 / −0.038 | −0.023 ± 0.014 | 0.047 | worse (significant) |
    | B_full − B_pilot | descriptive | 1 | +0.008 | — | 0.57 | none (one split) |

    Two verdicts changed with splits 3–4. E3₁₆ − E3₈ went from worse (−0.007) to not distinguishable (+0.003). E3raw − E3₈ went from not distinguishable (3 splits, +0.029, p = 0.071) to supported. The others kept their verdicts.
  - **Readings** (by the rules fixed beforehand):
    - **Embedding size (professor; RQ3).**
      - Both blocks beat B at every size tried, from 2 per relation to raw 64 (all supported).
      - Below 8: 8 does not measurably beat 2 or 4 for either block. By the third amendment's rule, **2 per relation (6 columns) is enough.**
      - Caveat to that: for DGI, 8 scored above 2 on all 3 splits (+0.021 ± 0.006) and failed only the CV test (p = 0.42), and the Z1 training-CV probe also rises from 2 to 8. Sizes 2 and 4 have 3 splits, not 5.
      - Above 8: DGI at 16 or raw 64 does not measurably beat 8. The untrained encoder at raw 64 does (+0.031 ± 0.013, supported over 5 splits).
      - The rule compares each size with 8 only, so it does not say whether raw 64 beats 2. Descriptively (not pre-registered), the untrained encoder's raw 64 is +0.033 above its size 2 on splits 0–2, higher on all three.
      - In short: two PCA components per relation already give most of the graph gain (E − B +0.11 to +0.12, against +0.14 to +0.16 at raw 64 on the same 3 splits), and only the untrained encoder gains measurably from keeping all 64 dimensions over 8.
    - **Training the GNN (professor; RQ3).** E2 − E3 is not supported at any size: 2 −0.015 and 4 −0.018 (3 splits); 8 +0.005, 16 +0.019 and raw 64 −0.009 (5 splits). By the "training helps" rule, DGI training does not measurably help over the same encoder with random weights at any size from 2 to 64. GraphSAGE trained on the labels is not built.
    - **Four features (professor; RQ1).**
      - The graph still adds to each split's four non-`f19` columns: neighbour means +0.144 and DGI +0.121 (both supported). Those gains are close to the gains on all 24 (+0.163 and +0.130 on the same splits).
      - On the four, DGI scores below neighbour means (−0.023, lower on all 3 splits, p = 0.047). That repeats the without-#36 result of 2026-10-20 (−0.033).
    - **Full budget (descriptive, one split).** B without `f19` at the full budget scores 0.525 [0.471, 0.585] on split 0, against the pilot's 0.516 (+0.008, CV p = 0.57). The pilot budget does not measurably understate B. Full-budget E1 and E2 were not run.
    - **Not pre-registered, descriptive only:** on splits 0–2, neighbour means (E1, 0.703) still score above every embedding rung. The best embedding rung there is E3 raw (0.695).
  - **Results DB:**
    - Summaries:
      - `20261029T023123Z_amazon_91fb7a` (four features + graph)
      - `20261029T025625Z_amazon_19faef` (E3 raw, 3 splits; replaced by the 5-split one)
      - `20261029T065310Z_amazon_d0bc69` (size 8, 5 splits)
      - `20261029T065523Z_amazon_27623e` (size 16, 5 splits)
      - `20261029T065903Z_amazon_551909` (raw 64, 5 splits)
      - `20261029T094738Z_amazon_409d22` (sizes 2 and 4)
      - `20261029T111325Z_amazon_956956` (full budget, descriptive)
    - Full-budget B: `20261029T095827Z_amazon-m1-own-no19_f961b5`.
    - The 33 pilot runs are tagged `rq-plan`; the summaries of 5 splits are also tagged `splits5`.
    - [`RQ_RESULTS.md`](RQ_RESULTS.md) is regenerated from all 17 summaries.
  - **Still to run** (each needs a go): full-budget E1 and E2 without `f19` on split 0 (about 1.5–2 h each); E1, N1 and N2 on splits 3–4, if the RQ1 and RQ2 verdicts should also rest on 5 splits (6 runs, about 1.5 h).
- **2026-10-29 ~11:30 UTC (fourth amendment, written BEFORE the runs it adds; every result above was known).** Caelen asked to queue the two full-budget runs the guard skipped. They are queued in `outputs/amazon/rq_logs/queue.txt` but not started; the runner starts on Caelen's go.
  - **Runs**, split 0 only, at the full Model A settings (6 classifiers, 100 trials), in this order: `m2_own_nbr_no19` (E1), then `m5_own_gnnssl_no19` (E2).
    - Both reuse split 0's `nbr_no19` and `gnnssl_no19` blocks, the ones their pilots used, so only the df-analyze budget differs from the pilots.
    - Job names `full2_…`, so last night's skip logs are kept.
  - **Guards:** the third amendment's, moved to the night of 2026-10-29/30. Each run starts no later than 10:00 UTC on 2026-10-30, and the summary no later than 13:00 UTC.
  - **Summary:** `full_no19_summary`, rerun with every complete pair. It is recorded as a new results-DB entry; the earlier entry (`20261029T111325Z_amazon_956956`, B only) stays. Pairs:
    - B_full − B_pilot, E1_full − B_full and E2_full − B_full, as in the third amendment;
    - added now: E1_full − E1_pilot and E2_full − E2_pilot (does the full budget change the graph rungs?).
  - **Descriptive only** (one split, no verdict): each run's test PR-AUC with its group-bootstrap 95% interval, each pair's paired Δ with its interval from the comparison page, and the 5-fold CV p next to it.
  - **What was known when this was written:** on split 0, the pilots scored E1 0.703 and E2 0.662, and full-budget B scored 0.525 (pilot 0.516).
- **2026-10-30 (fourth amendment, results).** On Caelen's go, the runner started at 01:46 UTC on 2026-10-30, and all three jobs exited 0 by 04:30 UTC. Every guard was met, and nothing was skipped. Both runs were on commit `59d6554` with a clean tree. Both passed the split and column-type checks: 99 columns for E1 and 48 for E2, none dropped. Model A decoded and reproduces its test probabilities in each, as do 18 of the 28 tuned models. Wall-clock time, including `select_model`, was 1 h 27 min for E1 and 1 h 14 min for E2.
  - **Scores** (split 0, test set; group-bootstrap 95% intervals; no skill 0.095):

    | Setup | Feature set | Model A | Test PR-AUC | AUROC | CV PR-AUC of Model A | Pilot, split 0 |
    |---|---|---|---|---|---|---|
    | B | `m1_own_no19` | CatBoost (none) | 0.525 [0.471, 0.585] | 0.894 [0.874, 0.910] | 0.570 ± 0.026 | 0.516 |
    | E1 | `m2_own_nbr_no19` | CatBoost (assoc) | 0.714 [0.667, 0.758] | 0.935 [0.919, 0.948] | 0.712 ± 0.036 | 0.703 |
    | E2 | `m5_own_gnnssl_no19` | CatBoost (embed_linear) | 0.646 [0.591, 0.701] | 0.919 [0.903, 0.935] | 0.692 ± 0.037 | 0.662 |

  - **Paired contrasts** (descriptive, as pre-registered):
    - Test Δ PR-AUC (B − A as written) with its paired group-bootstrap 95% interval (2,000 resamples of whole test groups) and the share of resamples above zero.
    - Next to it, the 5-fold training-CV Δ and its Nadeau–Bengio p.
    - `repeat_summary.py` still prints verdict labels for these rows ("supported", "worse"), because the mean > SD condition is undefined for one split. By this amendment they carry no verdict, and `RQ_RESULTS.md` marks them descriptive.

    | Contrast | Test Δ [95%] | Share > 0 | CV Δ | CV p |
    |---|---|---|---|---|
    | B_full − B_pilot | +0.008 [−0.009, +0.025] | 83% | +0.013 | 0.57 |
    | E1_full − B_full | +0.190 [+0.144, +0.231] | 100% | +0.142 | 0.0023 |
    | E2_full − B_full | +0.122 [+0.075, +0.166] | 100% | +0.122 | 0.017 |
    | E1_full − E1_pilot | +0.011 [−0.009, +0.032] | 86% | +0.006 | 0.51 |
    | E2_full − E2_pilot | −0.016 [−0.041, +0.008] | 12% | +0.004 | 0.79 |

  - **Readings** (one split, descriptive):
    - **The full budget changes no rung measurably.** B moves +0.008, E1 +0.011 and E2 −0.016 against their pilots. All three intervals cover zero, and so do the CV tests. On split 0, the pilot budget does not distort these contrasts.
    - **The graph gain survives the full budget.** E1 − B is +0.190, against +0.187 for the pilots on the same split. E2 − B is +0.122, against +0.146. Both paired intervals exclude zero.
      - In headroom terms (Δ / (1 − PR-AUC of B), with B = 0.525), E1 closes about 40% and E2 about 26%.
    - **Not pre-registered (point estimates; no paired interval was computed):** at the full budget E1 scores 0.068 above E2, against 0.041 for the pilots on split 0. Neighbour means stay ahead of DGI embeddings at both budgets.
    - Model A is CatBoost in all three full runs. The pilots chose LightGBM for B and E1, and CatBoost for E2.
    - The sanity-band warning (0.91, a band for all 25 features) fires on all three runs. That is expected without `f19`.
  - **Results DB:**
    - E1 full: `20261030T014631Z_amazon-m2-own-nbr-no19_71cfa9`.
    - E2 full: `20261030T031404Z_amazon-m5-own-gnnssl-no19_70f41b`.
    - Summary of the five pairs: `20261030T042833Z_amazon_39555d`. It extends `20261029T111325Z_amazon_956956`, which stays.
    - All three are tagged `rq-plan` and `full`. [`RQ_RESULTS.md`](RQ_RESULTS.md) is regenerated.
  - **Still to run** (each needs a go): E1, N1 and N2 on splits 3–4 (6 pilot runs, about 1.5 h), if the RQ1 and RQ2 verdicts should also rest on 5 splits.
- **2026-10-31 (fifth amendment: GNN classifiers, Model 6 and the E7 combination; written ~18:00 UTC, before any run of this round).** Caelen's decisions: [`GNN_MODELS_PLAN.md`](GNN_MODELS_PLAN.md) §7 and AGENTS.md's decision log.
  - **Feature variant:** every feature (Amazon all 25, YelpChi all 32), as published studies do. `f19`'s effect stays an observation (RQ1). No leak variants this round.
  - **Splits:** Amazon 0–4 (B, E1, E2, E3 exist on 0–2; 3–4 are added), YelpChi 0–2 (`split.n_repeats: 3`). Pilot df-analyze budget throughout: lgbm, lr, catboost; 10 trials.
  - **Setups:**

    | ID | Feature set | What |
    |---|---|---|
    | B, E1, E2, E3 | `pilot_m1_own`, `pilot_m2_own_nbr`, `pilot_m5_own_gnnssl`, `pilot_m5r_own_gnnrand` (E3 Amazon only) | as before |
    | E7 | `pilot_m7_own_nbr_gnnssl` | own + nbr + gnnssl (d = 8) |
    | E7r | `pilot_m7r_own_nbr_gnnrand` | own + nbr + gnnrand (Amazon only) |
    | E5 | `pilot_g5_sage` | heterogeneous GraphSAGE (mean per relation, root weight, 2 layers) |
    | E5g | `pilot_g5g_rgcn` | E5 with symmetric (R-GCN-style) normalisation |
    | E6 | `pilot_g6_mlp` | E5 with every edge removed (MLP) |
    | N3 | `pilot_g5n3_sage_rw` | E5 on N2's rewired graph, with E5's setting (Amazon only) |
    | E8 | `pilot_m8_own_nbr_gnnsup` | own + nbr + the best GNN's out-of-fold score (nested cross-fitting) |
    | E8s | `pilot_m8s_own_gnnsup` | own + that score (Amazon only) |

  - **GNN protocol** (GNN_MODELS_PLAN §2.2):
    - out-of-fold predictions on the saved grouped folds;
    - early stopping on a grouped, stratified fifth of each fit's training nodes (validation PR-AUC, patience 20, ≤ 300 epochs);
    - test = mean of the 5 fold models; 5 seeds, probabilities averaged;
    - grid hidden {64, 128} × dropout {0, 0.3} × lr {1e-3, 5e-3} (Adam, weight decay 5e-4), chosen per rung and split by mean fold PR-AUC with seed 0. N3 takes E5's setting. If one Amazon fit takes over 60 s, hidden 128 is dropped, decided on time only.
    - Threshold max F1 on the out-of-fold predictions, and the same metrics and bootstrap as Model A.
  - **Best GNN for E8, per split:** E5g if its mean per-fold out-of-fold PR-AUC gain over E5 exceeds one SE (SD/√5), else E5.
  - **Primary contrasts and expectations** (B − A as written; verdict rule of 2026-10-19; Holm within each family, reported beside the verdicts):

    | Family | Contrast | Expected |
    |---|---|---|
    | Amazon (5 splits) | E5 − E6 | supported |
    | | E5 − N3 | supported |
    | | E7 − E1 | not supported |
    | | E8 − E1 | not supported (little headroom with `f19`: E1 0.929) |
    | YelpChi (3 splits) | E1 − B | supported |
    | | E2 − B | supported |
    | | E5 − E6 | supported |
    | | E7 − E1 | not supported |
    | | E8 − E1 | not supported |

  - **Secondary** (verdicts reported, no family):
    - E5 − B, E6 − B, E5 − E1 (model families; expected E5 below E1, as in GADBench's finding that tree models with neighbour aggregates beat most GNNs);
    - E5g − E5, E7 − E2, E7 − E7r, E8 − E7, E8 − E5best, E8s − B, E8 − E8s;
    - the 5-split all-25 re-verdicts E1 − B, E2 − B, E2 − E1, E2 − E3.
  - **YelpChi audit:** a feature with univariate AUROC ≥ 0.85 is noted, not dropped (this round's feature rule).
  - **What was known when this was written:** Amazon all 25 on splits 0–2: B 0.918, E1 0.929, E2 0.929, E3 0.933. No GNN classifier, E7 or E8 result existed. YelpChi: only the 2026-10-08 smoke test (B 0.831).
- **2026-11-01 (fifth amendment, Amazon results).** The queue started at 17:29 UTC on 2026-10-31. The Amazon part (48 runs and 2 summaries) finished at 06:39 UTC on 2026-11-01; YelpChi is running.
  - **Run facts:**
    - Every entry was recorded from a clean tree: 2 at `19ba553`, 2 at `2ffb71d`, and 46 at `1bede69`, which only adds `scripts/gnn_report.py` to `2ffb71d`.
    - Every df-analyze run passed the split and column-type checks.
    - All 10 E8/E8s runs scored the shared CV with the nested out-of-fold `gnnsup` on each fold's fit rows.
    - The 60-s rule for hidden 128 never applied: the slowest grid point took 84 s for its 5 fold fits.
    - **One stop:** at 23:17 UTC, uv 0.12.19 aborted (exit 134, a panic in its own argument parser) on the `results_db.py add` after split 3's E5g. The run itself had finished and saved everything. Its entry was added by hand from those outputs, the job scripts now retry exit 134 only, and the runner restarted at 23:19 UTC.
  - **Scores** (test PR-AUC of Model A or the GNN rung, mean ± SD over the 5 splits; no skill 0.095; per split in [`GNN_RESULTS.md`](GNN_RESULTS.md)):

    | Setup | Test PR-AUC | CV PR-AUC |
    |---|---|---|
    | B (own) | 0.922 ± 0.007 | 0.924 |
    | E1 (+ nbr) | 0.933 ± 0.008 | 0.937 |
    | E2 (+ gnnssl) | 0.934 ± 0.007 | 0.935 |
    | E3 (+ gnnrand) | 0.934 ± 0.003 | 0.938 |
    | E7 (+ nbr + gnnssl) | 0.938 ± 0.007 | 0.937 |
    | E7r (+ nbr + gnnrand) | 0.936 ± 0.005 | 0.939 |
    | E8 (+ nbr + gnnsup) | 0.930 ± 0.007 | 0.937 |
    | E8s (+ gnnsup) | 0.928 ± 0.004 | 0.932 |
    | E5 (GraphSAGE) | 0.900 ± 0.014 | 0.905 |
    | E5g (R-GCN-style) | 0.911 ± 0.011 | 0.913 |
    | E6 (no edges) | 0.898 ± 0.013 | 0.904 |
    | N3 (rewired graph) | 0.887 ± 0.014 | 0.891 |

    - The GNN rungs' per-seed SD of test PR-AUC is at most 0.007.
    - **Best GNN for E8 (one-SE rule):** E5 on splits 0 and 3, E5g on splits 1, 2 and 4.
  - **Primary family (Amazon):** no contrast is supported. The smallest Holm p is 0.60 (E5 − N3).

    | Contrast | Mean Δ ± SD | Splits > 0 | CV Δ | CV p | Verdict | Expected |
    |---|---|---|---|---|---|---|
    | E5 − E6 | +0.002 ± 0.011 | 2 of 5 | +0.001 | 0.92 | not distinguishable | supported |
    | E5 − N3 | +0.013 ± 0.013 | 5 of 5 | +0.014 | 0.15 | not distinguishable | supported |
    | E7 − E1 | +0.005 ± 0.002 | 5 of 5 | −0.000 | 0.99 | not distinguishable | not supported |
    | E8 − E1 | −0.003 ± 0.008 | 2 of 5 | −0.000 | 0.99 | worse (lower mean) | not supported |

  - **Secondary** ("worse" with p ≥ 0.05 is a lower mean, not a significant drop):
    - **Supported:**
      - E1 − B: +0.011 ± 0.006, p 0.021.
      - E2 − B: +0.011 ± 0.002, p 0.028. This and E1 − B confirm the 3-split verdicts on 5 splits.
      - E8 − E5: +0.030 ± 0.014, p 0.0095.
    - **Significantly worse:**
      - E5 − E1: −0.033 ± 0.017, p 0.011.
      - E6 − B: −0.024 ± 0.011, p 0.043.
    - **Lower mean:**
      - E5 − B: −0.023, p 0.12.
      - E8 − E7: −0.008 ± 0.006, below on all 5 splits, CV Δ 0.000.
      - E2 − E3: −0.000.
    - **Not distinguishable:**
      - E5g − E5: +0.011 ± 0.004, above on all 5 splits, CV p 0.22.
      - E7 − E2: +0.004.
      - E7 − E7r: +0.002.
      - E8s − B: +0.005, p 0.24.
      - E8 − E8s: +0.002.
      - E2 − E1: +0.001.
  - **Readings** (pilot budget; all 25 features, so B starts at 0.92 and little headroom is left):
    - **The GNN classifiers trail df-analyze.** GraphSAGE is 0.023 below B and 0.033 below E1 (significant), as pre-registered for E5 − E1 and in line with GADBench.
    - **The edges do not measurably help the GNN.** Against its edge-free twin, E5 is ahead on only 2 of 5 splits. Against the rewired graph it is ahead on all 5, but mean Δ ≈ SD and CV p 0.15. Both "supported" expectations fail. With `f19` present, a node's own features carry nearly all of the signal (RQ1).
    - **Symmetric normalisation is ahead of the mean on every split** (+0.011, about the size of E1 − B). The CV test does not separate the two (p 0.22).
    - **Combining helps on the test set but not in CV.** E7 has the highest mean (0.938), and E7 − E1 is positive on all 5 test splits with mean > SD. Its CV Δ is zero (p 0.99), so it is not supported, as expected. E7 is also not distinguishable from E7r (DGI training adds nothing again; E2 − E3 likewise).
    - **Model 6's supervised score adds nothing over neighbour means.**
      - E8 − E1 is −0.003, and E8 is below E7 on every split.
      - On its own, the score lifts B by +0.005 (not distinguishable). That is about half the gain the neighbour means give.
      - The label-free aggregates already hold what the supervised GNN learns from the graph.
  - **Caveats:**
    - Pilot budget, with each side's Model A chosen on the same folds.
    - CV p values are two-sided, and Holm is reported only as a sensitivity check.
    - YelpChi, the validation dataset, is not in yet.
  - **Results DB:**
    - 48 pipeline entries, tags `gnn-plan` and `pilot`.
    - Primary summary `20261101T061101Z_amazon_0b023c`, secondary summary `20261101T061819Z_amazon_0ae1da`.
    - [`GNN_RESULTS.md`](GNN_RESULTS.md) is generated by `scripts/gnn_report.py`.
- **2026-11-02 (fifth amendment, YelpChi results).** YelpChi ran from 06:40 UTC on 2026-11-01 to 17:52 UTC on 2026-11-02, with two pauses on request: 18:25 UTC on 2026-11-01 to 05:18 UTC on 2026-11-02, and 11:45 to 17:44 UTC on 2026-11-02. All 24 runs and both summaries are in, and the queue is empty.
  - **Run facts:**
    - Every entry was recorded from a clean tree: 1 run at `1bede69`, 16 at `a064974`, 7 at `c060a88`, and the summaries at `c060a88` and `a63d6f6`. Across those commits the only code change is three lines in `scripts/gnn_report.py`, which no run uses.
    - All 15 df-analyze runs passed the split and column-type checks (32, 131, 56, 155 and 132 columns for B, E1, E2, E7 and E8).
    - All 3 E8 runs scored the shared CV with the nested out-of-fold `gnnsup` on each fold's fit rows.
    - The pauses lost no results: split 2's E5g kept its 6 saved grid settings, and the secondary summary kept its 15 finished comparison pages. The start times and durations of those two entries cover only the resumed part.
  - **Scores** (test PR-AUC of Model A or the GNN rung, mean ± SD over the 3 splits; no skill 0.145; per split in [`GNN_RESULTS.md`](GNN_RESULTS.md)):

    | Setup | Test PR-AUC | CV PR-AUC |
    |---|---|---|
    | B (own) | 0.838 ± 0.007 | 0.822 |
    | E1 (+ nbr) | 0.871 ± 0.013 | 0.860 |
    | E2 (+ gnnssl) | 0.884 ± 0.004 | 0.868 |
    | E7 (+ nbr + gnnssl) | 0.855 ± 0.021 | 0.845 |
    | E8 (+ nbr + gnnsup) | 0.817 ± 0.016 | 0.811 |
    | E5 (GraphSAGE) | 0.767 ± 0.007 | 0.761 |
    | E5g (R-GCN-style) | 0.767 ± 0.005 | 0.761 |
    | E6 (no edges) | 0.630 ± 0.020 | 0.625 |

    - The GNN rungs' per-seed SD of test PR-AUC is at most 0.006.
    - **Best GNN for E8 (one-SE rule):** E5 on splits 0 and 2, E5g on split 1.
  - **Primary family (YelpChi):** every verdict matches its expectation, and E8 − E1 goes further than expected (significantly below, not just unsupported). Holm changes no verdict.

    | Contrast | Mean Δ ± SD | Splits > 0 | CV Δ | CV p | Holm p | Verdict | Expected |
    |---|---|---|---|---|---|---|---|
    | E1 − B | +0.033 ± 0.020 | 3 of 3 | +0.037 | 0.0079 | 0.016 | supported | supported |
    | E2 − B | +0.046 ± 0.007 | 3 of 3 | +0.046 | 2.4e-06 | 9.5e-06 | supported | supported |
    | E5 − E6 | +0.136 ± 0.014 | 3 of 3 | +0.136 | 3.2e-10 | 1.6e-09 | supported | supported |
    | E7 − E1 | −0.016 ± 0.023 | 1 of 3 | −0.014 | 0.24 | 0.24 | worse (lower mean) | not supported |
    | E8 − E1 | −0.054 ± 0.029 | 0 of 3 | −0.049 | 0.0015 | 0.0044 | worse (significant) | not supported |

  - **Secondary** ("worse" with p ≥ 0.05 is a lower mean, not a significant drop; here every "worse" has p < 0.05):
    - **Supported:** E8 − E5: +0.050 ± 0.010, p 1.0e-05.
    - **Significantly worse:**
      - E5 − B: −0.071 ± 0.002, p 3.7e-05.
      - E6 − B: −0.207 ± 0.013, p 2.4e-13.
      - E5 − E1: −0.104 ± 0.019, p 9.6e-08.
      - E7 − E2: −0.029 ± 0.025, p 0.011.
      - E8 − E7: −0.038 ± 0.029, p 0.0091.
    - **Not distinguishable:**
      - E5g − E5: +0.000 ± 0.002, p 0.93.
      - E2 − E1: +0.013 ± 0.014, p 0.35.
  - **Check after the verdicts** (not pre-registered; training folds only; `scripts/tuned_vs_default.py`). `select_model` also scores every classifier at default settings on the saved folds. Best shared-CV PR-AUC, mean over the 3 splits:

    | Setup | Tuned | Default |
    |---|---|---|
    | B | 0.822 | 0.798 |
    | E1 | 0.860 | 0.873 |
    | E2 | 0.868 | 0.834 |
    | E7 | 0.845 | 0.870 |
    | E8 | 0.811 | 0.879 |

    - Tuning lifts B and E2 but lowers E1, E7 and E8. The largest case is LightGBM on E8, which falls from 0.876–0.880 at default settings to 0.775–0.792 tuned.
    - At default settings, E8 has the highest CV score of the five and E7 is level with E1. So in CV the shortfalls of E7 and E8 against E1 belong to the tuned models, not to the columns.
    - At default settings, E1 and E2 are still above B (+0.075 and +0.036), so the gain from graph features holds in CV either way. The order of E1 and E2 flips.
    - On Amazon, tuned and default differ by at most 0.003 for every setup, so this is specific to YelpChi.
    - **Not settled:**
      - No default-settings model was scored on the test set.
      - Which part of the tuning is responsible is not separated: the 10 trials, the balanced-accuracy objective, or df-analyze's random internal CV.
      - Model A stays the best tuned row (invariant 8), so the verdicts above stand as pre-registered.
  - **Readings** (pilot budget; all 32 features; B starts at 0.84, so there is more headroom than on Amazon):
    - **Graph features help df-analyze here too.** E1 − B and E2 − B are supported and survive Holm. The gains are larger than Amazon's (+0.033 and +0.046 against +0.011 and +0.011) and close 20% and 29% of the headroom.
    - **The edges matter to the GNN here.** E5 − E6 is +0.136 on YelpChi and +0.002 on Amazon. This fits, but does not prove, the plan's reading that `f19` leaves Amazon little room for a graph effect (GNN_MODELS_PLAN, risk 6).
    - **The GNN classifiers still trail df-analyze.** E5 is 0.071 below B and 0.104 below E1 (both significant), as pre-registered for E5 − E1 and as on Amazon.
    - **Symmetric normalisation does not replicate.** E5g − E5 is +0.000 here against +0.011 on Amazon, so the Amazon gap should not be read as a general gain.
    - **Neighbour means and DGI are not separated,** on either dataset (E2 − E1 +0.013 here, +0.001 on Amazon). DGI training was not tested on YelpChi (no E3).
    - **Combining did not help at the pilot budget, but this does not show that the columns hurt.** E7 is below E1 on 2 of 3 splits (not significant) and below E2 on all 3 (significant). E8 is below E1 and E7 on all 3 (significant). The check above ties these shortfalls, in CV, to the tuned models.
    - **Own features and neighbour means add to the GNN's score.** E8 − E5 is +0.050 (supported), as on Amazon (+0.030).
  - **Caveats:**
    - Pilot budget and 3 splits, with each side's Model A chosen on the same folds.
    - CV p values are two-sided, and Holm is reported only as a sensitivity check.
    - The tuned-versus-default check was added after seeing the verdicts, and each cell is the best of four feature selections, which is slightly optimistic for both columns.
    - The chosen GNN setting sits at a corner of the grid on every rung and split (hidden 128, dropout 0.3), so E5, E5g, E6 and `gnnsup` may be understated.
  - **Results DB:**
    - 24 pipeline entries, tags `gnn-plan` and `pilot`.
    - Primary summary `20261102T111849Z_yelpchi_7f5b15`, secondary summary `20261102T174428Z_yelpchi_2e772f`.
    - Tuned-versus-default checks `20261102T175623Z_yelpchi_b779cf` and `20261102T175623Z_amazon_dbcf85`.
- **2026-11-02 (default-settings sensitivity check on YelpChi: pre-registered, written before any default-settings model was scored on a test set).** Caelen approved the check after the entry above. It asks whether the tuned-versus-default pattern in CV carries over to the test set.
  - **Status:** a labelled sensitivity check. It replaces no verdict, Model A stays the best tuned row (invariant 8), and the test set chooses nothing (invariant 4).
  - **Setups:** B, E1, E2, E7 and E8 on YelpChi splits 0–2 (15 fits). The GNN rungs have no df-analyze tuning, so they are not included.
  - **Model, per setup and split:** the `settings == default` row with the highest shared-CV PR-AUC in `model_selection_cv.csv` (classifier and feature selection), first row on a tie. It is fit on the whole training export with df-analyze's default hyperparameters and seed 555, and predicts the test export. E8's training rows carry the out-of-fold `gnnsup`, as for its tuned Model A. Only PR-AUC is reported, so no threshold is needed.
  - **Contrasts and verdicts:** the 2026-10-19 rule, with test Δ from these fits and the CV p from the same default rows' saved out-of-fold predictions. No Holm, as this is not a confirmatory family.

    | Contrast | Default-settings CV Δ (known) | Expected |
    |---|---|---|
    | E1 − B | +0.075 | supported |
    | E2 − B | +0.037 | supported |
    | E7 − E1 | −0.002 | not supported |
    | E8 − E1 | +0.006 | not supported |
    | E2 − E1 | −0.038 | worse |
    | E7 − E2 | +0.036 | supported |
    | E8 − E7 | +0.008 | not supported |

  - **How it will be read, fixed now:**
    - **Shortfall tied to tuning:** at default settings, E7 − E1 and E8 − E1 each have a mean test Δ of at least −0.01, and neither has a negative CV Δ with p < 0.05.
    - **The combination helps:** E7 − E1 or E8 − E1 gets the verdict "supported". Even then it is a sensitivity result and needs a pre-registered run to count.
    - **Otherwise:** the CV pattern did not carry over, and the cause stays open.
  - **Code check:** the same script also refits each tuned Model A. If any of its test PR-AUCs differs from the pipeline's by more than 0.005, the check is reported as failed.
  - **Not answered, whatever the outcome:** which part of the tuning is responsible (the 10 trials, the balanced-accuracy objective, or the random internal CV).
- **2026-11-02 (default-settings sensitivity check on YelpChi: results).** Ran once, 18:03–18:05 UTC, at `c5d2932`, after the pre-registration above. By shared CV, all 15 fits took LightGBM with every column. The code check passed: each tuned Model A's refit is within 0.0032 of the pipeline's test PR-AUC (tolerance 0.005).
  - **Scores** (PR-AUC, mean over the 3 splits; no skill 0.145):

    | Setup | Test, tuned Model A | Test, default | CV, tuned | CV, default |
    |---|---|---|---|---|
    | B | 0.838 | 0.812 | 0.822 | 0.798 |
    | E1 | 0.871 | 0.883 | 0.860 | 0.873 |
    | E2 | 0.884 | 0.844 | 0.868 | 0.834 |
    | E7 | 0.855 | 0.879 | 0.845 | 0.870 |
    | E8 | 0.817 | 0.893 | 0.811 | 0.879 |

  - **Contrasts at default settings:**

    | Contrast | Mean test Δ ± SD | Splits > 0 | CV Δ | CV p | Verdict | Expected |
    |---|---|---|---|---|---|---|
    | E1 − B | +0.071 ± 0.008 | 3 of 3 | +0.075 | 1.6e-07 | supported | supported |
    | E2 − B | +0.032 ± 0.006 | 3 of 3 | +0.037 | 5.2e-05 | supported | supported |
    | E7 − E1 | −0.003 ± 0.006 | 1 of 3 | −0.002 | 0.52 | worse (lower mean) | not supported |
    | E8 − E1 | +0.010 ± 0.003 | 3 of 3 | +0.006 | 0.073 | not distinguishable | not supported |
    | E2 − E1 | −0.039 ± 0.009 | 0 of 3 | −0.038 | 1.6e-07 | worse (significant) | worse |
    | E7 − E2 | +0.035 ± 0.010 | 3 of 3 | +0.036 | 1.8e-07 | supported | supported |
    | E8 − E7 | +0.014 ± 0.007 | 3 of 3 | +0.008 | 0.0039 | supported | not supported |

  - **Outcome under the rule fixed beforehand: the shortfall is tied to tuning.** At default settings E7 − E1 is −0.003 and E8 − E1 is +0.010, both above −0.01, and neither has a negative CV Δ with p < 0.05. "The combination helps" is not met: E8 − E1 is above zero on all 3 splits, but its CV p is 0.073.
  - **Expectations:** six of seven match. E8 − E7 was expected to be unsupported and came out supported.
  - **Readings:**
    - **The tuned verdict on E8 should not be read as the supervised score hurting the model.** E8 falls from 0.893 to 0.817 when tuned, while E1 moves from 0.883 to 0.871. The verdicts of the YelpChi results entry stand as pre-registered, but for E7 and E8 they describe the pilot's tuning as much as the columns.
    - **The gain from graph features holds at both settings.** E1 − B and E2 − B are supported either way.
    - **The order of the setups does not.** Tuned: E2 > E1 > E7 > B > E8. Default: E8 > E1 > E7 > E2 > B. Tuning moves a setup's test PR-AUC by −0.076 to +0.040, as much as the contrasts being tested, so the pilot's tuning is not a stable basis for ranking YelpChi's setups against each other.
    - **Tuning is not simply bad.** It helps B (+0.026) and E2 (+0.040) on the test set and hurts E1 (−0.012), E7 (−0.024) and E8 (−0.076). The three it hurts are the ones with the neighbour-mean columns, which may be a clue, but that was not tested.
  - **Caveats:**
    - A sensitivity check on 3 splits, proposed after the tuned verdicts and the CV pattern were known.
    - It does not make E8 the best model. E8's default score is above every tuned setup, but comparing across settings was not pre-registered, and default settings were chosen by no one as the method.
    - Which part of the tuning is responsible is still not separated. Amazon's test sets were not scored this way; its CV shows no such gap (at most 0.003).
  - **Results DB:** contrasts `20261102T180319Z_yelpchi_833c47`, scores per setup and split `20261102T180319Z_yelpchi_3ec316`.
- **2026-11-02 (up to 100 trials on YelpChi: pre-registered, written before any of these runs started).** Caelen approved the runs after the entry above. They ask whether the pilot's 10 trials explain E7's and E8's shortfall against E1.
  - **Status:** a labelled sensitivity check. It replaces no verdict: the pilot verdicts of the YelpChi results entry stand as pre-registered. Within each run, Model A stays the best tuned row by shared CV (invariant 8), and the test set chooses nothing (invariant 4).
  - **Setups:** B, E1, E2, E7 and E8 on YelpChi splits 0–2 (15 runs), as new feature sets `t100_*`, so the pilot's folders are not touched. The GNN rungs have no df-analyze tuning, so they are not included.
  - **What changes:** only `df_analyze.htune_trials`, from 10 to 100. The classifiers (LightGBM, logistic regression, CatBoost), the balanced-accuracy objective, df-analyze's random internal CV, the prediction filter, the seed, the splits and the feature blocks stay the same. The `nbr`, `gnnssl` and `gnnsup` blocks are reused, not rebuilt. This is not the full Model A budget, which has six classifiers.
  - **"100 trials" means up to 100:** df-analyze ends a search early when it stops improving (in one full-budget Amazon run, these three classifiers' searches ended at 51–71 of 100), and it has time limits per search (CatBoost 30 min, LightGBM 60 min, logistic regression 15 min). How each search stopped is reported from `tuning_budget.csv`.
  - **Run-to-run noise:** df-analyze's feature selection is not seeded (see "Pipeline stages" in `AGENTS.md`). The best default row is scored in both runs with the same settings, so its CV difference between them shows the noise.
  - **Contrasts and verdicts:** the 2026-10-19 rule, from `compare_report.py` and `repeat_summary.py`, each side being that setup's Model A at up to 100 trials. No Holm, as this is not a confirmatory family.

    | Contrast | Test Δ at 10 trials (known) | Test Δ at default settings (known) | Expected at 100 |
    |---|---|---|---|
    | E1 − B | +0.033 | +0.071 | supported |
    | E2 − B | +0.046 | +0.032 | supported |
    | E7 − E1 | −0.016 | −0.003 | not supported |
    | E8 − E1 | −0.054 | +0.010 | not supported, with a smaller shortfall than at 10 trials |
    | E2 − E1 | +0.013 | −0.039 | not supported |
    | E7 − E2 | −0.029 | +0.035 | not supported |
    | E8 − E7 | −0.038 | +0.014 | not supported |

  - **The budget itself:** each setup at up to 100 trials minus the same setup at 10, under the same rule. Expected: E8 gains the most of the five. There is no firm expectation for which gains are supported.
  - **How it will be read, fixed now**, using the thresholds of the default-settings check:
    - **The trial count explains the shortfall:** at up to 100 trials, E7 − E1 and E8 − E1 each have a mean test Δ of at least −0.01, and neither has a negative CV Δ with p < 0.05.
    - **The trial count does not explain it:** E7 − E1 and E8 − E1 both fail that test.
    - **Partly:** one passes and the other fails. It is reported per contrast.
    - **The combination helps:** E7 − E1 or E8 − E1 gets the verdict "supported". Even then it is a sensitivity result and needs a pre-registered run to count.
  - **Not answered, whatever the outcome:** whether the balanced-accuracy objective or the random internal CV matters on its own. If the trial count does not explain the shortfall, those two remain candidates; they are not shown to be the cause.

---

## 12. Status and next steps

**State (2026-10-20).** Phase 2 is merged into `main` (PR #37). Models 2 (`nbr`) and 5 (`gnnssl`) are implemented as feature blocks, each also without the four #36 features (`_no36`), with the comparison page, a CV probe and the training-only checks. Results in §11:
- the full Model A both ways (0.919 vs 0.518);
- the six pilot rungs on 3 splits with pre-registered verdicts;
- the random-graph control;
- the #36 checks (a)–(b);
- embedding sensitivity.

`AGENTS.md` ("Current state") has the one-screen summary of what was run and what is known or unknown. Tested on Linux (WSL2) with an NVIDIA GPU; 200 tests pass with the `gnn` extra, and without it the GNN tests are skipped.

**Phase 3 status (2026-10-23, §11):** Task 1 (sizes and size sweep), Task 3a (untrained-GNN probe and pilot rungs) and Task 5 (ranking, probe, and pilot rungs for both readings of D3, `only36` and `top4`) are done at the pilot budget. Open for the group: D1 (Task 2b–c), D2 (Task 4, GraphSAGE), D3 (which four features: the results favour `top4`, though its fourth feature is unstable across splits), D4 (Task 6, datasets), D5 (backfilling), and whether any of the new rungs deserves a full-budget run.

**Next steps (phase 3; ask before starting):** the professor's feedback comes first ([`PHASE3_FEEDBACK_PLAN.md`](PHASE3_FEEDBACK_PLAN.md); Task 1 absorbs step 1 below). Record every run in the results database (`results/README.md`).
0. **Settle issue #36 locally first.** (a) and (b) ran on 2026-10-20 (§11): (a) found no helpful-vote partner for f19; (b) is consistent with rating statistics. (c)–(e) remain. (training/labelled rows, aggregates only, minutes of CPU; scripts in `scripts/`, outputs under `outputs/amazon/`):
    - (a) for every non-negative column c, does c / (c + f19) give < 0.2 for every fraud user and > 0.8 for every benign one? If so, c = helpful votes and f19 = unhelpful votes: leakage proven. Also check whether min(c + f19) ≥ 20, a minimum-vote filter;
    - (b) f15 ∈ {1, 1.5, …, 5}, f18 ∈ [1, 5], Spearman(f15, f18) > 0.8, and f18 / f13 against any five rating-level count columns (f18 = Σk·n_k / Σn_k, f13 = (n_4 + n_5) / Σn_k);
    - (c) sort all 25 columns into rating / vote / time / text families by such identities;
    - (d) read CARE-GNN's `amazon_preprocess.py` at `a64ff75` in a clone outside this repo;
    - (e) optionally rebuild the per-user statistics from McAuley & Leskovec's 2013 Musical Instruments reviews (SNAP, about 20 MB) and compare the distributions; that is distribution-level evidence only, since IDs are anonymised.

    Then, with the group, add a variant without every vote-family column (likely keeping f13, f15, f18) and run all rungs on it. That is the candidate Amazon headline.
1. Choose the embedding settings on training data only: a small grid (dim 16/32/64, epochs 100/300, PCA k 4/8/16), scored by the probe on the grouped CV folds. Two relations hit the 300-epoch cap.
2. Full-budget runs of `m2_own_nbr`, `m5_own_gnnssl` and the three `_no36` rungs, on the same machine as `m1_own`; about 2 h each.
3. `split.n_repeats: 3` for every compared rung, with the scaler and PCA refit per split. Report the per-repeat Δ and the Nadeau–Bengio corrected t-test over the 15 paired CV folds.
4. YelpChi: the same ladder.
5. Ablations: GraphMAE, or a homogeneous (`homo`) embedding.
6. The group's reading of #36 once both variants have been seen.

**Phase 1 hand-off (2026-10-14), kept for reference.**

**State of this branch (`feat/review-fraud-pivot`).** M0–M9 are done, and phase 1 is done: the full Amazon Model 1 run on the GPU (§11), the `export_model` stage, `predict.py`, and a bundle a teammate can use without retraining. The branch has been pushed and opened as a PR to `main`. Tested on Linux (WSL2, Debian) with an NVIDIA GPU; not tested on macOS (the CPU-only decode was simulated by hiding the GPU).

**Next steps.**
1. **One YelpChi validation run** (overnight): `uv run run.py --dataset yelpchi`. Estimated 5–8 h for df-analyze plus about 45 min for `select_model` (extrapolated). Then `export_model` and a bundle.
2. **Model 2** (neighbour-aggregate blocks per relation, label-blind builders as in `tests/test_label_blindness.py`). Then switch Amazon to `split.n_repeats: 3` for the Model 1 vs Model 2 comparison.
3. **#36:** the group decides whether to keep, drop, or report both ways the four Amazon features that may encode the label.

The setup steps below are kept for reference.

**Setup (local, with the GPU).** Follow `AGENTS.md`.
1. Setup:
   - `git fetch origin && git checkout feat/review-fraud-pivot && uv sync`
   - df-analyze at the pinned commit:
     ```
     git -C ../df-analyze fetch origin
     git -C ../df-analyze checkout 199e5638620693c267dac715784f1fd0e33fa796
     uv sync --locked --python '>=3.13.11,<3.14' --directory ../df-analyze
     ```
   - uv must be 0.9.16 or newer.
2. `./download_dataset.sh`: about 45 MB of zips, checksum-verified.
3. `uv run run.py --dataset amazon --to audit` (minutes). Check `outputs/amazon/reports/audit.md`:
   - the counts match;
   - the four shortcut flags reappear (`own__f13`, `own__f15`, `own__f18`, `own__f19`, AUROC 0.86–0.88).
4. **Full Amazon run:** `uv run run.py --dataset amazon` (done 2026-10-13: 2 h 05 min on an RTX 4060 laptop; §11).
   - Estimated 1–3 h on an RTX 4060. The pilot took 24 min with 10 trials on a CPU.
   - If it is interrupted: `uv run run.py --dataset amazon --resume`.
   - Then check that `split_check.json` and `type_check.json` exist and the report has no sanity-band warning.
   - Record the timings and the Model A result in §11.
5. **YelpChi, once:**
   - First `uv run run.py --dataset yelpchi --to audit` (minutes).
   - Then the full run overnight: `uv run run.py --dataset yelpchi`. Estimated 5–8 h for df-analyze plus about 45 min for `select_model`; this is extrapolated, not measured. kNN and GANDALF are the likely bottlenecks.
6. Share the outputs with `uv run pack_artifacts.py --dataset <name>`, outside git. The bundle includes every tuned model and the exported Model A (README, "Sharing outputs with the group").

**Still open** (§10.4–6): Model 6; the flagged Amazon features; the CFPB pilot's place in the report; and which dataset is the main one (see §11).

**After that:** Model 2 (neighbour-aggregate blocks per relation, label-blind builders as in `tests/test_label_blindness.py`). Then switch Amazon to `split.n_repeats: 3` for the Model 1 vs Model 2 comparison.
