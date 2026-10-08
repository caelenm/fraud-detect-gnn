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
- **Do not push the branch or open a PR until the user explicitly says so.** The repository is public.
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

---

## 12. Status and next steps (hand-off, 2026-10-08)

**State of this branch (`feat/review-fraud-pivot`).** M0–M9 are done. Every stage runs end to end, and the Amazon pilot passed on a CPU-only cloud machine (§11). Not yet done:
- the full runs;
- a run on a GPU;
- testing on macOS and WSL2.

**Next session (local, with the GPU).** Follow `AGENTS.md`.
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
4. **Full Amazon run:** `uv run run.py --dataset amazon`.
   - Estimated 1–3 h on an RTX 4060. The pilot took 24 min with 10 trials on a CPU.
   - If it is interrupted: `uv run run.py --dataset amazon --resume`.
   - Then check that `split_check.json` and `type_check.json` exist and the report has no sanity-band warning.
   - Record the timings and the Model A result in §11.
5. **YelpChi, once:**
   - First `uv run run.py --dataset yelpchi --to audit` (minutes).
   - Then the full run overnight: `uv run run.py --dataset yelpchi`. Estimated 5–8 h for df-analyze plus about 45 min for `select_model`; this is extrapolated, not measured. kNN and GANDALF are the likely bottlenecks.
6. Share the outputs with `uv run pack_artifacts.py --dataset <name>`, outside git.

**Still open** (§10.4–6): Model 6; the flagged Amazon features; the CFPB pilot's place in the report; and which dataset is the main one (see §11).

**After that:** Model 2 (neighbour-aggregate blocks per relation, label-blind builders as in `tests/test_label_blindness.py`). Then switch Amazon to `split.n_repeats: 3` for the Model 1 vs Model 2 comparison.
