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
- **2026-10-23 (phase 3 overnight: training-only probes, then pilot rungs; rules written before any pilot result).** Branch `exp/phase3-feedback`, on top of the phase 3 plan's branch (`claude/awesome-wozniak-wm548a`, not yet on `main`). The user asked for "the next set of experiments" overnight, which is the plan's "go" for the pilot rungs of Tasks 3a and 5. Split 0's training nodes and saved grouped folds only, unless stated; all probes use the fixed HistGradientBoosting of `models/probe.py`.
  - **Task 5.1, which four features (`scripts/top_features.py`, `probe/top_features.csv`).** Permutation importance (drop in held-out-fold PR-AUC, 10 shuffles, mean over 5 folds): `f19` 0.759 ± 0.036, `f00` 0.074 ± 0.014, `f23` 0.010 ± 0.010, `f18` 0.007 ± 0.006, then `f12` 0.007 ± 0.005 (ranks 3–5 are within noise). `f13` and `f15` have about zero importance: once `f18` and `f19` are known they add nothing. By univariate AUROC the top 4 are the four #36 features. **So D3's two readings differ on Amazon.** Both are run as pilots, so the group can choose without another night: `only36` (the four #36 features) and `top4` (`f00`, `f18`, `f19`, `f23`, the permutation top 4, fixed now from this table).
  - **Task 5.3 probe (`probe/only36_probe.csv`):** own / + nbr / + gnnssl: all 25 features 0.907 / 0.935 / 0.934; without #36 0.500 / 0.704 / 0.672; only36 0.876 / 0.918 / 0.912; top4 0.910 / 0.938 / 0.931. Four features carry the 25 features' result.
  - **Task 3a probe (`probe/gnnrand_probe.csv`):** own + gnnrand (the gnnssl encoder with its seeded initial weights, no training) over 5 initialisation seeds: all features 0.933 ± 0.004 (gnnssl 0.934, nbr 0.935); without #36 0.677 ± 0.007 (gnnssl 0.672, nbr 0.704). On CV, DGI training adds nothing over random weights.
  - **Task 1.1 (`scripts/embedding_sizes.py`, `gnn_embed/sizes.md`):** DGI gives 64 dimensions per relation (192 on Amazon); none is constant on the training nodes, but the effective rank (entropy of the singular values) is 27 / 8 / 15 (U-P-U / U-S-U / U-V-U; without #36 26 / 6 / 17). 90% of the variance needs 6 / 2 / 3 components, 99% needs 27 / 5 / 9. df-analyze receives 24 `gnnssl` columns (PCA 8 per relation) and both pilot `m5` Model As kept all 24 after feature selection; pilot `m2`'s Model A kept 13 of 78 `nbr` columns (all 66 without #36).
  - **Runs (pilot settings: lgbm, lr, catboost, 10 trials, `bal-acc`, seed 555; splits 0, 1, 2; one at a time on the RTX 4060 laptop):** `pilot_m1_own_only36`, `pilot_m2_own_nbr_only36`, `pilot_m5_own_gnnssl_only36`; `pilot_m5r_own_gnnrand`, `pilot_m5r_own_gnnrand_no36`; `pilot_m1_own_top4`, `pilot_m2_own_nbr_top4`, `pilot_m5_own_gnnssl_top4`. 24 runs. Every graph block keeps the plan's default settings (DGI 64, ≤ 300 epochs, PCA 8); gnnrand uses the run seed, so it is exactly the encoder gnnssl's training starts from. The size sweep (Task 1.2) is a training-CV check and changes no run.
  - **Contrasts** (B − A; test ΔPR-AUC per split from `compare_report.py`, verdict by the rule of 2026-10-19: Δ > 0 on all 3 splits, mean Δ > SD, Nadeau–Bengio p < 0.05 over the 15 CV folds; `scripts/repeat_summary.py --pairs`):
    - Task 3a (`compare/gnnrand_summary.md`): m5 − m5r (does DGI training add anything over random weights?) and m5r − m1, each with all features and without #36.
    - Task 5 (`compare/only36_summary.md`, `compare/top4_summary.md`): m1_only36 − m1 and m1_top4 − m1 (how much of the 25 features' result do four carry?), and within each variant m2 − m1, m5 − m1, m5 − m2.
  - **Reading, fixed now:** "worse" for m1_only36 − m1 means the other 21 features add something measurable; "not distinguishable" means the four carry the result. For m5 − m5r, anything but "supported" means DGI training is not shown to help. No pilot result is compared with a full-budget run.

---

## 12. Status and next steps

**State (2026-10-20).** Phase 2 is merged into `main` (PR #37). Models 2 (`nbr`) and 5 (`gnnssl`) are implemented as feature blocks, each also without the four #36 features (`_no36`), with the comparison page, a CV probe and the training-only checks. Results in §11:
- the full Model A both ways (0.919 vs 0.518);
- the six pilot rungs on 3 splits with pre-registered verdicts;
- the random-graph control;
- the #36 checks (a)–(b);
- embedding sensitivity.

`AGENTS.md` ("Current state") has the one-screen summary of what was run and what is known or unknown. Tested on Linux (WSL2) with an NVIDIA GPU; 179 tests pass with the `gnn` extra, and without it the GNN tests are skipped.

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
