# Dashboard plan: every method, per dataset, across datasets, and a live classify button

**Status: built on 2026-11-02.** How to run it is in [`DASHBOARD.md`](DASHBOARD.md). This is the plan it was built from, kept for the record; §9 lists the decisions the group's open questions got and where the build differs from the plan.

## 1. Goal

One local page the group can open to:

1. **See every method we tried** on a dataset: each setup (B, E1, E2, E3, E5, E5g, E6, E7, E7r, E8, E8s, N1–N3, and the feature variants all / without #36 / without `f19` / top 4 / #36 only), with test PR-AUC, CV PR-AUC, the no-skill line, the number of splits, the budget (pilot / full), and the pre-registered verdicts.
2. **Toggle between datasets** (Amazon ↔ YelpChi) with the whole page following the toggle.
3. **Compare across datasets**: the setups that ran on both, on a scale that accounts for their different base rates, and whether each contrast's verdict agrees.
4. **Classify a random sample** with the best model of each dataset: one button per dataset that draws a random *test* node, runs the saved best model on it live, and shows the probability, the decision at the saved threshold, the true label, and why (its most extreme features, and its graph neighbourhood for graph setups).

## 2. What it reuses

| Need | Existing piece |
|---|---|
| Aggregate results for every recorded run | `results/runs/` via `fraud_detect.results_db` |
| Verdict tables | `kind: compare` entries (from `scripts/repeat_summary.py`) |
| Setup names and descriptions | moved from the report scripts into `dashboard/registry.py` |
| Saved best model and its loader | `export_model` stage, `scripts/dfa/model_io.py`, `models/model_export.py` |
| Model input for a node | df-analyze's prepared `X_test_00.csv` |
| GNN rung predictions | `models/gnn_outputs.test_probabilities` |
| Per-node details | `report/web.top_features`, `report/compare.ego_network` and `ego_svg` |
| Styling | `report/report.css` |

## 3. Architecture

```
browser (127.0.0.1 only)
   │  GET /, /api/*        POST /api/classify
   ▼
dashboard server  ── project env (uv run scripts/dashboard.py)
   │  reads results/runs/ (aggregate)       reads outputs/<ds>/... (row-level, local)
   │
   └─ JSON lines over stdin/stdout
      ▼
model worker      ── df-analyze env (scripts/dfa/serve_model.py), started once,
                     loads each dataset's best model once, predicts in milliseconds
```

- **No new dependencies.** The standard library's `ThreadingHTTPServer`; one HTML file with vanilla JS and inline SVG.
- **Same rules as the comparison page:** not a pipeline stage, no external requests, every inserted value escaped, a group-only banner on the classify section.
- **Two environments, kept separate.** The worker is the only part in df-analyze's environment.
- **Local only.** The server binds `127.0.0.1`; the page is never published, committed or bundled.
- **Degrades gracefully.** Without `outputs/` or the exported model, the aggregate sections still work and the classify panel says what it needs.

## 4. Components

- **Registry** (`dashboard/registry.py`): one table every section reads, so a setup has one name. A test fails if a database entry maps to no setup.
- **Data layer** (`dashboard/data.py`): pure functions over `results/runs/`, no row-level data. Applies `supersedes`, keeps failed runs out of the means but visible, never mixes budgets, and lets a verdict on more splits replace one on fewer.
- **Methods section:** dataset toggle, Amazon variant toggle, budget filter, leaderboard chart, table, verdicts, coverage note.
- **Cross-dataset section:** normalised skill `(PR-AUC − no-skill) / (1 − no-skill)`, the gain over B per dataset, and a verdict agreement table, with the caveats beside them.
- **Classify panel:** buttons for a random test node, a random fraud and a random not-fraud; a seed; the live P(fraud) checked against the probability df-analyze saved; the decision at the saved threshold; the true label; the node's most extreme features; its neighbourhood per relation; what B said; a history strip and a tally.
- **Model worker** (`scripts/dfa/serve_model.py`): checks columns on every request; shares its loading code with `predict.py`; exits when stdin closes; restarted once if it dies.

## 5. Prerequisites

The exported models of the best setups, on split 0. They already existed from the pilot runs, so nothing had to be rerun.

## 6. Milestones and acceptance criteria

| # | Deliverable | Done when | Outcome |
|---|---|---|---|
| M1 | Registry and data layer | Every pipeline entry maps to a setup; `setup_summary` reproduces the means in `docs/GNN_RESULTS.md` and `docs/RQ_RESULTS.md` to 3 decimals; budgets never mix | Met; `tests/test_dashboard_committed.py` |
| M2 | Server and page, no model | `--no-model` works on a fresh clone with no `outputs/`; toggles change every chart; checked on Linux/WSL2 and macOS | Met on Linux/WSL2. **macOS was not checked** |
| M3 | Model worker | The worker's P(fraud) for every test row of each best setup matches df-analyze's saved probabilities within `tolerance_for`; `predict.py` still passes its tests | Met: 3,456, 3,456 and 18,383 rows, largest difference 0. The server repeats this check at every start |
| M4 | Classify panel | Each button returns in < 1 s; stratified draws only return the asked class; the verified mark shows; the GNN fallback works on a synthetic GNN output | Met: about 34 ms per click |
| M5 | Polish | B comparison on each draw, history strip, URL state, README section | Met, except that screenshots for slides were not made |

## 7. Tests

Synthetic data only, as `AGENTS.md` requires: `tests/test_dashboard_registry.py`, `test_dashboard_data.py`, `test_dashboard_checks.py`, `test_dashboard_server.py`, `test_dashboard_classify.py`, `test_dashboard_static.py` and `test_serve_model.py`. `test_dashboard_committed.py` reads the committed results database (aggregate numbers). `test_export_model_dfa.py` runs the real worker in df-analyze's environment when that environment is available.

## 8. Open decisions, as they were put to the group

- **D1.** What "best model" means.
- **D2.** Which Amazon variant the page opens on.
- **D3.** Whether to backfill the runs missing from `results/runs/`.
- **D4.** Plain standard library and vanilla JS, or Streamlit/Dash.
- **D5.** How to share it.

## 9. Decisions, and where the build differs

**Decided by the user on 2026-11-02:**

- **D1:** the setup with the highest mean shared-CV PR-AUC over its splits, at the largest budget that has every setup. The test set plays no part. The page lists the setups within one SD of it, and says so when there is no clear winner.
- **D2:** Amazon opens on all 25 features with a permanent `f19` warning, following the 2026-10-31 decision that the main comparison uses every feature. The plan had proposed without `f19`. Both models can be classified with.
- **D3:** yes. 20 earlier Amazon runs and phase 2's verdict table were recorded from their saved outputs (tag `backfill`). This turned out to be necessary, not optional: before it, Amazon's all-feature B, E1 and E2 had entries on splits 3–4 only, so the page would have shown B at 0.928 from two splits (0.922 over all five).
- **D4:** the plan's proposal, as the user asked for the plan to be implemented: plain standard library and vanilla JS.
- **D5:** local only, with setup instructions for Windows and WSL. No static snapshot was asked for.
- **The 100-trial check** the plan was waiting on was stopped: there was no time for it. It had no results, so the `t100` budget was removed.

**Where the build differs from the plan:**

- **Normalised skill is one row per setup, not a slope chart.** Amazon's values sit too close together for slope labels to stay readable.
- **The worker's requests name a model, not a dataset,** because Amazon has two.
- **"Worse" is split in two.** The verdict rule calls any negative mean "worse". The page shows those with p ≥ 0.05 as "lower mean, not significant", because some are as small as −0.00003.
- **"Why" is labelled as "most unusual features".** They are a node's highest values against the training nodes, not an explanation of the model, and the page says so.
- **Extra safeguards:** the server refuses requests addressed to any other host and sets a Content-Security-Policy, and it recomputes twelve sanity checks at start-up (see `DASHBOARD.md`).
