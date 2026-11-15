"""The words on the page, with their numbers taken from the database.

Anyone reading the dashboard should meet the limits of these results before the
results. Numbers come from data.py; a sentence whose number is missing is left
out rather than written from memory. The findings about f19 and the default
settings repeat what docs/RESEARCH_PLAN.md §11 records.
"""

from __future__ import annotations

from typing import Any

from fraud_detect.dashboard import data
from fraud_detect.dashboard import registry as reg

VERDICT_RULE = (
    "A contrast B − A is “supported” if the test Δ PR-AUC is above zero on every "
    "split, its mean is above its SD across splits, and the Nadeau–Bengio corrected "
    "p over the training CV folds is below 0.05. The rule was fixed on 2026-10-19, "
    "before the runs. Otherwise a positive mean is “not distinguishable from zero” "
    "and a negative mean is “worse”. This page shows a “worse” with p ≥ 0.05 as “lower "
    "mean, not significant”, because it may be a tiny difference. A verdict on more "
    "splits replaces one on fewer."
)
# Not runs the database could show, so they cannot be derived from it. Each
# repeats docs/RESEARCH_PLAN.md §10–§11; what the database can show is derived
# in not_run().
NOT_RUN_STATIC = (
    "Rebuilding the Amazon features from the raw reviews (V1). The column map that "
    "names f19 rests on CARE-GNN's preprocessing script and identities that hold in "
    "the data; none of them tests f19's column directly.",
    "HGT (declined 2026-10-31), and any dataset beyond YelpChi and Amazon.",
)
DATASET_NAMES = {"amazon": "Amazon", "yelpchi": "YelpChi"}


def _mean(rows: list[dict[str, Any]], display_id: str) -> dict[str, Any] | None:
    return next((r for r in rows if r["display_id"] == display_id), None)


def split_list(splits: list[int]) -> str:
    """[0, 1, 2] -> '0–2'; [0, 2] -> '0 and 2'; [3] -> '3'."""
    if len(splits) > 2 and splits == list(range(splits[0], splits[-1] + 1)):
        return f"{splits[0]}–{splits[-1]}"
    if len(splits) > 1:
        return ", ".join(map(str, splits[:-1])) + f" and {splits[-1]}"
    return "".join(map(str, splits))


def _name(dataset: str) -> str:
    return DATASET_NAMES.get(dataset, dataset)


def budget_note(view: dict[str, Any]) -> str:
    budgets, total = view["budgets"], view["n_runs"]
    where = [
        f"{_name(ds)} splits {split_list(splits)}"
        if len(splits) > 1
        else f"{_name(ds)} split {split_list(splits)}"
        for ds, splits in view["full_budget_splits"].items()
    ]
    return (
        f"{budgets[reg.PILOT]} of the {total} recorded runs use the pilot budget: "
        "LightGBM, logistic regression and CatBoost, 10 tuning trials. "
        f"{budgets[reg.FULL]} use the full budget (6 classifiers, 100 trials), on "
        f"{' and '.join(where) or 'no split'}; not every setup has one. Tuned on "
        "balanced accuracy, with df-analyze's random internal CV."
    )


def machines_note(view: dict[str, Any]) -> str:
    seeds = ", ".join(map(str, view["seeds"])) or "unknown"
    devices = view["devices"]
    if len(devices) <= 1:
        return (
            f"Every run used one machine ({', '.join(devices) or 'unknown'}) and one "
            f"base seed ({seeds}); only the splits differ. Treat a difference smaller "
            "than the spread across splits as noise."
        )
    parts = []
    for device in devices:
        counts = view["device_runs"].get(device, {})
        bits = [f"{counts[b]} {b}" for b in reg.BUDGETS if counts.get(b)]
        parts.append(f"{device} ({', '.join(bits)})")
    return (
        f"Runs come from {len(devices)} machines with the same settings and one base "
        f"seed ({seeds}): {'; '.join(parts)}. The machines do not give bit-identical "
        "results (DGI embeddings differ between them; cause not established)."
        + _replications(view["n_replications"])
        + " Treat a difference smaller than the spread across splits as noise."
    )


def _replications(n: int) -> str:
    if not n:
        return ""
    what = "run that was" if n == 1 else f"{n} runs that were"
    return (
        f" The {what} repeated on a second machine count once, by the earliest run; "
        "the repeats are kept as replications."
    )


def not_run(runs: Any) -> list[str]:
    """What was not run, from the database where it can say, then the rest."""
    items = []
    gaps = data.full_budget_gaps(runs)

    def label(g: dict[str, Any]) -> str:
        return f"{_name(g['dataset'])}, {reg.variant_label(g['variant'], g['dataset'])}"

    partial = [g for g in gaps if not g["none_at_full"]]
    if partial:
        items.append(
            "No full-budget run, only the pilot: "
            + "; ".join(f"{label(g)}: {', '.join(g['missing'])}" for g in partial)
            + "."
        )
    pilot_only = [g for g in gaps if g["none_at_full"]]
    if pilot_only:
        items.append(
            "Pilot budget only, for every setup: "
            + "; ".join(label(g) for g in pilot_only)
            + "."
        )
    items.append(
        "The GNN classifiers (E5, E5g, E6, N3) have no df-analyze budget, so they "
        "were not rerun at the full budget; their runs are recorded under the pilot."
    )
    for g in data.pilot_split_gaps(runs):
        by_n: dict[int, list[str]] = {}
        for f in g["fewer"]:
            by_n.setdefault(f["n_splits"], []).append(f["display_id"])
        text = "; ".join(f"{', '.join(ids)} on {n}" for n, ids in sorted(by_n.items()))
        items.append(
            f"At the pilot budget, {label(g)}, some setups have {g['most']} splits; "
            f"these have fewer: {text}."
        )
    return items + list(NOT_RUN_STATIC)


def general_caveats(view: dict[str, Any], failed: list[dict[str, Any]]) -> list[str]:
    notes = [
        budget_note(view),
        "The test set chose nothing. Model A is the tuned model with the highest "
        "PR-AUC in a shared, grouped CV on training data. Those CV scores are slightly "
        "optimistic, because the same folds chose Model A.",
        machines_note(view),
        "Comparing PR-AUC between datasets is not meaningful, since their base rates "
        "differ; the cross-dataset section uses normalised skill.",
    ]
    if view["n_backfilled"]:
        notes.append(
            f"{view['n_backfilled']} early runs were recorded in the database after the "
            "fact, from their saved outputs (tag “backfill”). None was rerun."
        )
    for item in failed:
        notes.append(f"Not finished: {item['description']}.")
    return notes


def amazon_f19(
    runs_all: list[dict[str, Any]],
    runs_no19: list[dict[str, Any]],
    budget: str = reg.PILOT,
    variant: str = reg.ALL,
) -> str | None:
    """The f19 warning, with this page's own numbers at `budget`."""
    with_f19, without = _mean(runs_all, "B"), _mean(runs_no19, "B")
    if with_f19 is None or without is None:
        return None
    this_view = (
        "and so does this view" if variant == reg.ALL else "and this view keeps f19 too"
    )
    return (
        "Amazon's f19 is the minimum number of unhelpful votes on a single review, "
        "built from the same votes that define the label, so it likely leaks the label. "
        f"B scores {with_f19['test_pr_auc']:.3f} with all 25 features and "
        f"{without['test_pr_auc']:.3f} without f19 (test PR-AUC, {budget} budget, "
        f"{with_f19['n_splits']} and {without['n_splits']} splits). Published Amazon "
        f"numbers use all 25 features, {this_view}; treat its scores as inflated "
        "by f19."
    )


def _order(rows: list[dict[str, Any]], key: str) -> str:
    return " > ".join(r["setup"] for r in sorted(rows, key=lambda r: -r[key]))


def full_budget_order(
    default_check: dict[str, Any], full_rows: list[dict[str, Any]]
) -> str | None:
    """The same setups' test PR-AUC order at the full budget, if all have one."""
    wanted = [r["setup"] for r in default_check["scores"]]
    full = {r["display_id"]: r for r in full_rows}
    if not all(s in full for s in wanted):
        return None
    rows = [{"setup": s, "test": full[s]["test_pr_auc"]} for s in wanted]
    most = max(full[s]["n_splits"] for s in wanted)
    short = [
        f"{s} on {full[s]['n_splits']}" for s in wanted if full[s]["n_splits"] < most
    ]
    splits = f"{most} splits" + (f"; {', '.join(short)}" if short else "")
    return (
        f"At the full budget (6 classifiers, 100 trials): {_order(rows, 'test')} "
        f"({splits}). It changes the classifiers as well as the trials, so it does "
        "not say which part of the tuning moves the order."
    )


def dataset_caveats(
    dataset: str,
    variant: str,
    runs_all: list[dict[str, Any]],
    runs_no19: list[dict[str, Any]],
    default_check: dict[str, Any] | None,
    budget: str = reg.PILOT,
    full_all: list[dict[str, Any]] | None = None,
) -> list[str]:
    """`runs_all` and `runs_no19` are the summaries at `budget`; `full_all` the
    full-budget summary with all features, for the default-settings note."""
    notes = []
    if dataset == "amazon":
        if variant in (reg.ALL, reg.TOP4, reg.ONLY36):
            warning = amazon_f19(runs_all, runs_no19, budget, variant)
            if warning:
                notes.append(warning)
        if variant == reg.NO19:
            notes.append(
                "Without f19, no column built from votes is left: CARE-GNN's "
                "preprocessing script and identities in the data map f19 as the only "
                "vote column among the 25 (2026-10-26). f13, f15 and f18 are rating "
                "statistics, not votes. That the saved features came from that script "
                "is very likely but unproven without the raw reviews."
            )
    if dataset == "yelpchi":
        notes.append(
            "YelpChi's labels come from Yelp's own spam filter, so they are not "
            "independent ground truth. They are user-level; every split keeps a user's "
            "reviews on one side."
        )
    if default_check is not None:
        rows = default_check["scores"]
        full = full_budget_order(default_check, full_all or [])
        notes.append(
            "The ranking of these setups depends on the pilot's tuning. Test PR-AUC "
            f"order when tuned: {_order(rows, 'test_tuned')}. At default settings: "
            f"{_order(rows, 'test_default')}. Both are {rows[0]['n_splits']} splits. "
            + (full + " " if full else "")
            + "The verdicts stand as pre-registered, but do not read them as a stable "
            "ranking."
        )
    return notes


def label_column_note() -> str:
    return (
        "E8 and E8s use a column built from a GNN trained on labels: its out-of-fold "
        "score, with nested cross-fitting. It is the one documented exception to the "
        "rule that labels enter only through the training loss (amended 2026-10-31)."
    )


def demo_notes() -> list[str]:
    return [
        "One draw is an illustration, not evidence. The whole-test-set numbers in each "
        "panel are the performance; a tally of a few clicks is not, and the fraud and "
        "not-fraud buttons pick the class on purpose, so their draws are not "
        "representative.",
        "“Most unusual features” are the node's highest values against the training "
        "nodes. They describe the node, not the model's reasoning: this is not a SHAP "
        "or causal explanation. SHAP explanations of the best models are on the "
        "“Explain a node” and “Explanation details” pages.",
        "Neighbour colours use training labels for this picture only. No model input "
        "uses any neighbour's label.",
    ]


def verdict_labels() -> dict[str, str]:
    return dict(data.VERDICT_LABELS)
