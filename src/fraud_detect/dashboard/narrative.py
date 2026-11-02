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
NOT_RUN = (
    "YelpChi at the full budget.",
    "Full-budget graph setups with all features on Amazon (only B has one).",
    "E1, N1 and N2 without f19 on Amazon splits 3–4.",
    "A variant that removes every vote-derived Amazon column, not just f19.",
    "HGT (declined 2026-10-31), and any dataset beyond YelpChi and Amazon.",
)


def _mean(rows: list[dict[str, Any]], display_id: str) -> dict[str, Any] | None:
    return next((r for r in rows if r["display_id"] == display_id), None)


def budget_note(view: dict[str, Any]) -> str:
    budgets, total = view["budgets"], view["n_runs"]
    where = ", ".join(view["full_budget_where"]) or "nowhere"
    return (
        f"{budgets[reg.PILOT]} of the {total} recorded runs use the pilot budget: "
        "LightGBM, logistic regression and CatBoost, 10 tuning trials. "
        f"Only {budgets[reg.FULL]} use the full budget (6 classifiers, 100 trials), on "
        f"{where}. Tuned on balanced accuracy, with df-analyze's random internal CV."
    )


def general_caveats(view: dict[str, Any], failed: list[dict[str, Any]]) -> list[str]:
    devices = ", ".join(view["devices"]) or "unknown"
    seeds = ", ".join(map(str, view["seeds"])) or "unknown"
    notes = [
        budget_note(view),
        "The test set chose nothing. Model A is the tuned model with the highest "
        "PR-AUC in a shared, grouped CV on training data. Those CV scores are slightly "
        "optimistic, because the same folds chose Model A.",
        f"Every run used one machine ({devices}) and one base seed ({seeds}); only the "
        "splits differ. Treat a difference smaller than the spread across splits as "
        "noise.",
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
    runs_all: list[dict[str, Any]], runs_no19: list[dict[str, Any]]
) -> str | None:
    """The f19 warning, with this page's own numbers."""
    with_f19, without = _mean(runs_all, "B"), _mean(runs_no19, "B")
    if with_f19 is None or without is None:
        return None
    return (
        "Amazon's f19 is the minimum number of unhelpful votes on a single review, "
        "built from the same votes that define the label, so it likely leaks the label. "
        f"B scores {with_f19['test_pr_auc']:.3f} with all 25 features and "
        f"{without['test_pr_auc']:.3f} without f19 (test PR-AUC, pilot budget, "
        f"{with_f19['n_splits']} and {without['n_splits']} splits). Published Amazon "
        "numbers use all 25 features, and so does this view; treat its scores as "
        "inflated by f19."
    )


def dataset_caveats(
    dataset: str,
    variant: str,
    runs_all: list[dict[str, Any]],
    runs_no19: list[dict[str, Any]],
    default_check: dict[str, Any] | None,
) -> list[str]:
    notes = []
    if dataset == "amazon":
        if variant in (reg.ALL, reg.TOP4, reg.ONLY36):
            warning = amazon_f19(runs_all, runs_no19)
            if warning:
                notes.append(warning)
        if variant == reg.NO19:
            notes.append(
                "Without f19, other columns may still be vote-derived; a variant "
                "without all of them was not run."
            )
    if dataset == "yelpchi":
        notes.append(
            "YelpChi's labels come from Yelp's own spam filter, so they are not "
            "independent ground truth. They are user-level; every split keeps a user's "
            "reviews on one side."
        )
    if default_check is not None:
        rows = default_check["scores"]
        tuned = sorted(rows, key=lambda r: -r["test_tuned"])
        default = sorted(rows, key=lambda r: -r["test_default"])
        order = " > ".join(r["setup"] for r in tuned)
        other = " > ".join(r["setup"] for r in default)
        notes.append(
            "The ranking of these setups depends on the pilot's tuning. Test PR-AUC "
            f"order when tuned: {order}. At default settings: {other}. Both are "
            f"{rows[0]['n_splits']} splits. The verdicts below stand as pre-registered, "
            "but do not read them as a stable ranking."
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
        "or causal explanation.",
        "Neighbour colours use training labels for this picture only. No model input "
        "uses any neighbour's label.",
    ]


def verdict_labels() -> dict[str, str]:
    return dict(data.VERDICT_LABELS)
