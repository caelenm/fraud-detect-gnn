"""Tuning-budget parsing from df-analyze's log."""

from __future__ import annotations

from fraud_detect.models.selection import parse_tuning_budget


def bar(done: int, total: int, elapsed: float, limit: int) -> str:
    """A tqdm progress bar as df-analyze's Optuna studies print it."""
    timing = f"00:10<00:10, 1.0s/it, {elapsed}/{limit} seconds"
    return f"Best trial: 1.: | {done}/{total} [{timing}]"


# Shaped like df-analyze's log: tqdm bars redraw with carriage returns.
LOG = (
    bar(57, 100, 40.33, 600) + "\n"  # feature-selection study before tuning
    "Tuning CatBoost Classifier for selection=none\n"
    "  0%|          | 0/100 [00:00<?, ?it/s]\r"
    + bar(1, 100, 6.1, 1800) + "\r" + bar(40, 100, 1790.0, 1800) + "\n"
    "Tuning LightGBM Classifier for selection=assoc\n"
    + bar(100, 100, 600.0, 3600) + "\n"
    "Tuning K-Neighbours Classifier for selection=pred\n"
    + bar(24, 100, 60.0, 1800) + "\n"
    "Tuning Dummy Classifier for selection=none\n"
)  # fmt: skip


def test_parse_tuning_budget():
    budget = parse_tuning_budget(LOG).set_index(["model", "selection"])
    assert len(budget) == 4  # the feature-selection bar before tuning is ignored
    cat = budget.loc[("CatBoost Classifier", "none")]
    assert cat["trials_completed"] == 40
    assert cat["time_limit_s"] == 1800
    assert cat["stopped_by"] == "time limit"
    assert budget.loc[("LightGBM Classifier", "assoc"), "stopped_by"] == "all trials"
    assert budget.loc[("K-Neighbours Classifier", "pred"), "stopped_by"] == "early stop"
    assert budget.loc[("Dummy Classifier", "none"), "stopped_by"] == "unknown"


def test_empty_log():
    assert parse_tuning_budget("no tuning here\n").empty
