"""Sanity checks on what the dashboard reports.

Each check re-derives something from the results database, or compares two
parts of it that were written separately, so a wrong number has to get past
more than one source. The page shows every check with its outcome, and
`scripts/dashboard.py --check` runs them from the command line. A failure is
shown in red at the top of the page, never hidden.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import results_db
from fraud_detect.dashboard import data
from fraud_detect.dashboard import registry as reg
from fraud_detect.models import contrasts as ct

# Deltas are recorded from the same probabilities as the run's own PR-AUC.
DELTA_TOLERANCE = 1e-6
SUMMARY_TOLERANCE = 1e-9
PILOT_TRIALS, PILOT_CLASSIFIERS = 10, 3
FULL_TRIALS, FULL_CLASSIFIERS = 100, 6
SHOWN_PROBLEMS = 8


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    problems: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "problems": list(self.problems[:SHOWN_PROBLEMS]),
            "n_problems": len(self.problems),
        }


def _result(name: str, passed: str, problems: list[str]) -> Check:
    if problems:
        return Check(name, False, f"{len(problems)} problem(s)", tuple(problems))
    return Check(name, True, passed)


def check_database(db_dir: Path, entries: list[data.Entry]) -> Check:
    problems = results_db.check_db(db_dir)
    return _result(
        "Results database is valid",
        f"{len(entries)} entries: aggregate columns only, valid UTC times, under the "
        "row limit",
        problems,
    )


def check_registry(entries: list[data.Entry]) -> Check:
    unknown = data.unmapped_feature_sets(entries)
    n = sum(e.meta["kind"] == "pipeline" for e in entries)
    return _result(
        "Every run maps to a setup",
        f"{n} pipeline entries, none unmapped",
        [f"unknown feature set {name}" for name in unknown],
    )


def check_supersedes(entries: list[data.Entry]) -> Check:
    ids = {e.run_id for e in entries}
    problems = [
        f"{e.run_id} supersedes missing entry {e.meta['supersedes']}"
        for e in entries
        if e.meta.get("supersedes") and e.meta["supersedes"] not in ids
    ]
    n = sum(e.superseded for e in entries)
    return _result(
        "Corrections point at real entries", f"{n} entries superseded", problems
    )


def check_no_duplicates(runs: pd.DataFrame) -> Check:
    dup = data.duplicate_runs(runs)
    return _result(
        "No run is counted twice",
        f"{len(data.usable_runs(runs))} usable runs, each (dataset, feature set, split) "
        "once",
        dup,
    )


def _nan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def check_metric_ranges(runs: pd.DataFrame) -> Check:
    problems = []
    usable = data.usable_runs(runs)
    for r in usable.itertuples():
        where = f"{r.dataset}/{r.feature_set}/split {r.split_repeat}"
        for name in ("test_pr_auc", "cv_pr_auc", "test_auroc", "no_skill", "threshold"):
            value = getattr(r, name)
            if _nan(value) or not 0 <= float(value) <= 1:
                problems.append(f"{where}: {name} = {value}")
        low, high = r.test_pr_auc_low, r.test_pr_auc_high
        if not _nan(low) and not _nan(high) and not low <= r.test_pr_auc <= high:
            problems.append(f"{where}: {r.test_pr_auc} is outside [{low}, {high}]")
        if (
            not _nan(r.no_skill)
            and not _nan(r.test_pr_auc)
            and r.test_pr_auc <= r.no_skill
        ):
            problems.append(f"{where}: PR-AUC {r.test_pr_auc} is not above no skill")
    return _result(
        "Metrics are in range",
        "PR-AUC, AUROC, thresholds and base rates in [0, 1]; each PR-AUC inside its "
        "interval and above no skill",
        problems,
    )


def check_frozen_test_set(runs: pd.DataFrame) -> Check:
    """Invariant 4: every setup on a split must see the same test set, so its
    size and base rate must be identical across feature sets."""
    problems = []
    usable = data.usable_runs(runs)
    n = 0
    for (dataset, split), group in usable.groupby(["dataset", "split_repeat"]):
        n += 1
        for column in ("n_test", "no_skill"):
            values = group[column].dropna().unique()
            if len(values) > 1:
                problems.append(
                    f"{dataset} split {split}: {column} takes values {values}"
                )
    return _result(
        "Each split has one test set",
        f"{n} (dataset, split) pairs: same test size and base rate for every setup",
        problems,
    )


def check_budget_labels(runs: pd.DataFrame) -> Check:
    """A run's name says its budget; its logged settings must agree."""
    problems = []
    n = 0
    for r in data.usable_runs(runs).itertuples():
        if reg.BY_ID[r.setup_id].gnn_rung:
            continue  # its df-analyze settings do not apply
        n += 1
        want = (
            (PILOT_TRIALS, PILOT_CLASSIFIERS)
            if r.budget == reg.PILOT
            else (FULL_TRIALS, FULL_CLASSIFIERS)
        )
        if (r.trials, r.n_classifiers) != want:
            problems.append(
                f"{r.run_id}: named {r.budget}, logged {r.trials} trials, "
                f"{r.n_classifiers} classifiers"
            )
    return _result(
        "Budget labels match the logged settings",
        f"{n} df-analyze runs: pilot = 10 trials and 3 classifiers, full = 100 and 6",
        problems,
    )


def check_model_a_chosen_by_cv(entries: list[data.Entry]) -> Check:
    """Invariant 8: Model A is the tuned row with the highest shared-CV PR-AUC,
    re-derived from each run's own results table."""
    problems = []
    n = 0
    for entry in entries:
        meta = entry.meta
        if meta["kind"] != "pipeline" or meta["status"] != "ok":
            continue
        table = entry.table()
        if "cv_pr_auc" in table.columns:
            table = table.dropna(subset=["cv_pr_auc"])
        if table.empty or "cv_pr_auc" not in table.columns:
            problems.append(f"{entry.run_id}: no shared-CV scores")
            continue
        n += 1
        best = table[table["cv_pr_auc"] == table["cv_pr_auc"].max()]
        names = {f"{r.model} ({r.selection})" for r in best.itertuples()}
        chosen = (meta.get("headline") or {}).get("model_a")
        if chosen not in names:
            problems.append(
                f"{entry.run_id}: Model A is {chosen}, highest CV is {sorted(names)}"
            )
    return _result(
        "Model A was chosen by shared CV, not by test",
        f"{n} runs: Model A has the highest shared-CV PR-AUC in its own results table",
        problems,
    )


def check_verdicts_recomputed(rows: list[dict[str, Any]]) -> Check:
    """Each stored verdict again from its stored deltas and p, by the rule."""
    problems = []
    for row in rows:
        again = ct.verdict(row["deltas"], row["p"])
        if again != row["verdict"]:
            problems.append(
                f"{row['run_id']} {row['b_id']} − {row['a_id']}: stored "
                f"{row['verdict']!r}, rule gives {again!r}"
            )
    return _result(
        "Verdicts follow the pre-registered rule",
        f"{len(rows)} contrast rows recomputed from their deltas and p",
        problems,
    )


def check_contrast_arithmetic(rows: list[dict[str, Any]]) -> Check:
    problems = []
    for row in rows:
        where = f"{row['run_id']} {row['b_id']} − {row['a_id']}"
        d = np.asarray(row["deltas"], dtype=float)
        if abs(d.mean() - row["mean"]) > SUMMARY_TOLERANCE:
            problems.append(f"{where}: mean {row['mean']} but deltas average {d.mean()}")
        if row["sd"] is not None and abs(d.std(ddof=1) - row["sd"]) > SUMMARY_TOLERANCE:
            problems.append(f"{where}: SD {row['sd']} but deltas give {d.std(ddof=1)}")
        if abs((row["pr_b"] - row["pr_a"]) - row["mean"]) > DELTA_TOLERANCE:
            problems.append(f"{where}: mean Δ differs from the two sides' mean scores")
        if not 0 <= row["p"] <= 1:
            problems.append(f"{where}: p = {row['p']}")
        if row["cv_folds"] != row["n_splits"] * 5:
            problems.append(
                f"{where}: {row['cv_folds']} CV folds, {row['n_splits']} splits"
            )
    return _result(
        "Contrast arithmetic adds up",
        f"{len(rows)} contrast rows: means and SDs match their deltas, p in range, 5 CV "
        "folds per split",
        problems,
    )


def check_contrasts_match_runs(rows: list[dict[str, Any]], runs: pd.DataFrame) -> Check:
    """Contrasts and runs are recorded separately; they must agree, split by split."""
    problems = []
    usable = data.usable_runs(runs)
    score = {
        (r.dataset, r.feature_set, r.split_repeat): r.test_pr_auc
        for r in usable.itertuples()
    }
    checked = unknown = 0
    for row in rows:
        for k, delta in enumerate(row["deltas"]):
            a = score.get((row["dataset"], row["a"], k))
            b = score.get((row["dataset"], row["b"], k))
            if a is None or b is None:
                unknown += 1
                continue
            checked += 1
            if abs((b - a) - delta) > DELTA_TOLERANCE:
                problems.append(
                    f"{row['run_id']} {row['b_id']} − {row['a_id']} split {k}: recorded "
                    f"{delta:+.6f}, runs give {b - a:+.6f}"
                )
    return _result(
        "Contrasts agree with the runs",
        f"{checked} per-split deltas equal the difference of the two runs' test PR-AUC"
        + (
            f"; {unknown} could not be checked (a run is not recorded)" if unknown else ""
        ),
        problems,
    )


def check_replacement(current: list[dict[str, Any]]) -> Check:
    """A replaced verdict must have had fewer or equally many splits."""
    problems = []
    n = 0
    for row in current:
        for old in row["replaces"]:
            n += 1
            if old["n_splits"] > row["n_splits"]:
                problems.append(
                    f"{row['b_id']} − {row['a_id']}: {old['n_splits']} splits replaced "
                    f"by {row['n_splits']}"
                )
    return _result(
        "No verdict was replaced by one on fewer splits",
        f"{n} earlier verdicts replaced, each shown beside its replacement",
        problems,
    )


def run_all(
    db_dir: Path,
    entries: list[data.Entry],
    runs: pd.DataFrame,
    all_rows: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[Check]:
    return [
        check_database(db_dir, entries),
        check_registry(entries),
        check_supersedes(entries),
        check_no_duplicates(runs),
        check_metric_ranges(runs),
        check_frozen_test_set(runs),
        check_budget_labels(runs),
        check_model_a_chosen_by_cv(entries),
        check_verdicts_recomputed(all_rows),
        check_contrast_arithmetic(all_rows),
        check_contrasts_match_runs(all_rows, runs),
        check_replacement(current),
    ]
