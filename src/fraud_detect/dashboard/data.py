"""The dashboard's numbers: pure functions over results/runs/.

Everything here is aggregate. It reads the committed results database and
nothing else, so a fresh clone with no outputs/ can show every section except
the classify panel. Nothing is recomputed from row-level data, and nothing is
chosen from a test score: best_setup uses the shared-CV PR-AUC only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import results_db
from fraud_detect.dashboard import registry as reg
from fraud_detect.models import contrasts as ct

ALPHA = 0.05
SUPPORTED = "supported"
NOT_DISTINGUISHABLE = "not_distinguishable"
WORSE_SIGNIFICANT = "worse_significant"
WORSE_MEAN = "worse_mean"
DESCRIPTIVE = "descriptive"
VERDICT_LABELS = {
    SUPPORTED: "supported",
    NOT_DISTINGUISHABLE: "not distinguishable from zero",
    WORSE_SIGNIFICANT: "worse (significant)",
    WORSE_MEAN: "lower mean, not significant",
    DESCRIPTIVE: "descriptive (one split, no verdict)",
}
PRIMARY = "primary"
OTHER = "other"
BACKFILL_TAG = "backfill"
DELTA_COLUMN = "test_delta_split"

# The questions the project set out to answer, as contrasts of setups (B − A),
# in the order they read best. A contrast that was not run is left out.
KEY_QUESTIONS = (
    ("Do neighbour means help over a node's own features?", "E1", "B"),
    ("Do DGI embeddings help over a node's own features?", "E2", "B"),
    ("Do untrained GNN embeddings help over a node's own features?", "E3", "B"),
    ("Does training the GNN encoder matter?", "E2", "E3"),
    (
        "Do neighbour means beat rewired-graph neighbour means (real structure)?",
        "E1",
        "N2",
    ),
    ("Does degree alone help?", "N1", "B"),
    ("Does a GNN beat the same network without edges?", "E5", "E6"),
    ("Does a GNN beat itself on a rewired graph?", "E5", "N3"),
    ("Does a GNN alone beat df-analyze on own features?", "E5", "B"),
    ("Does the m2 + m5 combination beat neighbour means alone?", "E7", "E1"),
    ("Does the supervised GNN score beat neighbour means alone?", "E8", "E1"),
)


class DashboardDataError(ValueError):
    """The results database cannot be read as it should; the message says why."""


@dataclass(frozen=True)
class Entry:
    run_id: str
    meta: dict[str, Any]
    path: Path
    superseded: bool

    @property
    def tags(self) -> list[str]:
        return list(self.meta.get("tags") or [])

    def table(self) -> pd.DataFrame:
        return pd.read_csv(self.path / results_db.RESULTS)


def load_entries(db_dir: Path) -> list[Entry]:
    """Every entry, oldest first. An entry named in another's `supersedes` is
    kept but marked: hidden from the numbers, never deleted."""
    metas = [(d, results_db.load_meta(d)) for d in results_db.entry_dirs(db_dir)]
    replaced = {m["supersedes"] for _, m in metas if m.get("supersedes")}
    return [Entry(d.name, m, d, d.name in replaced) for d, m in metas]


def unmapped_feature_sets(entries: list[Entry]) -> list[str]:
    """Pipeline entries whose feature set the registry cannot read."""
    names = {e.meta.get("feature_set") for e in entries if e.meta["kind"] == "pipeline"}
    return sorted(str(n) for n in names if not reg.is_known(str(n)))


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
RUN_COLUMNS = (
    "run_id",
    "dataset",
    "feature_set",
    "split_repeat",
    "status",
    "superseded",
    "replication",
    "df_analyze_run",
    "backfilled",
    "setup_id",
    "display_id",
    "family",
    "label",
    "variant",
    "embed_size",
    "budget",
    "sort_key",
    "test_pr_auc",
    "test_pr_auc_low",
    "test_pr_auc_high",
    "test_auroc",
    "no_skill",
    "cv_pr_auc",
    "cv_pr_auc_std",
    "model_a",
    "threshold",
    "n_train",
    "n_test",
    "device",
    "started_utc",
    "trials",
    "n_classifiers",
)


def _run_row(entry: Entry) -> dict[str, Any]:
    meta = entry.meta
    fs = reg.parse_feature_set(str(meta.get("feature_set")))
    head = meta.get("headline") or {}
    settings = meta.get("settings") or {}
    return {
        "run_id": entry.run_id,
        "dataset": meta.get("dataset"),
        "feature_set": fs.name,
        "split_repeat": int(meta.get("split_repeat") or 0),
        "status": meta["status"],
        "superseded": entry.superseded,
        "replication": False,
        "df_analyze_run": meta.get("df_analyze_run"),
        "backfilled": BACKFILL_TAG in entry.tags,
        "setup_id": fs.setup.setup_id,
        "display_id": fs.display_id,
        "family": fs.setup.family,
        "label": fs.setup.label,
        "variant": fs.variant,
        "embed_size": fs.embed_size,
        "budget": fs.budget,
        "sort_key": fs.sort_key,
        "test_pr_auc": head.get("test_pr_auc"),
        "test_pr_auc_low": head.get("test_pr_auc_low"),
        "test_pr_auc_high": head.get("test_pr_auc_high"),
        "test_auroc": head.get("test_auroc"),
        "no_skill": head.get("no_skill_pr_auc"),
        "cv_pr_auc": head.get("cv_pr_auc"),
        "cv_pr_auc_std": head.get("cv_pr_auc_std"),
        "model_a": head.get("model_a"),
        "threshold": head.get("threshold"),
        "n_train": head.get("n_train"),
        "n_test": head.get("n_test"),
        "device": meta.get("device"),
        "started_utc": meta.get("started_utc"),
        "trials": settings.get("df_analyze.htune_trials", settings.get("htune_trials")),
        "n_classifiers": len(settings.get("df_analyze.classifiers") or []),
    }


def runs_table(entries: list[Entry]) -> pd.DataFrame:
    """One row per pipeline entry whose feature set is known. Failed,
    superseded and replication runs are included and marked; usable_runs
    drops them."""
    rows = [
        _run_row(e)
        for e in entries
        if e.meta["kind"] == "pipeline" and reg.is_known(str(e.meta.get("feature_set")))
    ]
    table = pd.DataFrame(rows, columns=list(RUN_COLUMNS))
    for column in ("test_pr_auc", "test_auroc", "no_skill", "cv_pr_auc"):
        table[column] = pd.to_numeric(table[column])
    _mark_replications(table)
    return table


def _mark_replications(table: pd.DataFrame) -> None:
    """A later, separate df-analyze run of a (dataset, feature set, split) that
    already has one (e.g. a rerun on another machine) is a replication: kept
    and shown, but the earliest run is the one counted. The same df-analyze run
    recorded twice, or a run with no df_analyze_run, is not a replication and
    still fails the duplicate check."""
    ok = (table["status"] == "ok") & ~table["superseded"].astype(bool)
    keys = ["dataset", "feature_set", "split_repeat"]
    for _, group in table[ok].sort_values("started_utc").groupby(keys, sort=False):
        first = group.iloc[0]["df_analyze_run"]
        if not first:
            continue
        seen = {first}
        for index, run in group.iloc[1:].iterrows():
            if run["df_analyze_run"] and run["df_analyze_run"] not in seen:
                table.loc[index, "replication"] = True
                seen.add(run["df_analyze_run"])


def usable_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[
        (runs["status"] == "ok")
        & ~runs["superseded"].astype(bool)
        & ~runs["replication"].astype(bool)
    ]


def duplicate_runs(runs: pd.DataFrame) -> list[str]:
    """(dataset, feature set, split) recorded by more than one usable entry."""
    keys = ["dataset", "feature_set", "split_repeat"]
    usable = usable_runs(runs)
    dup = usable[usable.duplicated(keys, keep=False)]
    return sorted(
        {f"{r.dataset}/{r.feature_set}/split {r.split_repeat}" for r in dup.itertuples()}
    )


def normalised_skill(pr_auc: Any, no_skill: Any) -> Any:
    """(PR-AUC − no skill) / (1 − no skill): 0 at no skill, 1 at perfect, so
    datasets with different base rates share a scale."""
    return (pr_auc - no_skill) / (1 - no_skill)


def _sd(values: Any) -> float | None:
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1)) if len(v) > 1 else None


def setup_summary(
    runs: pd.DataFrame, dataset: str, budget: str, variant: str | None = None
) -> list[dict[str, Any]]:
    """Mean ± SD over splits per feature set, in registry order. One budget at
    a time, so a row can never mix pilot and full runs."""
    if budget not in reg.BUDGETS:
        raise DashboardDataError(f"Unknown budget {budget!r}")
    dup = duplicate_runs(runs)
    if dup:
        raise DashboardDataError(f"Recorded more than once: {dup}")
    usable = usable_runs(runs)
    rows = usable[(usable["dataset"] == dataset) & (usable["budget"] == budget)]
    if variant is not None:
        rows = rows[rows["variant"] == variant]
    out = []
    for name, group in rows.groupby("feature_set", sort=False):
        group = group.sort_values("split_repeat")
        first = group.iloc[0]
        setup = reg.BY_ID[first["setup_id"]]
        test = group["test_pr_auc"].to_numpy(float)
        cv = group["cv_pr_auc"].to_numpy(float)
        no_skill = group["no_skill"].to_numpy(float)
        out.append(
            {
                "feature_set": name,
                "setup_id": first["setup_id"],
                "display_id": first["display_id"],
                "family": first["family"],
                "label": first["label"],
                "variant": first["variant"],
                "embed_size": first["embed_size"],
                "budget": budget,
                "uses_graph": setup.uses_graph,
                "gnn_rung": setup.gnn_rung,
                "label_derived_column": setup.label_derived_column,
                "n_splits": len(group),
                "splits": [int(k) for k in group["split_repeat"]],
                "test_pr_auc": float(test.mean()),
                "test_pr_auc_sd": _sd(test),
                "test_per_split": [float(v) for v in test],
                "cv_pr_auc": float(cv.mean()),
                "cv_pr_auc_sd": _sd(cv),
                "cv_per_split": [float(v) for v in cv],
                "test_auroc": float(group["test_auroc"].mean()),
                "no_skill": float(no_skill.mean()),
                "normalised_skill": float(normalised_skill(test, no_skill).mean()),
                "normalised_skill_sd": _sd(normalised_skill(test, no_skill)),
                "model_a": [str(m) for m in group["model_a"]],
                "run_ids": list(group["run_id"]),
                "n_backfilled": int(group["backfilled"].sum()),
                "sort_key": first["sort_key"],
            }
        )
    out.sort(key=lambda r: r["sort_key"])
    for row in out:
        del row["sort_key"]
    return out


def budget_with_every_setup(runs: pd.DataFrame, dataset: str, variant: str) -> str | None:
    """The largest budget that has every setup of this variant on this dataset."""
    usable = usable_runs(runs)
    rows = usable[(usable["dataset"] == dataset) & (usable["variant"] == variant)]
    wanted = set(rows["feature_set"].map(lambda n: reg.parse_feature_set(n).display_id))
    if not wanted:
        return None
    for budget in (reg.FULL, reg.PILOT):
        got = rows[rows["budget"] == budget]["feature_set"]
        if set(got.map(lambda n: reg.parse_feature_set(n).display_id)) == wanted:
            return budget
    return None


def best_setup(
    runs: pd.DataFrame,
    dataset: str,
    variant: str,
    contrast_rows: list[dict] | None = None,
) -> dict[str, Any] | None:
    """The plan's rule (D1, decided 2026-11-02): the setup with the highest
    mean shared-CV PR-AUC over its splits, at the largest budget that has every
    setup, the first in registry order on a tie. Its Model A on split 0 is the
    one classified live, so a setup needs a split 0 run. The test score plays
    no part. `close` lists the setups within one SD (across splits) of the top,
    with any verdict that compares them, so the page need not imply a winner."""
    budget = budget_with_every_setup(runs, dataset, variant)
    if budget is None:
        return None
    rows = [r for r in setup_summary(runs, dataset, budget, variant) if 0 in r["splits"]]
    if not rows:
        return None
    top = max(rows, key=lambda r: r["cv_pr_auc"])  # max keeps the first on a tie
    margin = top["cv_pr_auc_sd"] or 0.0
    here = [r for r in contrast_rows or [] if r["dataset"] == dataset]
    close = []
    for row in rows:
        if row is top or top["cv_pr_auc"] - row["cv_pr_auc"] > margin:
            continue
        close.append(
            {
                "display_id": row["display_id"],
                "label": row["label"],
                "cv_pr_auc": row["cv_pr_auc"],
                "test_pr_auc": row["test_pr_auc"],
                "verdict": _verdict_between(here, top, row),
            }
        )
    close.sort(key=lambda r: -r["cv_pr_auc"])
    return {
        "dataset": dataset,
        "variant": variant,
        "budget": budget,
        "feature_set": top["feature_set"],
        "setup_id": top["setup_id"],
        "display_id": top["display_id"],
        "label": top["label"],
        "cv_pr_auc": top["cv_pr_auc"],
        "cv_pr_auc_sd": top["cv_pr_auc_sd"],
        "test_pr_auc": top["test_pr_auc"],
        "test_pr_auc_sd": top["test_pr_auc_sd"],
        "n_splits": top["n_splits"],
        "gnn_rung": top["gnn_rung"],
        "model_a_split0": top["model_a"][top["splits"].index(0)],
        "close": close,
        "rule": "Highest mean shared-CV PR-AUC over its splits, at the largest budget "
        "that has every setup. The test set played no part.",
    }


def _verdict_between(rows: list[dict], top: dict, other: dict) -> str | None:
    """The verdict of a contrast between two feature sets, read from `top`'s
    side, or None if no contrast compares them."""
    names = {top["feature_set"], other["feature_set"]}
    for row in rows:
        if {row["a"], row["b"]} == names:
            return f"{row['b_id']} − {row['a_id']}: {row['verdict_label']}"
    return None


# --------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------
def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p values, in the input order."""
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def verdict_kind(verdict: str, p: float, n_splits: int) -> str:
    """The rule's verdict, with 'worse' split in two: on its own it covers both
    a significant drop and a mean just below zero, which read very differently.
    One split cannot carry a verdict: it has no spread across splits."""
    if n_splits < 2:
        return DESCRIPTIVE
    if verdict == ct.SUPPORTED:
        return SUPPORTED
    if verdict == ct.NOT_DISTINGUISHABLE:
        return NOT_DISTINGUISHABLE
    if verdict == ct.WORSE:
        return WORSE_SIGNIFICANT if p < ALPHA else WORSE_MEAN
    raise DashboardDataError(f"Unknown verdict {verdict!r}")


def split_deltas(row: dict[str, Any]) -> list[float]:
    keys = sorted(
        (k for k in row if k.startswith(DELTA_COLUMN)),
        key=lambda k: int(k.removeprefix(DELTA_COLUMN)),
    )
    # A table's rows may cover different splits; a blank cell is not a delta.
    return [float(row[k]) for k in keys if pd.notna(row[k])]


def all_contrasts(entries: list[Entry]) -> list[dict[str, Any]]:
    """Every row of every usable `compare` entry, before replacement."""
    out = []
    for entry in entries:
        meta = entry.meta
        if meta["kind"] != "compare" or meta["status"] != "ok" or entry.superseded:
            continue
        table = entry.table()
        rows = table.to_dict("records")
        family = PRIMARY if "primary family" in meta["description"] else OTHER
        holm_p = holm([float(r["p"]) for r in rows]) if family == PRIMARY else None
        for i, row in enumerate(rows):
            a, b = reg.parse_feature_set(row["a"]), reg.parse_feature_set(row["b"])
            deltas = split_deltas(row)
            kind = verdict_kind(row["verdict"], float(row["p"]), len(deltas))
            same_variant = a.variant == b.variant
            out.append(
                {
                    "dataset": meta["dataset"],
                    "run_id": entry.run_id,
                    "started_utc": meta["started_utc"],
                    "description": meta["description"],
                    "family": family,
                    "a": a.name,
                    "b": b.name,
                    "a_id": a.display_id,
                    "b_id": b.display_id,
                    "a_setup": a.setup.setup_id,
                    "b_setup": b.setup.setup_id,
                    "a_label": a.setup.label,
                    "b_label": b.setup.label,
                    "a_variant": a.variant,
                    "b_variant": b.variant,
                    "a_budget": a.budget,
                    "b_budget": b.budget,
                    "same_variant": same_variant,
                    "same_budget": a.budget == b.budget,
                    "n_splits": len(deltas),
                    "deltas": deltas,
                    "mean": float(row["test_delta_mean"]),
                    "sd": None if len(deltas) < 2 else float(row["test_delta_sd"]),
                    "pr_a": float(row["test_pr_auc_a_mean"]),
                    "pr_b": float(row["test_pr_auc_b_mean"]),
                    "cv_folds": int(row["cv_folds"]),
                    "cv_delta": float(row["cv_delta_mean"]),
                    "p": float(row["p"]),
                    "holm_p": holm_p[i] if holm_p else None,
                    "verdict": row["verdict"],
                    "verdict_kind": kind,
                    "verdict_label": VERDICT_LABELS[kind],
                    "headroom": float(row["test_delta_mean"])
                    / (1 - float(row["test_pr_auc_a_mean"])),
                    "replaces": [],
                    "row_index": i,
                }
            )
    return out


def current_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (dataset, A, B): the one with the most splits, then the
    latest. A verdict on more splits replaces one on fewer (decided 2026-10-28);
    what it replaced stays visible in `replaces`."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["a"], row["b"]), []).append(row)
    out = []
    for group in groups.values():
        group.sort(key=lambda r: (r["n_splits"], r["started_utc"]))
        current = {**group[-1]}
        current["replaces"] = [
            {
                "n_splits": r["n_splits"],
                "verdict_label": r["verdict_label"],
                "mean": r["mean"],
                "p": r["p"],
                "run_id": r["run_id"],
            }
            for r in group[:-1]
        ]
        out.append(current)
    out.sort(key=lambda r: (r["dataset"], r["started_utc"], r["row_index"]))
    return out


def contrasts_for(
    rows: list[dict[str, Any]], dataset: str, variant: str
) -> dict[str, list[dict[str, Any]]]:
    """A dataset's current contrasts for one variant, by family. A contrast
    across variants appears under both; a one-split or cross-budget contrast is
    descriptive, kept apart so it never reads as a verdict."""
    out: dict[str, list[dict[str, Any]]] = {PRIMARY: [], OTHER: [], DESCRIPTIVE: []}
    for row in rows:
        if row["dataset"] != dataset or variant not in (
            row["a_variant"],
            row["b_variant"],
        ):
            continue
        if row["n_splits"] < 2 or not row["same_budget"]:
            out[DESCRIPTIVE].append(row)
        else:
            out[row["family"]].append(row)
    return out


def key_questions(
    rows: list[dict[str, Any]], dataset: str, variant: str
) -> list[dict[str, Any]]:
    """The project's questions, answered by the current pilot-budget contrasts
    between each pair of default-size setups on this variant."""
    found = []
    for question, b_id, a_id in KEY_QUESTIONS:
        for row in rows:
            if (
                row["dataset"] == dataset
                and row["a_variant"] == variant
                and row["b_variant"] == variant
                and row["a_budget"] == reg.PILOT
                and row["b_budget"] == reg.PILOT
                and row["a_id"] == a_id
                and row["b_id"] == b_id
            ):
                found.append({"question": question, **row})
                break
    return found


# --------------------------------------------------------------------------
# Sensitivity checks
# --------------------------------------------------------------------------
def default_settings_check(entries: list[Entry], dataset: str) -> dict[str, Any] | None:
    """The pre-registered default-settings check (RESEARCH_PLAN §11, 2026-11-02),
    from its two entries: scores per setup and contrasts at default settings."""
    scores, table, run_ids = None, None, []
    for entry in entries:
        if entry.meta["kind"] != "check" or entry.meta.get("dataset") != dataset:
            continue
        frame = entry.table()
        if "test_pr_auc_default" in frame.columns:
            scores = frame
            run_ids.append(entry.run_id)
        elif "verdict" in frame.columns and "contrast" in frame.columns:
            table = frame
            run_ids.append(entry.run_id)
    if scores is None or table is None:
        return None
    by_setup = []
    for setup, group in scores.groupby("setup", sort=False):
        by_setup.append(
            {
                "setup": setup,
                "test_tuned": float(group["test_pr_auc_model_a"].mean()),
                "test_default": float(group["test_pr_auc_default"].mean()),
                "cv_default": float(group["cv_pr_auc_default"].mean()),
                "n_splits": len(group),
            }
        )
    rows = []
    for r in table.to_dict("records"):
        deltas = split_deltas(r)
        kind = verdict_kind(r["verdict"], float(r["p"]), len(deltas))
        rows.append(
            {
                "contrast": str(r["contrast"]).replace(" - ", " − "),
                "deltas": deltas,
                "mean": float(r["test_delta_mean"]),
                "sd": float(r["test_delta_sd"]),
                "p": float(r["p"]),
                "verdict_kind": kind,
                "verdict_label": VERDICT_LABELS[kind],
            }
        )
    return {"scores": by_setup, "contrasts": rows, "run_ids": run_ids}


def tuned_vs_default_cv(entries: list[Entry], dataset: str) -> list[dict[str, Any]]:
    """Best shared-CV PR-AUC per setup with tuned and with default settings
    (scripts/tuned_vs_default.py), mean over splits. Training folds only."""
    for entry in entries:
        if entry.meta["kind"] != "check" or entry.meta.get("dataset") != dataset:
            continue
        frame = entry.table()
        if {"settings", "cv_pr_auc", "setup"} <= set(frame.columns):
            pivot = frame.pivot_table(
                index="setup", columns="settings", values="cv_pr_auc", sort=False
            )
            return [
                {
                    "setup": str(setup),
                    "tuned": float(row["tuned"]),
                    "default": float(row["default"]),
                }
                for setup, row in pivot.iterrows()
            ]
    return []


# --------------------------------------------------------------------------
# Across datasets
# --------------------------------------------------------------------------
def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["display_id"]: r for r in rows}


def graph_gain(
    runs: pd.DataFrame, dataset: str, budget: str, variant: str, setup: str
) -> dict[str, Any] | None:
    """Test PR-AUC of `setup` minus B on the splits both have, per split."""
    rows = _by_id(setup_summary(runs, dataset, budget, variant))
    if setup not in rows or "B" not in rows:
        return None
    a, b = rows["B"], rows[setup]
    shared = sorted(set(a["splits"]) & set(b["splits"]))
    if not shared:
        return None
    pa = dict(zip(a["splits"], a["test_per_split"], strict=True))
    pb = dict(zip(b["splits"], b["test_per_split"], strict=True))
    deltas = [pb[k] - pa[k] for k in shared]
    return {
        "dataset": dataset,
        "variant": variant,
        "setup": setup,
        "n_splits": len(shared),
        "splits": shared,
        "deltas": deltas,
        "mean": float(np.mean(deltas)),
        "sd": _sd(deltas),
        "min": float(np.min(deltas)),
        "max": float(np.max(deltas)),
    }


def _agreement(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    """Contrasts run on both datasets (same pair of setups, this variant), with
    both verdicts. Verdicts agree only if they are the same kind: a significant
    drop on one dataset and a tiny lower mean on the other do not agree."""
    per_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if (
            row["a_variant"] != variant
            or row["b_variant"] != variant
            or row["a_budget"] != reg.PILOT
            or row["b_budget"] != reg.PILOT
            or row["n_splits"] < 2
        ):
            continue
        per_pair.setdefault((row["a_id"], row["b_id"]), {})[row["dataset"]] = row
    out = []
    for (a_id, b_id), sides in per_pair.items():
        if not {"amazon", "yelpchi"} <= set(sides):
            continue
        am, ye = sides["amazon"], sides["yelpchi"]
        out.append(
            {
                "contrast": f"{b_id} − {a_id}",
                "b_label": am["b_label"],
                "a_label": am["a_label"],
                "amazon": {
                    k: am[k]
                    for k in (
                        "mean",
                        "sd",
                        "p",
                        "n_splits",
                        "verdict_kind",
                        "verdict_label",
                    )
                },
                "yelpchi": {
                    k: ye[k]
                    for k in (
                        "mean",
                        "sd",
                        "p",
                        "n_splits",
                        "verdict_kind",
                        "verdict_label",
                    )
                },
                "agrees": am["verdict_kind"] == ye["verdict_kind"],
            }
        )
    return out


def cross_dataset(
    runs: pd.DataFrame, contrast_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """The setups that ran on both datasets at the pilot budget with all
    features; each dataset's normalised skill and gain over B, with Amazon's
    without-f19 numbers beside them where they exist."""
    amazon = _by_id(setup_summary(runs, "amazon", reg.PILOT, reg.ALL))
    yelp = _by_id(setup_summary(runs, "yelpchi", reg.PILOT, reg.ALL))
    no19 = _by_id(setup_summary(runs, "amazon", reg.PILOT, reg.NO19))
    shared = [i for i in amazon if i in yelp]
    keep = (
        "test_pr_auc",
        "test_pr_auc_sd",
        "normalised_skill",
        "normalised_skill_sd",
        "n_splits",
        "no_skill",
        "cv_pr_auc",
    )
    setups = []
    for setup in shared:
        row = {
            "setup": setup,
            "label": amazon[setup]["label"],
            "family": amazon[setup]["family"],
            "amazon": {k: amazon[setup][k] for k in keep},
            "yelpchi": {k: yelp[setup][k] for k in keep},
            "amazon_no19": ({k: no19[setup][k] for k in keep} if setup in no19 else None),
        }
        setups.append(row)
    gains = []
    for setup in shared:
        if setup == "B":
            continue
        gains.append(
            {
                "setup": setup,
                "label": amazon[setup]["label"],
                "family": amazon[setup]["family"],
                "amazon": graph_gain(runs, "amazon", reg.PILOT, reg.ALL, setup),
                "yelpchi": graph_gain(runs, "yelpchi", reg.PILOT, reg.ALL, setup),
                "amazon_no19": graph_gain(runs, "amazon", reg.PILOT, reg.NO19, setup),
            }
        )
    return {
        "setups": setups,
        "gains": gains,
        "agreement": _agreement(contrast_rows, reg.ALL),
    }


# --------------------------------------------------------------------------
# What was run, and what was not
# --------------------------------------------------------------------------
def overview(entries: list[Entry], runs: pd.DataFrame) -> dict[str, Any]:
    usable = usable_runs(runs)
    budgets = {b: int((usable["budget"] == b).sum()) for b in reg.BUDGETS}
    kinds: dict[str, int] = {}
    for entry in entries:
        kinds[entry.meta["kind"]] = kinds.get(entry.meta["kind"], 0) + 1
    seeds = sorted(
        {
            e.meta["seed"]
            for e in entries
            if e.meta["kind"] == "pipeline" and e.meta.get("seed") is not None
        }
    )
    full = usable[usable["budget"] == reg.FULL]
    return {
        "n_entries": len(entries),
        "kinds": kinds,
        "n_runs": len(usable),
        "budgets": budgets,
        "datasets": {
            ds: int((usable["dataset"] == ds).sum())
            for ds in sorted(usable["dataset"].dropna().unique())
        },
        "devices": sorted(usable["device"].dropna().unique().tolist()),
        "seeds": seeds,
        "first_started": str(usable["started_utc"].min()) if len(usable) else None,
        "last_started": str(usable["started_utc"].max()) if len(usable) else None,
        "n_backfilled": int(usable["backfilled"].sum()),
        "full_budget_where": sorted(
            f"{r.dataset} split {r.split_repeat}"
            for r in full[["dataset", "split_repeat"]].drop_duplicates().itertuples()
        ),
    }


def failed_entries(entries: list[Entry]) -> list[dict[str, Any]]:
    """Runs that failed or were stopped: shown, never counted."""
    return [
        {
            "run_id": e.run_id,
            "kind": e.meta["kind"],
            "status": e.meta["status"],
            "dataset": e.meta.get("dataset"),
            "description": e.meta["description"],
            "notes": e.meta.get("notes"),
            "started_utc": e.meta["started_utc"],
        }
        for e in entries
        if e.meta["status"] != "ok"
    ]


def coverage(
    runs: pd.DataFrame, dataset: str, variant: str, feature_sets: list[str]
) -> dict[str, Any]:
    """For one dataset and variant: pilot setups with fewer splits than the
    dataset's most, and setups in the config with no recorded run there, so a
    missing row is never read as a missing result."""
    usable = usable_runs(runs)
    rows = usable[usable["dataset"] == dataset]
    per_set = rows.groupby("feature_set")["split_repeat"].nunique()
    pilot = rows[rows["budget"] == reg.PILOT]
    most = (
        int(per_set.loc[list(pilot["feature_set"].unique())].max()) if len(pilot) else 0
    )
    here = pilot[pilot["variant"] == variant]["feature_set"].unique()
    fewer = [
        {
            "display_id": reg.parse_feature_set(name).display_id,
            "n_splits": int(per_set[name]),
        }
        for name in sorted(here, key=lambda n: reg.parse_feature_set(n).sort_key)
        if per_set[name] < most
    ]
    recorded = set(rows["feature_set"])
    not_run = []
    for name in feature_sets:
        fs = reg.parse_feature_set(name)
        if name not in recorded and fs.variant == variant:
            not_run.append({"display_id": fs.display_id, "budget": fs.budget})
    not_run.sort(key=lambda r: (r["budget"], r["display_id"]))
    return {"most_splits": most, "fewer_splits": fewer, "not_run": not_run}
