"""Classifying one random test node with a dataset's best model.

Row-level, local only: nothing here is written to disk, and the server binds
127.0.0.1. A target is the best setup of a dataset and feature variant
(data.best_setup), its saved Model A on split 0, the prepared test table that
model takes, and the baseline B on the same test nodes.

Loading a target checks that everything lines up before any node is shown; each
check goes to the page. Then verify_live compares the running model with the
probabilities df-analyze saved, for every test row, and every click compares
again for its own row. Only test nodes can be drawn.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detect import columns as C
from fraud_detect.config import Paths, get_paths, with_dataset
from fraud_detect.dashboard import registry as reg
from fraud_detect.dashboard.checks import Check
from fraud_detect.dashboard.worker import ModelWorker, WorkerError
from fraud_detect.models import df_analyze, df_analyze_report
from fraud_detect.models import model_export as mx
from fraud_detect.report import compare, web

# Which variants get a classify panel (decided 2026-11-02: Amazon opens on all 25
# features, and without f19 is one click away; both models are offered).
TARGET_VARIANTS = {"amazon": (reg.ALL, reg.NO19), "yelpchi": (reg.ALL,)}
SPLIT = 0
N_TOP_FEATURES = 6
SCORE_TOLERANCE = 1e-9
DRAWS = ("any", "fraud", "benign")
MAX_SEED = 2**32 - 1
LIVE, SAVED = "live", "saved"


class TargetUnavailable(RuntimeError):
    """A target cannot be loaded; the message says what to run."""


class ClassifyError(RuntimeError):
    """A request or a row did not pass its checks; the message goes to the page."""


def target_key(dataset: str, variant: str) -> str:
    return f"{dataset}-{variant}"


def split_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """The config of split 0 of `dataset`."""
    cfg = copy.deepcopy(config)
    cfg["split"]["n_repeats"] = max(int(cfg["split"].get("n_repeats", 1)), SPLIT + 1)
    cfg["split"]["repeat"] = SPLIT
    return with_dataset(cfg, dataset)


@dataclass
class Target:
    key: str
    dataset: str
    variant: str
    best: dict[str, Any]
    side: compare.Side
    baseline: compare.Side | None
    y: np.ndarray  # test labels, test_ids.csv order
    node_ids: np.ndarray
    graph: compare.GraphContext | None
    relations: list[str]
    seed: int
    test_metrics: dict[str, Any]
    checks: list[Check] = field(default_factory=list)
    mode: str = SAVED
    prepared: pd.DataFrame | None = None  # X_test_00.csv, test_ids.csv order
    model_dir: Path | None = None
    model_meta: dict[str, Any] | None = None
    tolerance: float = 0.0
    live_verified: bool = False
    note: str = ""

    def status(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dataset": self.dataset,
            "variant": self.variant,
            "variant_label": reg.variant_label(self.variant, self.dataset),
            "best": self.best,
            "mode": self.mode,
            "live_verified": self.live_verified,
            "note": self.note,
            "n_test": len(self.y),
            "n_test_fraud": int(self.y.sum()),
            "threshold": self.side.threshold,
            "threshold_rule": "max F1 on out-of-fold training predictions",
            "model": self.side.summary["model_a"]["model"],
            "selection": self.side.summary["model_a"]["selection"],
            "test_metrics": self.test_metrics,
            "checks": [c.as_dict() for c in self.checks],
            "checks_ok": all(c.ok for c in self.checks),
        }


def _check(checks: list[Check], name: str, ok: bool, passed: str, failed: str) -> None:
    checks.append(Check(name, ok, passed if ok else failed))


def _require(checks: list[Check]) -> None:
    bad = [c for c in checks if not c.ok]
    if bad:
        raise TargetUnavailable("; ".join(f"{c.name}: {c.detail}" for c in bad))


def _load_side(cfg: dict[str, Any], feature_set: str) -> compare.Side:
    try:
        return compare.load_side(cfg, feature_set)
    except (compare.CompareError, OSError, ValueError, KeyError) as e:
        raise TargetUnavailable(
            f"{feature_set}: {e} This panel needs the run's outputs under outputs/ and "
            "data/processed/ (see docs/DASHBOARD.md)."
        ) from e


def whole_test_metrics(
    y: np.ndarray, prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Model A on the whole test set, saved predictions: the honest context for
    any one draw."""
    flagged = (prob >= threshold).astype(int)
    m = df_analyze_report.binary_metrics(y, prob, flagged)
    return {
        "pr_auc": float(average_precision_score(y, prob)),
        "f1": float(m["f1"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "tp": int(m["tp"]),
        "fp": int(m["fp"]),
        "fn": int(m["fn"]),
        "tn": int(m["tn"]),
        "no_skill": float(y.mean()),
    }


def load_target(
    config: dict[str, Any],
    dataset: str,
    variant: str,
    best: dict[str, Any],
    headline: dict[str, Any] | None,
    graphs: dict[str, compare.GraphContext] | None = None,
) -> Target:
    """A target, with its checks. Raises TargetUnavailable if a check fails or a
    file is missing. `headline` is the results database's entry for this run."""
    cfg = split_config(config, dataset)
    fs = best["feature_set"]
    p = get_paths({**cfg, "feature_set": fs})
    side = _load_side(cfg, fs)
    a = side.summary["model_a"]
    checks: list[Check] = []

    y = side.test_ids[C.LABEL].to_numpy(dtype=int)
    node_ids = side.test_ids[C.NODE_ID].to_numpy(dtype=int)
    _check(
        checks,
        "Test IDs are usable",
        len(np.unique(node_ids)) == len(node_ids) and set(np.unique(y)) <= {0, 1},
        f"{len(node_ids):,} distinct test nodes, labels 0 or 1",
        "duplicate test IDs or labels other than 0 and 1",
    )
    if not p.train_ids.is_file():
        raise TargetUnavailable(f"{p.train_ids} is missing; run the split stage.")
    train_ids = pd.read_csv(p.train_ids)[C.NODE_ID].to_numpy(dtype=int)
    overlap = np.intersect1d(train_ids, node_ids)
    _check(
        checks,
        "No test node was trained on",
        len(overlap) == 0,
        f"{len(train_ids):,} training nodes, none among the test nodes",
        f"{len(overlap)} nodes are in both train and test",
    )
    # The pipeline's saved probabilities give back its recorded score.
    score = float(average_precision_score(y, side.prob))
    _check(
        checks,
        "Saved predictions reproduce the recorded PR-AUC",
        abs(score - float(a["pr_auc"])) <= SCORE_TOLERANCE,
        f"PR-AUC {score:.4f} from {len(y):,} saved test probabilities equals "
        "model_a.json",
        f"{score:.6f} from saved probabilities, {a['pr_auc']:.6f} in model_a.json",
    )
    if headline is not None:
        same = (
            abs(float(headline["test_pr_auc"]) - float(a["pr_auc"])) <= SCORE_TOLERANCE
            and abs(float(headline["threshold"]) - side.threshold) <= SCORE_TOLERANCE
            and int(headline["n_test"]) == len(y)
        )
        _check(
            checks,
            "This run matches its results-database entry",
            same,
            "same PR-AUC, threshold and test size",
            "the entry and this run's outputs differ",
        )

    baseline = None
    base_name = _baseline_name(fs, variant)
    if base_name and base_name != fs:
        baseline = _load_side(cfg, base_name)
        try:
            compare.check_same_test_nodes(side, baseline)
        except compare.CompareError as e:
            checks.append(Check("Baseline B has the same test nodes", False, str(e)))
        else:
            checks.append(
                Check(
                    "Baseline B has the same test nodes",
                    True,
                    f"{base_name}: the same {len(y):,} nodes and labels",
                )
            )

    relations = list(cfg["datasets"][dataset]["relations"])
    graph = None
    if reg.parse_feature_set(fs).setup.uses_graph:
        graphs = graphs if graphs is not None else {}
        if dataset not in graphs:
            try:
                graphs[dataset] = compare.load_graph(p, relations)
            except (OSError, KeyError, ValueError) as e:
                raise TargetUnavailable(f"The graph of {dataset} is missing: {e}") from e
        graph = graphs[dataset]

    target = Target(
        key=target_key(dataset, variant),
        dataset=dataset,
        variant=variant,
        best=best,
        side=side,
        baseline=baseline,
        y=y,
        node_ids=node_ids,
        graph=graph,
        relations=relations,
        seed=int(cfg["seed"]),
        test_metrics=whole_test_metrics(y, side.prob, side.threshold),
        checks=checks,
    )
    if not best["gnn_rung"]:
        _load_live_inputs(target, p)
    else:
        target.note = (
            "This GNN needs the whole graph to predict, so this panel shows its saved "
            "test prediction, not a live one."
        )
    _require(checks)
    return target


def _baseline_name(feature_set: str, variant: str) -> str | None:
    """B's feature set for a variant, at the pilot budget."""
    if variant not in (reg.ALL, reg.NO19):
        return None
    suffix = "" if variant == reg.ALL else f"_{variant}"
    return f"{reg.PILOT_PREFIX}m1_own{suffix}"


def _load_live_inputs(target: Target, p: Paths) -> None:
    """The exported model's metadata and its prepared test table, each checked
    against the pipeline's run. Missing files leave the target on saved
    predictions, with a note saying which stage to run."""
    side, checks = target.side, target.checks
    a = side.summary["model_a"]
    try:
        meta = mx.read_metadata(p.model_dir / mx.MODEL_A_META)
    except mx.MetadataError as e:
        target.note = f"No exported model, so predictions are the saved ones: {e}"
        return
    export = df_analyze.find_export_dir(p.df_analyze_output_dir / side.run)
    prepared = pd.read_csv(export / "X_test_00.csv", float_precision="round_trip")
    y_export = pd.read_csv(export / "y_test_00.csv", index_col=0).iloc[:, 0].astype(int)
    _check(
        checks,
        "The exported model is this run's Model A",
        meta["run"] == side.run
        and meta["model"] == a["model"]
        and meta["selection"] == a["selection"]
        and abs(float(meta["threshold"]) - side.threshold) <= SCORE_TOLERANCE,
        f"{meta['model']} ({meta['selection']}), run {side.run}, same threshold",
        f"model_a.json says {meta['model']} ({meta['selection']}) from {meta['run']}; "
        f"the report says {a['model']} ({a['selection']}) from {side.run}",
    )
    _check(
        checks,
        "The model file is present",
        (p.model_dir / meta["model_file"]).is_file(),
        meta["model_file"],
        f"{meta['model_file']} is missing",
    )
    _check(
        checks,
        "The prepared test table lines up with the test IDs",
        len(prepared) == len(target.y) and np.array_equal(y_export.to_numpy(), target.y),
        f"{len(prepared):,} rows, labels equal test_ids.csv row by row",
        f"{len(prepared)} rows against {len(target.y)} test IDs, or different labels",
    )
    try:
        mx.check_columns(
            prepared.columns, meta["input_columns"], meta["prepared_columns"]
        )
    except mx.ColumnMismatchError as e:
        checks.append(Check("The prepared columns are the model's", False, str(e)))
    else:
        checks.append(
            Check(
                "The prepared columns are the model's",
                True,
                f"{len(prepared.columns)} columns, in order",
            )
        )
    target.prepared = prepared
    target.model_dir = p.model_dir.resolve()
    target.model_meta = meta
    target.tolerance = mx.tolerance_for(str(meta["model"]))


def verify_live(target: Target, worker: ModelWorker) -> Check:
    """The acceptance check: for every test row, the running model's P(fraud)
    equals what df-analyze saved, within the export check's tolerance. A target
    that passes classifies live; one that fails stays on saved predictions."""
    if target.prepared is None:
        raise ClassifyError(f"{target.key} has no exported model")
    prob = worker.predict(
        target.key,
        list(target.prepared.columns),
        target.prepared.to_numpy(dtype=float).tolist(),
    )
    worst = mx.max_abs_diff(prob, target.side.prob)
    ok = worst <= target.tolerance
    check = Check(
        "The running model reproduces every saved test probability",
        ok,
        f"{len(prob):,} rows, largest difference {worst:.2e}, tolerance "
        f"{target.tolerance:g}",
    )
    target.checks.append(check)
    target.live_verified = ok
    target.mode = LIVE if ok else SAVED
    if not ok:
        target.note = "The running model disagreed with the saved predictions, so this "
        target.note += "panel shows the saved ones."
    return check


def draw_row(y: np.ndarray, draw: str, rng: np.random.Generator) -> int:
    """A random row of the test set: any node, or one of the asked class."""
    if draw == "any":
        pool = np.arange(len(y))
    elif draw == "fraud":
        pool = np.flatnonzero(y == 1)
    elif draw == "benign":
        pool = np.flatnonzero(y == 0)
    else:
        raise ClassifyError(f"Unknown draw {draw!r}; use one of {DRAWS}")
    if len(pool) == 0:
        raise ClassifyError(f"The test set has no {draw} node")
    return int(rng.choice(pool))


def make_rng(draw: str, seed: int | None) -> np.random.Generator:
    """A seed makes a draw reproducible, for slides."""
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ClassifyError(f"The seed must be a whole number from 0 to {MAX_SEED}")
    if draw not in DRAWS:
        raise ClassifyError(f"Unknown draw {draw!r}; use one of {DRAWS}")
    return np.random.default_rng([seed, DRAWS.index(draw)])


def outcome_label(predicted: int, actual: int) -> str:
    return {
        (1, 1): "correct: caught fraud",
        (0, 0): "correct: not fraud",
        (1, 0): "false positive",
        (0, 1): "false negative",
    }[(predicted, actual)]


def _prediction(target: Target, row: int, worker: ModelWorker | None) -> dict[str, Any]:
    """P(fraud) for one row: live and compared with the saved value, or saved."""
    saved = float(target.side.prob[row])
    if target.mode != LIVE:
        return {
            "prob": saved,
            "mode": SAVED,
            "verified": False,
            "saved_prob": saved,
            "difference": None,
        }
    if worker is None or target.prepared is None:
        raise ClassifyError("The model worker is not running")
    try:
        prob = worker.predict(
            target.key,
            list(target.prepared.columns),
            [target.prepared.iloc[row].to_numpy(dtype=float).tolist()],
        )[0]
    except WorkerError as e:
        raise ClassifyError(str(e)) from e
    difference = abs(prob - saved)
    if difference > target.tolerance:
        raise ClassifyError(
            f"Node {target.node_ids[row]}: the running model says {prob}, df-analyze "
            f"saved {saved}. Stopped rather than showing a score that may be wrong."
        )
    return {
        "prob": float(prob),
        "mode": LIVE,
        "verified": True,
        "saved_prob": saved,
        "difference": float(difference),
    }


def _top_features(target: Target, row: int) -> list[dict[str, Any]]:
    side = target.side
    columns = list(side.used_columns)
    (features,) = web.top_features(
        side.test.iloc[[row]][columns], side.train[columns], N_TOP_FEATURES
    )
    return [
        {"name": compare.feature_label(name), "value": value, "percentile": pct}
        for name, value, pct in features
    ]


def _neighbourhoods(target: Target, node: int, label: int) -> list[dict[str, Any]]:
    if target.graph is None:
        return []
    out = []
    for relation in target.relations:
        ego = compare.ego_network(target.graph, relation, node, target.seed)
        share = ego["train_fraud_share"]
        out.append(
            {
                "relation": compare.relation_label(relation),
                "degree": ego["degree"],
                "n_shown": len(ego["shown"]),
                "n_train_neighbours": ego["n_train_neighbours"],
                "train_fraud_share": share,
                "svg": compare.ego_svg(node, label, ego),
            }
        )
    return out


def _baseline(
    target: Target, row: int, predicted: int, actual: int
) -> dict[str, Any] | None:
    base = target.baseline
    if base is None:
        return None
    prob = float(base.prob[row])
    flagged = int(prob >= base.threshold)
    return {
        "setup": "B",
        "label": reg.BY_ID["B"].label,
        "prob": prob,
        "threshold": base.threshold,
        "predicted": flagged,
        "outcome": outcome_label(flagged, actual),
        "changed": flagged != predicted,
        "source": "saved test prediction",
    }


def classify(
    target: Target, worker: ModelWorker | None, draw: str, seed: int | None
) -> dict[str, Any]:
    """Draw a test node and classify it. The result is row-level data: it goes
    to the page and nowhere else."""
    rng = make_rng(draw, seed)
    row = draw_row(target.y, draw, rng)
    node, actual = int(target.node_ids[row]), int(target.y[row])
    if (draw == "fraud" and actual != 1) or (draw == "benign" and actual != 0):
        raise ClassifyError(f"Asked for {draw}, drew label {actual}")
    pred = _prediction(target, row, worker)
    threshold = target.side.threshold
    predicted = int(pred["prob"] >= threshold)
    return {
        "target": target.key,
        "dataset": target.dataset,
        "variant": target.variant,
        "setup": target.best["display_id"],
        "setup_label": target.best["label"],
        "node_id": node,
        "draw": draw,
        "seed": seed,
        **pred,
        "threshold": threshold,
        "predicted": predicted,
        "actual": actual,
        "outcome": outcome_label(predicted, actual),
        "correct": predicted == actual,
        "confidence": pred["prob"] if predicted == 1 else 1.0 - pred["prob"],
        "top_features": _top_features(target, row),
        "neighbourhoods": _neighbourhoods(target, node, actual),
        "baseline": _baseline(target, row, predicted, actual),
    }
