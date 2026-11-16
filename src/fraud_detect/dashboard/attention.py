"""The dashboard's GAT attention page: where E10 puts its weight, overall and for
one random test node.

Reads what scripts/gat_attention.py saved under
outputs/<dataset>/pilot_g10_gat/attention/ (split 0): the aggregate summary, and
for every test node its strongest incoming edges per layer and relation.
Computes nothing new about the model. Row-level and local only, like the
explanations page: nothing here is written to disk, and the server binds
127.0.0.1.

The map is of a refit: GAT fits are not bit-identical between runs, so the
script refits E10's fold-0 model (trained on four of the five training folds)
and checks that its held-out fold scores within REFIT_TOLERANCE of the run's.

A neighbour's label is shown only when it is a training node (a label the model
was trained on). Test and unlabelled neighbours are shown as such, and no test
label other than the drawn node's own is ever sent to the page.

A model is offered only after its checks pass (each goes to the page).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C
from fraud_detect.config import Paths, get_paths, with_dataset
from fraud_detect.dashboard import registry as reg
from fraud_detect.dashboard.checks import Check
from fraud_detect.models import gnn_outputs

SPLIT = 0
FEATURE_SET = "pilot_g10_gat"
OUTPUT_DIR = "attention"
DATASETS = ("amazon", "yelpchi")
NEEDED = (
    "attention_meta.json",
    "attention_summary.csv",
    "attention_edges.parquet",
    "attention_targets.parquet",
)
DRAWS = ("any", "fraud", "benign", "true_positive", "false_positive", "false_negative")
WEIGHT_TOLERANCE = 1e-5  # softmax weights in float32, summed
MAX_SEED = 2**32 - 1
# Neighbour kinds the page may show; labels only for training nodes.
TRAIN_FRAUD, TRAIN_BENIGN, TEST, UNLABELLED = (
    "train_fraud",
    "train_benign",
    "test",
    "unlabelled",
)
TARGET_GROUPS = {"train fraud": "fraud", "train benign": "benign"}


class AttentionUnavailable(RuntimeError):
    """A model's attention cannot be offered; the message says what to run."""


class AttentionRequestError(ValueError):
    """A request did not pass its checks; the message goes to the page."""


def relation_label(relation: str) -> str:
    return "-".join(relation.upper())


def _check(checks: list[Check], name: str, ok: bool, passed: str, failed: str) -> None:
    checks.append(Check(name, bool(ok), passed if ok else failed))


def _split_paths(config: dict[str, Any], dataset: str) -> Paths:
    cfg = copy.deepcopy(config)
    cfg["split"]["n_repeats"] = max(int(cfg["split"].get("n_repeats", 1)), SPLIT + 1)
    cfg["split"]["repeat"] = SPLIT
    cfg = with_dataset(cfg, dataset)
    cfg["feature_set"] = FEATURE_SET
    return get_paths(cfg)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


@dataclass
class AttentionModel:
    dataset: str
    folder: Path
    meta: dict[str, Any]
    relations: list[str]
    summary_table: pd.DataFrame
    edges: pd.DataFrame  # sorted by target, layer, relation, rank
    edge_targets: np.ndarray  # edges["target"], for searchsorted
    targets: pd.DataFrame  # indexed by (target, layer, relation)
    test: pd.DataFrame  # node_id, label, prob (the run's), refit_prob; test order
    threshold: float
    kind: dict[int, str]  # training nodes' kind; others are test or unlabelled
    test_set: frozenset[int]
    checks: list[Check] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.dataset

    def summary(self) -> dict[str, Any]:
        """The model view: aggregates only."""
        t = self.summary_table
        focus = t[t["target_group"].isna()].copy()
        focus["relation_label"] = focus["relation"].map(relation_label)
        weights = t[t["target_group"].notna()].copy()
        heat = []
        for (layer, rel), part in focus[focus["head"] == "mean"].groupby(
            ["layer", "relation"], sort=True
        ):
            row = part.iloc[0]
            w = weights[(weights["layer"] == layer) & (weights["relation"] == rel)]
            entry = {
                "layer": int(layer),
                "relation": rel,
                "relation_label": relation_label(rel),
                "mean_degree": row["mean_degree_deg2plus"],
                "focus": row["entropy_ratio"],
                "max_alpha": row["max_alpha"],
                "uniform_max": row["uniform_max"],
                "top_vs_equal": row["max_alpha"] / row["uniform_max"]
                if row["uniform_max"]
                else None,
                "relation_to_root": w["relation_to_root"].iloc[0] if len(w) else None,
                "heads": [
                    float(part2["entropy_ratio"].iloc[0])
                    for _, part2 in focus[
                        (focus["layer"] == layer)
                        & (focus["relation"] == rel)
                        & (focus["head"] != "mean")
                    ].groupby("head", sort=True)
                ],
            }
            for group, short in TARGET_GROUPS.items():
                g = w[w["target_group"] == group]
                if len(g) and g["fraud_share_uniform"].iloc[0]:
                    entry[f"fraud_attention_{short}"] = g["fraud_share_attention"].iloc[0]
                    entry[f"fraud_uniform_{short}"] = g["fraud_share_uniform"].iloc[0]
                    entry[f"fraud_ratio_{short}"] = (
                        g["fraud_share_attention"].iloc[0]
                        / g["fraud_share_uniform"].iloc[0]
                    )
                    entry[f"fraud_n_{short}"] = int(g["n_targets_mixed"].iloc[0])
            heat.append(entry)
        check = self.meta.get("check", {})
        setting = self.meta.get("setting", {})
        return {
            "key": self.key,
            "dataset": self.dataset,
            "label": f"E10 GAT · {reg.variant_label(reg.ALL, self.dataset)}",
            "feature_set": FEATURE_SET,
            "relations": self.relations,
            "relation_labels": {r: relation_label(r) for r in self.relations},
            "layers": int(self.meta.get("layers", 2)),
            "heads": int(self.meta.get("heads", 1)),
            "top_k": int(self.meta.get("top_k", 0)),
            "setting": setting,
            "fold_model": self.meta.get("fold_model"),
            "best_epoch": self.meta.get("best_epoch"),
            "refit_fold_pr_auc": check.get("refit_fold_pr_auc"),
            "saved_fold_pr_auc": check.get("saved_fold_pr_auc"),
            "n_test": len(self.test),
            "n_fraud": int(self.test[C.LABEL].sum()),
            "threshold": self.threshold,
            "heat": heat,
            "checks": [c.as_dict() for c in self.checks],
            "draws": list(DRAWS),
        }

    # ---- One node -------------------------------------------------------
    def draw(self, draw: str, seed: Any) -> dict[str, Any]:
        if draw not in DRAWS:
            raise AttentionRequestError(f"Unknown draw {draw!r}; use one of {DRAWS}")
        if seed is None:
            seed = int(np.random.default_rng().integers(MAX_SEED))
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= MAX_SEED
        ):
            raise AttentionRequestError(
                f"The seed must be an integer from 0 to {MAX_SEED}"
            )
        label = self.test[C.LABEL].to_numpy()
        flagged = self.test["prob"].to_numpy() >= self.threshold
        masks = {
            "any": np.ones(len(label), dtype=bool),
            "fraud": label == 1,
            "benign": label == 0,
            "true_positive": flagged & (label == 1),
            "false_positive": flagged & (label == 0),
            "false_negative": ~flagged & (label == 1),
        }
        where = np.flatnonzero(masks[draw])
        if len(where) == 0:
            raise AttentionRequestError(
                f"No test node is a {draw.replace('_', ' ')} here"
            )
        rng = np.random.default_rng([seed, DRAWS.index(draw)])
        card = self.card(int(rng.choice(where)))
        card.update({"draw": draw, "seed": seed, "pool": len(where)})
        return card

    def neighbour_kind(self, node: int) -> str:
        if node in self.kind:
            return self.kind[node]
        return TEST if node in self.test_set else UNLABELLED

    def card(self, row: int) -> dict[str, Any]:
        r = self.test.iloc[row]
        node, label = int(r[C.NODE_ID]), int(r[C.LABEL])
        if node not in self.test_set or node in self.kind:
            raise AttentionRequestError("Only test nodes are shown")
        lo, hi = np.searchsorted(self.edge_targets, [node, node + 1])
        mine = self.edges.iloc[lo:hi]
        head_cols = [c for c in self.edges.columns if c.startswith("alpha_h")]
        layers = []
        for layer in range(1, int(self.meta.get("layers", 2)) + 1):
            rels = []
            for rel in self.relations:
                t = self.targets.loc[(node, layer, rel)]
                part = mine[(mine["layer"] == layer) & (mine["relation"] == rel)]
                shown = float(part["alpha"].sum())
                if abs(shown - float(t["shown_weight"])) > WEIGHT_TOLERANCE:
                    raise AttentionRequestError(
                        f"node {node}: the saved edges do not add up to its saved weight"
                    )
                degree = int(t["degree"])
                rels.append({
                    "relation": rel,
                    "relation_label": relation_label(rel),
                    "degree": degree,
                    "n_shown": int(t["n_shown"]),
                    "shown_weight": float(t["shown_weight"]),
                    "equal_weight": 1.0 / degree if degree else None,
                    "focus": None if pd.isna(t["focus"]) else float(t["focus"]),
                    "relation_to_root": None
                    if pd.isna(t["relation_to_root"])
                    else float(t["relation_to_root"]),
                    "neighbours": [
                        {
                            "node_id": int(e["neighbour"]),
                            "rank": int(e["rank"]),
                            "alpha": float(e["alpha"]),
                            "heads": [float(e[c]) for c in head_cols],
                            "kind": self.neighbour_kind(int(e["neighbour"])),
                        }
                        for _, e in part.iterrows()
                    ],
                })  # fmt: skip
            layers.append({"layer": layer, "relations": rels})
        flagged = bool(r["prob"] >= self.threshold)
        return {
            "model": self.key,
            "node_id": node,
            "label": label,
            "prob": float(r["prob"]),
            "refit_prob": float(r["refit_prob"]),
            "threshold": self.threshold,
            "flagged": flagged,
            "outcome": (
                ("true_positive" if label else "false_positive")
                if flagged
                else ("false_negative" if label else "true_negative")
            ),
            "layers": layers,
        }


def load_model(config: dict[str, Any], dataset: str) -> AttentionModel:
    """One dataset's GAT attention, with its checks. Raises AttentionUnavailable
    if a file is missing or a check fails."""
    if FEATURE_SET not in config.get("feature_sets", {}):
        raise AttentionUnavailable(f"{FEATURE_SET} is not a feature set in this config")
    p = _split_paths(config, dataset)
    folder = p.feature_set_dir / OUTPUT_DIR
    missing = [n for n in NEEDED if not (folder / n).is_file()]
    if missing:
        raise AttentionUnavailable(
            f"No finished attention read-out in {folder} (missing {', '.join(missing)}). "
            f"Run: uv run scripts/gat_attention.py --dataset {dataset} --split {SPLIT}"
        )
    meta = json.loads((folder / "attention_meta.json").read_text(encoding="utf-8"))
    checks: list[Check] = []
    _check(
        checks,
        "The read-out is of split 0 of E10",
        meta.get("dataset") == dataset
        and int(meta.get("split_repeat", -1)) == SPLIT
        and meta.get("feature_set") == FEATURE_SET,
        f"{dataset}, split {SPLIT}, {FEATURE_SET}",
        f"attention_meta.json says {meta.get('dataset')}, split "
        f"{meta.get('split_repeat')}, {meta.get('feature_set')}",
    )
    check = meta.get("check", {})
    _check(
        checks,
        "The refit matches the run's fold model",
        check.get("matches") is True,
        f"held-out fold PR-AUC {check.get('refit_fold_pr_auc', float('nan')):.4f} "
        f"against the run's {check.get('saved_fold_pr_auc') or float('nan'):.4f}",
        f"refit check: {check}",
    )

    test_ids = pd.read_csv(p.test_ids)
    train_ids = pd.read_csv(p.train_ids)
    test_nodes = test_ids[C.NODE_ID].to_numpy(dtype=int)
    relations = list(meta.get("relations") or config["datasets"][dataset]["relations"])
    n_layers = int(meta.get("layers", 2))

    targets = pd.read_parquet(folder / "attention_targets.parquet")
    targets["relation"] = targets["relation"].astype(str)
    edges = pd.read_parquet(folder / "attention_edges.parquet")
    edges["relation"] = edges["relation"].astype(str)
    edges = edges.sort_values(["target", "layer", "relation", "rank"], kind="stable")
    edges = edges.reset_index(drop=True)
    leaked = {"label", C.LABEL, "y", "prob"} & (set(targets) | set(edges))
    _check(
        checks,
        "The saved attention holds no labels",
        not leaked,
        "no label column in either file",
        f"label-like columns: {sorted(leaked)}",
    )
    expected = len(test_nodes) * n_layers * len(relations)
    per_target = targets.groupby("target").size()
    _check(
        checks,
        "The mapped nodes are exactly the test nodes",
        set(per_target.index) == set(test_nodes)
        and len(targets) == expected
        and bool((per_target == n_layers * len(relations)).all())
        and set(edges["target"]) <= set(test_nodes),
        f"{len(test_nodes):,} test nodes × {n_layers} layers × {len(relations)} "
        "relations",
        "the saved targets differ from test_ids.csv",
    )
    overlap = np.intersect1d(targets["target"].unique(), train_ids[C.NODE_ID])
    _check(
        checks,
        "No mapped node was trained on",
        len(overlap) == 0,
        f"none of the {len(train_ids):,} training nodes",
        f"{len(overlap)} mapped nodes are training nodes",
    )
    shown = edges.groupby(["target", "layer", "relation"])["alpha"].agg(["sum", "size"])
    joined = targets.set_index(["target", "layer", "relation"]).join(shown)
    joined[["sum", "size"]] = joined[["sum", "size"]].fillna(0)
    full = (joined["degree"] > 0) & (joined["degree"] <= int(meta.get("top_k", 0)))
    sums_ok = (
        bool((joined["size"] == joined["n_shown"]).all())
        and bool(
            ((joined["sum"] - joined["shown_weight"]).abs() <= WEIGHT_TOLERANCE).all()
        )
        and bool((joined["shown_weight"] <= 1 + WEIGHT_TOLERANCE).all())
        and bool(((joined.loc[full, "shown_weight"] - 1).abs() <= WEIGHT_TOLERANCE).all())
    )
    _check(
        checks,
        "Attention weights add up",
        sums_ok,
        "each node's weights over a relation sum to 1; the shown neighbours' share "
        "matches the saved one",
        "a node's saved weights do not add up",
    )
    head_cols = [c for c in edges.columns if c.startswith("alpha_h")]
    _check(
        checks,
        "The mean weight is the mean over heads",
        len(head_cols) == int(meta.get("heads", 0))
        and np.allclose(edges[head_cols].mean(axis=1), edges["alpha"], atol=1e-7),
        f"{len(head_cols)} heads",
        "the saved mean weight is not the heads' mean",
    )

    reports = p.model_reports_dir
    model_a = json.loads((reports / "model_a.json").read_text(encoding="utf-8"))
    try:
        run_prob = gnn_outputs.test_probabilities(reports, test_ids)
    except ValueError as e:
        raise AttentionUnavailable(str(e)) from e
    _check(
        checks,
        "The run's test scores line up with the test nodes",
        True,
        f"{len(run_prob):,} test scores in test_ids.csv order",
        "",
    )
    refit = targets.drop_duplicates("target").set_index("target")["refit_prob"]
    test = pd.DataFrame({
        C.NODE_ID: test_nodes,
        C.LABEL: test_ids[C.LABEL].to_numpy(dtype=int),
        "prob": run_prob,
        "refit_prob": refit.reindex(test_nodes).to_numpy(dtype=float),
    })  # fmt: skip
    kind = {
        int(n): TRAIN_FRAUD if int(y) == 1 else TRAIN_BENIGN
        for n, y in zip(train_ids[C.NODE_ID], train_ids[C.LABEL], strict=True)
    }
    model = AttentionModel(
        dataset=dataset,
        folder=folder,
        meta=meta,
        relations=relations,
        summary_table=pd.read_csv(folder / "attention_summary.csv"),
        edges=edges,
        edge_targets=edges["target"].to_numpy(),
        targets=targets.set_index(["target", "layer", "relation"]).sort_index(),
        test=test,
        threshold=float(model_a["model_a"]["threshold"]),
        kind=kind,
        test_set=frozenset(int(n) for n in test_nodes),
        checks=checks,
    )
    bad = [c for c in checks if not c.ok]
    if bad:
        raise AttentionUnavailable("; ".join(f"{c.name}: {c.detail}" for c in bad))
    return model
