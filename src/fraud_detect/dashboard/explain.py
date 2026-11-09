"""The dashboard's explainability page: SHAP and neighbour influence, per model and
for one random test node.

Reads what scripts/dfa/shap_explain.py and scripts/explain_report.py already
saved under outputs/<dataset>/<feature set>/explain/ (split 0); computes nothing
new about the model. Row-level and local only, like the classify panel: nothing
here is written to disk, and the server binds 127.0.0.1.

A model is offered only after its checks pass (each goes to the page): the SHAP
rows are exactly the split's test nodes, none of them a training node, the
saved margins match the saved probabilities, and base value plus SHAP gives the
margin. Each drawn card checks its own row's sum again before it is shown.
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
from fraud_detect.models import df_analyze
from fraud_detect.models import explain as X

SPLIT = 0
SHAP_PREFIX = "shap__"
GROUP_PREFIX = "group__"
# Which explanations the page offers, most wanted first: the first folder with
# finished outputs is used. Full budget where it was explained, pilot otherwise.
TARGETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("amazon", reg.NO19, ("m2_own_nbr_no19", "pilot_m2_own_nbr_no19")),
    ("amazon", reg.ALL, ("m2_own_nbr", "pilot_m2_own_nbr")),
    ("yelpchi", reg.ALL, ("m5_own_gnnssl", "pilot_m5_own_gnnssl")),
    ("yelpchi", "all-e1", ("m2_own_nbr", "pilot_m2_own_nbr")),
)
DRAWS = (
    "any",
    "fraud",
    "benign",
    "true_positive",
    "false_positive",
    "false_negative",
)
N_WATERFALL = 10
N_NEIGHBOURS = 5
N_NEAREST = 3
N_TOP = 15
N_STRATUM_TOP = 8
ADDITIVITY_TOLERANCE = 1e-3  # log-odds; the implementation check used 1e-4 per row
PROB_TOLERANCE = 1e-6
MAX_SEED = 2**32 - 1
STRATA_TABLES = {
    "fraud": "importance_fraud.csv",
    "flagged": "importance_flagged.csv",
    "false_positive": "importance_false_positive.csv",
    "false_negative": "importance_false_negative.csv",
}
STRATUM_LABELS = {
    "all": "all test nodes",
    "fraud": "fraud test nodes",
    "flagged": "flagged (score at or above the threshold)",
    "false_positive": "false positives",
    "false_negative": "false negatives (missed fraud)",
    "true_positive": "true positives",
}
KIND_LABELS = {
    "nbr": "neighbour mean",
    "gnnssl": "DGI embedding",
    "gnnrand": "untrained-GNN embedding",
    "deg": "degree",
    "nbrrw": "rewired neighbour mean",
    "gnnsup": "supervised GNN score",
}


class ExplainUnavailable(RuntimeError):
    """A model's explanation cannot be offered; the message says what to run."""


class ExplainRequestError(ValueError):
    """A request or a row did not pass its checks; the message goes to the page."""


def relation_label(relation: str | None) -> str:
    return "-".join(relation.upper()) if relation else ""


def column_label(info: X.ColumnInfo) -> str:
    """A readable name for one model input column."""
    rel = relation_label(info.relation)
    if info.kind == C.OWN_BLOCK:
        return f"{info.base_feature} (own)"
    if info.stat == "mean":
        return f"{info.base_feature}, mean over {rel} neighbours"
    if info.stat == "log1p_degree":
        return f"{rel} degree (log)"
    if info.kind == C.GNNSUP_BLOCK:
        return "supervised GNN score"
    what = info.block_column.partition(C.BLOCK_SEPARATOR)[2].partition("_")[2]
    return f"{KIND_LABELS.get(info.kind, info.kind)} {rel} {what}".strip()


def player_label(player: str) -> str:
    if player == C.OWN_BLOCK:
        return "own features"
    if player == X.DEGREE_PLAYER:
        return "degree (every relation)"
    kind, _, relation = player.partition("_")
    return f"{KIND_LABELS.get(kind, kind)}s, {relation_label(relation)}"


def target_key(dataset: str, variant: str) -> str:
    return f"{dataset}-{variant}"


def _split_paths(config: dict[str, Any], dataset: str, feature_set: str) -> Paths:
    cfg = copy.deepcopy(config)
    cfg["split"]["n_repeats"] = max(int(cfg["split"].get("n_repeats", 1)), SPLIT + 1)
    cfg["split"]["repeat"] = SPLIT
    cfg = with_dataset(cfg, dataset)
    cfg["feature_set"] = feature_set
    return get_paths(cfg)


def _sigmoid(margin: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-margin))


def _csv(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / "tables" / name
    if not path.is_file():
        path = folder / name
    return pd.read_csv(path) if path.is_file() else None


def _records(frame: pd.DataFrame | None, n: int | None = None) -> list[dict[str, Any]]:
    if frame is None:
        return []
    frame = frame if n is None else frame.head(n)
    return frame.to_dict(orient="records")


@dataclass
class Influence:
    """neighbour_influence.parquet, sorted by target for quick lookup."""

    table: pd.DataFrame
    targets: np.ndarray  # the sorted target column

    @classmethod
    def read(cls, path: Path) -> Influence:
        table = pd.read_parquet(path)
        table["relation"] = table["relation"].astype(str)
        table = table.sort_values(["target", "relation"], kind="stable")
        table = table.reset_index(drop=True)
        return cls(table, table["target"].to_numpy())

    def of(self, node: int) -> pd.DataFrame:
        lo, hi = np.searchsorted(self.targets, [node, node + 1])
        return self.table.iloc[lo:hi]


@dataclass
class ExplainModel:
    key: str
    dataset: str
    variant: str
    feature_set: str
    folder: Path
    meta: dict[str, Any]
    columns: list[str]  # df-analyze's prepared names, SHAP column order
    infos: list[X.ColumnInfo]
    rows: pd.DataFrame  # node_id, group_id, label, prob, margin (test, SHAP order)
    shap: np.ndarray
    usable: np.ndarray  # rows whose SHAP passed the implementation check
    base: float
    threshold: float
    x: np.ndarray | None  # prepared model inputs, same row order, or None
    group_shapley: pd.DataFrame | None  # indexed by node_id
    influence: Influence | None
    scores: pd.Series | None  # every node's Model A probability, by node_id
    train_shap: np.ndarray
    train_rows: pd.DataFrame  # node_id, label, prob
    relations: list[str]
    overall_rank: dict[str, int]
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def budget(self) -> str:
        return reg.parse_feature_set(self.feature_set).budget

    def label(self) -> str:
        fs = reg.parse_feature_set(self.feature_set)
        variant = reg.variant_label(fs.variant, self.dataset)
        return f"{fs.display_id} {fs.setup.label} · {variant}"

    def summary(self) -> dict[str, Any]:
        """The model view: aggregates only, from the saved tables."""
        folder = self.folder
        renamed = {
            c: column_label(i) for c, i in zip(self.columns, self.infos, strict=True)
        }
        blocks = {i.prepared: player_label(i.player) for i in self.infos}

        def named(frame: pd.DataFrame | None, n: int) -> list[dict[str, Any]]:
            if frame is None:
                return []
            out = frame.sort_values("rank").head(n).copy()
            out["label"] = out["name"].map(renamed).fillna(out["name"])
            out["block"] = out["name"].map(blocks).fillna("")
            return _records(out)

        groups = _csv(folder, "groups.csv")
        if groups is not None:
            groups = groups.assign(label=groups["name"].map(player_label))
        strata = {
            name: named(_csv(folder, table), N_STRATUM_TOP)
            for name, table in STRATA_TABLES.items()
        }
        interactions = _csv(folder, "interactions.csv")
        if interactions is not None:
            interactions = interactions.sort_values(
                "mean_abs_interaction", ascending=False
            ).head(6)
            for side in ("a", "b"):
                interactions[side] = (
                    interactions[side].map(renamed).fillna(interactions[side])
                )
        top_nodes = _csv(folder, "neighbour_top_nodes.csv")
        if top_nodes is not None:
            top_nodes = top_nodes.assign(
                relation_label=top_nodes["relation"].map(relation_label)
            )
        relations = _csv(folder, "neighbour_relations.csv")
        if relations is not None:
            relations = relations.assign(
                relation_label=relations["relation"].map(relation_label)
            )
        shap_meta = self.meta["sections"]["shap"]
        skipped = self.meta["sections"].get("neighbours", {}).get("skipped")
        return {
            "key": self.key,
            "dataset": self.dataset,
            "variant": self.variant,
            "label": self.label(),
            "feature_set": self.feature_set,
            "budget": self.budget,
            "budget_label": reg.BUDGET_LABELS[self.budget],
            "model": self.meta.get("model_cls") or self.meta.get("model"),
            "selection": self.meta.get("selection"),
            "threshold": self.threshold,
            "base_value": self.base,
            "base_prob": float(_sigmoid(np.array([self.base]))[0]),
            "n_test": len(self.rows),
            "n_usable": int(self.usable.sum()),
            "n_fraud": int(self.rows[C.LABEL].sum()),
            "n_columns": len(self.columns),
            "shap_mode": shap_meta.get("feature_perturbation"),
            "model_output": shap_meta.get("model_output"),
            "background": shap_meta.get("background"),
            "n_background": shap_meta.get("n_background"),
            "importance": named(_csv(folder, "importance.csv"), N_TOP),
            "strata": strata,
            "stratum_labels": STRATUM_LABELS,
            "groups": _records(groups),
            "correlated": _records(_csv(folder, "correlated.csv")),
            "roar": _records(_csv(folder, "roar_top_k_vs_random.csv")),
            "randomisation": _records(_csv(folder, "randomisation.csv")),
            "seed_stability": _records(_csv(folder, "seed_stability.csv")),
            "sensitivity": _records(_csv(folder, "sensitivity.csv")),
            "rashomon": _records(_csv(folder, "rashomon.csv")),
            "interactions": _records(interactions),
            "neighbour_relations": _records(relations),
            "neighbour_structure": _records(_csv(folder, "neighbour_structure.csv")),
            "neighbour_top_nodes": _records(top_nodes),
            "neighbour_redundancy": _records(
                _csv(folder, "neighbour_redundancy_summary.csv")
            ),
            "has_influence": self.influence is not None,
            "influence_note": skipped if isinstance(skipped, str) else "",
            "has_values": self.x is not None,
            "notes": self.notes,
            "checks": [c.as_dict() for c in self.checks],
            "draws": list(DRAWS),
        }

    # ---- One node -------------------------------------------------------
    def draw(self, draw: str, seed: int | None) -> dict[str, Any]:
        if draw not in DRAWS:
            raise ExplainRequestError(f"Unknown draw {draw!r}; use one of {DRAWS}")
        if seed is None:
            seed = int(np.random.default_rng().integers(MAX_SEED))
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= MAX_SEED
        ):
            raise ExplainRequestError(f"The seed must be an integer from 0 to {MAX_SEED}")
        masks = X.strata_masks(
            self.rows[C.LABEL].to_numpy(), self.rows["prob"].to_numpy(), self.threshold
        )
        mask = {"any": masks["all"], "benign": masks["not_fraud"]}.get(
            draw, masks.get(draw)
        )
        where = np.flatnonzero(mask & self.usable)
        if len(where) == 0:
            raise ExplainRequestError(f"No test node is a {draw.replace('_', ' ')} here")
        rng = np.random.default_rng([seed, DRAWS.index(draw)])
        row = int(rng.choice(where))
        card = self.card(row)
        card.update({"draw": draw, "seed": seed, "pool": len(where)})
        return card

    def card(self, row: int) -> dict[str, Any]:
        r = self.rows.iloc[row]
        node, label = int(r[C.NODE_ID]), int(r[C.LABEL])
        prob, margin = float(r["prob"]), float(r["margin"])
        values = self.shap[row]
        gap = abs(self.base + float(values.sum()) - margin)
        if not self.usable[row] or not np.isfinite(gap) or gap > ADDITIVITY_TOLERANCE:
            raise ExplainRequestError(
                f"Node {node}: base value plus SHAP misses the margin by {gap:.2g}; "
                "not shown"
            )
        flagged = prob >= self.threshold
        order = np.argsort(-np.abs(values), kind="stable")
        top = order[:N_WATERFALL]
        contributions = [
            {
                "column": self.columns[j],
                "label": column_label(self.infos[j]),
                "block": player_label(self.infos[j].player),
                "value": None if self.x is None else float(self.x[row, j]),
                "shap": float(values[j]),
                "overall_rank": self.overall_rank.get(self.columns[j]),
            }
            for j in top
        ]
        players = X.players_of(self.infos)
        summed, names = X.summed_by_group(values[None, :], players)
        exact = None
        if self.group_shapley is not None and node in self.group_shapley.index:
            exact_row = self.group_shapley.loc[node]
            exact = {
                n: float(exact_row[GROUP_PREFIX + n])
                for n in names
                if GROUP_PREFIX + n in exact_row
            }
        blocks = [
            {
                "name": n,
                "label": player_label(n),
                "summed_shap": float(summed[0, i]),
                "group_shapley": None if exact is None else exact.get(n),
            }
            for i, n in enumerate(names)
        ]
        distance = np.linalg.norm(self.train_shap - values, axis=1)
        nearest = self.train_rows.iloc[np.argsort(distance, kind="stable")[:N_NEAREST]]
        return {
            "model": self.key,
            "node_id": node,
            "label": label,
            "prob": prob,
            "margin": margin,
            "base_value": self.base,
            "base_prob": float(_sigmoid(np.array([self.base]))[0]),
            "threshold": self.threshold,
            "flagged": bool(flagged),
            "outcome": _outcome(bool(flagged), label),
            "additivity_gap": gap,
            "contributions": contributions,
            "rest": float(values[order[N_WATERFALL:]].sum()),
            "n_rest": int(max(0, len(values) - N_WATERFALL)),
            "blocks": blocks,
            "in_group_shapley_sample": exact is not None,
            "neighbours": self._neighbours(node),
            "nearest_training": [
                {
                    "node_id": int(n[C.NODE_ID]),
                    "label": int(n[C.LABEL]),
                    "prob": float(n["prob"]),
                    "distance": float(d),
                }
                for (_, n), d in zip(
                    nearest.iterrows(), np.sort(distance)[: len(nearest)], strict=True
                )
            ],
        }

    def _neighbours(self, node: int) -> dict[str, Any]:
        if self.influence is None:
            return {"available": False, "relations": []}
        mine = self.influence.of(node)
        out = []
        for relation in self.relations:
            part = mine[mine["relation"] == relation]
            if part.empty:
                out.append(
                    {"relation": relation, "label": relation_label(relation),
                     "degree": 0, "n_computed": 0, "top": [], "total_abs": 0.0}
                )  # fmt: skip
                continue
            part = part.assign(size=part["delta_prob"].abs()).sort_values(
                "size", ascending=False, kind="stable"
            )
            top = []
            for _, n in part.head(N_NEIGHBOURS).iterrows():
                neighbour = int(n["neighbour"])
                score = None
                if self.scores is not None and neighbour in self.scores.index:
                    score = float(self.scores.loc[neighbour])
                top.append(
                    {
                        "neighbour": neighbour,
                        "delta_prob": float(n["delta_prob"]),
                        "mean_part": float(n["delta_prob_mean"]),
                        "degree_part": float(n["delta_prob"] - n["delta_prob_mean"]),
                        "out_of_range": bool(n["out_of_range"]),
                        "neighbour_score": score,
                    }
                )
            out.append(
                {
                    "relation": relation,
                    "label": relation_label(relation),
                    "degree": int(part["degree"].iloc[0]),
                    "n_computed": len(part),
                    "total_abs": float(part["size"].sum()),
                    "top": top,
                }
            )
        return {"available": True, "relations": out}


def _outcome(flagged: bool, label: int) -> str:
    if flagged:
        return "true_positive" if label == 1 else "false_positive"
    return "false_negative" if label == 1 else "true_negative"


def _check(checks: list[Check], name: str, ok: bool, passed: str, failed: str) -> None:
    checks.append(Check(name, ok, passed if ok else failed))


def find_folder(config: dict[str, Any], dataset: str, candidates: tuple[str, ...]) -> str:
    """The first feature set whose explanation is finished enough for the page."""
    tried = []
    for fs in candidates:
        if fs not in config.get("feature_sets", {}):
            continue
        folder = _split_paths(config, dataset, fs).feature_set_dir / X.OUTPUT_DIR
        needed = ("explain_meta.json", "shap_test.parquet", "shap_train_sample.parquet")
        if (
            all((folder / n).is_file() for n in needed)
            and (folder / "tables" / "importance.csv").is_file()
        ):
            return fs
        tried.append(str(folder))
    raise ExplainUnavailable(
        "No finished explanation in " + ", ".join(tried) + ". Run "
        "scripts/dfa/shap_explain.py and scripts/explain_report.py (see "
        "docs/EXPLAINABILITY_PLAN.md)."
    )


def load_model(
    config: dict[str, Any], dataset: str, variant: str, candidates: tuple[str, ...]
) -> ExplainModel:
    """One model's explanation, with its checks. Raises ExplainUnavailable if a
    file is missing or a check fails."""
    fs = find_folder(config, dataset, candidates)
    p = _split_paths(config, dataset, fs)
    folder = p.feature_set_dir / X.OUTPUT_DIR
    meta = json.loads((folder / "explain_meta.json").read_text(encoding="utf-8"))
    checks: list[Check] = []
    notes: list[str] = []
    _check(
        checks,
        "The explanation is of split 0 of this feature set",
        meta.get("dataset") == dataset
        and int(meta.get("repeat", -1)) == SPLIT
        and meta.get("feature_set") == fs,
        f"{dataset}, split {SPLIT}, {fs}",
        f"explain_meta.json says {meta.get('dataset')}, split {meta.get('repeat')}, "
        f"{meta.get('feature_set')}",
    )
    frame = pd.read_parquet(folder / "shap_test.parquet")
    columns = [c.removeprefix(SHAP_PREFIX) for c in frame if c.startswith(SHAP_PREFIX)]
    info_json = json.loads(
        (p.df_analyze_input_dir / "feature_set.json").read_text(encoding="utf-8")
    )
    renames = {safe: c["block_column"] for safe, c in info_json["columns"].items()}
    infos = X.parse_columns(columns, renames)
    shap = frame[[SHAP_PREFIX + c for c in columns]].to_numpy(dtype=float)
    usable = np.isfinite(shap).all(axis=1)
    rows = frame[[C.NODE_ID, C.GROUP_ID, C.LABEL, "prob", "margin"]].reset_index(
        drop=True
    )
    node_ids = rows[C.NODE_ID].to_numpy(dtype=int)

    test_ids = pd.read_csv(p.test_ids)
    train_ids = pd.read_csv(p.train_ids)[C.NODE_ID].to_numpy(dtype=int)
    by_node = test_ids.set_index(C.NODE_ID)[C.LABEL]
    same_nodes = len(np.unique(node_ids)) == len(node_ids) and set(node_ids) == set(
        by_node.index
    )
    _check(
        checks,
        "The SHAP rows are exactly the test nodes",
        same_nodes
        and np.array_equal(by_node.loc[node_ids].to_numpy(dtype=int), rows[C.LABEL]),
        f"{len(node_ids):,} distinct test nodes with test_ids.csv's labels",
        "the SHAP table's nodes or labels differ from test_ids.csv",
    )
    overlap = np.intersect1d(node_ids, train_ids)
    _check(
        checks,
        "No explained test node was trained on",
        len(overlap) == 0,
        f"none of the {len(train_ids):,} training nodes",
        f"{len(overlap)} nodes are in both train and test",
    )
    prob_gap = float(np.max(np.abs(_sigmoid(rows["margin"].to_numpy()) - rows["prob"])))
    _check(
        checks,
        "Saved margins give the saved probabilities",
        prob_gap <= PROB_TOLERANCE,
        f"largest difference {prob_gap:.1e}",
        f"largest difference {prob_gap:.2g}",
    )
    base = float(meta["sections"]["shap"]["base_value"])
    gaps = np.abs(base + shap[usable].sum(axis=1) - rows["margin"].to_numpy()[usable])
    worst = float(gaps.max()) if len(gaps) else float("nan")
    _check(
        checks,
        "Base value plus SHAP gives each margin (implementation check)",
        bool(usable.any()) and worst <= ADDITIVITY_TOLERANCE,
        f"{int(usable.sum()):,} of {len(usable):,} rows, "
        f"largest gap {worst:.1e} log-odds",
        f"largest gap {worst:.2g} log-odds over {int(usable.sum())} rows",
    )
    if not usable.all():
        notes.append(
            f"{int((~usable).sum())} test nodes failed the implementation check when "
            "SHAP was computed (near a tree split threshold); they are never drawn."
        )

    x = None
    try:
        export = df_analyze.find_export_dir(
            p.df_analyze_output_dir / str(meta["df_analyze_run"])
        )
        prepared = pd.read_csv(export / "X_test_00.csv", float_precision="round_trip")
        y_export = pd.read_csv(export / "y_test_00.csv", index_col=0).iloc[:, 0]
        if list(prepared.columns) == columns and np.array_equal(
            y_export.to_numpy(dtype=int), rows[C.LABEL].to_numpy(dtype=int)
        ):
            x = prepared.to_numpy(dtype=float)
        else:
            notes.append(
                "df-analyze's prepared test table does not line up with the SHAP rows, "
                "so input values are not shown."
            )
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        notes.append(f"Input values are not shown: {e}")
    if x is not None:
        checks.append(
            Check(
                "Input values line up with the SHAP rows",
                True,
                "df-analyze's prepared test table: same columns, same labels row by row",
            )
        )

    group_shapley = None
    gpath = folder / "group_shapley.parquet"
    if gpath.is_file():
        group_shapley = pd.read_parquet(gpath).set_index(C.NODE_ID)
    influence = None
    ipath = folder / "neighbour_influence.parquet"
    if ipath.is_file():
        influence = Influence.read(ipath)
        unknown = np.setdiff1d(np.unique(influence.targets), node_ids)
        _check(
            checks,
            "Neighbour influence is for test nodes only",
            len(unknown) == 0,
            f"{len(np.unique(influence.targets)):,} target nodes, all in the test set",
            f"{len(unknown)} target nodes are not test nodes",
        )
    scores = None
    spath = folder / "node_scores.parquet"
    if spath.is_file():
        scores = pd.read_parquet(spath).set_index(C.NODE_ID)["prob"]

    train = pd.read_parquet(folder / "shap_train_sample.parquet")
    train_shap = train[[SHAP_PREFIX + c for c in columns]].to_numpy(dtype=float)
    keep = np.isfinite(train_shap).all(axis=1)
    not_train = np.setdiff1d(train[C.NODE_ID].to_numpy(dtype=int), train_ids)
    _check(
        checks,
        "Nearest neighbours in SHAP space are training nodes",
        len(not_train) == 0,
        f"a sample of {int(keep.sum()):,} training nodes",
        f"{len(not_train)} sampled nodes are not training nodes",
    )
    importance = pd.read_csv(folder / "tables" / "importance.csv")
    model = ExplainModel(
        key=target_key(dataset, variant),
        dataset=dataset,
        variant=variant,
        feature_set=fs,
        folder=folder,
        meta=meta,
        columns=columns,
        infos=infos,
        rows=rows,
        shap=shap,
        usable=usable,
        base=base,
        threshold=float(meta["threshold"]),
        x=x,
        group_shapley=group_shapley,
        influence=influence,
        scores=scores,
        train_shap=train_shap[keep],
        train_rows=train.loc[keep, [C.NODE_ID, C.LABEL, "prob"]].reset_index(drop=True),
        relations=list(config["datasets"][dataset]["relations"]),
        overall_rank=dict(
            zip(importance["name"], importance["rank"].astype(int), strict=True)
        ),
        checks=checks,
        notes=notes,
    )
    bad = [c for c in checks if not c.ok]
    if bad:
        raise ExplainUnavailable("; ".join(f"{c.name}: {c.detail}" for c in bad))
    return model
