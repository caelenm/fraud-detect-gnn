"""Dataset audit: counts and statistics only (plan §7, stage `audit`).

Everything here describes the data; nothing is used as a feature. Label-based
statistics (univariate AUROC, homophily, group purity) use training nodes
only, so the test set stays untouched.

- Counts against the config (plan §2); any difference stops the stage.
- Per feature: distinct values, range, missing values, whether it is
  constant, and its univariate AUROC on the training nodes, oriented to be at
  least 0.5. A feature at or above `shortcut_auroc` is flagged as a possible
  label shortcut. Flags are reported, never acted on (plan §10).
- Exact duplicate feature rows within train, within test and across
  train/test. Across must be 0 after the grouped split.
- Group sizes and label purity within multi-node groups.
- Per relation: edges, mean degree of labelled nodes, and two homophily
  measures on training-training edges:
    edge homophily: share of edges whose two ends have the same label;
    positive homophily: share of a positive node's edges that lead to another
    positive node. This is CARE-GNN's "label similarity" (simi_comp.py,
    mode 'pos'), which CARE-GNN computes over all nodes, counting Amazon's
    unlabelled nodes as negatives; ours excludes them and uses train only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_detect import columns as C

# CARE-GNN's published label similarities (README at the pinned commit).
CARE_GNN_LABEL_SIMILARITY = {
    "yelpchi": {"rur": 0.909, "rtr": 0.176, "rsr": 0.186, "homo": 0.184},
    "amazon": {"upu": 0.167, "usu": 0.056, "uvu": 0.053, "homo": 0.072},
}


def oriented_auroc(y: np.ndarray, x: np.ndarray) -> float:
    """AUROC of one feature as a score, flipped to be >= 0.5 (NaN if constant)."""
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def feature_table(
    block: pd.DataFrame, nodes: pd.DataFrame, spec: dict[str, Any], threshold: float
) -> pd.DataFrame:
    """One row per feature, from the training nodes."""
    train = nodes[nodes[C.SPLIT] == C.TRAIN]
    X = block.set_index(C.NODE_ID).loc[train[C.NODE_ID].to_numpy()]
    y = train[C.LABEL].to_numpy(dtype=np.int64)
    rows = []
    for name in X.columns:
        x = X[name].to_numpy(dtype=float)
        finite = x[~np.isnan(x)]
        auc = oriented_auroc(y[~np.isnan(x)], finite)
        rows.append(
            {
                "feature": name,
                "type": spec["columns"][name]["type"],
                "n_unique": int(len(np.unique(finite))),
                "min": float(finite.min()) if finite.size else np.nan,
                "max": float(finite.max()) if finite.size else np.nan,
                "n_nan": int(np.isnan(x).sum()),
                "constant": bool(len(np.unique(finite)) <= 1),
                "auroc_train": auc,
                "possible_shortcut": bool(auc >= threshold),
            }
        )
    return pd.DataFrame(rows)


def duplicate_counts(block: pd.DataFrame, nodes: pd.DataFrame) -> dict[str, int]:
    """Exact duplicate feature rows: rows that repeat an earlier row within
    train and within test, and test rows identical to some training row."""
    X = block.set_index(C.NODE_ID)

    def rows(part: str) -> pd.DataFrame:
        ids = nodes.loc[nodes[C.SPLIT] == part, C.NODE_ID].to_numpy()
        return X.loc[ids]

    train, test = rows(C.TRAIN), rows(C.TEST)
    train_keys = set(map(tuple, train.to_numpy()))
    return {
        "within_train": int(train.duplicated().sum()),
        "within_test": int(test.duplicated().sum()),
        "test_rows_identical_to_a_train_row": int(
            sum(tuple(r) in train_keys for r in test.to_numpy())
        ),
    }


def group_table(nodes: pd.DataFrame) -> pd.DataFrame:
    """Group size distribution of labelled nodes per part."""
    labelled = nodes[nodes[C.IS_LABELLED]]
    sizes = labelled.groupby([C.SPLIT, C.GROUP_ID]).size().rename("size").reset_index()
    table = sizes.groupby([C.SPLIT, "size"]).size().rename("n_groups").reset_index()
    table["n_nodes"] = table["size"] * table["n_groups"]
    return table


def group_purity(nodes: pd.DataFrame) -> dict[str, Any]:
    """Share of training nodes in multi-node groups whose label equals their
    group's majority label (labels are per user on YelpChi)."""
    train = nodes[nodes[C.SPLIT] == C.TRAIN]
    sizes = train.groupby(C.GROUP_ID)[C.NODE_ID].transform("size")
    multi = train[sizes > 1]
    if multi.empty:
        return {"n_nodes_in_multi_node_groups": 0, "purity": None}
    y = multi[C.LABEL].astype(int)
    majority = y.groupby(multi[C.GROUP_ID]).transform(lambda s: int(s.mean() >= 0.5))
    return {
        "n_nodes_in_multi_node_groups": len(multi),
        "n_multi_node_groups": int(multi[C.GROUP_ID].nunique()),
        "purity": float((y == majority).mean()),
    }


def relation_row(name: str, edges: np.ndarray, nodes: pd.DataFrame) -> dict[str, Any]:
    n = len(nodes)
    degree = np.bincount(edges.ravel(), minlength=n)
    labelled = nodes[C.IS_LABELLED].to_numpy()
    is_train = (nodes[C.SPLIT] == C.TRAIN).to_numpy()
    label = nodes[C.LABEL].fillna(-1).to_numpy(dtype=np.int64)
    src, dst = edges
    tt = is_train[src] & is_train[dst]
    ls, ld = label[src[tt]], label[dst[tt]]
    both_pos = int(((ls == 1) & (ld == 1)).sum())
    one_pos = int(((ls == 1) ^ (ld == 1)).sum())
    pos_ends = 2 * both_pos + one_pos  # directed edges leaving a positive node
    return {
        "relation": name,
        "n_edges": int(edges.shape[1]),
        "mean_degree_labelled": float(degree[labelled].mean()),
        "share_labelled_isolated": float((degree[labelled] == 0).mean()),
        "n_train_train_edges": int(tt.sum()),
        "edge_homophily_train": float((ls == ld).mean()) if tt.any() else np.nan,
        "positive_homophily_train": 2 * both_pos / pos_ends if pos_ends else np.nan,
    }


def count_table(found: dict[str, Any], expected: dict[str, Any]) -> list[tuple]:
    rows = [(k, found[k], expected[k]) for k in expected if k != "n_edges"]
    rows += [
        (f"edges: {r}", found["n_edges"].get(r), v)
        for r, v in expected["n_edges"].items()
    ]
    return rows


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if isinstance(v, float | np.floating):
        return "–" if np.isnan(v) else f"{v:.3f}"
    if isinstance(v, int | np.integer):
        return f"{int(v):,}"
    return str(v)


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join("---" for _ in frame.columns) + "|",
    ]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |"
              for row in frame.itertuples(index=False)]  # fmt: skip
    return "\n".join(lines)


def report(
    dataset: str,
    counts: list[tuple],
    split_summary: dict[str, Any],
    features: pd.DataFrame,
    duplicates: dict[str, int],
    groups: pd.DataFrame,
    purity: dict[str, Any],
    relations: pd.DataFrame,
    threshold: float,
) -> str:
    flagged = features[features["possible_shortcut"]]
    published = CARE_GNN_LABEL_SIMILARITY.get(dataset, {})
    rel = relations.copy()
    rel["care_gnn_label_similarity"] = rel["relation"].map(published)
    tr, te = split_summary["train"], split_summary["test"]
    lines = [
        f"# Dataset audit: {dataset}",
        "",
        "Counts and statistics only. Label-based statistics use training nodes only.",
        "",
        "## Counts (checked against the config)",
        "",
        markdown_table(pd.DataFrame(counts, columns=["count", "found", "expected"])),
        "",
        "## Split",
        "",
        f"- Train: {tr['n_nodes']:,} nodes, {tr['n_positive']:,} positive "
        f"({tr['positive_rate']:.4f}), {tr['n_groups']:,} groups",
        f"- Test: {te['n_nodes']:,} nodes, {te['n_positive']:,} positive "
        f"({te['positive_rate']:.4f}), {te['n_groups']:,} groups",
        f"- Unlabelled (graph context only): {split_summary['unlabelled']:,} nodes",
        "- Groups crossing train/test: "
        f"{split_summary['checks']['groups_in_both_train_and_test']}; groups in more "
        f"than one CV fold: {split_summary['checks']['groups_in_more_than_one_cv_fold']}",
        "",
        "## Possible label shortcuts",
        "",
        f"Features whose univariate AUROC on the training nodes is at least "
        f"{threshold} (oriented to be at least 0.5). A flag is a prompt to look, "
        "not a verdict, and no feature is dropped automatically (plan §10).",
        "",
        markdown_table(flagged[["feature", "type", "n_unique", "auroc_train"]])
        if not flagged.empty
        else "None.",
        "",
        "## Exact duplicate feature rows",
        "",
        f"- Within train (rows repeating an earlier train row): "
        f"{duplicates['within_train']:,}",
        f"- Within test: {duplicates['within_test']:,}",
        f"- Test rows identical to a training row: "
        f"{duplicates['test_rows_identical_to_a_train_row']:,} (must be 0)",
        "",
        "## Groups",
        "",
        "Group sizes of labelled nodes (YelpChi: a user's reviews; Amazon: users "
        "with identical feature rows).",
        "",
        markdown_table(groups),
        "",
        f"Label purity in multi-node training groups: "
        f"{_fmt(purity['purity'])} of {purity['n_nodes_in_multi_node_groups']:,} nodes "
        "carry their group's majority label.",
        "",
        "## Relations",
        "",
        "Homophily on training-training edges only. **positive_homophily_train** "
        "is CARE-GNN's \"label similarity\" (the share of a positive node's edges "
        "that lead to another positive node); CARE-GNN's published value, computed "
        "over all nodes, is shown next to it. Reported as a dataset statistic, "
        "never used as a feature.",
        "",
        markdown_table(rel),
        "",
        "## Every feature",
        "",
        markdown_table(features),
        "",
    ]
    return "\n".join(lines)
