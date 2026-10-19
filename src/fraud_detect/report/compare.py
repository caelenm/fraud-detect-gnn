"""Comparison page: two feature sets' Model A side by side, test node by node.

Built by scripts/compare_report.py. It is not a pipeline stage, because it
spans two feature sets (rungs of the model ladder), e.g. pilot_m1_own (A) and
pilot_m5_own_gnnssl (B). It reads only what the pipeline already wrote for
each side: model_a.json (Model A and its decision threshold, chosen on
out-of-fold training predictions), model_a_metrics.csv, df-analyze's saved
test probabilities, the df-analyze input tables and test_ids.csv; and, for
the node cards and the embedding view, nodes.parquet, the relation edge
lists and gnn_embed/raw_<relation>.npy. Nothing is fit on test data or
chosen from it: the only fits here are 2-D PCAs for a picture, on training
nodes.

Metrics per side: test PR-AUC and AUROC with group-bootstrap intervals,
Recall@K (K = the number of test frauds) and F1/precision/recall at the
side's own threshold. The headline is the paired difference
ΔPR-AUC = B − A: both models are scored on the same bootstrap resamples of
whole test groups, which also gives the share of resamples where B is ahead.

The page is one self-contained HTML file: inline CSS, inline SVG and a little
inline JavaScript (click a point or a gallery entry to see that node's
card). It makes no external requests. Every inserted value is HTML-escaped.
It contains row-level test data (node IDs, labels, scores, neighbourhoods),
so it shows a group-only banner and must never be committed or uploaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fraud_detect import columns as C
from fraud_detect.config import REPO_ROOT, Paths, get_paths
from fraud_detect.features.graph import adjacency
from fraud_detect.models import df_analyze_report, evaluation
from fraud_detect.report import web
from fraud_detect.runlog import git_commit

N_RESAMPLES = 2000
N_GALLERY = 20
N_RANDOM_CARDS = 200
N_TOP_FEATURES = 6
MAX_NEIGHBOURS = 12


class CompareError(RuntimeError):
    """The two sides cannot be compared; the message says why."""


# --------------------------------------------------------------------------
# Loading one side
# --------------------------------------------------------------------------
@dataclass
class Side:
    feature_set: str
    blocks: list[str]
    run: str
    summary: dict[str, Any]  # model_a.json
    prob: np.ndarray  # Model A's test P(fraud), in test_ids.csv order
    threshold: float  # chosen on out-of-fold training predictions
    test_ids: pd.DataFrame
    train: pd.DataFrame  # df-analyze input features under block names
    test: pd.DataFrame
    used_columns: list[str]  # the block columns Model A was fit on
    classifiers: list[str]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _block_column(name: str, renames: dict[str, str]) -> str:
    """Our block column for a df-analyze column (a binary column becomes an
    indicator like own_f22_1.0 in df-analyze; it maps back to own__f22)."""
    if name in renames:
        return renames[name]
    for safe, ours in renames.items():
        if name.startswith(safe + "_"):
            return ours
    return name


def load_side(config: dict[str, Any], feature_set: str) -> Side:
    """One feature set's finished Model A (pipeline stages through
    df_analyze_report), with its test probabilities and input tables."""
    if feature_set not in config["feature_sets"]:
        raise CompareError(
            f"Unknown feature set {feature_set!r}; choose from "
            + ", ".join(config["feature_sets"])
        )
    p = get_paths({**config, "feature_set": feature_set})
    summary_path = p.model_reports_dir / "model_a.json"
    if not summary_path.is_file():
        raise CompareError(
            f"No Model A for feature set {feature_set} ({summary_path} is missing). "
            f"Run `uv run run.py --dataset {config['dataset']} --set "
            f"feature_set={feature_set}` first."
        )
    summary = _read_json(summary_path)
    latest = p.df_analyze_output_dir / "latest.txt"
    if (
        not latest.is_file()
        or latest.read_text(encoding="utf-8").strip() != summary["run"]
    ):
        raise CompareError(
            f"{summary_path} does not describe the latest df-analyze run of "
            f"{feature_set}; rerun its report stages."
        )
    metrics = pd.read_csv(
        p.model_reports_dir / "model_a_metrics.csv", keep_default_na=False, na_values=[""]
    )
    chosen = df_analyze_report.choose_model_a(metrics)
    a = summary["model_a"]
    if (chosen["model"], str(chosen["selection"])) != (a["model"], a["selection"]):
        raise CompareError(
            f"{feature_set}: model_a_metrics.csv and model_a.json disagree"
        )
    outdir = p.df_analyze_output_dir / summary["run"]
    try:
        predictions, _, _ = df_analyze_report.load_run(outdir)
        entry = web.find_entry(predictions, a, df_analyze_report.MODEL_NAMES)
    except (df_analyze_report.ReportError, web.WebReportError) as e:
        raise CompareError(f"{feature_set}: {e}") from e
    prob = np.asarray(entry["probs_test"], dtype=float)[:, 1]
    test_ids = pd.read_csv(p.test_ids)
    info = _read_json(p.df_analyze_input_dir / "feature_set.json")
    renames = {safe: c["block_column"] for safe, c in info["columns"].items()}
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    test = pd.read_parquet(p.df_analyze_input_dir / "test.parquet")
    if len(prob) != len(test_ids) or not np.array_equal(
        test[C.TARGET].to_numpy(), test_ids[C.LABEL].to_numpy()
    ):
        raise CompareError(
            f"{feature_set}: its test predictions or test table do not line up with "
            f"{p.test_ids}"
        )
    return Side(
        feature_set=feature_set,
        blocks=list(info["blocks"]),
        run=str(summary["run"]),
        summary=summary,
        prob=prob,
        threshold=float(a["threshold"]),
        test_ids=test_ids,
        train=train.drop(columns=C.TARGET).rename(columns=renames),
        test=test.drop(columns=C.TARGET).rename(columns=renames),
        used_columns=sorted({_block_column(c, renames) for c in entry["selected_cols"]}),
        classifiers=sorted(set(metrics["model"]) - {df_analyze_report.BASELINE_MODEL}),
    )


def check_same_test_nodes(a: Side, b: Side) -> None:
    """Both sides must be scored on exactly the same test nodes and labels."""
    same = len(a.test_ids) == len(b.test_ids) and all(
        np.array_equal(a.test_ids[col].to_numpy(), b.test_ids[col].to_numpy())
        for col in (C.NODE_ID, C.LABEL)
    )
    if not same:
        raise CompareError(
            f"{a.feature_set} and {b.feature_set} were not scored on the same test "
            "nodes; they cannot be compared."
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def recall_at_k(y: np.ndarray, prob: np.ndarray, node_ids: np.ndarray) -> float:
    """Share of the test frauds among the K highest scores, K = their number
    (ties broken by node ID)."""
    k = int(y.sum())
    top = np.lexsort((node_ids, -prob))[:k]
    return float(y[top].sum() / k)


def side_metrics(
    y: np.ndarray,
    groups: np.ndarray,
    node_ids: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    flagged = (prob >= threshold).astype(int)
    m = df_analyze_report.binary_metrics(y, prob, flagged)
    return {
        **{
            metric: evaluation.group_bootstrap(
                y, groups, prob, metric=metric, n_resamples=n_resamples, seed=seed
            ).as_dict()
            for metric in ("pr_auc", "auroc")
        },
        "recall_at_k": recall_at_k(y, prob, node_ids),
        "k": int(y.sum()),
        "threshold": float(threshold),
        **{k: m[k] for k in ("f1", "precision", "recall", "tp", "fp", "fn", "tn")},
        "n_flagged": int(flagged.sum()),
    }


def paired_difference(
    y: np.ndarray,
    groups: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """metric(B) − metric(A) on the same group-bootstrap resamples, with the
    share of resamples in which B is ahead."""
    return evaluation.group_bootstrap(
        y, groups, prob_b, prob_a, metric=metric, n_resamples=n_resamples, seed=seed
    ).as_dict()


# Decision outcomes: (A flags, B flags) -> label, for fraud and other nodes.
OUTCOMES = {
    1: {(True, True): "both catch", (False, True): "B rescues",
        (True, False): "B breaks", (False, False): "both miss"},
    0: {(True, True): "both false alarm", (False, True): "B adds a false alarm",
        (True, False): "B removes a false alarm", (False, False): "both correct"},
}  # fmt: skip


def node_table(a: Side, b: Side) -> pd.DataFrame:
    """One row per test node: label, both probabilities and decisions, outcome."""
    y = a.test_ids[C.LABEL].to_numpy(dtype=int)
    table = pd.DataFrame(
        {
            C.NODE_ID: a.test_ids[C.NODE_ID].to_numpy(),
            "y": y,
            "pa": a.prob,
            "pb": b.prob,
            "flag_a": a.prob >= a.threshold,
            "flag_b": b.prob >= b.threshold,
        }
    )
    table["outcome"] = [
        OUTCOMES[int(t.y)][(bool(t.flag_a), bool(t.flag_b))] for t in table.itertuples()
    ]
    table["gap"] = (table["pb"] - table["pa"]).abs()
    return table


def outcome_counts(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = table["outcome"].value_counts()
    return {
        "fraud" if y == 1 else "not_fraud": {
            name: int(counts.get(name, 0)) for name in OUTCOMES[y].values()
        }
        for y in (1, 0)
    }


def gallery(table: pd.DataFrame, outcome: str, n: int) -> pd.DataFrame:
    """The `n` nodes with this outcome whose two scores differ most."""
    rows = table[table["outcome"] == outcome]
    return rows.sort_values(["gap", C.NODE_ID], ascending=[False, True]).head(n)


# --------------------------------------------------------------------------
# Graph context for the node cards
# --------------------------------------------------------------------------
@dataclass
class GraphContext:
    adjacency: dict[str, sp.csr_matrix]
    split: np.ndarray  # per node: train / test / unlabelled
    label: np.ndarray  # per node: 1, 0, or -1 when unlabelled


def load_graph(p: Paths, relations: list[str]) -> GraphContext:
    nodes = pd.read_parquet(p.nodes).sort_values(C.NODE_ID)
    n = len(nodes)
    adj = {}
    for r in relations:
        with np.load(p.relation_file(r)) as f:
            adj[r] = adjacency(f["edges"], n)
    return GraphContext(
        adjacency=adj,
        split=nodes[C.SPLIT].astype("string").fillna("").to_numpy(dtype=str),
        label=nodes[C.LABEL].astype("Int64").fillna(-1).to_numpy(dtype=int),
    )


def ego_network(g: GraphContext, relation: str, node: int, seed: int) -> dict[str, Any]:
    """A node's neighbourhood in one relation: true degree, its training
    neighbours' fraud share, and at most MAX_NEIGHBOURS neighbours drawn with a
    fixed seed (training ones carry their label; others are 'other')."""
    a = g.adjacency[relation]
    neighbours = a.indices[a.indptr[node] : a.indptr[node + 1]]
    in_train = g.split[neighbours] == C.TRAIN
    rng = np.random.default_rng([seed, int(node), sorted(g.adjacency).index(relation)])
    k = min(MAX_NEIGHBOURS, len(neighbours))
    shown = np.sort(rng.choice(neighbours, size=k, replace=False)) if k else neighbours
    kinds = []
    for v in shown:
        if g.split[v] != C.TRAIN:
            kinds.append("other")
        else:
            kinds.append("fraud" if g.label[v] == 1 else "benign")
    sub = a[shown][:, shown].tocoo()
    links = [(int(i), int(j)) for i, j in zip(sub.row, sub.col, strict=True) if i < j]
    n_train = int(in_train.sum())
    return {
        "relation": relation,
        "degree": int(len(neighbours)),
        "n_train_neighbours": n_train,
        "train_fraud_share": (
            float((g.label[neighbours[in_train]] == 1).mean()) if n_train else None
        ),
        "shown": [(int(v), kind) for v, kind in zip(shown, kinds, strict=True)],
        "links": links,
    }


def relation_label(relation: str) -> str:
    """upu -> U-P-U."""
    return "-".join(relation.upper())


def feature_label(column: str) -> str:
    """own__f13 -> own f13; nbr__upu_mean_f13 -> nbr U-P-U mean f13;
    gnnssl__upu_pc03 -> gnnssl U-P-U pc03."""
    block, _, rest = column.partition(C.BLOCK_SEPARATOR)
    if block == C.OWN_BLOCK:
        return f"own {rest}"
    relation, _, what = rest.partition("_")
    return f"{block} {relation_label(relation)} {what.replace('_', ' ')}"


# --------------------------------------------------------------------------
# Embedding view
# --------------------------------------------------------------------------
def projection_2d(
    x: np.ndarray, fit_rows: np.ndarray, seed: int, scale: bool
) -> np.ndarray:
    """2-D PCA (optionally after standardisation), both fit on training rows."""
    if scale:
        x = StandardScaler().fit(x[fit_rows]).transform(x)
    return (
        PCA(n_components=2, svd_solver="full", random_state=seed)
        .fit(x[fit_rows])
        .transform(x)
    )


# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------
def _f(v: float) -> str:
    return f"{v:.1f}"


def _line(cls: str, x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line class="{cls}" x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}"/>'
    )


def _text(x: float, y: float, text: str, anchor: str = "middle", cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return (
        f'<text{attr} x="{_f(x)}" y="{_f(y)}" text-anchor="{anchor}">'
        f"{escape(text)}</text>"
    )


def _svg_open(cls: str, size: int, label: str) -> str:
    return (
        f'<svg class="{cls}" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="{escape(label)}">'
    )


def scatter_svg(table: pd.DataFrame, ta: float, tb: float) -> str:
    """P(fraud | A) on x, P(fraud | B) on y; colour = true label; dashed lines =
    each side's threshold; one clickable circle per test node."""
    size, pad = 520, 44
    span = size - 2 * pad
    lo, hi = pad, size - pad

    def x(v: float) -> float:
        return pad + v * span

    def y(v: float) -> float:
        return size - pad - v * span

    parts = [
        _svg_open("plot scatter", size, "P(fraud) under A against B, every test node"),
        f'<rect class="frame" x="{pad}" y="{pad}" width="{span}" height="{span}"/>',
    ]
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts += [
            _line("grid", x(t), lo, x(t), hi),
            _line("grid", lo, y(t), hi, y(t)),
            _text(x(t), hi + 16, f"{t:g}"),
            _text(pad - 8, y(t) + 4, f"{t:g}", anchor="end"),
        ]
    parts += [
        _text(size / 2, size - 6, "P(fraud | A)", cls="axis-title"),
        f'<g transform="rotate(-90 12 {size / 2})">'
        + _text(12, size / 2, "P(fraud | B)", cls="axis-title")
        + "</g>",
        _line("thr", x(ta), lo, x(ta), hi),
        _line("thr", lo, y(tb), hi, y(tb)),
        _text(lo + 6, lo + 16, "B rescues", "start", "quad"),
        _text(hi - 6, lo + 16, "both catch", "end", "quad"),
        _text(lo + 6, hi - 8, "both miss", "start", "quad"),
        _text(hi - 6, hi - 8, "B breaks", "end", "quad"),
    ]
    # Not-fraud first, fraud on top.
    for row in table.sort_values(["y", C.NODE_ID]).itertuples():
        node = int(getattr(row, C.NODE_ID))
        parts.append(
            f'<circle class="pt y{int(row.y)}" cx="{_f(x(row.pa))}" '
            f'cy="{_f(y(row.pb))}" r="3" data-node="{node}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def embedding_svg(xy: np.ndarray, y: np.ndarray, is_test: np.ndarray, title: str) -> str:
    """Labelled nodes in a 2-D projection: training nodes filled, test nodes
    hollow, colour = label. The view is clipped to the central 98% of points."""
    size, pad = 360, 12
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    pos = (np.clip(xy, lo, hi) - lo) / span
    px = pad + pos[:, 0] * (size - 2 * pad)
    py = size - pad - pos[:, 1] * (size - 2 * pad)
    parts = [
        _svg_open("plot emb", size, title),
        f'<rect class="frame" x="0" y="0" width="{size}" height="{size}"/>',
    ]
    for i in np.lexsort((is_test, y)):  # benign first, fraud on top
        kind = "te" if is_test[i] else "tr"
        parts.append(
            f'<circle class="{kind} y{int(y[i])}" cx="{_f(px[i])}" '
            f'cy="{_f(py[i])}" r="2"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


NEIGHBOUR_TIPS = {
    "fraud": "training, fraud",
    "benign": "training, not fraud",
    "other": "test or unlabelled",
}


def ego_svg(node: int, label: int, ego: dict[str, Any]) -> str:
    """The node in the centre, its sampled neighbours on a circle; star edges
    to the centre and dashed edges between neighbours that are linked."""
    size, c, radius = 160, 80, 58
    shown = ego["shown"]
    m = len(shown)
    angles = [2 * np.pi * i / max(m, 1) - np.pi / 2 for i in range(m)]
    pts = [(c + radius * np.cos(a), c + radius * np.sin(a)) for a in angles]
    parts = [_svg_open("ego", size, relation_label(ego["relation"]) + " neighbourhood")]
    parts += [_line("link", *pts[i], *pts[j]) for i, j in ego["links"]]
    parts += [_line("star", c, c, px, py) for px, py in pts]
    for (v, kind), (px, py) in zip(shown, pts, strict=True):
        parts.append(
            f'<circle class="n-{kind}" cx="{_f(px)}" cy="{_f(py)}" r="5">'
            f"<title>node {v} · {escape(NEIGHBOUR_TIPS[kind])}</title></circle>"
        )
    parts.append(
        f'<circle class="center y{label}" cx="{c}" cy="{c}" r="8">'
        f"<title>node {node} (this test node)</title></circle></svg>"
    )
    return "".join(parts)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
EXTRA_CSS = """
.banner { background: var(--bad-soft); color: var(--bad); font-weight: 600;
  border: 1px solid color-mix(in srgb, var(--bad) 35%, transparent);
  border-radius: var(--radius); padding: 10px 14px; margin-bottom: 20px; }
.compare-grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  gap: 16px; align-items: start; }
@media (max-width: 900px) { .compare-grid { grid-template-columns: 1fr; } }
svg.plot { width: 100%; height: auto; display: block; }
svg .frame { fill: none; stroke: var(--border); }
svg .grid { stroke: var(--border); stroke-opacity: .6; }
svg text { fill: var(--muted); font-size: 11px; font-family: inherit; }
svg .axis-title { font-size: 12px; fill: var(--text); }
svg .quad { fill: var(--faint); font-weight: 600; }
svg .thr { stroke: var(--warn); stroke-width: 1.5; stroke-dasharray: 5 4; }
.pt { cursor: pointer; }
.pt.y0 { fill: var(--accent); fill-opacity: .3; }
.pt.y1 { fill: var(--bad); fill-opacity: .85; }
.pt.picked { stroke: var(--text); stroke-width: 2.5; fill-opacity: 1; }
.emb .tr.y0 { fill: var(--accent); fill-opacity: .3; }
.emb .tr.y1 { fill: var(--bad); fill-opacity: .8; }
.emb .te { fill: none; stroke-width: .9; }
.emb .te.y0 { stroke: var(--accent); stroke-opacity: .35; }
.emb .te.y1 { stroke: var(--bad); }
.ego { width: 100%; max-width: 180px; height: auto; display: block; margin: 0 auto; }
.ego .star { stroke: var(--border); stroke-width: 1.2; }
.ego .link { stroke: var(--faint); stroke-opacity: .5; stroke-dasharray: 2 2; }
.ego .n-fraud { fill: var(--bad); }
.ego .n-benign { fill: var(--accent); }
.ego .n-other { fill: var(--faint); fill-opacity: .6; }
.ego .center { stroke: var(--text); stroke-width: 2; }
.ego .center.y1 { fill: var(--bad-soft); }
.ego .center.y0 { fill: var(--accent-soft); }
.egos { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin-top: 8px; }
.ego-cap { font-size: 12px; color: var(--muted); text-align: center; margin-top: 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px;
  color: var(--muted); margin-top: 8px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 5px; vertical-align: -1px; }
.dot.fraud { background: var(--bad); } .dot.benign { background: var(--accent); }
.dot.other { background: var(--faint); }
.dot.hollow { background: transparent; border: 1.5px solid var(--bad); }
.dash { display: inline-block; width: 18px; border-top: 2px dashed var(--warn);
  margin-right: 5px; vertical-align: 3px; }
#card-panel { position: sticky; top: 12px; min-height: 200px; }
.card h3 { margin: 0; font-size: 16px; display: flex; gap: 8px; align-items: center;
  flex-wrap: wrap; }
.card h4 { margin: 16px 0 4px; font-size: 13px; color: var(--muted); font-weight: 600; }
.probs { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.probs > div { background: var(--surface-2); border-radius: 8px; padding: 8px 10px;
  font-size: 12.5px; }
.probs b { display: block; font-size: 18px; font-variant-numeric: tabular-nums;
  color: var(--text); }
.muted { color: var(--muted); }
.gallery-list { max-height: 520px; overflow-y: auto; }
.gallery-list button { all: unset; box-sizing: border-box; display: flex; gap: 10px;
  align-items: center; width: 100%; padding: 8px 16px; cursor: pointer;
  border-bottom: 1px solid var(--border); font-size: 13px;
  font-variant-numeric: tabular-nums; }
.gallery-list button:hover, .gallery-list button:focus-visible {
  background: var(--surface-2); }
.gallery-list .id { font-family: var(--mono); color: var(--muted); min-width: 6ch; }
.gallery-list .spacer { flex: 1; }
td.delta { font-weight: 600; }
"""

SCRIPT = """
(function () {
  var data = JSON.parse(document.getElementById("node-data").textContent);
  var index = {};
  data.ids.forEach(function (id, i) { index[id] = i; });
  var panel = document.getElementById("card-panel");
  var picked = null;
  function line(text, cls) {
    var el = document.createElement("p");
    if (cls) { el.className = cls; }
    el.textContent = text;
    return el;
  }
  function show(id) {
    if (!/^[0-9]+$/.test(id)) { return; }
    var tpl = document.getElementById("card-" + id);
    panel.replaceChildren();
    if (tpl) {
      panel.appendChild(tpl.content.cloneNode(true));
    } else if (index[id] !== undefined) {
      var i = index[id];
      var h = document.createElement("h3");
      h.textContent = "Node " + id + " · " + (data.y[i] ? "fraud" : "not fraud");
      panel.appendChild(h);
      panel.appendChild(line("P(fraud | A) " + data.pa[i].toFixed(3) +
        (data.pa[i] >= data.ta ? " (flagged)" : " (not flagged)") +
        " · P(fraud | B) " + data.pb[i].toFixed(3) +
        (data.pb[i] >= data.tb ? " (flagged)" : " (not flagged)")));
      panel.appendChild(line("Features and neighbourhoods are precomputed only for " +
        "the gallery nodes and a random sample of test nodes.", "muted small"));
    }
    if (picked) { picked.classList.remove("picked"); }
    picked = document.querySelector('circle.pt[data-node="' + id + '"]');
    if (picked) { picked.classList.add("picked"); picked.parentNode.appendChild(picked); }
  }
  document.addEventListener("click", function (e) {
    var el = e.target.closest ? e.target.closest("[data-node]") : null;
    if (!el) { return; }
    show(el.getAttribute("data-node"));
    if (el.tagName.toLowerCase() === "button") {
      panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });
  var first = document.querySelector(".gallery-list [data-node]");
  if (first) { show(first.getAttribute("data-node")); }
})();
"""


def _num(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{digits}f}"


def _ci(i: dict[str, Any]) -> str:
    return f"{_num(i['estimate'])} [{_num(i['low'])}, {_num(i['high'])}]"


def _signed(v: float) -> str:
    return f"{v:+.3f}"


def _json_for_script(data: Any) -> str:
    """JSON that cannot end the <script> element it sits in."""
    text = json.dumps(data, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _side_line(tag: str, side: Side) -> str:
    return f"{tag} · {side.feature_set}"


def _blocks_used(columns: list[str]) -> str:
    counts = pd.Series(
        [c.partition(C.BLOCK_SEPARATOR)[0] for c in columns]
    ).value_counts()
    return ", ".join(f"{n} {block}" for block, n in counts.items()) or "none"


def _signed_ci(i: dict[str, Any]) -> str:
    return f"{_signed(i['estimate'])} [{_signed(i['low'])}, {_signed(i['high'])}]"


def _metrics_table(m: dict[str, Any]) -> str:
    a, b = m["a"], m["b"]
    rows = [
        ("Test PR-AUC (95% CI)", _ci(a["pr_auc"]), _ci(b["pr_auc"]),
         _signed_ci(m["delta_pr_auc"])),
        ("Test AUROC (95% CI)", _ci(a["auroc"]), _ci(b["auroc"]),
         _signed_ci(m["delta_auroc"])),
        (f"Recall@K (K = {a['k']} test frauds)", _num(a["recall_at_k"]),
         _num(b["recall_at_k"]), _signed(b["recall_at_k"] - a["recall_at_k"])),
        ("Threshold (out-of-fold training predictions)", _num(a["threshold"], 4),
         _num(b["threshold"], 4), ""),
        ("F1 at its threshold", _num(a["f1"]), _num(b["f1"]), _signed(b["f1"] - a["f1"])),
        ("Precision at its threshold", _num(a["precision"]), _num(b["precision"]),
         _signed(b["precision"] - a["precision"])),
        ("Recall at its threshold", _num(a["recall"]), _num(b["recall"]),
         _signed(b["recall"] - a["recall"])),
        ("Test nodes flagged", f"{a['n_flagged']:,}", f"{b['n_flagged']:,}", ""),
    ]  # fmt: skip
    body = "".join(
        f"<tr><td>{escape(r[0])}</td><td>{escape(r[1])}</td><td>{escape(r[2])}</td>"
        f'<td class="delta">{escape(r[3])}</td></tr>'
        for r in rows
    )
    return (
        '<div class="panel table-scroll"><table><thead><tr><th class="l">Metric</th>'
        f"<th>A · {escape(m['a']['feature_set'])}</th>"
        f"<th>B · {escape(m['b']['feature_set'])}</th><th>B − A</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _settings_table(m: dict[str, Any]) -> str:
    rows = []
    for key in ("a", "b"):
        s = m[key]
        rows.append(
            "<tr>"
            f"<td>{key.upper()}</td><td class='l'>{escape(s['feature_set'])}</td>"
            f"<td class='l'>{escape(', '.join(s['blocks']))}</td>"
            f"<td class='l'>{escape(web.model_label(s['model']))} · "
            f"{escape(web.feature_set_label(s['selection_key']))}</td>"
            f"<td class='l'>{escape(_blocks_used(s['used_columns']))}</td>"
            f"<td>{escape(str(s['htune_trials']))}</td>"
            f"<td class='l'>{escape(', '.join(s['classifiers']))}</td>"
            f"<td class='l mono'>{escape(s['run'])}</td></tr>"
        )
    return (
        '<div class="panel table-scroll"><table><thead><tr><th>Side</th>'
        '<th class="l">Feature set</th><th class="l">Blocks</th>'
        '<th class="l">Model A</th><th class="l">Columns Model A used</th>'
        '<th>Trials</th><th class="l">Classifiers (+ dummy)</th>'
        f'<th class="l">df-analyze run</th></tr></thead><tbody>{"".join(rows)}'
        "</tbody></table></div>"
    )


def _outcomes_table(m: dict[str, Any]) -> str:
    out = m["outcomes"]
    rows = "".join(
        f"<tr><td class='l'>{escape(name)}</td><td>{n:,}</td></tr>"
        for group in ("fraud", "not_fraud")
        for name, n in out[group].items()
    )
    return (
        '<div class="panel table-scroll"><table><thead><tr><th class="l">Decision '
        "outcome (each side at its own threshold)</th><th>Test nodes</th></tr>"
        f"</thead><tbody>{rows}</tbody></table></div>"
    )


def _gallery(title: str, blurb: str, rows: pd.DataFrame) -> str:
    items = (
        "".join(
            f'<button type="button" data-node="{int(getattr(r, C.NODE_ID))}">'
            f'<span class="id">{int(getattr(r, C.NODE_ID))}</span>'
            f"<span>A {_num(r.pa)} → B {_num(r.pb)}</span><span class='spacer'></span>"
            f'<span class="badge">|Δ| {_num(r.gap)}</span></button>'
            for r in rows.itertuples()
        )
        or '<p class="muted panel-pad">None.</p>'
    )
    return (
        '<div class="panel conf-card"><div class="conf-head">'
        f'<h3>{escape(title)} <span class="badge">{len(rows)}</span></h3>'
        f"<p>{escape(blurb)}</p></div>"
        f'<div class="gallery-list">{items}</div></div>'
    )


def _card(
    row: Any,
    a: Side,
    b: Side,
    chips: list[tuple],
    egos: list[dict[str, Any]],
) -> str:
    node, label = int(getattr(row, C.NODE_ID)), int(row.y)
    outcome_kind = (
        "good"
        if row.outcome in ("B rescues", "B removes a false alarm")
        else ("bad" if row.outcome in ("B breaks", "B adds a false alarm") else "")
    )
    chip_html = "".join(
        f"<span>{escape(feature_label(name))} <b>{escape(web._param(float(value)))}</b> "
        f'<span class="pct">p{100 * float(pct):.0f}</span></span>'
        for name, value, pct in chips
    )
    ego_html = []
    for ego in egos:
        share = ego["train_fraud_share"]
        share_text = (
            "no training neighbours"
            if share is None
            else (f"{ego['n_train_neighbours']:,} training neighbours, {share:.0%} fraud")
        )
        ego_html.append(
            f"<div>{ego_svg(node, label, ego)}<div class='ego-cap'>"
            f"<b>{escape(relation_label(ego['relation']))}</b> · degree "
            f"{ego['degree']:,} · {len(ego['shown'])} shown<br>{escape(share_text)}"
            "</div></div>"
        )
    return (
        f'<template id="card-{node}"><div class="card">'
        f"<h3>Node {node} <span class='badge {'bad' if label else ''}'>"
        f"{'fraud' if label else 'not fraud'}</span>"
        f"<span class='badge {outcome_kind}'>{escape(row.outcome)}</span></h3>"
        '<div class="probs">'
        f"<div><span class='muted'>{escape(_side_line('A', a))}</span>"
        f"<b>{escape(web.prob_text(row.pa))}</b>"
        f"{'flagged' if row.flag_a else 'not flagged'} (threshold {_num(a.threshold)})"
        "</div>"
        f"<div><span class='muted'>{escape(_side_line('B', b))}</span>"
        f"<b>{escape(web.prob_text(row.pb))}</b>"
        f"{'flagged' if row.flag_b else 'not flagged'} (threshold {_num(b.threshold)})"
        "</div></div>"
        "<h4>Highest-ranked features (value, training percentile)</h4>"
        f'<div class="facts features">{chip_html}</div>'
        "<h4>Neighbourhood per relation (at most 12 neighbours, seeded sample)</h4>"
        f'<div class="egos">{"".join(ego_html)}</div>'
        '<div class="legend"><span><span class="dot fraud"></span>training, fraud</span>'
        '<span><span class="dot benign"></span>training, not fraud</span>'
        '<span><span class="dot other"></span>test or unlabelled (label hidden)</span>'
        "</div></div></template>"
    )


@dataclass
class Comparison:
    html: str
    metrics: dict[str, Any]


def build(
    config: dict[str, Any],
    a_name: str,
    b_name: str,
    n_resamples: int = N_RESAMPLES,
    n_random_cards: int = N_RANDOM_CARDS,
    n_gallery: int = N_GALLERY,
) -> Comparison:
    """Everything for one comparison page (see the module docstring)."""
    seed = int(config["seed"])
    a, b = load_side(config, a_name), load_side(config, b_name)
    check_same_test_nodes(a, b)
    y = a.test_ids[C.LABEL].to_numpy(dtype=int)
    groups = a.test_ids[C.GROUP_ID].to_numpy()
    ids = a.test_ids[C.NODE_ID].to_numpy()

    def describe(side: Side) -> dict[str, Any]:
        model_a = side.summary["model_a"]
        return {
            "feature_set": side.feature_set,
            "blocks": side.blocks,
            "run": side.run,
            "model": model_a["model"],
            "selection": model_a["selection"],
            "selection_key": web.feature_set_key(
                model_a["selection"], model_a.get("embed_selector", "")
            ),
            "htune_trials": side.summary.get("htune_trials"),
            "classifiers": side.classifiers,
            "used_columns": side.used_columns,
            **side_metrics(y, groups, ids, side.prob, side.threshold, n_resamples, seed),
        }

    table = node_table(a, b)
    metrics = {
        "dataset": config["dataset"],
        "n_test": len(y),
        "n_test_fraud": int(y.sum()),
        "test_positive_rate": float(y.mean()),
        "a": describe(a),
        "b": describe(b),
        "delta_pr_auc": paired_difference(y, groups, a.prob, b.prob, "pr_auc",
                                          n_resamples, seed),
        "delta_auroc": paired_difference(y, groups, a.prob, b.prob, "auroc",
                                         n_resamples, seed),
        "outcomes": outcome_counts(table),
        "bootstrap": {"resamples": n_resamples, "seed": seed,
                      "unit": "whole test split groups"},
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(REPO_ROOT),
    }  # fmt: skip

    rescues = gallery(table, "B rescues", n_gallery)
    breaks = gallery(table, "B breaks", n_gallery)
    picked = set(rescues[C.NODE_ID]) | set(breaks[C.NODE_ID])
    rest = np.setdiff1d(ids, np.fromiter(picked, dtype=ids.dtype, count=len(picked)))
    rng = np.random.default_rng(seed)
    random_ids = rng.choice(rest, size=min(n_random_cards, len(rest)), replace=False)
    card_ids = sorted(picked | {int(v) for v in random_ids})

    p = get_paths({**config, "feature_set": a_name})
    relations = list(config["datasets"][config["dataset"]]["relations"])
    graph = load_graph(p, relations)
    # Features of both sides (own once), ranked by training percentile.
    union_train = a.train.join(b.train.drop(columns=a.train.columns, errors="ignore"))
    union_test = a.test.join(b.test.drop(columns=a.test.columns, errors="ignore"))
    position = {int(v): i for i, v in enumerate(ids)}
    chips = web.top_features(
        union_test.iloc[[position[v] for v in card_ids]], union_train, N_TOP_FEATURES
    )
    card_rows = table.set_index(C.NODE_ID, drop=False).loc[card_ids]
    cards = "".join(
        _card(row, a, b, node_chips,
              [ego_network(graph, r, int(getattr(row, C.NODE_ID)), seed)
               for r in relations])
        for row, node_chips in zip(card_rows.itertuples(index=False), chips, strict=True)
    )  # fmt: skip

    embedding = _embedding_section(a, b, p, relations, seed)
    html = _render(metrics, a, b, table, rescues, breaks, cards, embedding)
    return Comparison(html=html, metrics=metrics)


def _embedding_section(
    a: Side, b: Side, p: Paths, relations: list[str], seed: int
) -> str:
    uses = [s.feature_set for s in (a, b) if C.GNNSSL_BLOCK in s.blocks]
    raw_path = p.raw_embedding(relations[0])
    intro = '<section><h2>Embedding view</h2><p class="lede">A picture, not evidence. '
    if not uses or not raw_path.is_file():
        return (
            intro + "Neither feature set uses the self-supervised GNN embeddings "
            "(gnnssl), so there is nothing to show here.</p></section>"
        )
    nodes = pd.read_parquet(p.nodes).sort_values(C.NODE_ID)
    split = nodes[C.SPLIT].astype("string").fillna("").to_numpy(dtype=str)
    labelled = np.isin(split, [C.TRAIN, C.TEST])
    y = nodes[C.LABEL].astype("Int64").fillna(0).to_numpy(dtype=int)[labelled]
    is_test = split[labelled] == C.TEST
    fit = split == C.TRAIN
    raw = np.load(raw_path)
    own_block = pd.read_parquet(p.block_file(C.OWN_BLOCK)).sort_values(C.NODE_ID)
    own_x = own_block.drop(columns=C.NODE_ID).to_numpy(dtype=float)
    rel = relation_label(relations[0])
    emb = projection_2d(raw, fit, seed, scale=False)[labelled]
    own_xy = projection_2d(own_x, fit, seed, scale=True)[labelled]
    return (
        intro + f"Every labelled node in two 2-D projections, each a PCA fit on the "
        f"training nodes only: left, the {escape(rel)} DGI embedding "
        f"(gnn_embed/{escape(raw_path.name)}, before the PCA that compresses it for "
        "df-analyze); right, the own features (standardised on training nodes). "
        "Filled: training nodes; hollow: test nodes; red: fraud; blue: not fraud. "
        "Unlabelled nodes are left out; points beyond the central 98% are drawn on "
        f"the edge. Used by: {escape(', '.join(uses))}.</p>"
        '<div class="grid-2">'
        f'<div class="panel panel-pad"><h3 class="sub">{escape(rel)} DGI embedding '
        f"(2-D PCA)</h3>{embedding_svg(emb, y, is_test, rel + ' embedding')}</div>"
        '<div class="panel panel-pad"><h3 class="sub">Own features (2-D PCA)</h3>'
        f"{embedding_svg(own_xy, y, is_test, 'own features')}</div>"
        "</div></section>"
    )


def _render(
    m: dict[str, Any],
    a: Side,
    b: Side,
    table: pd.DataFrame,
    rescues: pd.DataFrame,
    breaks: pd.DataFrame,
    cards: str,
    embedding: str,
) -> str:
    css = web.CSS_PATH.read_text(encoding="utf-8") + EXTRA_CSS
    d = m["delta_pr_auc"]
    node_data = {
        "ids": [int(v) for v in table[C.NODE_ID]],
        "y": [int(v) for v in table["y"]],
        "pa": [round(float(v), 6) for v in table["pa"]],
        "pb": [round(float(v), 6) for v in table["pb"]],
        "ta": a.threshold,
        "tb": b.threshold,
    }
    tiles = "".join(
        [
            web._tile("A: " + a.feature_set, _num(m["a"]["pr_auc"]["estimate"]),
                      f"test PR-AUC · 95% CI {_num(m['a']['pr_auc']['low'])}–"
                      f"{_num(m['a']['pr_auc']['high'])}"),
            web._tile("B: " + b.feature_set, _num(m["b"]["pr_auc"]["estimate"]),
                      f"test PR-AUC · 95% CI {_num(m['b']['pr_auc']['low'])}–"
                      f"{_num(m['b']['pr_auc']['high'])}"),
            web._tile("ΔPR-AUC (B − A)", _signed(d["estimate"]),
                      f"paired 95% CI {_signed(d['low'])} to {_signed(d['high'])} · B "
                      f"ahead in {d['share_above_zero']:.0%} of resamples",
                      hero=True, badge="headline"),
            web._tile("Test set", f"{m['n_test']:,}",
                      f"nodes · {m['n_test_fraud']:,} fraud · no-skill PR-AUC "
                      f"{_num(m['test_positive_rate'])}"),
        ]
    )  # fmt: skip
    commit = m.get("git_commit") or "unknown"
    meta = "".join(
        f"<span>{escape(k)}</span><b>{escape(v)}</b>"
        for k, v in (
            ("Dataset", m["dataset"]),
            ("Generated", m["generated_utc"]),
            ("Git commit", commit[:10]),
            ("Bootstrap", f"{m['bootstrap']['resamples']:,} group resamples, "
                          f"seed {m['bootstrap']['seed']}"),
        )
    )  # fmt: skip
    trials = {str(m[k]["htune_trials"]) for k in ("a", "b")}
    budget_note = (
        "Both sides used the same tuning budget"
        if len(trials) == 1
        else "<b>The two sides used different tuning budgets</b>"
    )
    body = (
        '<header class="top"><div><div class="eyebrow">CS555 · Graph fraud detection '
        f"· {escape(m['dataset'])}</div><h1>Model comparison: "
        f"{escape(a.feature_set)} vs {escape(b.feature_set)}</h1></div>"
        f'<div class="meta">{meta}</div></header>'
        '<div class="banner">Contains row-level test data. Group only: do not upload '
        "or commit.</div>"
        "<section><h2>Headline</h2>"
        '<p class="lede">Each side is its feature set\'s Model A, chosen by grouped '
        "shared-CV PR-AUC on the training set; its threshold was chosen on its own "
        "out-of-fold training predictions. Both are scored on the same frozen test "
        "nodes. ΔPR-AUC resamples whole test groups for both models at once; it "
        "covers test-set sampling only, not other splits or seeds. "
        f"{budget_note} (trials per model: {escape(', '.join(sorted(trials)))}).</p>"
        f'<div class="tiles">{tiles}</div></section>'
        f"<section><h2>Metrics</h2>{_metrics_table(m)}</section>"
        f"<section><h2>Run settings</h2>{_settings_table(m)}</section>"
        "<section><h2>Test node by test node</h2>"
        '<p class="lede">Every test node, P(fraud) under A against P(fraud) under B. '
        "Dashed lines: each side's threshold. The corner names describe fraud nodes "
        "(red); for the other nodes (blue) the upper and right regions are false "
        "alarms. Click a point or a gallery entry to see that node.</p>"
        '<div class="compare-grid"><div class="panel panel-pad">'
        f"{scatter_svg(table, a.threshold, b.threshold)}"
        '<div class="legend"><span><span class="dot fraud"></span>fraud</span>'
        '<span><span class="dot benign"></span>not fraud</span>'
        '<span><span class="dash"></span>threshold</span></div></div>'
        '<div class="panel panel-pad" id="card-panel"><p class="muted">Click a point '
        "or a gallery entry.</p></div></div>"
        f'<div style="margin-top:16px">{_outcomes_table(m)}</div></section>'
        "<section><h2>Where the two disagree on fraud nodes</h2>"
        '<p class="lede">The fraud nodes on which the two decisions differ, ranked by '
        "how far apart the two probabilities are.</p>"
        '<div class="grid-2">'
        + _gallery(
            "B rescues", "Missed by A (below its threshold), caught by B.", rescues
        )
        + _gallery("B breaks", "Caught by A, missed by B.", breaks)
        + "</div></section>"
        + embedding
        + HOW_TO_READ
        + f"<footer>Generated by <code>uv run scripts/compare_report.py --dataset "
        f"{escape(m['dataset'])} --a {escape(a.feature_set)} --b "
        f"{escape(b.feature_set)}</code> · Contains row-level test data: group only, "
        "never commit or upload.</footer>"
        + cards
        + f'<script type="application/json" id="node-data">{_json_for_script(node_data)}'
        "</script>"
    )
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Model Comparison</title><style>{css}</style></head>"
        f'<body><div class="wrap">{body}</div><script>{SCRIPT}</script></body></html>\n'
    )


HOW_TO_READ = """
<section><h2>How to read this page</h2>
<ul class="notes panel">
<li><b>A and B</b> are two rungs of the model ladder: the same df-analyze pipeline on
two feature sets. Each side's Model A was chosen on the training set only; test
numbers never chose anything.</li>
<li><b>ΔPR-AUC</b> is B's test PR-AUC minus A's. Its interval comes from resampling
whole test groups (sets of identical Amazon users, or YelpChi users) and scoring both
models on each resample, so the two models' shared luck cancels out. The share of
resamples with B ahead is not a p-value.</li>
<li><b>Recall@K</b> is the share of test frauds among the K highest-scored test nodes,
K being the number of test frauds. It needs no threshold.</li>
<li><b>Thresholds</b> differ between the sides: each was chosen on its own out-of-fold
training predictions (max F1).</li>
<li><b>Neighbourhoods</b> show at most 12 neighbours per relation, drawn with a fixed
seed. Only training neighbours are coloured by their label; the share of training
neighbours that are fraud describes the graph and is never a model input.</li>
<li><b>One split, one seed.</b> Intervals cover test-set sampling only. Amazon is
small; differences of one or two points are within its split-to-split variation.</li>
</ul></section>
"""
