# ruff: noqa: E501  (long lines of HTML and CSS)
"""Explanation page and aggregate tables, from scripts/dfa/shap_explain.py's outputs.

Built by scripts/explain_report.py. It is not a pipeline stage, so it adds no
outputs that could make a finished run or a bundle rerun. It reads
outputs/<dataset>[_repeat<k>]/<feature set>/explain/ and writes, in the same folder:

- tables/*.csv, aggregate tables, and db_summary.csv, the one table for the
  results database (at most 1,000 rows, no IDs, no labels, no predictions);
- explain.html, one self-contained page with inline CSS and SVG, no scripts and no
  external requests, every value HTML-escaped. It holds row-level test data (case
  cards, top neighbour IDs), so it carries a group-only banner and is never
  committed or uploaded.

The wording follows docs/EXPLAINABILITY_PLAN.md: exact Shapley values of a stated
game, additivity as an implementation check only, faithfulness tested separately
and limited for correlated features, and neighbour influence as necessity-style
influence under mean aggregation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detect import columns as C
from fraud_detect.features.graph import adjacency
from fraud_detect.models import explain as X
from fraud_detect.models import explain_eval as E
from fraud_detect.report.web import CSS_PATH
from fraud_detect.runstate import atomic_write_csv

SHAP_COLUMN = "shap__"
GROUP_COLUMN = "group__"
N_TOP = 10
N_TOP_NODES = 10
N_OVERLAP = 100
STRATA = ("flagged", "true_positive", "false_positive", "false_negative")
STRATUM_LABELS = {
    "all": "All test nodes",
    "fraud": "Fraud",
    "not_fraud": "Not fraud",
    "flagged": "Flagged (above the threshold)",
    "true_positive": "True positives",
    "false_positive": "False positives",
    "false_negative": "False negatives",
}
CARD_TITLES = {
    "true_positive": "True positive",
    "false_positive": "False positive",
    "false_negative": "False negative",
}


class ExplainReportError(RuntimeError):
    """The explanation outputs are incomplete; the message says what is missing."""


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
@dataclass
class Summary:
    meta: dict[str, Any]
    importance: pd.DataFrame  # one row per column, all test nodes
    strata: dict[str, pd.DataFrame]
    groups: pd.DataFrame  # block importance: summed SHAP and group Shapley
    correlated: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    cards: list[dict[str, Any]] = field(default_factory=list)
    audit: pd.DataFrame | None = None
    audit_meta: dict[str, Any] | None = None
    split_agreement: pd.DataFrame | None = None


def read_meta(folder: Path) -> dict[str, Any]:
    path = folder / "explain_meta.json"
    if not path.is_file():
        raise ExplainReportError(
            f"{path} is missing: run scripts/dfa/shap_explain.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _shap_frame(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    path = folder / "shap_test.parquet"
    if not path.is_file():
        raise ExplainReportError(f"{path} is missing: run the shap section first")
    frame = pd.read_parquet(path)
    columns = [c.removeprefix(SHAP_COLUMN) for c in frame if c.startswith(SHAP_COLUMN)]
    # A row that failed the implementation check holds NaN; no summary uses it.
    frame = frame[frame[[SHAP_COLUMN + c for c in columns]].notna().all(axis=1)]
    return frame.reset_index(drop=True), columns


def _renames(feature_set_json: Path) -> dict[str, str]:
    info = json.loads(feature_set_json.read_text(encoding="utf-8"))
    return {safe: c["block_column"] for safe, c in info["columns"].items()}


def _optional_csv(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / name
    return pd.read_csv(path, keep_default_na=False) if path.is_file() else None


def importance_summary(
    folder: Path, feature_set_json: Path, correlated_features: list[str], seed: int
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Per-column importance overall and per stratum, block importance (summed SHAP
    next to group Shapley) and the correlated group's importance, with intervals."""
    frame, columns = _shap_frame(folder)
    meta = read_meta(folder)
    values = frame[[SHAP_COLUMN + c for c in columns]].to_numpy(dtype=float)
    groups = frame[C.GROUP_ID].to_numpy()
    infos = X.parse_columns(columns, _renames(feature_set_json))
    importance = X.importance_table(values, columns, groups, seed=seed)
    masks = X.strata_masks(
        frame[C.LABEL].to_numpy(), frame["prob"].to_numpy(), float(meta["threshold"])
    )
    strata = {}
    for name, mask in masks.items():
        if name != "all" and mask.sum() >= 2:
            strata[name] = X.importance_table(
                values[mask], columns, groups[mask], seed=seed
            )
    players = X.players_of(infos)
    summed, names = X.summed_by_group(values, players)
    block = X.importance_table(summed, names, groups, seed=seed).assign(
        measure="summed SHAP",
        n_columns=lambda t: t["name"].map(lambda n: len(players[n])),
    )
    tables = [block]
    path = folder / "group_shapley.parquet"
    if path.is_file():
        shapley = pd.read_parquet(path)
        cols = [c for c in shapley if c.startswith(GROUP_COLUMN)]
        names_g = [c.removeprefix(GROUP_COLUMN) for c in cols]
        exact = X.importance_table(
            shapley[cols].to_numpy(dtype=float), names_g, shapley[C.GROUP_ID].to_numpy(),
            seed=seed,
        )  # fmt: skip
        tables.append(
            exact.assign(
                measure="group Shapley",
                n_columns=lambda t: t["name"].map(lambda n: len(players[n])),
            )
        )
    correlated_columns = X.correlated_group_columns(infos, correlated_features)
    if correlated_columns:
        combined, names_c = X.summed_by_group(values, correlated_columns)
        correlated = X.importance_table(combined, names_c, groups, seed=seed)
    else:
        correlated = pd.DataFrame()
    return importance, strata, pd.concat(tables, ignore_index=True), correlated


def neighbour_tables(
    folder: Path,
    threshold: float,
    edges: dict[str, np.ndarray] | None,
) -> dict[str, pd.DataFrame]:
    """Per-relation influence, the most influential neighbours, and whether simple
    structure reproduces the ranking. `edges`: each relation's edge list."""
    path = folder / "neighbour_influence.parquet"
    if not path.is_file():
        return {}
    table = pd.read_parquet(path)
    table["relation"] = table["relation"].astype(str)
    mean_part = table["delta_prob_mean"].abs()
    degree_part = (table["delta_prob"] - table["delta_prob_mean"]).abs()
    work = table.assign(
        abs_total=table["delta_prob"].abs().astype(float),
        abs_mean=mean_part.astype(float),
        abs_degree=degree_part.astype(float),
        abs_x_degree=table["delta_prob"].abs().astype(float) * table["degree"],
    )
    per_relation = (
        work.groupby("relation", sort=False)
        .agg(
            n_targets=("target", "nunique"),
            n_influences=("target", "size"),
            mean_abs_delta_prob=("abs_total", "mean"),
            mean_abs_mean_part=("abs_mean", "mean"),
            mean_abs_degree_part=("abs_degree", "mean"),
            mean_abs_x_degree=("abs_x_degree", "mean"),
            share_out_of_range=("out_of_range", "mean"),
        )
        .reset_index()
    )
    # Per target: over influences, a target with many neighbours would count many times.
    per_target = work.drop_duplicates(["relation", "target"])
    medians = per_target.groupby("relation", sort=False)["degree"].median()
    per_relation.insert(3, "median_target_degree", per_relation["relation"].map(medians))
    out = {"neighbour_relations": per_relation}
    shap_frame, _ = _shap_frame(folder)
    flagged = shap_frame.loc[shap_frame["prob"] >= threshold, C.NODE_ID].to_numpy()
    nodes = X.aggregate_influence(table, flagged_targets=flagged)
    if edges:
        scores = pd.read_parquet(folder / "node_scores.parquet").set_index(C.NODE_ID)[
            "prob"
        ]
        is_flagged = (scores.sort_index().to_numpy() >= threshold).astype(float)
        rows = []
        for relation, part in nodes.groupby("relation", sort=False):
            a = adjacency(edges[relation], len(is_flagged))
            ids = part["neighbour"].to_numpy()
            degree = np.asarray(a.sum(axis=1)).ravel()[ids]
            near_flagged = (a @ is_flagged)[ids]
            sum_abs = part["sum_abs"].to_numpy()
            top = set(part.nlargest(N_OVERLAP, "sum_abs")["neighbour"])
            by_degree = set(ids[np.argsort(-degree, kind="stable")[:N_OVERLAP]])
            rows.append(
                {
                    "relation": relation,
                    "n_neighbours": len(part),
                    "spearman_sum_vs_degree": X.spearman(sum_abs, degree),
                    "spearman_sum_vs_targets": X.spearman(
                        sum_abs, part["n_targets"].to_numpy()
                    ),
                    "spearman_sum_vs_flagged_neighbours": X.spearman(
                        sum_abs, near_flagged
                    ),
                    "spearman_normalised_vs_degree": X.spearman(
                        part["mean_abs_x_degree"].to_numpy(), degree
                    ),
                    f"top{N_OVERLAP}_overlap_with_highest_degree": len(top & by_degree)
                    / max(1, len(top)),
                }
            )
        out["neighbour_structure"] = pd.DataFrame(rows)
    out["neighbour_top_nodes"] = (
        nodes.sort_values("sum_abs", ascending=False)
        .groupby("relation", sort=False)
        .head(N_TOP_NODES)
    )
    redundancy = _optional_csv(folder, "neighbour_redundancy.csv")
    if redundancy is not None and len(redundancy):
        for column in ("spearman_loo_vs_shapley", "shapley_abs_sum", "loo_abs_sum"):
            redundancy[column] = pd.to_numeric(redundancy[column], errors="coerce")
        out["neighbour_redundancy_summary"] = (
            redundancy.groupby("relation", sort=False)
            .agg(
                n_targets=("degree", "size"),
                median_degree=("degree", "median"),
                median_spearman=("spearman_loo_vs_shapley", "median"),
                share_same_top_neighbour=("same_top_neighbour", "mean"),
                mean_shapley_abs_sum=("shapley_abs_sum", "mean"),
                mean_loo_abs_sum=("loo_abs_sum", "mean"),
            )
            .reset_index()
        )
    return out


def summarise(
    folder: Path,
    feature_set_json: Path,
    correlated_features: list[str],
    edges: dict[str, np.ndarray] | None = None,
    seed: int = 555,
) -> Summary:
    meta = read_meta(folder)
    importance, strata, groups, correlated = importance_summary(
        folder, feature_set_json, correlated_features, seed
    )
    tables: dict[str, pd.DataFrame] = {}
    for key, name in {
        "sensitivity": "sensitivity.csv",
        "interactions": "interactions.csv",
        "rashomon": "rashomon.csv",
        "randomisation": "randomisation.csv",
        "seed_stability": "seed_stability.csv",
    }.items():
        table = _optional_csv(folder, name)
        if table is not None:
            tables[key] = table
    roar = _optional_csv(folder, "roar.csv")
    if roar is not None:
        extended = _optional_csv(folder, "roar_extended.csv")
        if extended is not None:
            extra = extended[extended["kind"] != "baseline"]
            roar = pd.concat([roar, extra], ignore_index=True)
        for column in ("fold", "k", "draw", "pr_auc", "drop"):
            roar[column] = pd.to_numeric(roar[column])
        summary = E.roar_summary(roar)
        tables["roar_summary"] = summary
        tables["roar_top_k_vs_random"] = E.shap_beats_random(summary)
    tables.update(refit_recurrence(folder, feature_set_json))
    tables.update(neighbour_tables(folder, float(meta["threshold"]), edges))
    cards_path = folder / "case_cards.json"
    cards = (
        json.loads(cards_path.read_text(encoding="utf-8"))["cards"]
        if cards_path.is_file()
        else []
    )
    return Summary(meta, importance, strata, groups, correlated, tables, cards)


def refit_recurrence(folder: Path, feature_set_json: Path) -> dict[str, pd.DataFrame]:
    """How much survives refitting: how many fits (Model A, its refits with other
    seeds, other good models) have each column in their top 10, and each block's
    share of total mean |SHAP| across those fits. Test-node bootstraps hold the
    model fixed, so they cannot show this."""
    fits: dict[str, pd.Series] = {}
    # A refit with df-analyze's own seed reproduces Model A, so it is not another fit.
    for name, skip in (
        ("seed_importance.csv", ("df-analyze's seed",)),
        ("rashomon_importance.csv", ("Model A",)),
    ):
        table = _optional_csv(folder, name)
        if table is None:
            continue
        table = (
            table.set_index("column").apply(pd.to_numeric, errors="coerce").fillna(0.0)
        )
        for fit in table.columns:
            if fit not in skip:
                fits[fit] = table[fit]
    if len(fits) < 2:
        return {}
    wide = pd.DataFrame(fits).fillna(0.0)
    in_top = pd.DataFrame({f: wide[f].rank(ascending=False) <= N_TOP for f in wide})
    recurrence = pd.DataFrame(
        {
            "name": wide.index,
            "fits_in_top10": in_top.sum(axis=1).to_numpy(),
            "n_fits": wide.shape[1],
            "mean_rank": wide.rank(ascending=False).mean(axis=1).to_numpy(),
        }
    ).sort_values(["fits_in_top10", "mean_rank"], ascending=[False, True])
    infos = X.parse_columns(list(wide.index), _renames(feature_set_json))
    players = pd.Series([c.player for c in infos], index=wide.index)
    shares = wide.groupby(players).sum() / wide.sum()
    block = pd.DataFrame(
        {
            "name": shares.index,
            "min_share": shares.min(axis=1).to_numpy(),
            "max_share": shares.max(axis=1).to_numpy(),
            "n_fits": wide.shape[1],
        }
    ).sort_values("max_share", ascending=False)
    return {
        "top10_recurrence": recurrence[recurrence["fits_in_top10"] > 0],
        "block_share_across_fits": block,
    }


def split_agreement(frames: dict[str, pd.Series]) -> pd.DataFrame:
    """Rank agreement between every pair of splits' importance vectors."""
    names = list(frames)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append({"a": a, "b": b, **X.rank_agreement(frames[a], frames[b])})
    return pd.DataFrame(rows)


def write_tables(summary: Summary, folder: Path) -> Path:
    """The aggregate tables, and db_summary.csv for the results database. Tables
    naming node IDs stay out of db_summary.csv."""
    tables = folder / "tables"
    atomic_write_csv(tables / "importance.csv", summary.importance)
    atomic_write_csv(tables / "groups.csv", summary.groups)
    if len(summary.correlated):
        atomic_write_csv(tables / "correlated.csv", summary.correlated)
    for name, frame in summary.strata.items():
        atomic_write_csv(tables / f"importance_{name}.csv", frame)
    for name, frame in summary.tables.items():
        atomic_write_csv(tables / f"{name}.csv", frame)
    if summary.split_agreement is not None:
        atomic_write_csv(tables / "split_agreement.csv", summary.split_agreement)
    long = [
        summary.importance.head(N_TOP * 3).assign(section="importance", stratum="all"),
        summary.groups.assign(section="block importance", stratum="all"),
    ]
    for name, frame in summary.strata.items():
        long.append(frame.head(N_TOP).assign(section="importance", stratum=name))
    if len(summary.correlated):
        long.append(summary.correlated.assign(section="correlated group", stratum="all"))
    for key in ("sensitivity", "seed_stability", "rashomon", "randomisation",
                "top10_recurrence", "block_share_across_fits",
                "interactions", "roar_summary", "roar_top_k_vs_random",
                "neighbour_relations", "neighbour_structure",
                "neighbour_redundancy_summary"):  # fmt: skip
        if key in summary.tables:
            long.append(summary.tables[key].assign(section=key))
    if summary.split_agreement is not None:
        long.append(summary.split_agreement.assign(section="split_agreement"))
    combined = pd.concat(long, ignore_index=True)
    first = ["section", "stratum", "name"]
    combined = combined[first + [c for c in combined if c not in first]]
    path = tables / "db_summary.csv"
    atomic_write_csv(path, combined)
    return path


# --------------------------------------------------------------------------
# Page: formats
# --------------------------------------------------------------------------
EXTRA_CSS = """
:root { --series-1: #2a78d6; --series-2: #eb6834; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --series-1: #3987e5; --series-2: #d95926; }
}
:root[data-theme="dark"] { --series-1: #3987e5; --series-2: #d95926; }
svg.chart { display: block; width: 100%; height: auto; max-width: 860px; }
svg.chart text { font: 12px system-ui, sans-serif; fill: var(--muted); }
svg.chart text.v { fill: var(--text); }
svg.chart .axis { stroke: var(--border); stroke-width: 1; }
svg.chart .grid { stroke: var(--border); stroke-width: 1; stroke-dasharray: 2 3; }
svg.chart .s1 { fill: var(--series-1); }
svg.chart .s2 { fill: var(--series-2); }
svg.chart .l1 { stroke: var(--series-1); fill: none; stroke-width: 2; }
svg.chart .l2 { stroke: var(--series-2); fill: none; stroke-width: 2; }
svg.chart .w { stroke: var(--text); stroke-width: 1; opacity: .55; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12.5px; color: var(--muted); margin: 6px 0 10px; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.legend i.s1 { background: var(--series-1); }
.legend i.s2 { background: var(--series-2); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }
.card { border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.card h4 { margin: 0 0 4px; font-size: 14px; }
.card ul { margin: 6px 0; padding-left: 18px; font-size: 12.5px; }
.prose { font-size: 14px; max-width: 80ch; }
.prose li { margin-bottom: 4px; }
h2 { margin-top: 28px; }
td.wrap-ok { white-space: normal; }
"""


def _num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if np.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _table(frame: pd.DataFrame, columns: dict[str, str], digits: int = 3) -> str:
    head = "".join(
        f"<th{' class=l' if i == 0 else ''}>{escape(title)}</th>"
        for i, title in enumerate(columns.values())
    )
    body = []
    for _, row in frame.iterrows():
        cells = []
        for i, key in enumerate(columns):
            value = row.get(key, "")
            is_text = isinstance(value, str) and not _is_number(value)
            text = escape(str(value)) if is_text else _num(value, digits)
            if isinstance(value, bool | np.bool_):
                text = "yes" if value else "no"
            elif isinstance(value, int | np.integer):
                text = str(int(value))
            cells.append(f"<td{' class=l' if i == 0 else ''}>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------
# Page: charts (inline SVG, thin marks, native tooltips, no scripts)
# --------------------------------------------------------------------------
def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    power = 10.0 ** np.floor(np.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if step * power >= value:
            return float(step * power)
    return float(10 * power)


def bar_chart(
    rows: list[dict[str, Any]],
    series: list[tuple[str, str]],
    axis_title: str,
    label_width: int = 230,
) -> str:
    """Horizontal bars with interval whiskers. rows: {label, <key>: (value, low,
    high)}; series: [(key, name)], up to two."""
    width, right, bar, gap = 860, 56, 11, 2
    group = len(series) * (bar + gap) + 10
    top = 6
    height = top + group * len(rows) + 34
    largest = max(
        (r[key][2] for r in rows for key, _ in series if r.get(key) is not None),
        default=1.0,
    )
    span = _nice_max(float(largest))
    plot = width - label_width - right

    def x(value: float) -> float:
        return label_width + plot * max(0.0, value) / span

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(axis_title)}">'
    ]
    bottom = top + group * len(rows)
    for tick in np.linspace(0, span, 5):
        parts.append(
            f'<line class="grid" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" y2="{bottom}"/>'
            f'<text x="{x(tick):.1f}" y="{bottom + 14}" text-anchor="middle">{tick:.3g}</text>'
        )
    parts.append(
        f'<text x="{label_width + plot / 2:.1f}" y="{bottom + 30}" text-anchor="middle">'
        f"{escape(axis_title)}</text>"
    )
    for i, row in enumerate(rows):
        y0 = top + i * group
        parts.append(
            f'<text class="v" x="{label_width - 8}" y="{y0 + group / 2 + 1:.1f}" '
            f'text-anchor="end" dominant-baseline="middle">{escape(str(row["label"]))}</text>'
        )
        for j, (key, name) in enumerate(series):
            if row.get(key) is None:
                continue
            value, low, high = row[key]
            y = y0 + 4 + j * (bar + gap)
            tip = f"{row['label']}, {name}: {value:.3f} [{low:.3f}, {high:.3f}]"
            parts.append(
                f'<g><title>{escape(tip)}</title><rect class="s{j + 1}" x="{label_width}" '
                f'y="{y}" width="{max(1.0, x(value) - label_width):.1f}" height="{bar}" rx="3"/>'
                f'<line class="w" x1="{x(low):.1f}" x2="{x(high):.1f}" y1="{y + bar / 2}" '
                f'y2="{y + bar / 2}"/></g>'
                f'<text class="v" x="{x(high) + 6:.1f}" y="{y + bar / 2 + 1}" '
                f'dominant-baseline="middle">{value:.3f}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def legend(series: list[str]) -> str:
    items = "".join(
        f'<span><i class="s{i + 1}"></i>{escape(name)}</span>'
        for i, name in enumerate(series)
    )
    return f'<div class="legend">{items}</div>'


def roar_chart(table: pd.DataFrame) -> str:
    """Fall in held-out PR-AUC against k, for top-k and random removal, mean over
    folds with whiskers of one SD between folds."""
    top = table[table["kind"] == "top_k"].sort_values("k")
    random = table[table["kind"] == "random"].sort_values("k")
    if top.empty:
        return ""
    ks = sorted(top["k"].unique())
    width, height, left, right, upper, lower = 860, 270, 70, 90, 12, 46
    values = pd.concat([top, random])
    high = float((values["mean_drop"] + values["sd_drop"].fillna(0)).max())
    low = float(min(0.0, (values["mean_drop"] - values["sd_drop"].fillna(0)).min()))
    ceiling = _nice_max(high)
    floor = -_nice_max(-low) if low < 0 else 0.0
    plot_w, plot_h = width - left - right, height - upper - lower

    def px(k: int) -> float:
        return left + plot_w * (ks.index(k) + 0.5) / len(ks)

    def py(v: float) -> float:
        return upper + plot_h * (ceiling - v) / (ceiling - floor)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Fall in cross-validated PR-AUC against columns removed">'
    ]
    step = _nice_max((ceiling - floor) / 5)
    first, count = (
        np.ceil(floor / step),
        int(np.floor(ceiling / step) - np.ceil(floor / step)),
    )
    for tick in step * (first + np.arange(count + 1)):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{py(tick):.1f}" '
            f'y2="{py(tick):.1f}"/><text x="{left - 8}" y="{py(tick) + 4:.1f}" '
            f'text-anchor="end">{tick:.3g}</text>'
        )
    for k in ks:
        parts.append(
            f'<text x="{px(k):.1f}" y="{height - lower + 18}" text-anchor="middle">{k}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">'
        "columns removed (k), retrained each time</text>"
    )
    for index, (frame, name) in enumerate(((top, "top k by SHAP"), (random, "k random"))):
        points = [
            (px(int(r["k"])), py(float(r["mean_drop"]))) for _, r in frame.iterrows()
        ]
        path = " ".join(f"{a:.1f},{b:.1f}" for a, b in points)
        parts.append(f'<polyline class="l{index + 1}" points="{path}"/>')
        for (a, b), (_, r) in zip(points, frame.iterrows(), strict=True):
            sd = 0.0 if pd.isna(r["sd_drop"]) else float(r["sd_drop"])
            tip = f"{name}, k={int(r['k'])}: {r['mean_drop']:.3f} (SD {sd:.3f})"
            parts.append(
                f"<g><title>{escape(tip)}</title>"
                f'<line class="w" x1="{a:.1f}" x2="{a:.1f}" y1="{py(r["mean_drop"] + sd):.1f}" '
                f'y2="{py(r["mean_drop"] - sd):.1f}"/>'
                f'<circle class="s{index + 1}" cx="{a:.1f}" cy="{b:.1f}" r="4.5"/></g>'
            )
        a, b = points[-1]
        parts.append(
            f'<text class="v" x="{a + 10:.1f}" y="{b + 4:.1f}">{escape(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def waterfall(card: dict[str, Any], names: dict[str, str] | None = None) -> str:
    """One node's log-odds from the base value to its margin. Blue pushes away from
    fraud, orange towards it; every row also says its sign in text."""
    items = [
        (c["column"], float(c["shap"]), float(c["value"])) for c in card["contributions"]
    ]
    items.append(("all other columns", float(card["rest"]), float("nan")))
    start = float(card["base_value"])
    level = [start]
    for _, value, _ in items:
        level.append(level[-1] + value)
    low, high = min(level), max(level)
    pad = max(0.2, (high - low) * 0.06)
    low, high = low - pad, high + pad
    width, left, right, row_h, top = 560, 190, 60, 20, 8
    height = top + row_h * len(items) + 34

    def x(value: float) -> float:
        return left + (width - left - right) * (value - low) / (high - low)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="SHAP waterfall for node {int(card["node_id"])}">'
    ]
    bottom = top + row_h * len(items)
    for tick in np.linspace(low, high, 4):
        parts.append(
            f'<line class="grid" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" y2="{bottom}"/>'
            f'<text x="{x(tick):.1f}" y="{bottom + 13}" text-anchor="middle">{tick:.1f}</text>'
        )
    parts.append(
        f'<text x="{left + (width - left - right) / 2:.1f}" y="{bottom + 28}" '
        'text-anchor="middle">log-odds of fraud</text>'
    )
    for i, (name, value, raw) in enumerate(items):
        y = top + i * row_h
        a, b = level[i], level[i + 1]
        klass = "s2" if value >= 0 else "s1"
        tip = (
            f"{name} = {raw:.4g}: {value:+.3f}"
            if not np.isnan(raw)
            else f"{name}: {value:+.3f}"
        )
        short = name if len(name) <= 28 else name[:27] + "…"
        parts.append(
            f'<text class="v" x="{left - 6}" y="{y + 11}" text-anchor="end" '
            f'dominant-baseline="middle">{escape(short)}</text>'
            f'<g><title>{escape(tip)}</title><rect class="{klass}" x="{x(min(a, b)):.1f}" '
            f'y="{y + 4}" width="{max(1.0, abs(x(b) - x(a))):.1f}" height="12" rx="3"/></g>'
            f'<text x="{x(max(a, b)) + 4:.1f}" y="{y + 11}" dominant-baseline="middle">'
            f"{value:+.2f}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Page: sections
# --------------------------------------------------------------------------
def _caveats(meta: dict[str, Any]) -> str:
    items = [
        "This explains the model, not the cause of fraud.",
        "In controlled studies with fraud analysts, explanations did not reliably "
        "improve decisions once the experiment matched deployment (Jesus et al., FAccT "
        "2021; Amarasinghe et al., AAAI 2024), and they can raise over-reliance on the "
        "model (Bansal et al., CHI 2021). Nothing here claims otherwise.",
    ]
    if meta["dataset"] == "amazon":
        items.append(
            "Amazon's f19 is a probable label leak (issue #36): the label is built from "
            "helpfulness votes, and f19 looks vote-derived."
        )
    if meta["dataset"] == "yelpchi":
        items.append(
            "YelpChi's labels come from Yelp's filter, so they are pseudo-labels. R-U-R "
            "neighbours are the same author's other reviews."
        )
    body = "".join(f"<li>{escape(i)}</li>" for i in items)
    return f'<div class="notice"><ul style="margin:0;padding-left:18px">{body}</ul></div>'


def _game_note(shap_meta: dict[str, Any]) -> str:
    n_rows = shap_meta.get("n_test_rows", "?")
    if shap_meta.get("feature_perturbation") == "interventional":
        return (
            "SHAP values are exact Shapley values of the <strong>interventional</strong> "
            f"game against {shap_meta.get('n_background', '?')} training rows, in log-odds, "
            f"on {n_rows} test nodes."
        )
    return (
        "SHAP values here are <strong>path-dependent</strong> TreeSHAP, in log-odds, on "
        f"{n_rows} test nodes. That is a different game from the interventional one on "
        "the other pages, used because interventional TreeSHAP failed the implementation "
        "check for this model, so do not compare the two directly."
    )


def _excluded_note(check: dict[str, Any]) -> str:
    n = int(check.get("n_rows_excluded", 0))
    if n == 0:
        return ""
    return (
        f" {n} test nodes failed that check, by up to "
        f"{float(check['max_error_excluded']):.2f} log-odds, and are left out of every "
        "summary: shap 0.54's interventional TreeSHAP can send a LightGBM row the wrong "
        "way at a split when its value and the threshold agree in single precision."
    )


def _header(summary: Summary) -> str:
    m = summary.meta
    shap_meta = m["sections"].get("shap", {})
    check = shap_meta.get("implementation_check", {})
    selector = f" ({m['embed_selector']})" if m.get("embed_selector") else ""
    title = f"{m['dataset']}, split {m['repeat']}: {m['feature_set']}"
    return f"""
<div class="eyebrow">Explanation of Model A: for the group only, never commit or upload</div>
<h1>{escape(title)}</h1>
<p class="lede">Model A is {escape(m["model"])}, selection {escape(str(m["selection"]))}{escape(selector)},
{m["n_input_columns"]} input columns, {escape(m["budget"])} df-analyze budget, chosen by shared grouped-CV
PR-AUC. Threshold {_num(m["threshold"])} from out-of-fold training predictions. {_game_note(shap_meta)} Largest
additivity error
{float(check.get("max_additivity_error", float("nan"))):.1e}: this is an implementation check, not evidence
that the explanation is faithful.{_excluded_note(check)}</p>
"""


def _headline(summary: Summary) -> str:
    groups = summary.groups
    exact = groups[groups["measure"] == "group Shapley"].set_index("name")
    summed = groups[groups["measure"] == "summed SHAP"].set_index("name")
    if exact.empty:
        order = list(summed.index)
    else:
        order = list(exact.sort_values("mean_abs_shap", ascending=False).index)
    rows = []
    for name in order:
        row: dict[str, Any] = {
            "label": f"{name} ({int(summed.loc[name, 'n_columns'])} cols)"
        }
        for key, frame in (("exact", exact), ("summed", summed)):
            if name in frame.index:
                r = frame.loc[name]
                row[key] = (float(r["mean_abs_shap"]), float(r["low"]), float(r["high"]))
        rows.append(row)
    series = [("exact", "group Shapley"), ("summed", "summed SHAP")]
    chart = bar_chart(rows, series, "mean |attribution| per test node, log-odds")
    n_rows = int(exact["n_rows"].iloc[0]) if not exact.empty else 0
    return f"""
<h2>Does the graph matter to this model?</h2>
<p class="prose">Mean size of each feature block's contribution, with group-bootstrap 95% intervals.
<strong>Group Shapley</strong> treats the block as one player (all {2 ** len(rows)} coalitions, on {n_rows}
stratified test nodes). <strong>Summed SHAP</strong> adds the block's per-column values. They differ by how
each game shares interactions across blocks. Sizes are not additive across blocks when signs differ.</p>
<div class="panel panel-pad">{legend([s[1] for s in series])}{chart}</div>
{_roar_section(summary)}
"""


def _roar_section(summary: Summary) -> str:
    if "roar_summary" not in summary.tables:
        return ""
    table = summary.tables["roar_summary"]
    versus = summary.tables["roar_top_k_vs_random"]
    # A gap counts only if it exceeds the spread between folds of the top-k removal.
    spread = table[table["kind"] == "top_k"].set_index("k")["sd_drop"]
    clear = versus[versus["gap"] > versus["k"].map(spread).fillna(0.0)]
    verdict = (
        "The top-k removal did not separate clearly from random at any k."
        if clear.empty
        else "The top-k removal hurt more than random by more than one SD between folds at k = "
        + ", ".join(str(int(k)) for k in clear["k"])
        + "."
    )
    added = (
        " Values of k above 8 and the all-graph removal were added after the plan's k = 1, "
        "2, 4, 8 showed no separation on Amazon, so read them as follow-up."
        if (table["k"] > 8).any()
        else ""
    )
    blocks = table[table["kind"].isin(["block", "group"])].sort_values(
        "mean_drop", ascending=False
    )
    return f"""
<h3>Retrain after removal (training CV only)</h3>
<p class="prose">Each fold ranks columns with SHAP from a model fit without that fold, on that model's own
rows; the model is then retrained with its hyperparameters fixed, and the held-out fold's PR-AUC is compared
with the fold's baseline. Attributions that identify what the model needs should hurt more than random removal.
{
        escape(verdict)
    } A large attribution with little fall means other columns can stand in for that one, so
attribution here does not measure necessity.{
        escape(added)
    } Whiskers are one SD between folds, and only 5
folds stand behind each point.</p>
<div class="panel panel-pad">{legend(["top k by SHAP", "k random"])}{
        roar_chart(table)
    }</div>
<div class="panel" style="margin-top:12px">{
        _table(
            blocks,
            {
                "name": "Block or group removed",
                "k": "Columns",
                "mean_drop": "Mean fall in PR-AUC",
                "sd_drop": "SD between folds",
                "n_folds": "Folds",
            },
        )
    }</div>
"""


def _features(summary: Summary) -> str:
    top = summary.importance.head(N_TOP)
    rows = [
        {"label": r["name"], "v": (r["mean_abs_shap"], r["low"], r["high"])}
        for _, r in top.iterrows()
    ]
    chart = bar_chart(rows, [("v", "mean |SHAP|")], "mean |SHAP| per test node, log-odds")
    unstable = int(summary.importance["unstable"].sum())
    recurrence = ""
    if "top10_recurrence" in summary.tables:
        table = summary.tables["top10_recurrence"]
        n_fits = int(table["n_fits"].iloc[0])
        always = int((table["fits_in_top10"] == n_fits).sum())
        recurrence = f"""<h3>What survives refitting</h3>
<p class="prose">The intervals above resample test nodes with the model held fixed, so they say nothing about
refitting. Across {
            n_fits
        } fits (Model A, refits with other seeds, and other good models), {
            always
        } columns stay
in the top {
            N_TOP
        } every time. Read per-column ranks below those as unsettled. Each block's share of the total
mean |SHAP| across the same fits:</p>
<div class="panel">{
            _table(
                summary.tables["block_share_across_fits"],
                {
                    "name": "Block",
                    "min_share": "Lowest share",
                    "max_share": "Highest share",
                },
            )
        }</div>
<div class="panel" style="margin-top:12px">{
            _table(
                table.head(15),
                {
                    "name": "Column",
                    "fits_in_top10": "Fits with it in the top 10",
                    "n_fits": "Fits",
                    "mean_rank": "Mean rank",
                },
                digits=1,
            )
        }</div>"""
    correlated = ""
    if len(summary.correlated):
        correlated = f"""<p class="prose">Credit is unreliable between correlated columns, so issue #36's
columns are also shown as one group:</p><div class="panel">{
            _table(
                summary.correlated,
                {
                    "name": "Group",
                    "mean_abs_shap": "Mean |summed SHAP|",
                    "low": "Low",
                    "high": "High",
                },
            )
        }</div>"""
    strata = []
    for name in STRATA:
        if name in summary.strata:
            frame = summary.strata[name].head(5)
            strata.append(
                {
                    "stratum": STRATUM_LABELS[name],
                    "n": int(frame["n_rows"].iloc[0]),
                    "top": ", ".join(
                        f"{r['name']} ({r['mean_signed_shap']:+.2f})"
                        for _, r in frame.iterrows()
                    ),
                }
            )
    strata_frame = pd.DataFrame(strata)
    return f"""
<h2>Which columns drive the score?</h2>
<p class="prose">The top {N_TOP} of {len(summary.importance)} columns. {
        unstable
    } columns have a rank interval
wider than 5 places, so treat their exact order as unstable.</p>
<div class="panel panel-pad">{chart}</div>
{recurrence}
{correlated}
<h3>Where it matters: flagged nodes and errors</h3>
<p class="prose">PR-AUC lives among high-scoring nodes, so averages alone mislead. Top 5 columns per subset,
with the signed mean contribution in brackets.</p>
<div class="panel">{
        _table(
            strata_frame,
            {"stratum": "Subset", "n": "Nodes", "top": "Top columns (signed mean)"},
        )
        if len(strata_frame)
        else ""
    }</div>
"""


def _checks(summary: Summary) -> str:
    parts = []
    t = summary.tables
    if "sensitivity" in t:
        parts.append(
            "<h3>Background and SHAP mode</h3>"
            '<p class="prose">Against the main run. Interventional and path-dependent '
            "TreeSHAP are different games, so disagreement is information, not an error.</p>"
            + '<div class="panel">'
            + _table(
                t["sensitivity"],
                {
                    "comparison": "Run",
                    "n_background": "Background rows",
                    "kendall_tau": "Kendall tau",
                    "top10_jaccard": "Top-10 Jaccard",
                    "top10_max_rank_move": "Largest top-10 rank move",
                },
            )  # fmt: skip
            + "</div>"
        )
    if "seed_stability" in t or summary.split_agreement is not None:
        parts.append("<h3>Stability across model seeds and splits</h3>")
    if "seed_stability" in t:
        parts.append(
            '<div class="panel">'
            + _table(
                t["seed_stability"],
                {
                    "refit": "Refit",
                    "max_margin_gap_vs_model_a": "Largest margin gap vs Model A",
                    "kendall_tau": "Kendall tau vs Model A",
                    "top10_jaccard": "Top-10 Jaccard",
                    "top10_max_rank_move": "Largest top-10 rank move",
                },
            )  # fmt: skip
            + "</div>"
        )
    if summary.split_agreement is not None:
        parts.append(
            '<p class="prose">Between split repeats, each with its own Model A:</p>'
            + '<div class="panel">'
            + _table(
                summary.split_agreement,
                {
                    "a": "Split",
                    "b": "Split",
                    "kendall_tau": "Kendall tau",
                    "top10_jaccard": "Top-10 Jaccard",
                    "top10_max_rank_move": "Largest top-10 rank move",
                },
            )  # fmt: skip
            + "</div>"
        )
    if "rashomon" in t:
        parts.append(
            "<h3>Other good models</h3>"
            '<p class="prose">Tuned tree models within one SD of Model A\'s shared-CV PR-AUC, '
            "refit with their tuned settings. Agreement is on a sample of test nodes.</p>"
            + '<div class="panel">'
            + _table(
                t["rashomon"],
                {
                    "model_cls": "Model",
                    "selection": "Selection",
                    "role": "Role",
                    "cv_pr_auc": "CV PR-AUC",
                    "kendall_tau": "Kendall tau vs A",
                    "top10_jaccard": "Top-10 Jaccard",
                },
            )  # fmt: skip
            + "</div>"
        )
    if "randomisation" in t:
        parts.append(
            "<h3>Sanity check: permuted training labels</h3>"
            '<p class="prose">Refit on permuted labels, with attributions on a held-out '
            "training fold. The ranking should lose its structure. A tree still splits on "
            "noise, so total attribution falls without reaching zero.</p>"
            + '<div class="panel">'
            + _table(
                t["randomisation"],
                {
                    "run": "Run",
                    "pr_auc": "Held-out PR-AUC",
                    "prevalence": "No-skill",
                    "total_mean_abs": "Total mean |SHAP|",
                    "spearman_with_true": "Spearman with true model",
                },
            )  # fmt: skip
            + "</div>"
        )
    if "interactions" in t:
        own_graph = t["interactions"][
            t["interactions"]["own_x_graph"].astype(str) == "True"
        ]
        parts.append(
            "<h3>Interactions</h3>"
            '<p class="prose">Path-dependent interaction values, the only mode shap provides '
            "them in, so they are not the same game as the main run. Pairs among the top 5 "
            "columns and the degree columns, strongest first.</p>"
            + '<div class="panel">'
            + _table(
                pd.concat(
                    [own_graph.head(3), t["interactions"].head(3)]
                ).drop_duplicates(),
                {
                    "a": "Column",
                    "b": "Column",
                    "mean_abs_interaction": "Mean |interaction|",
                    "own_x_graph": "Own x graph",
                    "n_rows": "Nodes",
                },
            )  # fmt: skip
            + "</div>"
        )
    if not parts:
        return ""
    return "<h2>How far to trust these attributions</h2>" + "".join(parts)


def _neighbours(summary: Summary) -> str:
    t = summary.tables
    if "neighbour_relations" not in t:
        return ""
    relations = t["neighbour_relations"]
    encoder = summary.meta.get("sections", {}).get("neighbours_encoder")
    if encoder:
        intro = f"""<strong>Necessity-style neighbour influence through the DGI encoder, approximate</strong>: how far
one target's score falls when one neighbour is removed from one relation and that relation's embedding columns
are recomputed, with the encoder and PCA held fixed. Nothing is retrained, so this describes these weights on a
smaller graph, not a model that never saw the edge. It is not "the important nodes". Each target scores at most
{int(encoder["max_neighbours_per_target"])} of its neighbours, a seeded random sample, so a node's total counts
only the removals that were sampled, and a target's most influential neighbour may not be among them."""
        columns = {
            "relation": "Relation",
            "n_targets": "Targets",
            "n_influences": "Influences",
            "median_target_degree": "Median degree",
            "mean_abs_delta_prob": "Mean |Δ prob|",
            "mean_abs_x_degree": "|Δ prob| × degree",
            "share_out_of_range": "Outside training range",
        }
    else:
        intro = """<strong>Necessity-style neighbour influence under mean aggregation</strong>: how far one
target's score falls when one neighbour is removed from one relation and its neighbour columns are recomputed
as in training. It is not "the important nodes". Two neighbours that carry the same signal each look harmless
alone, and under a mean a neighbour's weight is about 1/degree, so totals favour hubs. The mean part is who
the neighbour is; the degree part is how many neighbours there are."""
        columns = {
            "relation": "Relation",
            "n_targets": "Targets",
            "n_influences": "Influences",
            "median_target_degree": "Median degree",
            "mean_abs_delta_prob": "Mean |Δ prob|",
            "mean_abs_mean_part": "Mean part",
            "mean_abs_degree_part": "Degree part",
            "mean_abs_x_degree": "|Δ prob| × degree",
            "share_out_of_range": "Outside training range",
        }
    parts = [
        f"""
<h2>Which neighbours move a node's score?</h2>
<p class="prose">{intro}</p>
<div class="panel">{_table(relations, columns, digits=5)}</div>
<p class="prose small">"Outside training range" is the share of removals with a recomputed column outside that
column's training minimum and maximum. A row can pass this and still be an unusual combination.</p>"""
    ]
    if "neighbour_redundancy_summary" in t:
        parts.append(
            "<h3>Does leave-one-out miss redundant neighbours?</h3>"
            '<p class="prose">Sampled Shapley over each target\'s neighbours, on small '
            "neighbourhoods only (3 to 30). Low agreement means neighbours overlap, and "
            "leave-one-out understates them.</p>"
            + '<div class="panel">'
            + _table(
                t["neighbour_redundancy_summary"],
                {
                    "relation": "Relation",
                    "n_targets": "Targets",
                    "median_degree": "Median degree",
                    "median_spearman": "Median Spearman",
                    "share_same_top_neighbour": "Same top neighbour",
                    "mean_shapley_abs_sum": "Mean sum |Shapley|",
                    "mean_loo_abs_sum": "Mean sum |LOO|",
                },
            )  # fmt: skip
            + "</div>"
        )
    if "neighbour_structure" in t:
        overlap = [
            c
            for c in t["neighbour_structure"]
            if c.endswith("overlap_with_highest_degree")
        ]
        parts.append(
            "<h3>Is it just centrality?</h3>"
            '<p class="prose">If a simple count reproduces the influence ranking, the ranking '
            "adds little beyond structure. Flagged neighbours use the model's scores, not "
            "labels.</p>"
            + '<div class="panel">'
            + _table(
                t["neighbour_structure"],
                {
                    "relation": "Relation",
                    "n_neighbours": "Neighbours",
                    "spearman_sum_vs_degree": "Sum vs degree",
                    "spearman_sum_vs_targets": "Sum vs targets",
                    "spearman_sum_vs_flagged_neighbours": "Sum vs flagged neighbours",
                    "spearman_normalised_vs_degree": "Normalised vs degree",
                    overlap[0]: f"Top {N_OVERLAP} overlap with highest degree",
                },
            )  # fmt: skip
            + "</div>"
        )
    if "neighbour_top_nodes" in t:
        parts.append(
            f"<h3>Largest totals, top {N_TOP_NODES} per relation</h3>"
            + '<div class="panel">'
            + _table(
                t["neighbour_top_nodes"],
                {
                    "neighbour": "Node",
                    "relation": "Relation",
                    "n_targets": "Targets",
                    "sum_abs": "Sum |Δ prob|",
                    "mean_abs": "Mean |Δ prob|",
                    "mean_abs_x_degree": "|Δ prob| × degree",
                    "towards_fraud": "Towards fraud",
                    "away_from_fraud": "Away from fraud",
                    "sum_abs_flagged": "On flagged targets",
                },
                digits=4,
            )  # fmt: skip
            + "</div>"
        )
    return "".join(parts)


def _card(card: dict[str, Any]) -> str:
    title = CARD_TITLES.get(card["stratum"], card["stratum"])
    margin = float(card["margin"])
    base_prob = 1 / (1 + np.exp(-float(card["base_value"])))
    neighbours = "".join(
        f"<li>{int(n['neighbour'])} via {escape(n['relation'])} (degree {int(n['degree'])}): "
        f"removing it moves the score by {-float(n['delta_prob']):+.4f}</li>"
        for n in card["neighbours"]
    )
    nearest = "".join(
        f"<li>{int(n['node_id'])}: training label {int(n['label'])}, score "
        f"{float(n['prob']):.3f}</li>"
        for n in card["nearest_training"]
    )
    change = card["counterfactual"]
    verb = (
        "setting it to the background median"
        if change["kind"] == "feature"
        else "removing it"
    )
    outcome = (
        "crosses the threshold"
        if change["crosses_threshold"]
        else "is the largest single move tested and does not cross the threshold"
    )
    return f"""<div class="card">
<h4>{escape(title)}: node {int(card["node_id"])}</h4>
<div class="small">Score {float(card["prob"]):.3f} (log-odds {margin:+.2f}); base {base_prob:.3f}; threshold
{float(card["threshold"]):.3f}</div>
{waterfall(card)}
{f'<div class="small"><strong>Most influential neighbours</strong></div><ul>{neighbours}</ul>' if neighbours else ""}
<div class="small"><strong>Nearest training nodes by SHAP values</strong></div><ul>{nearest}</ul>
<div class="small"><strong>What would change the score (model behaviour, not advice):</strong>
{escape(str(change["name"]))}: {verb} gives score {float(change["new_prob"]):.3f}, which {outcome}.</div>
</div>"""


def _cards(summary: Summary) -> str:
    if not summary.cards:
        return ""
    body = "".join(_card(c) for c in summary.cards)
    return f"""
<h2>Worked examples</h2>
<p class="prose">Seeded random picks from each outcome, for understanding the model's behaviour and failures.
False negatives are by definition below the threshold. Orange pushes towards fraud, blue away from it.</p>
<div class="cards">{body}</div>
"""


def _audit(summary: Summary) -> str:
    if summary.audit is None or summary.audit_meta is None:
        return ""
    m = summary.audit_meta
    top = summary.audit.head(N_TOP)
    return f"""
<h2>Leakage audit: the model that keeps f19</h2>
<p class="prose">{escape(m["feature_set"])} ({escape(m["budget"])} budget, {
        escape(m["model"])
    }). This run
confirms what a leaky model leans on. It is not an explanation of fraud behaviour.</p>
<div class="panel">{
        _table(
            top,
            {
                "name": "Column",
                "mean_abs_shap": "Mean |SHAP|",
                "low": "Low",
                "high": "High",
                "rank": "Rank",
                "rank_low": "Rank low",
                "rank_high": "Rank high",
            },
        )
    }</div>
"""


ATTENTION = """
<h2>How this differs from attention</h2>
<div class="prose"><ul>
<li><strong>SHAP</strong> is computed after training, outside the model, from the trained model's inputs and
outputs. Here it gives exact Shapley values of the interventional game defined by a stated background. Local
accuracy holds by construction; faithfulness is tested above, and it stays limited for correlated features.</li>
<li><strong>Attention weights</strong> are parameters of a layer inside an architecture such as GAT or HGT, and
describe how that layer mixes neighbours' messages. They are not, by default, a faithful measure of input
importance (Jain &amp; Wallace, NAACL 2019); whether they can serve as one depends on the test (Wiegreffe &amp;
Pinter, EMNLP 2019).</li>
<li><strong>No model in this ladder uses attention.</strong> GraphSAGE and the R-GCN-style rung use mean or
symmetric normalisation, and HGT was not run, so this comparison is conceptual, not a result.</li>
<li>For GNN embedding columns, SHAP attributes to embedding dimensions, not to edges, which is why the block
level is the only reading reported for them.</li>
</ul></div>
"""


def render_page(summary: Summary) -> str:
    css = CSS_PATH.read_text(encoding="utf-8") + EXTRA_CSS
    m = summary.meta
    title = f"Explanation: {m['dataset']} {m['feature_set']}"
    body = "".join(
        [
            _header(summary),
            _caveats(m),
            _headline(summary),
            _features(summary),
            _neighbours(summary),
            _checks(summary),
            _cards(summary),
            _audit(summary),
            ATTENTION,
        ]
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{css}</style></head>"
        f'<body><div class="wrap">{body}</div></body></html>'
    )
