"""Explaining a tree Model A: column grouping, summaries and neighbour influence.

Plain numpy, pandas and scipy logic, imported by scripts/dfa/shap_explain.py
(df-analyze's environment, where the models load) and scripts/explain_report.py
(this project's environment), and tested on synthetic data in this project's
environment. Nothing here imports a model library. Models arrive as a callable
from an (n, c) float array to n log-odds.

What is computed here, and what it is not (docs/EXPLAINABILITY_PLAN.md):

- SHAP values are exact Shapley values of the interventional game defined by a
  stated background sample. Local accuracy holds by construction, so an
  additivity check only catches implementation errors (wrong output space,
  column order). It is not evidence of faithfulness; explain_eval.py tests that.
- Neighbour influence is necessity-style influence under mean aggregation: how
  one target's score moves when one neighbour is taken out of one relation. It
  says nothing about neighbours that are redundant with each other (the
  sampled-Shapley check measures that), and removals can give inputs outside
  the training data (the out-of-range flag measures one version of that).
- Influence through a GNN encoder is the same question for embedding columns: the
  encoder and PCA stay fixed while one edge is removed, so it describes these
  weights on a smaller graph, not what retraining without that edge would give.

Nothing here takes a label for a graph computation (AGENTS.md invariant 1):
labels enter only the summaries, which split test nodes by outcome for reporting.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from fraud_detect import columns as C

OUTPUT_DIR = "explain"
DEGREE_PLAYER = "degree"
# One decimal place of slack, so rounding in a CSV never marks a value out of range.
RANGE_SLACK = 1e-9

Margin = Callable[[np.ndarray], np.ndarray]


class ExplainError(RuntimeError):
    """The explanation cannot be trusted as asked; the message says why."""


# --------------------------------------------------------------------------
# Column names and players
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnInfo:
    """One model input column. `player` is its block for group Shapley and
    summed importance: own, nbr_<relation>, degree or <gnn kind>_<relation>."""

    prepared: str  # df-analyze's name, e.g. nbr_no19_upu_mean_f00 or own_f22_1.0
    block_column: str  # ours, e.g. nbr_no19__upu_mean_f00 or own__f22
    kind: str  # own, nbr, deg, nbrrw, gnnssl, gnnrand, gnnsup
    relation: str | None
    stat: str | None  # mean, log1p_degree or pc (a GNN embedding column)
    base_feature: str | None  # the own feature behind it, e.g. f00
    player: str


def block_column_of(name: str, renames: Mapping[str, str]) -> str:
    """Our block column for a df-analyze column. df-analyze turns a binary column
    into one indicator named after a level (own_f22_1.0), which maps back to
    own__f22; everything else keeps its name (`renames`: feature_set.json)."""
    if name in renames:
        return renames[name]
    matches = [safe for safe in renames if name.startswith(safe + "_")]
    if not matches:
        raise ExplainError(f"Column {name!r} is not in the feature set's column list")
    return renames[max(matches, key=len)]


def parse_column(prepared: str, block_column: str) -> ColumnInfo:
    block, sep, rest = block_column.partition(C.BLOCK_SEPARATOR)
    if not sep:
        raise ExplainError(f"{block_column!r} has no block prefix")
    kind = block.split("_")[0]
    if kind not in C.BLOCK_KINDS:
        raise ExplainError(f"{block_column!r} is in an unknown block kind {kind!r}")
    if kind == C.OWN_BLOCK:
        return ColumnInfo(prepared, block_column, kind, None, None, rest, C.OWN_BLOCK)
    if kind == C.GNNSUP_BLOCK:
        return ColumnInfo(prepared, block_column, kind, None, None, None, kind)
    relation, _, what = rest.partition("_")
    if what == "log1p_degree":
        return ColumnInfo(
            prepared, block_column, kind, relation, "log1p_degree", None, DEGREE_PLAYER
        )
    if what.startswith("mean_"):
        return ColumnInfo(
            prepared, block_column, kind, relation, "mean", what.removeprefix("mean_"),
            f"{kind}_{relation}",
        )  # fmt: skip
    return ColumnInfo(
        prepared, block_column, kind, relation, "pc", None, f"{kind}_{relation}"
    )


def parse_columns(
    prepared: Sequence[str], renames: Mapping[str, str]
) -> list[ColumnInfo]:
    return [parse_column(name, block_column_of(name, renames)) for name in prepared]


def players_of(infos: Sequence[ColumnInfo]) -> dict[str, list[int]]:
    """Each player's column positions, own first, then in order of appearance."""
    players: dict[str, list[int]] = {}
    for i, info in enumerate(infos):
        players.setdefault(info.player, []).append(i)
    order = sorted(players, key=lambda p: (p != C.OWN_BLOCK, p == DEGREE_PLAYER))
    return {p: players[p] for p in order}


def correlated_group_columns(
    infos: Sequence[ColumnInfo], features: Sequence[str]
) -> dict[str, list[int]]:
    """Columns of a set of correlated own features (issue #36's f13, f15, f18,
    f19), as own columns alone and with their neighbour means. Credit is split
    unreliably between correlated columns, so they are also reported as one."""
    wanted = {f.removeprefix(f"{C.OWN_BLOCK}{C.BLOCK_SEPARATOR}") for f in features}
    own = [
        i
        for i, c in enumerate(infos)
        if c.kind == C.OWN_BLOCK and c.base_feature in wanted
    ]
    every = [i for i, c in enumerate(infos) if c.base_feature in wanted]
    groups: dict[str, list[int]] = {}
    if own:
        groups["#36 group (own)"] = own
    if len(every) > len(own):
        groups["#36 group (own + neighbour means)"] = every
    return groups


def summed_by_group(
    values: np.ndarray, groups: Mapping[str, Sequence[int]]
) -> tuple[np.ndarray, list[str]]:
    """Row-wise sum of SHAP values within each group of columns. The sum is
    additive, but it is not the Shapley value of the group as one player:
    cross-group interactions stay split as the per-column game splits them."""
    names = list(groups)
    out = np.column_stack([values[:, list(groups[g])].sum(axis=1) for g in names])
    return out, names


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def stratified_sample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorted positions of about n rows, keeping each class's share (at least one
    row of each class). All rows if n is at least their number."""
    y = np.asarray(y)
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    picked = []
    for value in np.unique(y):
        where = np.flatnonzero(y == value)
        take = max(1, round(n * len(where) / len(y)))
        picked.append(rng.choice(where, size=min(take, len(where)), replace=False))
    return np.sort(np.concatenate(picked))


# --------------------------------------------------------------------------
# Implementation check
# --------------------------------------------------------------------------
def near_threshold(rows: np.ndarray, thresholds: Mapping[int, np.ndarray]) -> np.ndarray:
    """Rows with a value that a split threshold of its column equals in single
    precision. See implementation_check: shap can send such a row the wrong way. In
    a test row that spoils one row; in a background row it spoils every row a little,
    so such rows are kept out of the background."""
    near = np.zeros(len(rows), dtype=bool)
    for column, values in thresholds.items():
        single = np.asarray(rows)[:, column].astype(np.float32)
        near |= np.isin(single, np.asarray(values).astype(np.float32))
    return near


def implementation_check(
    values: np.ndarray,
    base: float,
    margin: np.ndarray,
    background_margin: np.ndarray | None = None,
    tolerance: float = 1e-4,
    max_bad_share: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    """base + sum of SHAP values must equal the model's margin (and, for an
    interventional run, base must be the mean margin over the background).
    Holds by construction for correct code, so it finds implementation errors,
    such as a wrong output space or column order; it does not show the
    explanation is faithful.

    Returns a mask of rows that miss, for the caller to exclude, and a summary. A
    few rows can miss for a known reason: shap 0.54's interventional TreeSHAP sends a
    LightGBM row the wrong way at a split when the value and threshold agree in
    single precision (found by this check: about 1 row in 1,000 on these datasets,
    cleared by moving such values apart in single precision, not in double). More
    than `max_bad_share` of the rows means something else is wrong: raises
    ExplainError."""
    scale = max(1.0, float(np.abs(margin).max()))
    errors = np.abs(base + values.sum(axis=1) - margin)
    bad = errors > tolerance * scale
    out = {
        "max_additivity_error": float(errors[~bad].max())
        if (~bad).any()
        else float("nan"),
        "tolerance": tolerance * scale,
        "n_rows_excluded": int(bad.sum()),
        "max_error_excluded": float(errors[bad].max()) if bad.any() else 0.0,
    }
    if bad.mean() > max_bad_share:
        raise ExplainError(
            f"base + sum of SHAP misses the model's margin on {int(bad.sum())} of "
            f"{len(bad)} rows, by up to {float(errors.max()):.3g} (tolerance "
            f"{tolerance * scale:.3g}): wrong output space or columns?"
        )
    if background_margin is not None:
        gap = abs(float(np.mean(background_margin)) - base)
        out["base_vs_background_mean"] = gap
        if gap > tolerance * scale:
            raise ExplainError(
                f"The base value is {gap:.3g} from the mean margin over the "
                "background, so the explainer did not use this background"
            )
    return bad, out


# --------------------------------------------------------------------------
# Summaries with group-bootstrap intervals
# --------------------------------------------------------------------------
def strata_masks(
    y: np.ndarray, prob: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    """Test-node subsets to summarise. `threshold` comes from out-of-fold
    training predictions, so no subset is chosen with test labels' help."""
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(prob) >= threshold
    return {
        "all": np.ones(len(y), dtype=bool),
        "fraud": y == 1,
        "not_fraud": y == 0,
        "flagged": flagged,
        "true_positive": flagged & (y == 1),
        "false_positive": flagged & (y == 0),
        "false_negative": ~flagged & (y == 1),
    }


def importance_table(
    values: np.ndarray,
    names: Sequence[str],
    groups: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    unstable_span: int = 5,
) -> pd.DataFrame:
    """Mean |SHAP| per column, with a percentile interval from resampling whole
    split groups, the signed mean, and the column's rank with its interval. A
    column whose rank interval spans more than `unstable_span` places is
    flagged unstable. Row-level input; the output has one row per column."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(groups):
        raise ExplainError("One group per row of SHAP values is required")
    _, index = np.unique(np.asarray(groups), return_inverse=True)
    index = index.ravel()
    n_groups = int(index.max()) + 1 if len(index) else 0
    if n_groups == 0:
        raise ExplainError("No rows to summarise")
    abs_sums = np.zeros((n_groups, values.shape[1]))
    np.add.at(abs_sums, index, np.abs(values))
    sizes = np.bincount(index, minlength=n_groups).astype(float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        n_groups, np.full(n_groups, 1.0 / n_groups), size=n_resamples
    )
    means = (counts @ abs_sums) / (counts @ sizes)[:, None]
    ranks = (-means).argsort(axis=1).argsort(axis=1) + 1
    alpha = (1.0 - level) / 2
    estimate = np.abs(values).mean(axis=0)
    rank_low, rank_high = np.quantile(ranks, [alpha, 1 - alpha], axis=0)
    table = pd.DataFrame(
        {
            "name": list(names),
            "mean_abs_shap": estimate,
            "low": np.quantile(means, alpha, axis=0),
            "high": np.quantile(means, 1 - alpha, axis=0),
            "mean_signed_shap": values.mean(axis=0),
            "rank": (-estimate).argsort().argsort() + 1,
            "rank_low": rank_low,
            "rank_high": rank_high,
        }
    )
    table["unstable"] = (table["rank_high"] - table["rank_low"]) > unstable_span
    table["n_rows"] = len(values)
    return table.sort_values("rank").reset_index(drop=True)


def rank_agreement(a: pd.Series, b: pd.Series, k: int = 10) -> dict[str, float]:
    """How much two importance vectors (same index) agree: Kendall's tau over
    every column, the Jaccard overlap of their top k, and the largest rank move
    of a's top k. A diagnostic: the two may measure different things."""
    common = a.index.intersection(b.index)
    a, b = a.loc[common].astype(float), b.loc[common].astype(float)
    k = min(k, len(common))
    top_a = set(a.nlargest(k).index)
    top_b = set(b.nlargest(k).index)
    rank_a = a.rank(ascending=False, method="first")
    rank_b = b.rank(ascending=False, method="first")
    tau = (
        stats.kendalltau(a.to_numpy(), b.to_numpy()).statistic if len(common) > 1 else 1.0
    )
    return {
        "kendall_tau": float(tau),
        f"top{k}_jaccard": len(top_a & top_b) / len(top_a | top_b),
        f"top{k}_max_rank_move": float((rank_a - rank_b).abs().loc[list(top_a)].max()),
        "n_columns": len(common),
    }


# --------------------------------------------------------------------------
# Neighbour influence under mean aggregation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RelationLayout:
    """Where one relation's neighbour columns sit in the model's input."""

    relation: str
    mean_positions: np.ndarray  # model columns holding a neighbour mean
    feature_index: np.ndarray  # for each, its own feature's column in F
    degree_position: int | None  # the log1p(degree) model column, if kept

    @property
    def positions(self) -> np.ndarray:
        if self.degree_position is None:
            return self.mean_positions
        return np.append(self.mean_positions, self.degree_position)


def neighbour_layouts(
    infos: Sequence[ColumnInfo], own_features: Sequence[str]
) -> dict[str, RelationLayout]:
    """One layout per relation with `nbr` columns in the model's input. Feature
    selection may have dropped some columns; those need no update."""
    where = {f.removeprefix(f"{C.OWN_BLOCK}{C.BLOCK_SEPARATOR}"): i
             for i, f in enumerate(own_features)}  # fmt: skip
    relations = sorted(
        {c.relation for c in infos if c.kind == C.NBR_BLOCK and c.relation}
    )
    layouts = {}
    for relation in relations:
        means, features = [], []
        degree = None
        for i, c in enumerate(infos):
            if c.kind != C.NBR_BLOCK or c.relation != relation:
                continue
            if c.stat == "log1p_degree":
                degree = i
            else:
                means.append(i)
                features.append(where[c.base_feature])
        layouts[relation] = RelationLayout(
            relation, np.array(means, dtype=int), np.array(features, dtype=int), degree
        )
    return layouts


def check_layout(
    x: np.ndarray,
    targets: np.ndarray,
    own: np.ndarray,
    adjacency: sp.csr_matrix,
    layout: RelationLayout,
    tolerance: float = 1e-6,
) -> None:
    """The model's neighbour columns must be what the graph and own features
    give, or a removal would change something other than one neighbour."""
    degree = np.asarray(adjacency[targets].sum(axis=1)).ravel()
    if layout.degree_position is not None:
        gap = np.abs(x[:, layout.degree_position] - np.log1p(degree)).max()
        if gap > tolerance:
            raise ExplainError(f"{layout.relation}: degree column is off by {gap:.3g}")
    if len(layout.mean_positions):
        sums = adjacency[targets] @ own[:, layout.feature_index]
        means = np.divide(
            sums, degree[:, None], out=np.zeros_like(sums, dtype=float),
            where=degree[:, None] > 0,
        )  # fmt: skip
        scale = np.maximum(1.0, np.abs(means))
        gap = (np.abs(x[:, layout.mean_positions] - means) / scale).max()
        if gap > tolerance:
            raise ExplainError(
                f"{layout.relation}: neighbour-mean columns differ from the graph and "
                f"own features by {gap:.3g}, so the model's rows are not lined up"
            )


def rows_without_each_neighbour(
    x_row: np.ndarray,
    neighbours: np.ndarray,
    own: np.ndarray,
    layout: RelationLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """One target's model row, once per neighbour with that neighbour removed:
    (a) mean only, with the degree column held at its original value; (b) the
    whole removal, as build_nbr_block would compute it on the reduced graph."""
    d = len(neighbours)
    used = own[neighbours][:, layout.feature_index]
    if d > 1:
        after = (used.sum(axis=0, keepdims=True) - used) / (d - 1)
    else:
        after = np.zeros_like(used, dtype=float)
    mean_only = np.tile(x_row, (d, 1))
    mean_only[:, layout.mean_positions] = after
    whole = mean_only.copy()
    if layout.degree_position is not None:
        whole[:, layout.degree_position] = np.log1p(d - 1)
    return mean_only, whole


def training_ranges(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each model column's training minimum and maximum."""
    return np.nanmin(x_train, axis=0), np.nanmax(x_train, axis=0)


def out_of_range(
    rows: np.ndarray, positions: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Whether a row has a value outside the training range in `positions`. One
    per-column version of "the model never saw this input": a row can be in range
    in every column and still be an unusual combination."""
    part = rows[:, positions]
    return ((part < low[positions] - RANGE_SLACK)
            | (part > high[positions] + RANGE_SLACK)).any(axis=1)  # fmt: skip


def _sigmoid(margin: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-margin))


def _influence_frame(
    pending: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    layout: RelationLayout,
    base: np.ndarray,
    target_ids: np.ndarray,
    margin_fn: Margin,
    ranges: tuple[np.ndarray, np.ndarray],
) -> pd.DataFrame:
    """Score a batch of (row, neighbours, mean-only rows, whole-removal rows)."""
    mean_only = np.vstack([p[2] for p in pending])
    whole = np.vstack([p[3] for p in pending])
    margin_mean = margin_fn(mean_only)
    has_degree = layout.degree_position is not None
    margin_whole = margin_fn(whole) if has_degree else margin_mean
    rows = np.concatenate([np.full(len(p[1]), p[0]) for p in pending])
    before = base[rows]
    prob = _sigmoid(before)
    degrees = np.concatenate([np.full(len(p[1]), len(p[1])) for p in pending])
    return pd.DataFrame(
        {
            "target": target_ids[rows].astype(np.int32),
            "neighbour": np.concatenate([p[1] for p in pending]).astype(np.int32),
            "relation": layout.relation,
            "degree": degrees.astype(np.int32),
            "delta_margin": (before - margin_whole).astype(np.float32),
            "delta_prob": (prob - _sigmoid(margin_whole)).astype(np.float32),
            "delta_prob_mean": (prob - _sigmoid(margin_mean)).astype(np.float32),
            "out_of_range": out_of_range(whole, layout.positions, *ranges),
        }
    )


def neighbour_influence(
    margin_fn: Margin,
    x: np.ndarray,
    target_ids: np.ndarray,
    own: np.ndarray,
    adjacency: Mapping[str, sp.csr_matrix],
    layouts: Mapping[str, RelationLayout],
    ranges: tuple[np.ndarray, np.ndarray],
    batch_rows: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """For every target (a row of `x`, node `target_ids`) and neighbour in each
    relation, how far the target's score falls when that neighbour is removed.
    Positive means the neighbour pushed the score towards fraud.

    `delta_prob` and `delta_margin` are the whole removal; `delta_prob_mean` is
    the mean-only part (who the neighbour is), and the rest is the degree part
    (how many neighbours there are). Yields chunks of one row per (target,
    neighbour, relation). Row-level: never commit it."""
    base = margin_fn(x)
    for relation, layout in layouts.items():
        a = adjacency[relation]
        check_layout(x, target_ids, own, a, layout)
        pending: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        held = 0
        for row, node in enumerate(target_ids):
            neighbours = a.indices[a.indptr[node] : a.indptr[node + 1]]
            if len(neighbours) == 0:
                continue
            mean_only, whole = rows_without_each_neighbour(
                x[row], neighbours, own, layout
            )
            pending.append((row, neighbours, mean_only, whole))
            held += len(neighbours)
            if held >= batch_rows:
                yield _influence_frame(
                    pending, layout, base, target_ids, margin_fn, ranges
                )
                pending, held = [], 0
        if pending:
            yield _influence_frame(pending, layout, base, target_ids, margin_fn, ranges)


def neighbour_shapley(
    margin_fn: Margin,
    x_row: np.ndarray,
    neighbours: np.ndarray,
    own: np.ndarray,
    layout: RelationLayout,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Sampled Shapley values of one target's neighbours in one relation: each
    coalition's mean and degree are what build_nbr_block gives on that subset.
    Antithetic permutations, so a pair of orders covers each neighbour's early
    and late positions. In margin units; the values sum to f(all) - f(none)."""
    d = len(neighbours)
    used = own[neighbours][:, layout.feature_index].astype(float)
    rng = np.random.default_rng(seed)
    phi = np.zeros(d)
    orders: list[np.ndarray] = []
    for _ in range((n_permutations + 1) // 2):
        order = rng.permutation(d)
        orders.extend([order, order[::-1]])
    for order in orders:
        rows = np.tile(x_row, (d + 1, 1))
        running = np.cumsum(used[order], axis=0)
        sizes = np.arange(1, d + 1)[:, None]
        rows[0, layout.mean_positions] = 0.0
        rows[1:, layout.mean_positions] = running / sizes
        if layout.degree_position is not None:
            rows[:, layout.degree_position] = np.log1p(np.arange(d + 1))
        values = margin_fn(rows)
        phi[order] += np.diff(values)
    return phi / len(orders)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation; NaN when either side is constant."""
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def aggregate_influence(
    influence: pd.DataFrame, flagged_targets: np.ndarray | None = None
) -> pd.DataFrame:
    """One row per (neighbour, relation), so hubs do not win on count alone: the
    sum and mean per target of |influence|, the number of targets, the signed
    sum in each direction, and the mean of |influence| x degree, which undoes the
    1/degree dilution under mean aggregation. Row-level: never commit it."""
    work = influence[["neighbour", "relation", "target", "degree", "delta_prob"]].copy()
    work["abs_"] = work["delta_prob"].abs().astype(float)
    work["towards_fraud"] = work["delta_prob"].clip(lower=0).astype(float)
    work["away_from_fraud"] = (-work["delta_prob"]).clip(lower=0).astype(float)
    work["abs_x_degree"] = work["abs_"] * work["degree"]
    grouped = work.groupby(["neighbour", "relation"], sort=False, observed=True)
    out = grouped.agg(
        n_targets=("target", "size"),
        sum_abs=("abs_", "sum"),
        mean_abs=("abs_", "mean"),
        mean_abs_x_degree=("abs_x_degree", "mean"),
        towards_fraud=("towards_fraud", "sum"),
        away_from_fraud=("away_from_fraud", "sum"),
    ).reset_index()
    if flagged_targets is not None:
        sub = work[work["target"].isin(flagged_targets)]
        flagged = sub.groupby(["neighbour", "relation"], observed=True)["abs_"].sum()
        out = out.merge(
            flagged.rename("sum_abs_flagged").reset_index(),
            on=["neighbour", "relation"],
            how="left",
        )
        out["sum_abs_flagged"] = out["sum_abs_flagged"].fillna(0.0)
    return out.sort_values("sum_abs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Neighbour influence through a GNN encoder
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EncoderWeights:
    """One relation's 2-layer GCN encoder (graph/ssl.py) and PCA, as arrays. w1 and
    w2 are torch Linear weights, (out, in); s1 and s2 are the PReLU slopes."""

    w1: np.ndarray
    b1: np.ndarray
    s1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    s2: np.ndarray
    pca_components: np.ndarray  # (k, embedding size)
    pca_mean: np.ndarray


def _prelu(h: np.ndarray, slope: np.ndarray) -> np.ndarray:
    return np.where(h >= 0, h, slope * h)


def _gcn_degree(a: sp.csr_matrix) -> np.ndarray:
    """The degree in A + I, which GCN normalisation uses."""
    return np.asarray(a.sum(axis=1), dtype=float).ravel() + 1.0


def _first_layer(
    enc: EncoderWeights, x: np.ndarray, a: sp.csr_matrix
) -> tuple[np.ndarray, np.ndarray]:
    """W1 x per node, and the first layer's value before its activation."""
    root = np.sqrt(_gcn_degree(a))
    a_hat = (
        sp.diags(1 / root)
        @ (a + sp.identity(a.shape[0], format="csr"))
        @ sp.diags(1 / root)
    )
    x1 = x @ enc.w1.T
    return x1, a_hat @ x1 + enc.b1


def encoder_forward(enc: EncoderWeights, x: np.ndarray, a: sp.csr_matrix) -> np.ndarray:
    """Every node's embedding: graph/ssl.py's GCNEncoder in numpy."""
    root = np.sqrt(_gcn_degree(a))
    a_hat = (
        sp.diags(1 / root)
        @ (a + sp.identity(a.shape[0], format="csr"))
        @ sp.diags(1 / root)
    )
    h = _prelu(a_hat @ (x @ enc.w1.T) + enc.b1, enc.s1)
    return _prelu(a_hat @ (h @ enc.w2.T) + enc.b2, enc.s2)


def embeddings_without_each_neighbour(
    enc: EncoderWeights,
    x1: np.ndarray,
    pre1: np.ndarray,
    a: sp.csr_matrix,
    degree: np.ndarray,
    target: int,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The target's embedding, then once per candidate with that edge removed and
    the encoder's weights held fixed.

    Removing edge (i, j) lowers the GCN degree of i and j, so it also changes the
    first layer of every neighbour of i (their weights on i, and on j if adjacent),
    which is why this is not a small update to i alone. Equal to encoder_forward on
    the reduced graph, at the cost of one neighbourhood rather than the whole graph."""
    neighbours = a.indices[a.indptr[target] : a.indptr[target + 1]]
    d_i = degree[target]
    d_k = degree[neighbours]

    def embed(
        k: np.ndarray, pre_k: np.ndarray, pre_i: np.ndarray, d: float
    ) -> np.ndarray:
        weight = (1 / np.sqrt(d * degree[k]))[:, None]
        total = ((_prelu(pre_k, enc.s1) @ enc.w2.T) * weight).sum(axis=0)
        return _prelu(total + (_prelu(pre_i, enc.s1) @ enc.w2.T) / d + enc.b2, enc.s2)

    base = embed(neighbours, pre1[neighbours], pre1[target], d_i)
    near = a[neighbours][:, candidates].toarray().astype(float)
    out = np.empty((len(candidates), len(base)))
    for c, j in enumerate(candidates):
        keep = neighbours != j
        k = neighbours[keep]
        d_new, d_j, d_j_new = d_i - 1, degree[j], degree[j] - 1
        roots = d_k[keep]
        change_i = 1 / np.sqrt(roots * d_new) - 1 / np.sqrt(roots * d_i)
        change_j = near[keep, c] * (
            1 / np.sqrt(roots * d_j_new) - 1 / np.sqrt(roots * d_j)
        )
        pre_k = pre1[k] + change_i[:, None] * x1[target] + change_j[:, None] * x1[j]
        pre_i = (
            (x1[k] / np.sqrt(d_new * roots)[:, None]).sum(axis=0)
            + x1[target] / d_new
            + enc.b1
        )
        out[c] = embed(k, pre_k, pre_i, d_new)
    return base, out


def encoder_layout(
    infos: Sequence[ColumnInfo], relation: str
) -> tuple[np.ndarray, np.ndarray]:
    """Where one relation's kept PCA columns sit in the model's input, and which
    component each one is. Feature selection may have dropped some."""
    positions, components = [], []
    for i, c in enumerate(infos):
        if c.kind == C.GNNSSL_BLOCK and c.relation == relation:
            positions.append(i)
            components.append(int(c.block_column.rsplit("_pc", 1)[1]) - 1)
    return np.array(positions, dtype=int), np.array(components, dtype=int)


def encoder_influence(
    margin_fn: Margin,
    x: np.ndarray,
    target_ids: np.ndarray,
    own_scaled: np.ndarray,
    a: sp.csr_matrix,
    enc: EncoderWeights,
    infos: Sequence[ColumnInfo],
    relation: str,
    ranges: tuple[np.ndarray, np.ndarray],
    max_neighbours: int,
    seed: int,
    tolerance: float = 1e-3,
) -> tuple[pd.DataFrame, float]:
    """For each target, how far its score falls when one neighbour is removed from
    one relation and its PCA columns for that relation are recomputed, with the
    encoder and PCA held fixed (nothing is retrained). Targets with more than
    `max_neighbours` neighbours get a seeded random sample of them, so a neighbour's
    total counts only the removals that were scored. Returns the table and the
    largest gap between the recomputed PCA columns and the model's own, which stops
    the run above `tolerance`, so a removal changes nothing but that one neighbour."""
    positions, components = encoder_layout(infos, relation)
    if len(positions) == 0:
        raise ExplainError(f"{relation}: the model has no embedding columns")
    degree = _gcn_degree(a)
    x1, pre1 = _first_layer(enc, own_scaled, a)
    z = encoder_forward(enc, own_scaled, a)[target_ids]
    pcs = (z - enc.pca_mean) @ enc.pca_components.T
    gap = float(np.abs(pcs[:, components] - x[:, positions]).max())
    if gap > tolerance:
        raise ExplainError(
            f"{relation}: PCA columns rebuilt from the encoder differ from the model's "
            f"input by {gap:.3g}, so its rows are not lined up"
        )
    rng = np.random.default_rng(seed)
    rows, targets, neighbours_out, degrees, n_scored = [], [], [], [], []
    for row, node in enumerate(target_ids):
        neighbours = a.indices[a.indptr[node] : a.indptr[node + 1]]
        if len(neighbours) == 0:
            continue
        chosen = (
            np.sort(rng.choice(neighbours, max_neighbours, replace=False))
            if len(neighbours) > max_neighbours
            else neighbours
        )
        base, without = embeddings_without_each_neighbour(
            enc, x1, pre1, a, degree, int(node), chosen
        )
        changed = np.tile(x[row], (len(chosen), 1))
        # A difference from the same calculation, so rounding in the saved columns
        # cannot look like influence.
        moved = (without - base) @ enc.pca_components.T
        changed[:, positions] = x[row, positions] + moved[:, components]
        rows.append(changed)
        targets.append(np.full(len(chosen), row))
        neighbours_out.append(chosen)
        degrees.append(np.full(len(chosen), len(neighbours)))
        n_scored.append(np.full(len(chosen), len(chosen)))
    changed = np.vstack(rows)
    which = np.concatenate(targets)
    before = margin_fn(x)[which]
    after = margin_fn(changed)
    table = pd.DataFrame(
        {
            "target": target_ids[which].astype(np.int32),
            "neighbour": np.concatenate(neighbours_out).astype(np.int32),
            "relation": relation,
            "degree": np.concatenate(degrees).astype(np.int32),
            "n_scored": np.concatenate(n_scored).astype(np.int32),
            "delta_margin": (before - after).astype(np.float32),
            "delta_prob": (_sigmoid(before) - _sigmoid(after)).astype(np.float32),
            # An encoder has no mean and degree parts to separate.
            "delta_prob_mean": np.full(len(changed), np.nan, dtype=np.float32),
            "out_of_range": out_of_range(changed, positions, *ranges),
        }
    )
    return table, gap
