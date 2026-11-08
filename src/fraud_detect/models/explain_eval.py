"""Tests of the explanations: group Shapley, retrain-after-removal, sanity checks.

Additivity only shows the arithmetic is right, so these ask whether the
attributions mean anything:

- group_shapley: the exact Shapley value of each block treated as one player.
  It differs from the sum of per-column SHAP values by how the two games split
  cross-block interactions.
- roar_lite: retrain without the top-k columns and compare the fall in held-out
  PR-AUC with removing k random columns (Hooker et al., NeurIPS 2019). It retrains
  so that the model never sees inputs it was not trained on. The ranking of each
  fold comes from a model fit without that fold, on that model's own fit rows.
- randomisation_check: refit on permuted training labels; the ranking should
  lose its structure (Adebayo et al., NeurIPS 2018).

All three take callables, so they run on toy models in tests and on df-analyze's
models in scripts/dfa/shap_explain.py. Everything that uses labels here is inside
the training set's saved CV folds (AGENTS.md invariant 4).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detect.models.explain import Margin, spearman

Fit = Callable[[pd.DataFrame, np.ndarray, int], Any]  # X, y, seed -> fitted model
Prob = Callable[[Any, pd.DataFrame], np.ndarray]  # model, X -> P(fraud)
# model, the rows to explain -> mean |attribution| per column of those rows
Importance = Callable[[Any, pd.DataFrame], pd.Series]


# --------------------------------------------------------------------------
# Exact group Shapley
# --------------------------------------------------------------------------
def group_shapley(
    margin_fn: Margin,
    x: np.ndarray,
    background: np.ndarray,
    players: Mapping[str, Sequence[int]],
    chunk_rows: int = 200,
) -> tuple[np.ndarray, list[str], float]:
    """Exact interventional Shapley values with blocks of columns as players.

    v_x(S) is the model's mean margin over the background with the columns of
    players in S set to x's values. Evaluates all 2^p coalitions, so keep p small
    (about 5). Returns (n, p) local values, the player names, and the base value;
    each row's values sum to f(x) - base."""
    names = list(players)
    p = len(names)
    if p > 12:
        raise ValueError(f"{p} players means {2**p} coalitions; group the columns")
    m = len(background)
    weights = np.array(
        [
            math.factorial(s) * math.factorial(p - s - 1) / math.factorial(p)
            for s in range(p)
        ]
    )
    phi = np.zeros((len(x), p))
    for start in range(0, len(x), chunk_rows):
        part = x[start : start + chunk_rows]
        n = len(part)
        values = np.zeros((2**p, n))
        for mask in range(2**p):
            rows = np.tile(background, (n, 1))
            for i in range(p):
                if mask >> i & 1:
                    cols = list(players[names[i]])
                    rows[:, cols] = np.repeat(part[:, cols], m, axis=0)
            values[mask] = margin_fn(rows).reshape(n, m).mean(axis=1)
        for i in range(p):
            others = [j for j in range(p) if j != i]
            for size in range(p):
                for subset in combinations(others, size):
                    mask = sum(1 << j for j in subset)
                    phi[start : start + n, i] += weights[size] * (
                        values[mask | (1 << i)] - values[mask]
                    )
    base = float(margin_fn(background).mean())
    return phi, names, base


# --------------------------------------------------------------------------
# Retrain after removal
# --------------------------------------------------------------------------
def roar_lite(
    x: pd.DataFrame,
    y: np.ndarray,
    fold: np.ndarray,
    fit: Fit,
    prob: Prob,
    importance: Importance,
    players: Mapping[str, Sequence[str]],
    groups: Mapping[str, Sequence[str]] | None = None,
    top_ks: Sequence[int] = (1, 2, 4, 8),
    n_random: int = 5,
    seed: int = 0,
    only_folds: Sequence[int] | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> pd.DataFrame:
    """One row per (fold, removal): held-out PR-AUC after retraining without some
    columns, and its fall from that fold's baseline. Removals: each player, each
    extra group, the top k columns by attribution, and k random columns.

    The ranking uses each fold's baseline model and its own fit rows, so the
    held-out fold plays no part in choosing what is removed. Hyperparameters stay
    fixed by `fit`. Random draws use the same seeds in every fold. `only_folds`
    runs some folds, so a caller can save each as it finishes."""
    y = np.asarray(y, dtype=int)
    rows: list[dict[str, Any]] = []
    columns = list(x.columns)
    for k_fold in np.unique(fold) if only_folds is None else only_folds:
        held = fold == k_fold
        x_fit, y_fit = x[~held], y[~held]
        x_val, y_val = x[held], y[held]
        baseline_model = fit(x_fit, y_fit, seed)
        baseline = float(average_precision_score(y_val, prob(baseline_model, x_val)))
        ranking = importance(baseline_model, x_fit).reindex(columns).fillna(0.0)
        order = list(ranking.sort_values(ascending=False).index)
        removals: list[tuple[str, str, int, int, list[str]]] = []
        for name, cols in {**players, **(groups or {})}.items():
            if len(cols) < len(columns):
                kind = "block" if name in players else "group"
                removals.append((kind, name, len(cols), 0, list(cols)))
        for k in top_ks:
            if k >= len(columns):
                continue
            removals.append(("top_k", f"top {k}", k, 0, order[:k]))
            for draw in range(n_random):
                rng = np.random.default_rng([seed, k, draw])
                picked = list(rng.choice(columns, size=k, replace=False))
                removals.append(("random", f"random {k}", k, draw, picked))
        rows.append(
            {
                "fold": int(k_fold),
                "kind": "baseline",
                "name": "baseline",
                "k": 0,
                "draw": 0,
                "pr_auc": baseline,
                "drop": 0.0,
            }
        )
        for kind, name, k, draw, drop in removals:
            removed = set(drop)
            keep = [c for c in columns if c not in removed]
            model = fit(x_fit[keep], y_fit, seed)
            value = float(average_precision_score(y_val, prob(model, x_val[keep])))
            rows.append(
                {
                    "fold": int(k_fold),
                    "kind": kind,
                    "name": name,
                    "k": k,
                    "draw": draw,
                    "pr_auc": value,
                    "drop": baseline - value,
                }
            )
        log(f"fold {k_fold}: baseline PR-AUC {baseline:.4f}, {len(removals)} removals")
    return pd.DataFrame(rows)


def roar_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over folds of the fall in PR-AUC for each removal. Random draws
    are averaged within a fold first, so the SD is between folds in every row."""
    per_fold = (
        table.groupby(["kind", "name", "k", "fold"], sort=False, observed=True)["drop"]
        .mean()
        .reset_index()
    )
    out = (
        per_fold.groupby(["kind", "name", "k"], sort=False, observed=True)["drop"]
        .agg(mean_drop="mean", sd_drop=lambda s: s.std(ddof=1), n_folds="size")
        .reset_index()
    )
    return out


def shap_beats_random(summary: pd.DataFrame) -> pd.DataFrame:
    """For each k, whether removing the top k columns hurts more than k random."""
    top = summary[summary["kind"] == "top_k"].set_index("k")["mean_drop"]
    random = summary[summary["kind"] == "random"].set_index("k")["mean_drop"]
    frame = pd.DataFrame({"top_k_drop": top, "random_drop": random}).reset_index()
    frame["gap"] = frame["top_k_drop"] - frame["random_drop"]
    frame["top_k_hurts_more"] = frame["gap"] > 0
    return frame


# --------------------------------------------------------------------------
# Model-randomisation sanity check
# --------------------------------------------------------------------------
def randomisation_check(
    x: pd.DataFrame,
    y: np.ndarray,
    fold: np.ndarray,
    fit: Fit,
    prob: Prob,
    importance: Importance,
    n_permutations: int = 3,
    seed: int = 0,
    max_rows: int = 1000,
) -> pd.DataFrame:
    """Fit on all but the first fold with the true labels and with permuted ones,
    then attribute on the held-out fold. Attributions on the permuted models
    should not resemble the true model's: low Spearman with its ranking and little
    held-out PR-AUC above the prevalence. A tree on noise labels still splits, so
    its total attribution shrinks without reaching zero. Training rows only."""
    y = np.asarray(y, dtype=int)
    held = fold == np.unique(fold)[0]
    x_fit, x_val = x[~held], x[held]
    y_fit, y_val = y[~held], y[held]
    explain = x_val.iloc[:max_rows]
    true_model = fit(x_fit, y_fit, seed)
    true_imp = importance(true_model, explain).reindex(x.columns).fillna(0.0)
    rows = [
        {
            "run": "true labels",
            "pr_auc": float(average_precision_score(y_val, prob(true_model, x_val))),
            "total_mean_abs": float(true_imp.sum()),
            "spearman_with_true": 1.0,
            "prevalence": float(y_val.mean()),
        }
    ]
    for i in range(n_permutations):
        rng = np.random.default_rng([seed, i])
        model = fit(x_fit, rng.permutation(y_fit), seed)
        imp = importance(model, explain).reindex(x.columns).fillna(0.0)
        rows.append(
            {
                "run": f"permuted labels {i}",
                "pr_auc": float(average_precision_score(y_val, prob(model, x_val))),
                "total_mean_abs": float(imp.sum()),
                "spearman_with_true": spearman(true_imp.to_numpy(), imp.to_numpy()),
                "prevalence": float(y_val.mean()),
            }
        )
    return pd.DataFrame(rows)
