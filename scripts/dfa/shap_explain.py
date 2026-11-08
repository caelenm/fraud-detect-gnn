"""Explain an exported tree Model A: SHAP, its quality tests and neighbour influence.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT, with shap added for the run
only, so df-analyze's lock file stays as it is. From anywhere:

    uv run --directory "$DF_ANALYZE_DIR" --python '>=3.13.11,<3.14' --with shap \
        python <repo>/scripts/dfa/shap_explain.py \
        --dataset amazon --feature-set m2_own_nbr_no19

It writes outputs/<dataset>[_repeat<k>]/<feature set>/explain/, one output per
section, each written atomically and skipped on a later run unless --force, so
stopping and rerunning loses at most the section in progress. Those files are
row-level and are never committed. scripts/explain_report.py summarises them.

Sections (--sections picks some):

- shap: interventional TreeSHAP of every test node against a seeded, stratified
  background of training rows, in log-odds, with the implementation check.
- sensitivity: other background seeds and sizes, and path-dependent TreeSHAP.
- interactions: path-dependent interaction values, the only mode shap has them in.
- group_shapley: exact Shapley values of the feature blocks as players.
- rashomon: the same SHAP for other tuned models within Model A's CV interval.
- roar: retrain after removal, inside the training set's saved folds.
- roar_extended: the same with larger removals and all graph columns together.
- randomisation: refit on permuted labels, inside the training folds.
- seeds: refit Model A with other seeds; SHAP on a sample of test nodes.
- neighbours: influence of each neighbour, for models with `nbr` columns.
- neighbours_encoder: the same through the DGI encoder, for models with `gnnssl`
  columns. Only when named, after scripts/gnn_encoder_export.py.
- cards: a few worked test nodes for the group-only page.

Nothing here tunes or selects anything from the test set. Test nodes are only
explained, and the threshold comes from out-of-fold training predictions.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from model_io import (
    REPO_ROOT,
    load_model_a,
    read_model_meta,
    read_table,
    read_target,
    setup_paths,
)

setup_paths()

from fraud_detect import columns as C  # noqa: E402
from fraud_detect.config import (  # noqa: E402
    DEFAULT_CONFIG,
    Paths,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.features.graph import adjacency  # noqa: E402
from fraud_detect.models import explain as X  # noqa: E402
from fraud_detect.models import explain_eval as E  # noqa: E402
from fraud_detect.runlog import git_commit  # noqa: E402
from fraud_detect.runstate import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)

INTERVENTIONAL = "interventional"
PATH_DEPENDENT = "tree_path_dependent"
TREE_MODELS = {"catboost": "catboost", "lgbm": "lgbm"}
MODEL_MODULES = {"CatBoostClassifier": "catboost", "LightGBMClassifier": "lgbm"}
SEED_ARGUMENT = {
    "CatBoostClassifier": "random_seed",
    "LightGBMClassifier": "random_state",
}
SHAP_COLUMN = "shap__"
GROUP_COLUMN = "group__"
# Sizes: fixed here and recorded with every output.
N_TRAIN_SAMPLE = 2000
N_INTERACTION_ROWS = 500
N_INTERACTION_TOP = 5
N_GROUP_ROWS = 1000
N_GROUP_BACKGROUND = 200
N_SAMPLE_ROWS = 1000
N_SAMPLE_BACKGROUND = 500
N_RANKING_ROWS = 1000
N_RANKING_BACKGROUND = 200
N_REDUNDANCY_TARGETS = 200
MAX_REDUNDANCY_DEGREE = 30
N_REDUNDANCY_PERMUTATIONS = 64
N_CARDS = {"true_positive": 3, "false_positive": 3, "false_negative": 2}
N_CARD_FEATURES = 10
N_CARD_NEIGHBOURS = 5
N_CARD_NEAREST = 3


# --------------------------------------------------------------------------
# The run's inputs
# --------------------------------------------------------------------------
@dataclass
class Context:
    args: argparse.Namespace
    config: dict[str, Any]
    paths: Paths
    meta: dict[str, Any]
    model: Any
    columns: list[str]
    infos: list[X.ColumnInfo]
    players: dict[str, list[int]]
    groups: dict[str, list[int]]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_ids: pd.DataFrame
    test_ids: pd.DataFrame
    fold: np.ndarray
    threshold: float
    out: Path

    @property
    def background_rows(self) -> np.ndarray:
        return X.stratified_sample(self.y_train, self.args.background, self.args.seed)

    def background(self, n: int | None = None, seed: int | None = None) -> np.ndarray:
        rows = X.stratified_sample(
            self.y_train,
            self.args.background if n is None else n,
            self.args.seed if seed is None else seed,
        )
        return self.x_train.to_numpy(dtype=float)[rows]


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_context(args: argparse.Namespace) -> Context:
    config = load_config(args.config)
    config["split"]["n_repeats"] = max(int(config["split"]["n_repeats"]), args.repeat + 1)
    config["split"]["repeat"] = args.repeat
    config = with_dataset(config, args.dataset)
    config["feature_set"] = args.feature_set
    p = get_paths(config)
    meta, warnings = read_model_meta(p.model_dir)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if meta["model"] not in TREE_MODELS:
        raise SystemExit(
            f"Model A of {args.feature_set} is {meta['model']}, not a LightGBM or "
            "CatBoost model. TreeSHAP does not apply: stop and ask."
        )
    results = repo_path(meta["results_dir"])
    columns = list(meta["input_columns"])
    x_train = read_table(results / "X_train_00.csv")
    x_test = read_table(results / "X_test_00.csv")
    y_train = read_target(results / "y_train_00.csv").to_numpy(dtype=int)
    y_test = read_target(results / "y_test_00.csv").to_numpy(dtype=int)
    train_ids = pd.read_csv(p.train_ids)
    test_ids = pd.read_csv(p.test_ids)
    if not np.array_equal(train_ids[C.LABEL].to_numpy(), y_train) or not np.array_equal(
        test_ids[C.LABEL].to_numpy(), y_test
    ):
        raise SystemExit("df-analyze's exported rows do not line up with the saved split")
    fold_of = pd.read_csv(p.cv_folds).set_index(C.NODE_ID)["fold"]
    info = json.loads(
        (p.df_analyze_input_dir / "feature_set.json").read_text(encoding="utf-8")
    )
    renames = {safe: c["block_column"] for safe, c in info["columns"].items()}
    infos = X.parse_columns(columns, renames)
    variants = config.get("feature_variants") or {}
    correlated = (
        X.correlated_group_columns(infos, variants.get("no36", []))
        if args.dataset == "amazon"
        else {}
    )
    out = p.feature_set_dir / X.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return Context(
        args=args,
        config=config,
        paths=p,
        meta=meta,
        model=load_model_a(p.model_dir, meta),
        columns=columns,
        infos=infos,
        players=X.players_of(infos),
        groups=correlated,
        x_train=x_train[columns],
        x_test=x_test[columns],
        y_train=y_train,
        y_test=y_test,
        train_ids=train_ids,
        test_ids=test_ids,
        fold=fold_of.loc[train_ids[C.NODE_ID].to_numpy()].to_numpy(dtype=int),
        threshold=float(meta["threshold"]),
        out=out,
    )


# --------------------------------------------------------------------------
# Models and SHAP
# --------------------------------------------------------------------------
def margin_function(model: Any) -> X.Margin:
    """The model's raw log-odds for a float array."""
    if hasattr(model, "booster_"):
        booster = model.booster_
        return lambda rows: np.asarray(booster.predict(rows, raw_score=True), dtype=float)
    return lambda rows: np.asarray(
        model.predict(rows, prediction_type="RawFormulaVal"), dtype=float
    )


def prob_of(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame))[:, 1]


def _matrix(values: Any) -> np.ndarray:
    """shap gives a binary classifier's values as (n, c), (n, c, 2) or a list."""
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values, dtype=float)
    return values[:, :, 1] if values.ndim == 3 else values


def _explain_chunk(
    model: Any, background: np.ndarray | None, part: np.ndarray
) -> tuple[np.ndarray, float]:
    if background is None:
        explainer = shap.TreeExplainer(
            model, feature_perturbation=PATH_DEPENDENT, model_output="raw"
        )
    else:
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.TreeExplainer(
            model, data=masker, feature_perturbation=INTERVENTIONAL, model_output="raw"
        )
        if explainer.data.shape[0] != len(background):
            kept = explainer.data.shape[0]
            raise X.ExplainError(f"shap kept {kept} of {len(background)} background rows")
    values = _matrix(explainer.shap_values(part, check_additivity=False))
    return values, float(np.ravel(explainer.expected_value)[-1])


def split_thresholds(model: Any) -> dict[int, np.ndarray]:
    """Each column's split thresholds, for LightGBM models (X.near_threshold)."""
    if not hasattr(model, "booster_"):
        return {}
    found: dict[int, set[float]] = {}
    stack = [t["tree_structure"] for t in model.booster_.dump_model()["tree_info"]]
    while stack:
        node = stack.pop()
        if "split_feature" in node:
            found.setdefault(int(node["split_feature"]), set()).add(
                float(node["threshold"])
            )
            stack.extend((node["left_child"], node["right_child"]))
    return {column: np.array(sorted(values)) for column, values in found.items()}


def shap_values(
    model: Any, rows: np.ndarray, background: np.ndarray | None, n_jobs: int
) -> tuple[np.ndarray, float]:
    """SHAP values and base value. background=None means path-dependent TreeSHAP;
    otherwise interventional against exactly that background."""
    chunks = np.array_split(rows, max(1, min(n_jobs * 2, len(rows))))
    done = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_explain_chunk)(model, background, part) for part in chunks
    )
    return np.vstack([d[0] for d in done]), done[0][1]


def checked_shap(
    model: Any, rows: np.ndarray, background: np.ndarray | None, n_jobs: int
) -> tuple[np.ndarray, float, dict[str, float]]:
    """SHAP values after the implementation check, with any row that misses set to
    NaN, so no summary can use it (see X.implementation_check). Background rows that
    shap could misroute are left out first, since they would spoil every row."""
    if background is not None:
        background = background[~X.near_threshold(background, split_thresholds(model))]
    values, base = shap_values(model, rows, background, n_jobs)
    margin = margin_function(model)
    bad, check = X.implementation_check(
        values, base, margin(rows), None if background is None else margin(background)
    )
    check["n_background_used"] = 0 if background is None else len(background)
    values[bad] = np.nan
    return values, base, check


def mean_abs(values: np.ndarray, columns: list[str]) -> pd.Series:
    """Mean |SHAP| per column, over the rows that passed the implementation check."""
    return pd.Series(np.nanmean(np.abs(values), axis=0), index=columns)


def model_class(name: str) -> type:
    module = importlib.import_module(f"df_analyze.models.{MODEL_MODULES[name]}")
    return getattr(module, name)


def parse_params(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        return params
    try:
        return json.loads(params)
    except json.JSONDecodeError:
        return ast.literal_eval(params)


def make_fit(model_cls: str, params: dict[str, Any]) -> E.Fit:
    """Refit with df-analyze's own class and these tuned settings; only the seed
    changes, so the same hyperparameters are compared throughout. seed=None keeps
    whatever seed df-analyze's class uses."""
    cls = model_class(model_cls)
    seed_argument = SEED_ARGUMENT[model_cls]

    def fit(x: pd.DataFrame, y: np.ndarray, seed: int | None) -> Any:
        wrapper = cls()
        wrapper.refit_tuned(
            X=x,
            y=pd.Series(np.asarray(y), name=C.TARGET, index=x.index),
            tuned_args={**params, **({} if seed is None else {seed_argument: seed})},
        )
        return wrapper.tuned_model

    return fit


def ranking_importance(n_jobs: int, seed: int) -> E.Importance:
    """Mean |interventional SHAP| on a sample of the given rows, with a background
    from the same rows, so a ranking only ever sees the rows it is handed."""

    def importance(model: Any, rows: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(seed)
        values = rows.to_numpy(dtype=float)
        picked = rng.choice(len(values), min(N_RANKING_ROWS, len(values)), replace=False)
        chosen = rng.choice(
            len(values), min(N_RANKING_BACKGROUND, len(values)), replace=False
        )
        attributions, _, _ = checked_shap(model, values[picked], values[chosen], n_jobs)
        return mean_abs(attributions, list(rows.columns))

    return importance


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def section_shap(ctx: Context) -> dict[str, Any]:
    n_jobs = ctx.args.n_jobs
    background = None if ctx.args.path_dependent else ctx.background()
    x = ctx.x_test.to_numpy(dtype=float)
    margin = margin_function(ctx.model)
    gap = np.abs(1 / (1 + np.exp(-margin(x))) - prob_of(ctx.model, ctx.x_test)).max()
    if gap > 1e-6:
        raise X.ExplainError(
            f"Raw margins and predicted probabilities differ by {gap:.3g}"
        )
    values, base, check = checked_shap(ctx.model, x, background, n_jobs)
    frame = pd.DataFrame(values, columns=[SHAP_COLUMN + c for c in ctx.columns])
    frame.insert(0, "margin", margin(x))
    frame.insert(0, "prob", prob_of(ctx.model, ctx.x_test))
    frame.insert(0, C.LABEL, ctx.y_test)
    frame.insert(0, C.GROUP_ID, ctx.test_ids[C.GROUP_ID].to_numpy())
    frame.insert(0, C.NODE_ID, ctx.test_ids[C.NODE_ID].to_numpy())
    atomic_write_parquet(ctx.out / "shap_test.parquet", frame)
    # Training rows for the nearest-node comparison on the case cards.
    picked = X.stratified_sample(ctx.y_train, N_TRAIN_SAMPLE, ctx.args.seed + 1)
    rows = ctx.x_train.to_numpy(dtype=float)[picked]
    train_values, _, _ = checked_shap(ctx.model, rows, background, n_jobs)
    train = pd.DataFrame(train_values, columns=[SHAP_COLUMN + c for c in ctx.columns])
    train.insert(0, "prob", prob_of(ctx.model, ctx.x_train.iloc[picked]))
    train.insert(0, C.LABEL, ctx.y_train[picked])
    train.insert(0, C.NODE_ID, ctx.train_ids[C.NODE_ID].to_numpy()[picked])
    atomic_write_parquet(ctx.out / "shap_train_sample.parquet", train)
    return {
        "feature_perturbation": PATH_DEPENDENT if background is None else INTERVENTIONAL,
        "model_output": "raw (log-odds)",
        "n_background": check["n_background_used"],
        "n_background_drawn": 0 if background is None else len(background),
        "background_seed": ctx.args.seed,
        "background": "stratified sample of training rows",
        "n_test_rows": len(x),
        "base_value": base,
        "implementation_check": check,
    }


def section_sensitivity(ctx: Context) -> dict[str, Any]:
    main = pd.read_parquet(ctx.out / "shap_test.parquet")
    values = main[[SHAP_COLUMN + c for c in ctx.columns]].to_numpy()
    reference = mean_abs(values, ctx.columns)
    x = ctx.x_test.to_numpy(dtype=float)
    seed = ctx.args.seed
    runs = [
        (
            f"background seed {seed + 100}",
            INTERVENTIONAL,
            ctx.background(seed=seed + 100),
        ),
        (
            f"background seed {seed + 200}",
            INTERVENTIONAL,
            ctx.background(seed=seed + 200),
        ),
        ("background size 200", INTERVENTIONAL, ctx.background(n=200)),
        ("path-dependent TreeSHAP", PATH_DEPENDENT, None),
    ]
    rows = []
    wide = {"interventional (main)": reference}
    for name, mode, background in runs:
        other, base, check = checked_shap(ctx.model, x, background, ctx.args.n_jobs)
        importance = mean_abs(other, ctx.columns)
        wide[name] = importance
        rows.append(
            {
                "comparison": name,
                "feature_perturbation": mode,
                "n_background": check["n_background_used"],
                "base_value": base,
                "max_additivity_error": check["max_additivity_error"],
                "n_rows_excluded": check["n_rows_excluded"],
                **X.rank_agreement(reference, importance),
            }
        )
    atomic_write_csv(ctx.out / "sensitivity.csv", pd.DataFrame(rows))
    table = pd.DataFrame(wide).rename_axis("column").reset_index()
    atomic_write_csv(ctx.out / "sensitivity_importance.csv", table)
    return {"n_comparisons": len(rows)}


def section_interactions(ctx: Context) -> dict[str, Any]:
    picked = X.stratified_sample(ctx.y_test, N_INTERACTION_ROWS, ctx.args.seed)
    rows = ctx.x_test.iloc[picked]
    explainer = shap.TreeExplainer(
        ctx.model, feature_perturbation=PATH_DEPENDENT, model_output="raw"
    )
    values = explainer.shap_interaction_values(rows.to_numpy(dtype=float))
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 4:
        values = values[:, :, :, 1]
    main = pd.read_parquet(ctx.out / "shap_test.parquet")
    importance = mean_abs(
        main[[SHAP_COLUMN + c for c in ctx.columns]].to_numpy(), ctx.columns
    )
    top = list(importance.nlargest(N_INTERACTION_TOP).index)
    degree = [c.prepared for c in ctx.infos if c.player == X.DEGREE_PLAYER]
    chosen = list(dict.fromkeys(top + degree))
    where = {c: i for i, c in enumerate(ctx.columns)}
    by_name = {c.prepared: c for c in ctx.infos}
    table = []
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            # shap splits a pair's interaction equally between (a, b) and (b, a).
            pair = 2 * values[:, where[a], where[b]]
            kinds = {by_name[a].player == C.OWN_BLOCK, by_name[b].player == C.OWN_BLOCK}
            table.append(
                {
                    "a": a,
                    "b": b,
                    "mean_abs_interaction": float(np.abs(pair).mean()),
                    "own_x_graph": kinds == {True, False},
                    "n_rows": len(rows),
                    "feature_perturbation": PATH_DEPENDENT,
                }
            )
    frame = pd.DataFrame(table).sort_values("mean_abs_interaction", ascending=False)
    atomic_write_csv(ctx.out / "interactions.csv", frame)
    return {"n_rows": len(rows), "mode": PATH_DEPENDENT, "columns": chosen}


def section_group_shapley(ctx: Context) -> dict[str, Any]:
    picked = X.stratified_sample(ctx.y_test, N_GROUP_ROWS, ctx.args.seed)
    x = ctx.x_test.to_numpy(dtype=float)[picked]
    background = ctx.background(n=N_GROUP_BACKGROUND)
    margin = margin_function(ctx.model)
    phi, names, base = E.group_shapley(margin, x, background, ctx.players)
    worst = float(np.abs(phi.sum(axis=1) + base - margin(x)).max())
    if worst > 1e-6 * max(1.0, float(np.abs(margin(x)).max())):
        raise X.ExplainError(f"Group Shapley values miss the margin by {worst:.3g}")
    frame = pd.DataFrame(phi, columns=[GROUP_COLUMN + n for n in names])
    frame.insert(0, C.GROUP_ID, ctx.test_ids[C.GROUP_ID].to_numpy()[picked])
    frame.insert(0, C.NODE_ID, ctx.test_ids[C.NODE_ID].to_numpy()[picked])
    atomic_write_parquet(ctx.out / "group_shapley.parquet", frame)
    return {
        "players": {n: len(ctx.players[n]) for n in names},
        "n_coalitions": 2 ** len(names),
        "n_rows": len(x),
        "n_background": len(background),
        "base_value": base,
        "max_additivity_error": worst,
    }


def _model_key(row: Any) -> tuple[str, str, str]:
    return (str(row["model_cls"]), str(row["selection"]), str(row["embed_selector"]))


def section_rashomon(ctx: Context) -> dict[str, Any]:
    """Other tuned tree models whose shared-CV PR-AUC is within one SD (over folds)
    of Model A's, refit on the training set. They are not chosen from test data."""
    cv = pd.read_csv(
        ctx.paths.model_reports_dir / "model_selection_cv.csv", keep_default_na=False
    )
    cv = cv[(cv["settings"] == "tuned") & (cv["model_cls"].isin(MODEL_MODULES))]
    cv = cv.sort_values("pr_auc_mean", ascending=False)
    label = (
        "embed" if ctx.meta["selection"].startswith("embed") else ctx.meta["selection"]
    )
    a_rows = cv[
        (cv["model_cls"] == ctx.meta["model_cls"])
        & (cv["selection"] == label)
        & (cv["embed_selector"].astype(str) == str(ctx.meta.get("embed_selector") or ""))
    ]
    if len(a_rows) != 1:
        raise X.ExplainError("Model A is not a single row of model_selection_cv.csv")
    a = a_rows.iloc[0]
    floor = float(a["pr_auc_mean"]) - float(a["pr_auc_std"])
    others = cv[(cv["pr_auc_mean"] >= floor) & (cv.index != a.name)].head(2)
    results = repo_path(ctx.meta["results_dir"])
    predictions = json.loads(
        (results / "prediction_results_00.json").read_text(encoding="utf-8")
    )["predictions"]
    entries = {
        (
            str(e["model_cls"]),
            str(e["selection"]),
            str(e.get("embed_select_model") or ""),
        ): e
        for e in predictions
    }
    x_train = read_table(results / "X_train_00.csv")
    x_test = read_table(results / "X_test_00.csv")
    picked = X.stratified_sample(ctx.y_test, N_SAMPLE_ROWS, ctx.args.seed)
    bg_rows = X.stratified_sample(ctx.y_train, N_SAMPLE_BACKGROUND, ctx.args.seed)
    a_values, _, _ = checked_shap(
        ctx.model,
        x_test[ctx.columns].to_numpy(dtype=float)[picked],
        x_train[ctx.columns].to_numpy(dtype=float)[bg_rows],
        ctx.args.n_jobs,
    )
    reference = mean_abs(a_values, ctx.columns)
    rows = [
        {
            "model_cls": a["model_cls"],
            "selection": a["selection"],
            "embed_selector": a["embed_selector"],
            "role": "Model A",
            "cv_pr_auc": float(a["pr_auc_mean"]),
            "cv_pr_auc_sd": float(a["pr_auc_std"]),
            "n_columns": len(ctx.columns),
        }
    ]
    wide = {"Model A": reference}
    for _, other in others.iterrows():
        entry = entries[_model_key(other)]
        cols = list(entry["selected_cols"])
        fit = make_fit(entry["model_cls"], parse_params(entry["params"]))
        model = fit(x_train[cols], ctx.y_train, int(ctx.meta["seed"]))
        values, _, _ = checked_shap(
            model,
            x_test[cols].to_numpy(dtype=float)[picked],
            x_train[cols].to_numpy(dtype=float)[bg_rows],
            ctx.args.n_jobs,
        )
        importance = (
            mean_abs(values, cols).reindex(reference.index.union(cols)).fillna(0.0)
        )
        name = (
            f"{other['model_cls']} {other['selection']} {other['embed_selector']}".strip()
        )
        wide[name] = importance
        base = reference.reindex(importance.index).fillna(0.0)
        rows.append(
            {
                "model_cls": other["model_cls"],
                "selection": other["selection"],
                "embed_selector": other["embed_selector"],
                "role": "within Model A's CV interval",
                "cv_pr_auc": float(other["pr_auc_mean"]),
                "cv_pr_auc_sd": float(other["pr_auc_std"]),
                "n_columns": len(cols),
                **X.rank_agreement(base, importance),
            }
        )
    atomic_write_csv(ctx.out / "rashomon.csv", pd.DataFrame(rows))
    wide_frame = pd.DataFrame(wide).fillna(0.0).rename_axis("column").reset_index()
    atomic_write_csv(ctx.out / "rashomon_importance.csv", wide_frame)
    return {"n_other_models": len(rows) - 1, "cv_floor": floor}


def _roar(
    ctx: Context,
    stem: str,
    players: dict[str, list[str]],
    groups: dict[str, list[str]],
    top_ks: list[int],
) -> dict[str, Any]:
    fit = make_fit(ctx.meta["model_cls"], parse_params(ctx.meta["tuned_params"]))
    importance = ranking_importance(ctx.args.n_jobs, ctx.args.seed)
    parts = []
    timings = {}
    for k in np.unique(ctx.fold):
        path = ctx.out / f"{stem}_fold{int(k)}.csv"
        if path.is_file() and not ctx.args.force:
            parts.append(pd.read_csv(path))
            continue
        start = time.monotonic()
        table = E.roar_lite(
            ctx.x_train,
            ctx.y_train,
            ctx.fold,
            fit,
            prob_of,
            importance,
            players=players,
            groups=groups,
            top_ks=top_ks,
            n_random=ctx.args.roar_random,
            seed=ctx.args.seed,
            only_folds=[int(k)],
            log=print,
        )
        atomic_write_csv(path, table)
        parts.append(table)
        timings[int(k)] = round(time.monotonic() - start, 1)
    table = pd.concat(parts, ignore_index=True)
    atomic_write_csv(ctx.out / f"{stem}.csv", table)
    return {"fold_seconds": timings, "n_fits": len(table)}


def section_roar(ctx: Context) -> dict[str, Any]:
    players = {n: [ctx.columns[i] for i in idx] for n, idx in ctx.players.items()}
    groups = {n: [ctx.columns[i] for i in idx] for n, idx in ctx.groups.items()}
    return _roar(ctx, "roar", players, groups, ctx.args.roar_k)


def section_roar_extended(ctx: Context) -> dict[str, Any]:
    """Larger removals, added after the plan's k = 1, 2, 4, 8 showed no separation
    from random on Amazon, where those are at most 8% of the columns, and all graph
    columns together, the removal that compares with B. Training folds only."""
    graph = [c.prepared for c in ctx.infos if c.player != C.OWN_BLOCK]
    groups = {"all graph columns": graph} if graph else {}
    return _roar(ctx, "roar_extended", {}, groups, ctx.args.roar_k_extended)


def section_randomisation(ctx: Context) -> dict[str, Any]:
    fit = make_fit(ctx.meta["model_cls"], parse_params(ctx.meta["tuned_params"]))
    table = E.randomisation_check(
        ctx.x_train,
        ctx.y_train,
        ctx.fold,
        fit,
        prob_of,
        ranking_importance(ctx.args.n_jobs, ctx.args.seed),
        n_permutations=3,
        seed=ctx.args.seed,
        max_rows=N_RANKING_ROWS,
    )
    atomic_write_csv(ctx.out / "randomisation.csv", table)
    return {"n_runs": len(table)}


def section_seeds(ctx: Context) -> dict[str, Any]:
    fit = make_fit(ctx.meta["model_cls"], parse_params(ctx.meta["tuned_params"]))
    picked = X.stratified_sample(ctx.y_test, N_SAMPLE_ROWS, ctx.args.seed)
    x = ctx.x_test.to_numpy(dtype=float)[picked]
    background = ctx.background(n=N_SAMPLE_BACKGROUND)
    base_values, _, _ = checked_shap(ctx.model, x, background, ctx.args.n_jobs)
    reference = mean_abs(base_values, ctx.columns)
    wide = {"Model A": reference}
    rows = []
    # The first refit keeps df-analyze's seed, so it shows how much of the movement
    # below is the refit itself rather than the seed.
    runs = [("df-analyze's seed", None)] + [
        (f"seed {ctx.args.seed + i}", ctx.args.seed + i) for i in (1, 2, 3)
    ]
    for name, seed in runs:
        model = fit(ctx.x_train, ctx.y_train, seed)
        values, _, _ = checked_shap(model, x, background, ctx.args.n_jobs)
        importance = mean_abs(values, ctx.columns)
        wide[name] = importance
        gap = float(
            np.abs(margin_function(model)(x) - margin_function(ctx.model)(x)).max()
        )
        rows.append(
            {
                "refit": name,
                "max_margin_gap_vs_model_a": gap,
                **X.rank_agreement(reference, importance),
            }
        )
    atomic_write_csv(ctx.out / "seed_stability.csv", pd.DataFrame(rows))
    table = pd.DataFrame(wide).rename_axis("column").reset_index()
    atomic_write_csv(ctx.out / "seed_importance.csv", table)
    return {"n_seeds": len(rows), "n_rows": len(x), "n_background": len(background)}


def _own_matrix(ctx: Context) -> tuple[np.ndarray, list[str]]:
    own = pd.read_parquet(ctx.paths.block_file(C.OWN_BLOCK))
    if not np.array_equal(own[C.NODE_ID].to_numpy(), np.arange(len(own))):
        raise X.ExplainError("The own block's rows are not in node order")
    names = [c for c in own.columns if c != C.NODE_ID]
    return own[names].to_numpy(dtype=float), names


def _load_adjacency(ctx: Context, relations: list[str], n: int) -> dict[str, Any]:
    out = {}
    for relation in relations:
        with np.load(ctx.paths.relation_file(relation)) as f:
            out[relation] = adjacency(f["edges"], n)
    return out


def all_node_matrix(
    ctx: Context, own: np.ndarray, own_names: list[str], adj: dict[str, Any]
) -> np.ndarray:
    """The model's input for every node, built from the own block and the graph, so
    each node can be scored for the structural baseline. Matches df-analyze's
    prepared test rows, which is checked: it writes these columns unchanged, apart
    from a binary column's indicator, and has no saved preprocessing object."""
    where = {f.removeprefix(f"{C.OWN_BLOCK}{C.BLOCK_SEPARATOR}"): i
             for i, f in enumerate(own_names)}  # fmt: skip
    out = np.zeros((len(own), len(ctx.infos)))
    embeddings = None
    for i, c in enumerate(ctx.infos):
        if c.kind == C.OWN_BLOCK:
            values = own[:, where[c.base_feature]]
            if c.prepared == c.block_column.replace(C.BLOCK_SEPARATOR, "_"):
                out[:, i] = values
            else:  # an indicator such as own_f22_1.0
                level = float(c.prepared.rsplit("_", 1)[1])
                out[:, i] = np.isclose(values, level).astype(float)
        elif c.kind == C.NBR_BLOCK:
            degree = np.asarray(adj[c.relation].sum(axis=1)).ravel()
            if c.stat == "log1p_degree":
                out[:, i] = np.log1p(degree)
            else:
                sums = adj[c.relation] @ own[:, where[c.base_feature]]
                out[:, i] = np.divide(
                    sums, degree, out=np.zeros(len(own)), where=degree > 0
                )
        elif c.kind == C.GNNSSL_BLOCK:
            if embeddings is None:
                embeddings = pd.read_parquet(ctx.paths.block_file(C.GNNSSL_BLOCK))
            out[:, i] = embeddings[c.block_column].to_numpy(dtype=float)
        else:
            raise X.ExplainError(f"Cannot rebuild column {c.prepared} for every node")
    ids = ctx.test_ids[C.NODE_ID].to_numpy()
    gap = np.abs(out[ids] - ctx.x_test.to_numpy(dtype=float))
    scale = np.maximum(1.0, np.abs(out[ids]))
    if (gap / scale).max() > 1e-6:
        raise X.ExplainError(
            f"Columns rebuilt from the own block and graph differ from df-analyze's "
            f"prepared test rows by {(gap / scale).max():.3g}"
        )
    return out


def section_neighbours(ctx: Context) -> dict[str, Any]:
    own, own_names = _own_matrix(ctx)
    layouts = X.neighbour_layouts(ctx.infos, own_names)
    if not layouts:
        atomic_write_json(
            ctx.out / "neighbours_skipped.json",
            {"reason": "This model has no neighbour-mean columns to recompute."},
        )
        return {"skipped": True}
    adj = _load_adjacency(ctx, list(layouts), len(own))
    margin = margin_function(ctx.model)
    every = all_node_matrix(ctx, own, own_names, adj)
    scores = 1 / (1 + np.exp(-margin(every)))
    atomic_write_parquet(
        ctx.out / "node_scores.parquet",
        pd.DataFrame({C.NODE_ID: np.arange(len(own)), "prob": scores}),
    )
    picked = (
        X.stratified_sample(ctx.y_test, ctx.args.max_targets, ctx.args.seed)
        if ctx.args.max_targets
        else np.arange(len(ctx.y_test))
    )
    ids = ctx.test_ids[C.NODE_ID].to_numpy()[picked]
    x = ctx.x_test.to_numpy(dtype=float)[picked]
    ranges = X.training_ranges(ctx.x_train.to_numpy(dtype=float))
    chunks = list(X.neighbour_influence(margin, x, ids, own, adj, layouts, ranges))
    table = pd.concat(chunks, ignore_index=True)
    atomic_write_parquet(ctx.out / "neighbour_influence.parquet", table)
    # Is leave-one-out missing redundant neighbours? A sample of small neighbourhoods.
    rng = np.random.default_rng(ctx.args.seed)
    rows = []
    for relation, layout in layouts.items():
        a = adj[relation]
        degree = np.asarray(a[ids].sum(axis=1)).ravel()
        small = np.flatnonzero((degree >= 3) & (degree <= MAX_REDUNDANCY_DEGREE))
        chosen = rng.choice(small, min(N_REDUNDANCY_TARGETS, len(small)), replace=False)
        for position in chosen:
            node = ids[position]
            neighbours = a.indices[a.indptr[node] : a.indptr[node + 1]]
            phi = X.neighbour_shapley(
                margin, x[position], neighbours, own, layout,
                N_REDUNDANCY_PERMUTATIONS, int(rng.integers(2**31)),
            )  # fmt: skip
            _, whole = X.rows_without_each_neighbour(x[position], neighbours, own, layout)
            loo = margin(x[[position]])[0] - margin(whole)
            rows.append(
                {
                    "relation": relation,
                    "degree": len(neighbours),
                    "spearman_loo_vs_shapley": X.spearman(loo, phi),
                    "shapley_abs_sum": float(np.abs(phi).sum()),
                    "loo_abs_sum": float(np.abs(loo).sum()),
                    "same_top_neighbour": int(
                        np.abs(loo).argmax() == np.abs(phi).argmax()
                    ),
                }
            )
    atomic_write_csv(ctx.out / "neighbour_redundancy.csv", pd.DataFrame(rows))
    return {
        "relations": list(layouts),
        "n_targets": len(ids),
        "n_targets_available": len(ctx.y_test),
        "n_rows": len(table),
        "n_redundancy_targets": len(rows),
        "n_redundancy_permutations": N_REDUNDANCY_PERMUTATIONS,
    }


def section_neighbours_encoder(ctx: Context) -> dict[str, Any]:
    """Neighbour influence through the DGI encoder, for models with gnnssl columns.
    Needs scripts/gnn_encoder_export.py's weights, checked there against the saved
    embeddings and here against the model's input columns."""
    path = ctx.out / "encoder_weights.npz"
    if not path.is_file():
        raise X.ExplainError(
            f"{path} is missing: run scripts/gnn_encoder_export.py first."
        )
    relations = sorted({c.relation for c in ctx.infos if c.kind == C.GNNSSL_BLOCK})
    picked = (
        X.stratified_sample(ctx.y_test, ctx.args.max_targets, ctx.args.seed)
        if ctx.args.max_targets
        else np.arange(len(ctx.y_test))
    )
    ids = ctx.test_ids[C.NODE_ID].to_numpy()[picked]
    x = ctx.x_test.to_numpy(dtype=float)[picked]
    ranges = X.training_ranges(ctx.x_train.to_numpy(dtype=float))
    margin = margin_function(ctx.model)
    own, own_names = _own_matrix(ctx)
    all_relations = [
        c.relation for c in ctx.infos if c.kind == C.NBR_BLOCK and c.relation
    ]
    every = all_node_matrix(
        ctx, own, own_names, _load_adjacency(ctx, all_relations, len(own))
    )
    atomic_write_parquet(
        ctx.out / "node_scores.parquet",
        pd.DataFrame(
            {C.NODE_ID: np.arange(len(own)), "prob": 1 / (1 + np.exp(-margin(every)))}
        ),
    )
    adj = _load_adjacency(ctx, relations, len(own))
    tables, gaps = [], {}
    with np.load(path) as saved:
        own_scaled = saved["own_scaled"].astype(float)
        for relation in relations:
            enc = X.EncoderWeights(
                **{
                    field: saved[f"{relation}/{field}"].astype(float)
                    for field in X.EncoderWeights.__dataclass_fields__
                }
            )
            table, gaps[relation] = X.encoder_influence(
                margin, x, ids, own_scaled, adj[relation], enc, ctx.infos, relation,
                ranges, ctx.args.max_neighbours, ctx.args.seed,
            )  # fmt: skip
            tables.append(table)
            print(f"  {relation}: {len(table):,} removals", flush=True)
    atomic_write_parquet(
        ctx.out / "neighbour_influence.parquet", pd.concat(tables, ignore_index=True)
    )
    return {
        "relations": relations,
        "n_targets": len(ids),
        "n_targets_available": len(ctx.y_test),
        "max_neighbours_per_target": ctx.args.max_neighbours,
        "pca_column_gap": gaps,
        "weights": "encoder and PCA held fixed; nothing retrained",
    }


def _counterfactual(
    ctx: Context, row: int, values: np.ndarray, table: pd.DataFrame, margin: X.Margin
) -> dict[str, Any]:
    """The largest single change that crosses the threshold, among setting one of
    the top columns to its background median and removing one neighbour. A
    description of model behaviour, not advice."""
    x = ctx.x_test.to_numpy(dtype=float)
    medians = np.median(ctx.background(), axis=0)
    prob = float(1 / (1 + np.exp(-margin(x[[row]])[0])))
    was_flagged = prob >= ctx.threshold
    top = np.argsort(-np.abs(values))[:N_CARD_FEATURES]
    candidates = []
    for j in top:
        changed = x[[row]].copy()
        changed[0, j] = medians[j]
        new = float(1 / (1 + np.exp(-margin(changed)[0])))
        candidates.append(("feature", ctx.columns[j], float(medians[j]), new))
    for _, n in table.iterrows():
        name = f"{n['neighbour']} ({n['relation']})"
        candidates.append(("neighbour", name, None, prob - float(n["delta_prob"])))
    crossing = [c for c in candidates if (c[3] >= ctx.threshold) != was_flagged]
    pool = crossing or candidates
    kind, name, value, new = max(pool, key=lambda c: abs(c[3] - prob))
    return {
        "kind": kind,
        "name": name,
        "to_value": value,
        "new_prob": new,
        "crosses_threshold": bool(crossing),
    }


def section_cards(ctx: Context) -> dict[str, Any]:
    main = pd.read_parquet(ctx.out / "shap_test.parquet")
    train = pd.read_parquet(ctx.out / "shap_train_sample.parquet")
    shap_cols = [SHAP_COLUMN + c for c in ctx.columns]
    meta = json.loads((ctx.out / "explain_meta.json").read_text(encoding="utf-8"))
    base = float(meta["sections"]["shap"]["base_value"])
    influence_path = ctx.out / "neighbour_influence.parquet"
    influence = pd.read_parquet(influence_path) if influence_path.is_file() else None
    margin = margin_function(ctx.model)
    masks = X.strata_masks(
        main[C.LABEL].to_numpy(), main["prob"].to_numpy(), ctx.threshold
    )
    checked = main[shap_cols].notna().all(axis=1).to_numpy()
    rng = np.random.default_rng(ctx.args.seed)
    x = ctx.x_test.to_numpy(dtype=float)
    cards = []
    train_shap = train[shap_cols].to_numpy()
    for stratum, count in N_CARDS.items():
        where = np.flatnonzero(masks[stratum] & checked)
        for row in rng.choice(where, min(count, len(where)), replace=False):
            values = main.iloc[row][shap_cols].to_numpy(dtype=float)
            order = np.argsort(-np.abs(values))
            top = order[:N_CARD_FEATURES]
            node = int(main.iloc[row][C.NODE_ID])
            neighbours = pd.DataFrame()
            if influence is not None:
                mine = influence[influence["target"] == node]
                neighbours = (
                    mine.assign(size=mine["delta_prob"].abs())
                    .sort_values("size", ascending=False)
                    .groupby("relation", observed=True)
                    .head(N_CARD_NEIGHBOURS)
                )
            distance = np.linalg.norm(train_shap - values, axis=1)
            nearest = train.iloc[np.argsort(distance)[:N_CARD_NEAREST]]
            cards.append(
                {
                    "stratum": stratum,
                    "node_id": node,
                    "label": int(main.iloc[row][C.LABEL]),
                    "prob": float(main.iloc[row]["prob"]),
                    "margin": float(main.iloc[row]["margin"]),
                    "base_value": base,
                    "threshold": ctx.threshold,
                    "contributions": [
                        {
                            "column": ctx.columns[j],
                            "value": float(x[row, j]),
                            "shap": float(values[j]),
                        }
                        for j in top
                    ],
                    "rest": float(values[order[N_CARD_FEATURES:]].sum()),
                    "neighbours": [
                        {
                            "neighbour": int(n["neighbour"]),
                            "relation": str(n["relation"]),
                            "degree": int(n["degree"]),
                            "delta_prob": float(n["delta_prob"]),
                            "delta_prob_mean": float(n["delta_prob_mean"]),
                        }
                        for _, n in neighbours.iterrows()
                    ],
                    "nearest_training": [
                        {
                            "node_id": int(n[C.NODE_ID]),
                            "label": int(n[C.LABEL]),
                            "prob": float(n["prob"]),
                        }
                        for _, n in nearest.iterrows()
                    ],
                    "counterfactual": _counterfactual(
                        ctx, int(row), values, neighbours, margin
                    ),
                }
            )
    atomic_write_json(ctx.out / "case_cards.json", {"cards": cards})
    return {"n_cards": len(cards)}


SECTIONS = {
    "shap": (section_shap, "shap_test.parquet"),
    "sensitivity": (section_sensitivity, "sensitivity.csv"),
    "interactions": (section_interactions, "interactions.csv"),
    "group_shapley": (section_group_shapley, "group_shapley.parquet"),
    "rashomon": (section_rashomon, "rashomon.csv"),
    "roar": (section_roar, "roar.csv"),
    "roar_extended": (section_roar_extended, "roar_extended.csv"),
    "randomisation": (section_randomisation, "randomisation.csv"),
    "seeds": (section_seeds, "seed_stability.csv"),
    "neighbours": (section_neighbours, "neighbour_influence.parquet"),
    "neighbours_encoder": (section_neighbours_encoder, "neighbour_influence.parquet"),
    "cards": (section_cards, "case_cards.json"),
}


def read_meta(ctx: Context) -> dict[str, Any]:
    path = ctx.out / "explain_meta.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    packages = ("shap", "lightgbm", "catboost", "numpy", "pandas", "scikit-learn")
    return {
        "dataset": ctx.args.dataset,
        "repeat": ctx.args.repeat,
        "feature_set": ctx.args.feature_set,
        "budget": "pilot" if ctx.args.feature_set.startswith("pilot_") else "full",
        "model": ctx.meta["model"],
        "model_cls": ctx.meta["model_cls"],
        "selection": ctx.meta["selection"],
        "embed_selector": ctx.meta.get("embed_selector"),
        "n_input_columns": len(ctx.columns),
        "threshold": ctx.threshold,
        "threshold_rule": ctx.meta["threshold_rule"],
        "df_analyze_run": ctx.meta["run"],
        "df_analyze_commit": ctx.meta["df_analyze_commit"],
        "git_commit": git_commit(REPO_ROOT),
        "seed": ctx.args.seed,
        "package_versions": {name: metadata.version(name) for name in packages},
        "players": {n: len(i) for n, i in ctx.players.items()},
        "correlated_groups": {n: len(i) for n, i in ctx.groups.items()},
        "sections": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--repeat", type=int, default=0, help="split repeat")
    parser.add_argument("--sections", nargs="+", choices=list(SECTIONS))
    parser.add_argument("--force", action="store_true", help="redo finished sections")
    parser.add_argument("--background", type=int, default=1000)
    parser.add_argument(
        "--path-dependent",
        action="store_true",
        help="shap section: path-dependent TreeSHAP, a different game, for a model where "
        "interventional TreeSHAP fails the implementation check. Recorded in the outputs",
    )
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--n-jobs", type=int, default=16, help="processes for SHAP")
    parser.add_argument("--roar-k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--roar-k-extended", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--roar-random", type=int, default=5, help="random draws per k")
    parser.add_argument(
        "--max-targets", type=int, default=0, help="neighbours: a sample of test nodes"
    )
    parser.add_argument(
        "--max-neighbours",
        type=int,
        default=30,
        help="neighbours_encoder: a seeded sample of each target's neighbours",
    )
    args = parser.parse_args()
    ctx = load_context(args)
    meta = read_meta(ctx)
    print(
        f"{args.dataset} split {args.repeat} {args.feature_set}: {ctx.meta['model']} "
        f"({ctx.meta['selection']}), {len(ctx.columns)} columns, "
        f"players {list(ctx.players)}"
    )
    # The encoder section has its own prerequisite, so it runs only when asked for.
    default = [s for s in SECTIONS if s != "neighbours_encoder"]
    for name in args.sections or default:
        run, output = SECTIONS[name]
        if (ctx.out / output).is_file() and not args.force:
            print(f"{name}: done earlier, kept")
            continue
        start = time.monotonic()
        details = run(ctx)
        seconds = round(time.monotonic() - start, 1)
        meta["sections"][name] = {"seconds": seconds, **details}
        atomic_write_json(ctx.out / "explain_meta.json", meta)
        print(f"{name}: {seconds} s {details}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except X.ExplainError as e:
        print(f"ERROR: {e} Stop and ask.", file=sys.stderr)
        sys.exit(3)
