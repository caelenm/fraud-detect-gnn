"""
=====================================================================
SYNTHETIC TEST DATA. NOT REAL YELPCHI OR AMAZON DATA.
=====================================================================

Every row used by the test suite is built by a function in this module and
nowhere else. All values are invented: random feature values, made-up node
IDs and labels, and graphs drawn at random. Nothing describes a real review,
user or model.

Nothing here is written to data/ or outputs/. Tests that need files write
them to pytest's temporary directory (`tmp_path`), which is deleted
automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detect import columns as C
from fraud_detect import results_db
from fraud_detect.models import contrasts

# (config name, df-analyze estimator class) for fake df-analyze results.
FAKE_DFA_MODELS = [("dummy", "DummyClassifier"), ("lgbm", "LightGBMClassifier"),
                   ("catboost", "CatBoostClassifier"),
                   ("lr", "LRClassifier")]  # fmt: skip


def fake_df_analyze_results(
    y_test: np.ndarray, cv_scores: dict[tuple[str, str], float], seed: int = 0
) -> tuple[list[dict], pd.DataFrame]:
    """Invented df-analyze outputs: `prediction_results` entries and a
    `tuned_models` table, one per (model, selection) key in `cv_scores`.
    Probabilities are random noise plus a model-specific signal; they describe
    no real model."""
    rng = np.random.default_rng(seed)
    classes = dict(FAKE_DFA_MODELS)
    entries, tuned = [], []
    for i, ((model, selection), score) in enumerate(cv_scores.items()):
        signal = 0.0 if model == "dummy" else 0.15 * (i + 1)
        p1 = np.clip(rng.uniform(0, 0.6, len(y_test)) + signal * y_test, 0, 1)
        entries.append(
            {
                "model_cls": classes[model],
                "selection": selection,
                "embed_select_model": "linear" if selection == "embed" else None,
                "metric": "Accuracy",
                "score": score,
                "probs_test": np.column_stack([1 - p1, p1]).tolist(),
                "preds_test": (p1 > 0.5).astype(int).tolist(),
                "params": "{'n_estimators': 50, 'learning_rate': 0.1}",
                "selected_cols": ["own_f00", "own_f01"],
            }
        )
        tuned.append(
            {
                "selection": selection,
                "embed_selector": "linear" if selection == "embed" else "none",
                "model": model,
                "params": "{}",
                "metric": "acc",
                "score": score,
                "test_idx": 0,
            }
        )
    return entries, pd.DataFrame(tuned)


def fake_test_meta(n: int, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    """Invented test-node details for the web report, in test order: node IDs
    and each node's top features as (name, value, percentile) tuples."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            C.NODE_ID: [9000 + i for i in range(n)],
            "top_features": [
                [(f"own__f{j:02d}", float(rng.uniform()), float(rng.uniform()))
                 for j in range(n_features)]
                for _ in range(n)
            ],
        }
    )  # fmt: skip


def fake_shared_cv(
    pr_auc: dict[tuple[str, str], float],
    std: float = 0.01,
    default_offset: float = -0.05,
) -> pd.DataFrame:
    """Invented output of scripts/dfa/cv_select.py: for each (model, selection)
    key, a tuned row and a default-settings row whose PR-AUC is the tuned one
    plus `default_offset`. The scores describe no real model."""
    classes = dict(FAKE_DFA_MODELS)
    rows = []
    for (model, selection), score in pr_auc.items():
        for settings, value in (("tuned", score), ("default", score + default_offset)):
            rows.append(
                {
                    "model_cls": classes[model],
                    "selection": selection,
                    "embed_selector": "linear" if selection == "embed" else "",
                    "settings": settings,
                    "pr_auc_mean": value,
                    "pr_auc_std": std,
                    "auroc_mean": 0.5 + value / 2,
                    "bal_acc_mean": 0.5,
                    "error": "",
                }
            )
    return pd.DataFrame(rows)


def fake_model_tables(
    n_train: int = 180, n_test: int = 120, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Invented df-analyze input tables: continuous, ordinal and binary `own__`
    columns plus `target`, shaped like the df_analyze_input stage's output.
    Features are random noise, so none predicts the label."""
    rng = np.random.default_rng(seed)

    def table(n: int) -> pd.DataFrame:
        y = (rng.random(n) < 0.2).astype(int)
        y[:2] = [0, 1]  # both classes
        return pd.DataFrame(
            {
                "own__f00": rng.normal(size=n),
                "own__f01": rng.uniform(size=n),
                "own__f02": rng.integers(0, 6, n),
                "own__f03": rng.integers(0, 2, n),
                "target": y,
            }
        )

    return table(n_train), table(n_test)


def fake_care_gnn_mat(
    n_nodes: int = 60, n_features: int = 5, unlabelled_prefix: int = 8, seed: int = 0
) -> dict:
    """Invented contents of a CARE-GNN-style .mat file (for scipy.io.savemat):
    sparse `features`, a 1 x n `label` row (0 for the unlabelled prefix, as in
    Amazon), two symmetric relations and their union `homo`.

    Planted structure for the tests: labelled nodes p and p+1 have identical
    feature rows (a duplicate group); relation `net_aaa` links nodes in pairs
    (2k, 2k+1) for k < 10, so those pairs are its connected components."""
    import scipy.sparse as sp  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    features = rng.integers(0, 9, size=(n_nodes, n_features)).astype(float)
    features[:, 0] = np.arange(n_nodes)  # unique rows unless planted below
    p = unlabelled_prefix
    features[p + 1] = features[p]
    labels = (rng.random(n_nodes) < 0.3).astype(float)
    labels[:p] = 0  # unlabelled nodes are stored as 0
    labels[p], labels[p + 2] = 1, 0  # both classes among labelled nodes

    def symmetric(pairs: list[tuple[int, int]]) -> sp.csc_matrix:
        rows = [a for a, b in pairs] + [b for a, b in pairs]
        cols = [b for a, b in pairs] + [a for a, b in pairs]
        return sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes,) * 2)

    pairs_a = [(2 * k, 2 * k + 1) for k in range(10)]
    pairs_b = [(int(a), int(b)) for a, b in rng.integers(0, n_nodes, (30, 2)) if a != b]
    a, b = symmetric(pairs_a), symmetric(pairs_b)
    homo = ((a + b) > 0).astype(float)
    return {
        "features": sp.csc_matrix(features),
        "label": labels.reshape(1, -1),
        "net_aaa": a,
        "net_bbb": b,
        "homo": sp.csc_matrix(homo),
    }


def fake_node_table(
    n: int = 400, unlabelled: int = 0, group_size: int = 3, seed: int = 0
) -> pd.DataFrame:
    """Invented node table shaped like the load stage's output: the first
    `unlabelled` nodes have no label; labelled nodes come in groups of
    `group_size` consecutive nodes that share one label (like a YelpChi user's
    reviews), about 15% positive."""
    rng = np.random.default_rng(seed)
    groups = np.arange(n) // group_size
    group_label = (rng.random(groups.max() + 1) < 0.15).astype(int)
    label = pd.array(group_label[groups], dtype="Int8")
    is_labelled = np.arange(n) >= unlabelled
    label[~is_labelled] = pd.NA
    split = pd.array([None] * n, dtype="string")
    split[~is_labelled] = C.UNLABELLED
    return pd.DataFrame(
        {
            C.NODE_ID: np.arange(n, dtype=np.int64),
            C.LABEL: label,
            C.IS_LABELLED: is_labelled,
            C.GROUP_ID: groups.astype(np.int64),
            C.SPLIT: split,
            C.CV_FOLD: pd.array([None] * n, dtype="Int8"),
        }
    )


def fake_block(name: str, n: int = 10, width: int = 3, seed: int = 0) -> pd.DataFrame:
    """Invented feature block: node_id plus <name>__x0 ... columns of noise."""
    rng = np.random.default_rng(seed)
    block = pd.DataFrame(rng.normal(size=(n, width))).add_prefix(f"{name}__x")
    block.insert(0, C.NODE_ID, np.arange(n, dtype=np.int64))
    return block


def fake_embeddings(
    n: int, relations: tuple[str, ...], dim: int = 8, seed: int = 0
) -> dict[str, np.ndarray]:
    """Invented raw node embeddings (float32 noise), one (n, dim) per relation,
    shaped like the gnn_embed stage's raw_<relation>.npy."""
    rng = np.random.default_rng(seed)
    return {r: rng.normal(size=(n, dim)).astype(np.float32) for r in relations}


def fake_edges(n: int, n_pairs: int, seed: int = 0) -> np.ndarray:
    """Invented undirected edge list over nodes 0..n-1, shaped like the load
    stage's output: (2, E) int32, src < dst, no self-loops, no repeats."""
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, n, size=(2, n_pairs))
    pairs = pairs[:, pairs[0] != pairs[1]]
    return np.unique(np.sort(pairs, axis=0), axis=1).astype(np.int32)


def fake_feature_graph(
    n: int = 60,
    width: int = 4,
    relations: tuple[str, ...] = ("aaa", "bbb"),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Invented graph for the graph-block tests: a node table whose first n/10
    nodes are unlabelled and whose labelled nodes are split about 60/40 into
    train/test at random; an own block (own__f00, ...) of noise; and one random
    edge list per relation."""
    rng = np.random.default_rng(seed)
    nodes = fake_node_table(n=n, unlabelled=n // 10, seed=seed)
    labelled = nodes[C.IS_LABELLED].to_numpy()
    nodes.loc[labelled, C.SPLIT] = np.where(
        rng.random(int(labelled.sum())) < 0.6, C.TRAIN, C.TEST
    )
    own = fake_block(C.OWN_BLOCK, n=n, width=width, seed=seed)
    own.columns = [C.NODE_ID, *(f"own__f{j:02d}" for j in range(width))]
    edges = {r: fake_edges(n, 3 * n, seed + i) for i, r in enumerate(relations)}
    return nodes, own, edges


def fake_inferred_types(kinds: dict[str, str]) -> pd.DataFrame:
    """Invented df-analyze inspection/inferred_types.csv: index feature_name,
    columns user, inferred, reason (as written by InspectionResults.basic_df)."""
    frame = pd.DataFrame(
        {
            "user": ["ord" if k.startswith("user-ord") else "" for k in kinds.values()],
            "inferred": list(kinds.values()),
            "reason": ["fake reason"] * len(kinds),
        },
        index=pd.Index(list(kinds), name="feature_name"),
    )
    return frame


def fake_model_a_metadata(model_file: str = "model_a.joblib") -> dict:
    """Invented model_a.json contents (export_model stage); describes no real
    model."""
    return {
        "format_version": 1,
        "dataset": "fake",
        "feature_set": "m1_own",
        "run": "20260101T000000Z",
        "results_dir": "outputs/fake/m1_own/df_analyze/run/results/test00",
        "model": "lgbm",
        "model_cls": "LightGBMClassifier",
        "selection": "none",
        "embed_selector": "",
        "selection_label": "none",
        "input_columns": ["own_f00", "own_f01"],
        "prepared_columns": ["own_f00", "own_f01", "own_f02"],
        "tuned_params": {"n_estimators": 50},
        "threshold": 0.3,
        "threshold_rule": "max_f1 on out-of-fold training predictions",
        "seed": 555,
        "df_analyze_commit": "0" * 40,
        "source": "decoded",
        "model_file": model_file,
        "model_format": "joblib",
        "verification": {"max_abs_diff_native": 0.0, "tolerance": 1e-9},
        "package_versions": {"python": "3.13.0", "lightgbm": "4.6.0"},
    }


def fake_oof_predictions(
    keys: list[tuple[str, str]], y_train: np.ndarray, folds: np.ndarray, seed: int = 0
) -> pd.DataFrame:
    """Invented cv_select.py out-of-fold predictions: for each (model,
    selection) key, a tuned and a default-settings row per training row."""
    rng = np.random.default_rng(seed)
    classes = dict(FAKE_DFA_MODELS)
    frames = []
    for model, selection in keys:
        for settings in ("tuned", "default"):
            prob = np.clip(rng.uniform(0, 0.6, len(y_train)) + 0.3 * y_train, 0, 1)
            frames.append(
                pd.DataFrame(
                    {
                        "model_cls": classes[model],
                        "selection": selection,
                        "embed_selector": "",
                        "settings": settings,
                        "row": np.arange(len(y_train)),
                        "fold": folds,
                        "prob": prob,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def fake_results_table(n_rows: int = 3, seed: int = 0) -> pd.DataFrame:
    """Invented aggregate results (one row per model), shaped like
    model_a_metrics.csv, for the results database. The numbers describe no
    real run."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "model": [f"fake_model_{i}" for i in range(n_rows)],
            "selection": ["none"] * n_rows,
            "cv_pr_auc": rng.uniform(0.1, 0.9, n_rows),
            "cv_pr_auc_std": rng.uniform(0.0, 0.05, n_rows),
            "pr_auc": rng.uniform(0.1, 0.9, n_rows),
            "auroc": rng.uniform(0.5, 1.0, n_rows),
        }
    )


def fake_model_a_summary(dataset: str = "amazon", feature_set: str = "m1_own") -> dict:
    """Invented reports/model_a.json of the df_analyze_report stage."""
    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "run": "20990101T000000Z",
        "htune_trials": 3,
        "seed": 1,
        "n_train": 30,
        "n_test": 20,
        "test_positive_rate": 0.25,
        "model_a": {
            "model": "fake_model_0",
            "selection": "none",
            "pr_auc": 0.5,
            "auroc": 0.7,
            "cv_pr_auc": 0.45,
            "cv_pr_auc_std": 0.02,
            "f1": 0.4,
            "threshold": 0.3,
        },  # fmt: skip
        "intervals": [
            {"metric": "pr_auc", "estimate": 0.5, "low": 0.4, "high": 0.6},
            {"metric": "auroc", "estimate": 0.7, "low": 0.6, "high": 0.8},
        ],
        "sanity_warning": None,
    }


# --------------------------------------------------------------------------
# Results-database entries for the dashboard tests. Aggregate numbers, all
# invented; each helper writes one entry under `db_dir` (a tmp_path folder).
# --------------------------------------------------------------------------
_FAKE_DEVICE = {"os": "Fake", "platform": "fake", "wsl": False, "python": "3.13",
                "cpu": "fake", "gpus": []}  # fmt: skip


def fake_entry(db_dir: Path, run_id: str, table: pd.DataFrame, **meta) -> str:
    """One database entry, through the same check as a real one."""
    fields = {
        "schema_version": results_db.SCHEMA_VERSION,
        "run_id": run_id,
        "recorded_utc": "2026-01-01T00:00:00Z",
        "started_utc": "2026-01-01T00:00:00Z",
        "device": "fake-laptop",
        "device_details": _FAKE_DEVICE,
        "description": "Invented entry for a test",
        "kind": "pipeline",
        "status": "ok",
        **meta,
    }
    results_db.add_entry(db_dir, fields, table)
    return run_id


def fake_model_table(model: str, selection: str, cv: float, test: float) -> pd.DataFrame:
    """Invented results.csv of a pipeline run: the named Model A, and a dummy
    with a lower shared-CV score."""
    return pd.DataFrame(
        {
            "model": [model, "dummy"],
            "selection": [selection, "none"],
            "cv_pr_auc": [cv, cv / 4],
            "pr_auc": [test, test / 4],
        }
    )


def fake_run_entry(
    db_dir: Path,
    dataset: str,
    feature_set: str,
    split: int = 0,
    test: float = 0.8,
    cv: float = 0.8,
    no_skill: float = 0.1,
    model: str = "lgbm",
    selection: str = "none",
    n_test: int = 100,
    n_train: int = 150,
    table: pd.DataFrame | None = None,
    **meta,
) -> str:
    """A pipeline entry with invented numbers, at settings that match its name."""
    pilot = feature_set.startswith("pilot_")
    settings = {
        "df_analyze.htune_trials": 10 if pilot else 100,
        "df_analyze.classifiers": ["a", "b", "c"] if pilot else list("abcdef"),
    }
    suffix = f"{dataset}_{feature_set}_{split}_{meta.pop('tag_id', '')}"
    return fake_entry(
        db_dir,
        f"20260101T000000Z_{suffix}",
        fake_model_table(model, selection, cv, test) if table is None else table,
        dataset=dataset,
        feature_set=feature_set,
        split_repeat=split,
        seed=555,
        settings=meta.pop("settings", settings),
        headline={
            "model_a": f"{model} ({selection})",
            "test_pr_auc": test,
            "test_pr_auc_low": test - 0.02,
            "test_pr_auc_high": test + 0.02,
            "test_auroc": min(test + 0.05, 1.0),
            "no_skill_pr_auc": no_skill,
            "cv_pr_auc": cv,
            "cv_pr_auc_std": 0.01,
            "threshold": 0.4,
            "n_train": n_train,
            "n_test": n_test,
        },
        **meta,
    )


def fake_contrast_row(
    a: str, b: str, deltas: list[float], p: float = 0.01, pr_a: float = 0.8, **overrides
) -> dict:
    """One invented repeat_summary.py row, consistent unless overridden."""
    d = np.asarray(deltas, dtype=float)
    row = {
        "variant": "-",
        "contrast": f"{b} - {a}",
        "a": a,
        "b": b,
        **{f"test_delta_split{k}": v for k, v in enumerate(deltas)},
        "test_delta_mean": float(d.mean()),
        "test_delta_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "test_pr_auc_a_mean": pr_a,
        "test_pr_auc_a_sd": 0.01,
        "test_pr_auc_b_mean": pr_a + float(d.mean()),
        "test_pr_auc_b_sd": 0.01,
        "cv_folds": 5 * len(d),
        "cv_delta_mean": float(d.mean()),
        "corrected_t": 3.0,
        "p": p,
        "verdict": contrasts.verdict(list(d), p),
    }
    row.update(overrides)
    return row


def fake_compare_entry(
    db_dir: Path,
    dataset: str,
    rows: list[dict],
    name: str,
    description: str = "Invented verdicts",
    started: str = "2026-01-02T00:00:00Z",
    **meta,
) -> str:
    return fake_entry(
        db_dir,
        f"{started.replace('-', '').replace(':', '')}_{dataset}_{name}",
        pd.DataFrame(rows),
        kind="compare",
        dataset=dataset,
        description=description,
        started_utc=started,
        **meta,
    )


def fake_ladder_db(db_dir: Path) -> None:
    """A small invented database covering both datasets, every family and both
    budgets, with contrasts that agree with its runs."""
    amazon = {"pilot_m1_own": (0.80, 0.81), "pilot_m2_own_nbr": (0.82, 0.86),
              "pilot_m5_own_gnnssl": (0.83, 0.84), "pilot_g5_sage": (0.75, 0.74),
              "pilot_g6_mlp": (0.72, 0.71), "pilot_m1_own_no19": (0.50, 0.52),
              "pilot_m2_own_nbr_no19": (0.65, 0.66)}  # fmt: skip
    yelp = {"pilot_m1_own": (0.70, 0.71), "pilot_m2_own_nbr": (0.76, 0.75),
            "pilot_m5_own_gnnssl": (0.72, 0.78), "pilot_g5_sage": (0.65, 0.64),
            "pilot_g6_mlp": (0.55, 0.54)}  # fmt: skip
    for k, shift in enumerate((-0.01, 0.0, 0.01)):
        for fs, (test, cv) in amazon.items():
            fake_run_entry(
                db_dir, "amazon", fs, k, test + shift, cv + shift, no_skill=0.1
            )
        for fs, (test, cv) in yelp.items():
            fake_run_entry(db_dir, "yelpchi", fs, k, test + shift, cv + shift,
                           no_skill=0.15, n_test=200, n_train=300)  # fmt: skip
    fake_run_entry(db_dir, "amazon", "m1_own", 0, 0.805, 0.815, no_skill=0.1,
                   tags=["backfill", "full"])  # fmt: skip
    a, b = "pilot_m1_own", "pilot_m2_own_nbr"
    fake_compare_entry(
        db_dir, "amazon",
        [fake_contrast_row(a, b, [0.02, 0.02, 0.02], p=0.01, pr_a=0.80),
         fake_contrast_row("pilot_g6_mlp", "pilot_g5_sage", [0.03, 0.03, 0.03], p=0.2,
                           pr_a=0.72),
         fake_contrast_row("pilot_m1_own_no19", "pilot_m2_own_nbr_no19",
                           [0.15, 0.15, 0.15], p=0.001, pr_a=0.50)],
        "amazon_primary",
        description="Invented primary family of contrasts",
    )  # fmt: skip
    fake_compare_entry(
        db_dir, "yelpchi",
        [fake_contrast_row(a, b, [0.06, 0.06, 0.06], p=0.001, pr_a=0.70),
         fake_contrast_row("pilot_g6_mlp", "pilot_g5_sage", [0.10, 0.10, 0.10], p=0.001,
                           pr_a=0.55)],
        "yelp_primary",
        description="Invented primary family of contrasts",
    )  # fmt: skip
