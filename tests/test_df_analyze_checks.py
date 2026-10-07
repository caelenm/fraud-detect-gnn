"""df-analyze column names, the inferred-type check, and select_model's folds.

df-analyze itself is never run here: its outputs are faked (tests/synthetic.py)
and written to pytest's tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from helpers import fake_dataset
from synthetic import fake_inferred_types

from fraud_detect import columns as C
from fraud_detect import pipeline
from fraud_detect.cli import main
from fraud_detect.config import get_paths
from fraud_detect.models import df_analyze as dfa

EXPECTED = {"own_f00": "continuous", "own_f01": "ordinal", "own_f02": "binary"}


# ---- Column names ----------------------------------------------------------
def test_names_match_df_analyze_sanitizing():
    # Checked against df-analyze's sanitize_names at the pinned commit.
    assert dfa.df_analyze_name("own__f00") == "own_f00"
    assert dfa.df_analyze_name("nbr_rur__f01") == "nbr_rur_f01"
    assert dfa.df_analyze_name("p(q)[r]{s}") == "p_q_r_s"
    assert dfa.df_analyze_name("x__") == "x"
    assert dfa.df_analyze_names(["own__f00", "own__f01"]) == {
        "own__f00": "own_f00",
        "own__f01": "own_f01",
    }
    with pytest.raises(ValueError, match="both become"):
        dfa.df_analyze_names(["own__f00", "own_f00"])
    with pytest.raises(ValueError, match="target"):
        dfa.df_analyze_names(["target__"])


# ---- Inferred types ----------------------------------------------------------
def test_matching_types_pass_and_coercions_are_reported():
    inferred = fake_inferred_types(
        {"own_f00": "cont", "own_f01": "ord-coerce", "own_f02": "bin"}
    )
    report = dfa.check_inferred_types(inferred, EXPECTED, list(EXPECTED))
    assert report["all_types_match"] and report["n_features"] == 3
    assert report["dropped_or_coerced"] == ["own_f01: ord-coerce (fake reason)"]


@pytest.mark.parametrize(
    "kinds, exported, message",
    [
        # An ordinal inferred as categorical (would be one-hot encoded).
        ({"own_f00": "cont", "own_f01": "cat", "own_f02": "bin"}, None, "ordinal"),
        # A high-cardinality continuous column flagged as an identifier.
        ({"own_f00": "id", "own_f01": "ord", "own_f02": "bin"}, None, "own_f00"),
        # A column missing from the inspection.
        ({"own_f00": "cont", "own_f01": "ord"}, None, "missing from df-analyze"),
        # A column df-analyze typed that we never gave it.
        ({**{"own_f00": "cont", "own_f01": "ord", "own_f02": "bin"}, "zz": "cont"},
         None, "not ours"),
        # Typed correctly but absent from the exported training table.
        ({"own_f00": "cont", "own_f01": "ord", "own_f02": "bin"},
         ["own_f00", "own_f01"], "exported X_train"),
    ],
)  # fmt: skip
def test_type_mismatches_and_dropped_features_fail(kinds, exported, message):
    with pytest.raises(dfa.TypeCheckError, match=message):
        dfa.check_inferred_types(
            fake_inferred_types(kinds), EXPECTED, exported or list(EXPECTED)
        )


def test_type_check_reads_a_fake_inferred_types_csv(tmp_path):
    inspection = tmp_path / "train" / "hash" / "inspection"
    inspection.mkdir(parents=True)
    fake_inferred_types({"own_f00": "id", "own_f01": "ord", "own_f02": "bin"}).to_csv(
        inspection / "inferred_types.csv"
    )
    inferred = pd.read_csv(dfa.find_inferred_types(tmp_path), index_col=0)
    with pytest.raises(
        dfa.TypeCheckError, match="own_f00: ours continuous, df-analyze id"
    ):
        dfa.check_inferred_types(inferred, EXPECTED, list(EXPECTED))


# ---- select_model uses the saved grouped folds ----------------------------
def test_select_model_passes_the_saved_folds_in_train_order(tmp_path, monkeypatch):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    assert main(["--config", str(path), "--to", "df_analyze_input"]) == 0
    p = get_paths(config)
    # A verified df-analyze run (faked): what select_model needs to start.
    run = p.df_analyze_output_dir / "20260101T000000Z"
    export = run / "train" / "hash" / "results" / "test00"
    export.mkdir(parents=True)
    (export / "X_test_00.csv").write_text("placeholder\n")
    (run / "df_analyze.log").write_text("no tuning lines\n")
    for check in (pipeline.SPLIT_CHECK, pipeline.TYPE_CHECK):
        (run / check).write_text("{}")
    (p.df_analyze_output_dir / "latest.txt").write_text(run.name + "\n")

    seen = {}

    def fake_cv_select(ctx, script, args, log_path):
        assert script.endswith("cv_select.py")
        opts = dict(zip(args[::2], args[1::2], strict=False))
        seen["folds"] = pd.read_csv(opts["--folds-file"])
        pd.DataFrame(
            [{"model_cls": "LightGBMClassifier", "selection": "none",
              "embed_selector": "", "settings": "tuned", "pr_auc_mean": 0.5,
              "error": ""}]
        ).to_csv(opts["--out"], index=False)  # fmt: skip

    monkeypatch.setattr(pipeline, "_run_dfa", fake_cv_select)
    assert main(["--config", str(path), "--only", "select_model"]) == 0
    train_ids = pd.read_csv(p.train_ids)
    saved = pd.read_csv(p.cv_folds).set_index(C.NODE_ID)["fold"]
    folds = seen["folds"]
    # One fold per training row, in the train table's row order, equal to the
    # saved grouped folds (never folds made up by the CV script).
    assert folds[C.NODE_ID].tolist() == train_ids[C.NODE_ID].tolist()
    assert np.array_equal(folds["fold"], saved.loc[train_ids[C.NODE_ID]])
    groups = train_ids.assign(fold=folds["fold"]).groupby(C.GROUP_ID)["fold"]
    assert groups.nunique().max() == 1


def test_df_analyze_input_uses_df_analyze_names(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=120)
    assert main(["--config", str(path), "--to", "df_analyze_input"]) == 0
    p = get_paths(config)
    train = pd.read_parquet(p.df_analyze_input_dir / "train.parquet")
    assert list(train.columns) == [f"own_f{j:02d}" for j in range(5)] + [C.TARGET]
    info = json.loads((p.df_analyze_input_dir / "feature_set.json").read_text())
    assert info["columns"]["own_f03"]["block_column"] == "own__f03"
    # Every name is already what df-analyze would use, so it renames nothing.
    assert all(dfa.df_analyze_name(c) == c for c in train.columns)
