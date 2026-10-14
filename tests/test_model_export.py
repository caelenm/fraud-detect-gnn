"""Model A export: input-column checks, model_a.json, and the refit decision.

These test the logic shared with scripts/dfa/export_model.py and predict.py
in the project environment. The same scripts are run for real, in
df-analyze's environment, by test_export_model_dfa.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from helpers import tmp_config
from synthetic import fake_model_a_metadata

from fraud_detect.config import get_paths
from fraud_detect.models import model_export as mx
from fraud_detect.pipeline import export_outputs, get_stage

INPUT = ["own_f02", "own_f00"]
PREPARED = ["own_f00", "own_f01", "own_f02"]


def test_columns_in_the_models_order_pass():
    assert mx.check_columns(INPUT, INPUT, PREPARED) == INPUT


def test_the_full_prepared_table_passes_and_gives_the_models_columns():
    assert mx.check_columns(PREPARED, INPUT, PREPARED) == INPUT


@pytest.mark.parametrize(
    ("found", "message"),
    [
        (["own_f00", "own_f02"], "reordered"),  # model's columns, wrong order
        (["own_f00", "own_f02", "own_f01"], "reordered"),  # prepared, wrong order
        (["own_f02"], "missing"),
        (["own_f02", "own_f00", "target"], "unexpected"),
        (["own_f02", "own_f00", "own_f02"], "reordered or repeated"),
    ],
)
def test_column_mismatch_fails_loudly(found, message):
    with pytest.raises(mx.ColumnMismatchError, match=message):
        mx.check_columns(found, INPUT, PREPARED)


def test_metadata_round_trips(tmp_path):
    meta = fake_model_a_metadata()
    path = tmp_path / mx.MODEL_A_META
    mx.write_metadata(path, meta)
    assert mx.read_metadata(path) == meta


def test_incomplete_or_unknown_metadata_is_refused(tmp_path):
    meta = fake_model_a_metadata()
    del meta["input_columns"]
    with pytest.raises(mx.MetadataError, match="input_columns"):
        mx.write_metadata(tmp_path / "a.json", meta)
    path = tmp_path / mx.MODEL_A_META
    mx.write_metadata(path, fake_model_a_metadata())
    path.write_text(
        path.read_text().replace('"format_version": 1', '"format_version": 9')
    )
    with pytest.raises(mx.MetadataError, match="format"):
        mx.read_metadata(path)
    with pytest.raises(mx.MetadataError, match="export_model"):
        mx.read_metadata(tmp_path / "missing.json")


class FittedModel:
    """Stand-in for a decoded df-analyze model (no data, fixed outputs)."""

    def __init__(self, probs: np.ndarray, tuned_model: object = "fitted") -> None:
        self.probs = probs
        self.tuned_model = object() if tuned_model == "fitted" else tuned_model

    def predict_proba(self, X: object) -> np.ndarray:
        return self.probs


SAVED = np.array([[0.9, 0.1], [0.2, 0.8]])


def check(model: object) -> dict:
    return mx.check_combination(model, lambda m: m.predict_proba(None), SAVED, 1e-9)


def test_a_string_placeholder_triggers_the_refit_path():
    # What df-analyze saves (fail_safe=str) for an object jsonpickle cannot encode.
    result = check("<df_analyze.models.gandalf.GandalfEstimator object at 0x7f00>")
    assert result["decoded"] is False and result["ok"] is False
    assert mx.model_a_source(result) == "refit"


def test_a_placeholder_inside_the_model_triggers_the_refit_path():
    result = check(FittedModel(SAVED, tuned_model="<lightgbm.LGBMClassifier object>"))
    assert result["decoded"] is False
    assert mx.model_a_source(result) == "refit"


def test_a_model_that_reproduces_its_probabilities_is_used_as_decoded():
    result = check(FittedModel(SAVED.copy()))
    assert result == {"decoded": True, "intact": True, "max_abs_diff": 0.0,
                      "labels_match": None, "ok": True, "error": ""}  # fmt: skip
    assert mx.model_a_source(result) == "decoded"


def test_a_model_with_unrebuilt_parts_triggers_the_refit_path():
    # Reproducing the probabilities is not enough if part of the model could
    # not be rebuilt from the file (a placeholder somewhere inside it).
    result = mx.check_combination(
        FittedModel(SAVED.copy()), lambda m: m.predict_proba(None), SAVED, 1e-9,
        placeholders=1,
    )  # fmt: skip
    assert result["decoded"] and not result["intact"] and not result["ok"]
    assert result["max_abs_diff"] == 0.0 and "could not be rebuilt" in result["error"]
    assert mx.model_a_source(result) == "refit"


def test_labels_must_match_too():
    def run(labels: list[int]) -> dict:
        return mx.check_combination(
            FittedModel(SAVED.copy()), lambda m: m.predict_proba(None), SAVED, 1e-9,
            predict_labels=lambda m: np.array(labels), saved_labels=[0, 1],
        )  # fmt: skip

    assert run([0, 1])["ok"] and run([0, 1])["labels_match"] is True
    wrong = run([1, 1])
    assert wrong["labels_match"] is False and not wrong["ok"]
    assert "labels differ" in wrong["error"]


def test_different_probabilities_or_a_failing_model_trigger_the_refit_path():
    off = check(FittedModel(SAVED + 1e-6))
    assert off["decoded"] and not off["ok"] and "differ" in off["error"]
    assert mx.model_a_source(off) == "refit"

    class Broken(FittedModel):
        def predict_proba(self, X: object) -> np.ndarray:
            raise RuntimeError("checkpoint not found")

    broken = check(Broken(SAVED))
    assert broken["decoded"] and not broken["ok"] and "checkpoint" in broken["error"]
    assert mx.max_abs_diff(SAVED, SAVED[:1]) == math.inf


def test_tolerances_and_native_formats():
    assert mx.tolerance_for("lgbm") == mx.CPU_TOLERANCE
    assert mx.tolerance_for("catboost") == mx.GPU_TOLERANCE
    assert mx.native_format("catboost") == "cbm" and mx.native_format("lr") == "joblib"
    with pytest.raises(ValueError, match="not implemented"):
        mx.native_format("gandalf")
    assert mx.selection_label("embed", "linear") == "embed_linear"
    assert mx.selection_label("pred", "") == "pred"


def test_version_differences_are_reported():
    recorded = {"python": "3.13.0", "lightgbm": "4.6.0"}
    assert mx.version_warnings(recorded, dict(recorded)) == []
    warnings = mx.version_warnings(recorded, {"python": "3.13.0", "lightgbm": "4.7.0"})
    assert len(warnings) == 1 and "lightgbm 4.7.0" in warnings[0]


def test_stage_outputs_include_the_model_file_named_in_model_a_json(tmp_path):
    _, config = tmp_config(tmp_path)
    p = get_paths(config)
    stage = get_stage("export_model")
    assert [o.name for o in export_outputs(p)] == [mx.MODELS_CHECK, mx.MODEL_A_META]
    p.model_dir.mkdir(parents=True)
    mx.write_metadata(p.model_dir / mx.MODEL_A_META, fake_model_a_metadata())
    (p.model_dir / mx.MODELS_CHECK).write_text("model\n", encoding="utf-8")
    assert export_outputs(p)[-1] == p.model_dir / "model_a.joblib"
    assert not stage.is_done(p)  # the model file itself is missing
    (p.model_dir / "model_a.joblib").write_bytes(b"placeholder")
    assert stage.is_done(p)
