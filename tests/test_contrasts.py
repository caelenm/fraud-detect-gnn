"""Contrasts across split repeats and the pre-registered verdict (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fraud_detect.models import contrasts


def test_corrected_t_test_matches_the_formula_and_is_wider_than_plain():
    d = np.array([0.02, 0.01, 0.03, -0.01, 0.02, 0.015, 0.0, 0.025, 0.01, 0.02])
    out = contrasts.corrected_t_test(d, 0.25)
    t = d.mean() / np.sqrt((1 / 10 + 0.25) * d.var(ddof=1))
    assert out["t"] == pytest.approx(t) and out["df"] == 9
    assert out["p"] == pytest.approx(2 * stats.t.sf(abs(t), 9))
    plain = stats.ttest_1samp(d, 0.0).pvalue
    assert out["p"] > plain  # the correction is conservative


def test_verdict_rule():
    assert contrasts.verdict([0.02, 0.03, 0.025], p=0.01) == contrasts.SUPPORTED
    # one split negative
    assert contrasts.verdict([0.02, -0.01, 0.03], 0.01) == contrasts.NOT_DISTINGUISHABLE
    # mean not above its SD
    assert contrasts.verdict([0.001, 0.05, 0.002], 0.01) == contrasts.NOT_DISTINGUISHABLE
    # p too large
    assert contrasts.verdict([0.02, 0.03, 0.025], 0.2) == contrasts.NOT_DISTINGUISHABLE
    assert contrasts.verdict([-0.02, -0.03, 0.01], 0.01) == contrasts.WORSE


def oof_table() -> pd.DataFrame:
    """Invented out-of-fold predictions for two tuned models and a default."""
    rows = []
    rng = np.random.default_rng(0)
    for cls, sel, selector, settings in (
        ("LightGBMClassifier", "embed", "linear", "tuned"),
        ("LightGBMClassifier", "embed", "lgbm", "tuned"),
        ("LightGBMClassifier", "embed", "linear", "default"),
        ("CatBoostClassifier", "none", np.nan, "tuned"),
    ):
        for row in range(20):
            rows.append({"model_cls": cls, "selection": sel, "embed_selector": selector,
                         "settings": settings, "row": row, "fold": row % 4,
                         "prob": rng.uniform()})  # fmt: skip
    return pd.DataFrame(rows)


def test_model_a_rows_and_fold_scores():
    oof = oof_table()
    rows = contrasts.model_a_oof(
        oof, {"model": "lgbm", "selection": "embed", "embed_selector": "linear"}
    )
    expected = oof[(oof.embed_selector == "linear") & (oof.settings == "tuned")]
    assert rows["prob"].tolist() == expected.sort_values("row")["prob"].tolist()
    cat = contrasts.model_a_oof(
        oof, {"model": "catboost", "selection": "none", "embed_selector": ""}
    )
    assert len(cat) == 20
    y = (np.arange(20) // 4) % 2  # both classes in every fold
    folds = contrasts.fold_pr_auc(rows, y)
    assert list(folds.index) == [0, 1, 2, 3] and folds.between(0, 1).all()
    with pytest.raises(ValueError):
        contrasts.model_a_oof(oof, {"model": "knn", "selection": "none"})
