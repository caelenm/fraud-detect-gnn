"""The comparison page (report/compare.py, scripts/compare_report.py).

The dataset is a fake CARE-GNN file and every df-analyze run is faked
(tests/helpers.py); df-analyze and torch never run. Everything is in tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import replace

import numpy as np
import pytest
from helpers import fake_dataset, fake_df_analyze_run, fake_gnnssl_outputs

from fraud_detect import columns as C
from fraud_detect.cli import main
from fraud_detect.config import REPO_ROOT, get_paths
from fraud_detect.report import compare

# B's Model A (catboost) is given a stronger fake signal than A's (lgbm).
CV_A = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.6}
CV_B = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.5, ("catboost", "none"): 0.7}


def finish(path, config, feature_set: str, cv: dict) -> None:
    """Pipeline through df_analyze_input, a faked df-analyze run, then the
    report stage, for one feature set."""
    args = ["--config", str(path), "--set", f"feature_set={feature_set}"]
    assert main([*args, "--to", "df_analyze_input"]) == 0
    fake_df_analyze_run(get_paths({**config, "feature_set": feature_set}), cv)
    assert main([*args, "--only", "df_analyze_report"]) == 0


@pytest.fixture
def two_runs(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    finish(path, config, "m1_own", CV_A)
    finish(path, config, "m2_own_nbr", CV_B)
    return path, config


def load_script():
    spec = importlib.util.spec_from_file_location(
        "compare_report", REPO_ROOT / "scripts" / "compare_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_page_builds_offline_from_synthetic_runs(two_runs):
    _, config = two_runs
    result = compare.build(config, "m1_own", "m2_own_nbr", n_resamples=200,
                           n_random_cards=10)  # fmt: skip
    html, m = result.html, result.metrics
    # Self-contained: no resource or link of any kind leaves the file.
    assert "http://" not in html and "https://" not in html
    assert re.search(r"\b(src|href)\s*=", html) is None
    assert "@import" not in html and "url(" not in html
    assert "Group only: do not upload or commit" in html
    assert html.count("<script") == 2  # the node data and the inline script
    # Every gallery node has a card; cards hold one neighbourhood per relation.
    gallery_ids = re.findall(r'<button type="button" data-node="(\d+)"', html)
    assert gallery_ids and all(f'id="card-{i}"' in html for i in gallery_ids)
    assert "A-A-A" in html and "B-B-B" in html
    assert html.count('class="pt ') == m["n_test"]
    assert "nothing to show here" in html  # neither side uses gnnssl
    for side in ("a", "b"):
        assert m[side]["pr_auc"]["low"] <= m[side]["pr_auc"]["estimate"]
        assert 0 <= m[side]["recall_at_k"] <= 1
    assert m["a"]["model"] == "lgbm" and m["b"]["model"] == "catboost"
    assert sum(m["outcomes"]["fraud"].values()) == m["n_test_fraud"]


def test_a_better_b_gives_a_positive_paired_difference():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.15).astype(int)
    groups = np.arange(600) // 2
    prob_a = rng.uniform(size=600)  # no skill
    prob_b = np.clip(0.6 * y + rng.uniform(0, 0.5, 600), 0, 1)  # real skill
    d = compare.paired_difference(y, groups, prob_a, prob_b, "pr_auc", 300, seed=1)
    assert d["estimate"] > 0 and d["low"] > 0 and d["share_above_zero"] > 0.95
    reverse = compare.paired_difference(y, groups, prob_b, prob_a, "pr_auc", 300, 1)
    assert reverse["estimate"] == pytest.approx(-d["estimate"])


def test_recall_at_k_and_labels():
    y = np.array([1, 0, 1, 0, 0])
    prob = np.array([0.9, 0.8, 0.1, 0.7, 0.0])
    assert compare.recall_at_k(y, prob, np.arange(5)) == 0.5  # top 2: one fraud
    assert compare.feature_label("nbr__upu_mean_f13") == "nbr U-P-U mean f13"
    assert compare.feature_label("gnnssl__uvu_pc03") == "gnnssl U-V-U pc03"
    assert compare.feature_label("own__f02") == "own f02"


def test_different_test_nodes_stop_the_comparison(two_runs, monkeypatch, capsys):
    path, config = two_runs
    a = compare.load_side(config, "m1_own")
    shuffled = a.test_ids.assign(**{C.NODE_ID: a.test_ids[C.NODE_ID].to_numpy()[::-1]})
    with pytest.raises(compare.CompareError, match="same test"):
        compare.check_same_test_nodes(a, replace(a, test_ids=shuffled))

    real = compare.load_side

    def mismatched(config, feature_set):
        side = real(config, feature_set)
        return replace(side, test_ids=shuffled) if feature_set == "m2_own_nbr" else side

    monkeypatch.setattr(compare, "load_side", mismatched)
    script = load_script()
    args = ["--config", str(path), "--a", "m1_own", "--b", "m2_own_nbr"]
    assert script.main(args) == 2
    assert "same test" in capsys.readouterr().err
    out = get_paths(config).compare_dir("m1_own", "m2_own_nbr")
    assert not (out / "compare.html").exists()


def test_script_writes_page_and_metrics(two_runs):
    path, config = two_runs
    script = load_script()
    args = ["--config", str(path), "--a", "m1_own", "--b", "m2_own_nbr",
            "--resamples", "100", "--random-cards", "5"]  # fmt: skip
    assert script.main(args) == 0
    out = get_paths(config).compare_dir("m1_own", "m2_own_nbr")
    metrics = json.loads((out / "compare_metrics.json").read_text(encoding="utf-8"))
    assert metrics["delta_pr_auc"]["n_resamples"] <= 100
    assert "share_above_zero" in metrics["delta_pr_auc"]
    assert (out / "compare.html").stat().st_size > 0
    assert script.main(["--config", str(path), "--a", "m1_own", "--b", "nope"]) == 2


def test_embedding_view_when_a_side_uses_gnnssl(two_runs):
    path, config = two_runs
    p = get_paths({**config, "feature_set": "m5_own_gnnssl"})
    fake_gnnssl_outputs(p, ("aaa", "bbb"), n_nodes=200)
    finish(path, config, "m5_own_gnnssl", CV_B)
    html = compare.build(config, "m1_own", "m5_own_gnnssl", n_resamples=50,
                         n_random_cards=5).html  # fmt: skip
    assert "A-A-A DGI embedding (2-D PCA)" in html and "Own features (2-D PCA)" in html
    assert "gnnssl A-A-A pc0" in html or "gnnssl B-B-B pc0" in html
