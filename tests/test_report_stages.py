"""The report stages end to end on synthetic df-analyze outputs.

The dataset is a fake CARE-GNN file and the df-analyze run is faked
(tests/helpers.py); df-analyze itself never runs. Everything is in tmp_path.
"""

from __future__ import annotations

import json

import pytest
import yaml
from helpers import fake_dataset, fake_df_analyze_run

from fraud_detect.cli import main
from fraud_detect.config import get_paths

CV = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.6, ("catboost", "none"): 0.7}


@pytest.fixture
def finished_run(tmp_path):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    assert main(["--config", str(path), "--to", "df_analyze_input"]) == 0
    p = get_paths(config)
    fake_df_analyze_run(p, CV)
    return path, p


def test_report_and_web_report_render_from_synthetic_outputs(finished_run):
    path, p = finished_run
    assert main(["--config", str(path), "--from", "df_analyze_report"]) == 0
    summary = json.loads((p.model_reports_dir / "model_a.json").read_text())
    a = summary["model_a"]
    assert a["model"] == "catboost"  # highest shared-CV PR-AUC
    # The threshold came from the out-of-fold training predictions.
    assert summary["model_a_threshold"]["n_train_rows"] == summary["n_train"]
    assert a["threshold"] == pytest.approx(summary["model_a_threshold"]["threshold"])
    pr = summary["intervals"][0]
    assert pr["metric"] == "pr_auc" and pr["low"] <= a["pr_auc"] <= pr["high"]
    report = (p.model_reports_dir / "model_a_report.md").read_text()
    assert "group-bootstrap" in report and "Decision threshold" in report

    page = p.web_report.read_text()
    assert "Node " in page and "95% CI" in page and "own__f0" in page
    assert "<script" not in page and "complaint" not in page.lower()


def test_report_warns_below_the_sanity_band(finished_run, capsys):
    path, p = finished_run
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    # The fake models are nearly perfect; only a perfect PR-AUC passes this.
    config["datasets"]["fake"]["sanity_band"] = {"pr_auc": 1.0, "auroc": 1.0}
    config["report"]["sanity_margin"] = 0.0
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["--config", str(path), "--only", "df_analyze_report"]) == 0
    assert "WARNING" in capsys.readouterr().out
    summary = json.loads((p.model_reports_dir / "model_a.json").read_text())
    assert "pipeline bug" in summary["sanity_warning"]
