"""shap_explain.py's SHAP calls for real, in df-analyze's environment with shap.

Skipped when df-analyze's clone (at the pinned commit) or uv is not available, or if
shap cannot be fetched. The subprocess takes several seconds to start.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from fraud_detect.config import REPO_ROOT, load_config
from fraud_detect.config import df_analyze_dir as find_df_analyze
from fraud_detect.external import df_analyze_commit


def test_interventional_shap_uses_the_whole_background_on_both_tree_libraries() -> None:
    config = load_config()
    dfa = find_df_analyze(config)
    if (
        shutil.which("uv") is None
        or not (dfa / "df-analyze.py").is_file()
        or df_analyze_commit(dfa) != config["df_analyze"]["commit"]
    ):
        pytest.skip("df-analyze (at the pinned commit) or uv is not available")
    script = REPO_ROOT / "tests" / "dfa" / "tiny_shap_check.py"
    done = subprocess.run(
        [
            "uv", "run", "--directory", str(dfa), "--python", ">=3.13.11,<3.14",
            "--with", "shap", "python", str(script),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "DF_ANALYZE_DIR": str(dfa)},
        check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(x for x in done.stdout.splitlines() if x.startswith("RESULT "))
    results = json.loads(line.removeprefix("RESULT "))
    assert set(results) == {"LightGBMClassifier", "CatBoostClassifier"}
    for result in results.values():
        assert result["margin_vs_prob"] < 1e-6
        assert result["shape"] == [50, 3]
        assert result["additivity"] < 1e-4
        # The base is the mean over the whole background, which shap's default masker
        # would have subsampled to 100 of these 300 rows.
        assert abs(result["base"] - result["background_mean"]) < 1e-6
        # The two modes are different games, so they are recorded and never mixed.
        assert result["modes_differ"]
        assert result["top_column"] in {"a", "b"}
