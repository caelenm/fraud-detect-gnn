"""run.py stage selection."""

from __future__ import annotations

import pytest

from fraud_detect.cli import main, select_stages
from fraud_detect.pipeline import STAGE_NAMES


def test_select_all_by_default():
    assert select_stages(None, None, None) == STAGE_NAMES


def test_select_range_and_only():
    assert select_stages("sample", "split", None) == ["sample", "split"]
    assert select_stages(None, None, "embed") == ["embed"]


def test_invalid_selections():
    with pytest.raises(ValueError):
        select_stages("split", "sample", None)
    with pytest.raises(ValueError):
        select_stages("load", None, "embed")


def test_list_runs_without_side_effects(capsys):
    assert main(["--list"]) == 0
    assert "df_analyze" in capsys.readouterr().out
