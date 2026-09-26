"""The df-analyze clone must be at the pinned commit."""

from __future__ import annotations

import subprocess

import pytest

from fraud_detect.external import (
    ExternalToolError,
    check_df_analyze_dir,
    df_analyze_commit,
)


def fake_clone(tmp_path):
    """An empty git repo containing a placeholder script (no data)."""
    (tmp_path / "df-embed.py").write_text("# placeholder\n")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    return df_analyze_commit(tmp_path)


def test_matching_commit_passes(tmp_path):
    commit = fake_clone(tmp_path)
    check_df_analyze_dir(tmp_path, "df-embed.py", commit)


def test_other_commit_stops_with_instructions(tmp_path):
    fake_clone(tmp_path)
    with pytest.raises(ExternalToolError, match="git -C .* checkout 0000"):
        check_df_analyze_dir(tmp_path, "df-embed.py", "0000")


def test_check_can_be_disabled(tmp_path):
    fake_clone(tmp_path)
    check_df_analyze_dir(tmp_path, "df-embed.py", None)


def test_missing_script_stops(tmp_path):
    with pytest.raises(ExternalToolError, match="not found"):
        check_df_analyze_dir(tmp_path, "df-embed.py", None)
