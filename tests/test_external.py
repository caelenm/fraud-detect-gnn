"""The df-analyze clone must be at the pinned commit."""

from __future__ import annotations

import subprocess

import pytest

from fraud_detect.external import (
    ExternalToolError,
    check_df_analyze_dir,
    df_analyze_commit,
)


@pytest.fixture(autouse=True)
def recent_uv(monkeypatch):
    """Tests must not depend on the uv installed on the machine running them."""
    import fraud_detect.external as ext

    monkeypatch.setattr(ext, "uv_version", lambda: ext.MIN_UV_VERSION)


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


def test_uv_command_pins_python(tmp_path, monkeypatch):
    """The subprocess command passes --python so uv cannot pick 3.14."""
    import fraud_detect.external as ext

    fake_clone(tmp_path)
    seen = {}

    class FakeProc:
        stdout = iter(["done\n"])

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(ext.subprocess, "Popen", fake_popen)
    ext.run_df_analyze_script(
        tmp_path, "df-embed.py", ["--x"], tmp_path / "log.txt",
        python=">=3.13.11,<3.14",
    )  # fmt: skip
    cmd = seen["cmd"]
    assert cmd[cmd.index("--python") + 1] == ">=3.13.11,<3.14"
    assert cmd.index("--python") < cmd.index("df-embed.py")


def test_parse_uv_version():
    from fraud_detect.external import parse_uv_version

    assert parse_uv_version("uv 0.9.7") == (0, 9, 7)
    assert parse_uv_version("uv 0.12.19 (x86_64-unknown-linux-gnu)") == (0, 12, 19)
    assert parse_uv_version("something else") is None


def test_old_uv_stops_with_upgrade_help(tmp_path, monkeypatch):
    import fraud_detect.external as ext

    fake_clone(tmp_path)
    monkeypatch.setattr(ext, "uv_version", lambda: (0, 9, 7))
    with pytest.raises(ExternalToolError, match="uv tool install"):
        check_df_analyze_dir(tmp_path, "df-embed.py", None)
    monkeypatch.setattr(ext, "uv_version", lambda: (0, 9, 16))
    check_df_analyze_dir(tmp_path, "df-embed.py", None)
