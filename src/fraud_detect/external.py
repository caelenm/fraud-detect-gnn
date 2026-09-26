"""Running df-analyze scripts in df-analyze's own uv environment.

df-analyze is never imported into this project's environment. Its scripts are
run as subprocesses with `uv run --directory <df-analyze clone>`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class ExternalToolError(RuntimeError):
    """Raised when df-analyze is missing or one of its scripts fails."""


def df_analyze_commit(dfa_dir: Path) -> str | None:
    """The df-analyze clone's checked-out commit, or None if it is not a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(dfa_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def check_df_analyze_dir(
    dfa_dir: Path, script: str, expected_commit: str | None = None
) -> None:
    if not (dfa_dir / script).is_file():
        raise ExternalToolError(
            f"{script} not found in {dfa_dir}. Clone df-analyze there (see README "
            "Setup), or set DF_ANALYZE_DIR or `df_analyze.dir` in the config."
        )
    if shutil.which("uv") is None:
        raise ExternalToolError("`uv` is not on PATH; it is needed to run df-analyze.")
    if expected_commit:
        actual = df_analyze_commit(dfa_dir)
        if actual != expected_commit:
            raise ExternalToolError(
                f"df-analyze in {dfa_dir} is at commit {actual}, but this pipeline is "
                f"tested against {expected_commit}. Run:\n"
                f"  git -C {dfa_dir} fetch origin\n"
                f"  git -C {dfa_dir} checkout {expected_commit}\n"
                f"  uv sync --locked --directory {dfa_dir}\n"
                "or set `df_analyze.commit: null` in the config to skip this check."
            )


def run_df_analyze_script(
    dfa_dir: Path,
    script: str,
    args: list[str],
    log_path: Path,
    expected_commit: str | None = None,
) -> list[str]:
    """Run `uv run --directory dfa_dir python <script> <args>`, tee output to a
    log file, and raise if it fails. Returns the command that was run."""
    check_df_analyze_dir(dfa_dir, script, expected_commit)
    cmd = ["uv", "run", "--directory", str(dfa_dir), "python", script, *args]
    env = dict(os.environ)
    # Our own virtual environment must not leak into df-analyze's.
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONUNBUFFERED"] = "1"  # stream progress instead of buffering it
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise ExternalToolError(f"{script} exited with code {code}. See {log_path}")
    return cmd
