"""Recording each df-analyze model's tuning budget.

df-analyze stops each tuning run at a per-model time limit, so the number of
Optuna trials a model actually gets depends on how slow it is per trial. The
`select_model` stage records this next to the shared cross-validation scores
(scripts/dfa/cv_select.py) so the comparison's budget is visible.
"""

from __future__ import annotations

import re

import pandas as pd

SECTION = re.compile(r"^Tuning (?P<model>.+) for selection=(?P<selection>\S+)\s*$")
# tqdm progress, e.g. "5/5 [04:04<00:00, 48.92s/it, 244.58/3600 seconds]"
PROGRESS = re.compile(
    r"(?P<done>\d+)/(?P<total>\d+) \[[^\]]*?, "
    r"(?P<elapsed>[\d.]+)/(?P<limit>\d+) seconds\]"
)
# Fraction of the time limit after which a stop counts as a timeout.
TIMEOUT_FRACTION = 0.98


def parse_tuning_budget(log_text: str) -> pd.DataFrame:
    """Trials completed and time used by each df-analyze tuning run.

    Parses the progress bars in df-analyze's log. `stopped_by` is "all trials"
    when every requested trial ran, "time limit" when the run used (nearly)
    all of its time limit, and otherwise "early stop" (df-analyze's patience
    rule or an exhausted search grid).
    """
    rows: list[dict] = []
    current: dict | None = None
    for line in log_text.replace("\r", "\n").splitlines():
        header = SECTION.match(line.strip())
        if header:
            current = {"model": header["model"], "selection": header["selection"]}
            rows.append(current)
            continue
        if current is None:
            continue  # progress bars before tuning belong to feature selection
        for m in PROGRESS.finditer(line):
            done = int(m["done"])
            if done >= current.get("trials_completed", -1):
                current["trials_completed"] = done
                current["trials_requested"] = int(m["total"])
                current["elapsed_s"] = float(m["elapsed"])
                current["time_limit_s"] = int(m["limit"])
    for row in rows:
        if "trials_completed" not in row:
            row["stopped_by"] = "unknown"
        elif row["trials_completed"] >= row["trials_requested"]:
            row["stopped_by"] = "all trials"
        elif row["elapsed_s"] >= TIMEOUT_FRACTION * row["time_limit_s"]:
            row["stopped_by"] = "time limit"
        else:
            row["stopped_by"] = "early stop"
    columns = [
        "model", "selection", "trials_completed", "trials_requested",
        "elapsed_s", "time_limit_s", "stopped_by",
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=columns)
