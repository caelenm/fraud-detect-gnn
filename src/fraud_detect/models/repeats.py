"""Model A across independently seeded grouped splits (split.n_repeats).

Amazon is small, so its scores depend noticeably on the split (plan §4).
With `split.n_repeats > 1` the pipeline runs once per split repeat; this
module collects each repeat's model_a.json and reports the mean and standard
deviation of Model A's test metrics, with the model each repeat chose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

METRICS = ("pr_auc", "auroc", "f1", "precision", "recall")


def summarize(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± SD (ddof=1) of Model A's test metrics over repeats."""
    if len(summaries) < 2:
        raise ValueError("A repeat summary needs at least two repeats")
    out: dict[str, Any] = {
        "n_repeats": len(summaries),
        "runs": [s["run"] for s in summaries],
        "model_a": [
            f"{s['model_a']['model']} ({s['model_a']['selection']})" for s in summaries
        ],
        "test_positive_rate": [s["test_positive_rate"] for s in summaries],
        "metrics": {},
    }
    for metric in METRICS:
        values = np.array([float(s["model_a"][metric]) for s in summaries])
        out["metrics"][metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return out


def markdown(summary: dict[str, Any], dataset: str, feature_set: str) -> str:
    n = summary["n_repeats"]
    lines = [
        f"# Model A over {n} grouped splits: {dataset}, feature set {feature_set}",
        "",
        "Each repeat is an independently seeded grouped split with its own "
        "df-analyze run and Model A choice. Mean ± SD (n - 1) of Model A's test "
        "metrics; F1, precision and recall at each repeat's out-of-fold threshold.",
        "",
        "| metric | mean | SD | " + " | ".join(f"repeat {k}" for k in range(n)) + " |",
        "|---|---|---|" + "|".join("---" for _ in range(n)) + "|",
    ]
    for metric, s in summary["metrics"].items():
        values = " | ".join(f"{v:.4f}" for v in s["values"])
        lines.append(f"| {metric} | {s['mean']:.4f} | {s['sd']:.4f} | {values} |")
    lines += [
        "",
        "Model A per repeat: " + ", ".join(summary["model_a"]),
        "",
        "No-skill PR-AUC per repeat (test positive rate): "
        + ", ".join(f"{r:.4f}" for r in summary["test_positive_rate"]),
        "",
    ]
    return "\n".join(lines)
