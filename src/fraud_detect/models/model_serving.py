"""Answering prediction requests with exported models, one JSON object per line.

scripts/dfa/serve_model.py runs this in df-analyze's environment; the dashboard
talks to it through dashboard/worker.py. Like model_export.py this uses only the
standard library, numpy and pandas, so all of it is plain logic, tested in the
project environment with a fake model.

Requests, each on one line:

    {"op": "info"}
    {"op": "predict", "model": "<key>", "columns": [...], "rows": [[...], ...]}

Every request gets exactly one answer line, with `"ok": true` or
`"ok": false, "error": "..."`. A bad request never stops the server.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TextIO

import numpy as np
import pandas as pd

from fraud_detect.models import model_export as mx


class RequestError(ValueError):
    """A request that cannot be answered; the message goes back to the caller."""


@dataclass
class LoadedModel:
    key: str
    meta: dict[str, Any]  # model_a.json
    estimator: Any  # has predict_proba(table)
    warnings: list[str] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        return {
            "model": self.meta["model"],
            "selection": self.meta["selection"],
            "feature_set": self.meta["feature_set"],
            "dataset": self.meta["dataset"],
            "threshold": float(self.meta["threshold"]),
            "n_input_columns": len(self.meta["input_columns"]),
            "warnings": list(self.warnings),
        }


def table_from_request(request: dict[str, Any]) -> pd.DataFrame:
    """The request's rows as a float table; every row must have every column."""
    columns, rows = request.get("columns"), request.get("rows")
    if not isinstance(columns, list) or not columns:
        raise RequestError("`columns` must be a non-empty list")
    if not isinstance(rows, list) or not rows:
        raise RequestError("`rows` must be a non-empty list")
    if any(not isinstance(r, list) or len(r) != len(columns) for r in rows):
        raise RequestError(f"Every row must have {len(columns)} values")
    try:
        return pd.DataFrame(rows, columns=columns, dtype=float)
    except (TypeError, ValueError) as e:
        raise RequestError(f"Rows must be numbers: {e}") from e


def predict(model: LoadedModel, table: pd.DataFrame) -> list[float]:
    """P(fraud) per row. Columns are checked first: a model given columns in the
    wrong order silently predicts nonsense, so a mismatch is an error, never a
    silent reorder. The output is checked too."""
    try:
        columns = mx.check_columns(
            table.columns, model.meta["input_columns"], model.meta["prepared_columns"]
        )
    except mx.ColumnMismatchError as e:
        raise RequestError(str(e)) from e
    output = np.asarray(model.estimator.predict_proba(table[columns]), dtype=float)
    if output.ndim != 2 or output.shape != (len(table), 2):
        raise RequestError(
            f"The model returned shape {output.shape}, not ({len(table)}, 2)"
        )
    prob = output[:, 1]
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise RequestError("The model returned probabilities outside [0, 1]")
    return [float(p) for p in prob]


def handle_request(request: Any, models: dict[str, LoadedModel]) -> dict[str, Any]:
    if not isinstance(request, dict):
        return {"ok": False, "error": "A request must be a JSON object"}
    op = request.get("op")
    try:
        if op == "info":
            return {"ok": True, "models": {k: m.info() for k, m in models.items()}}
        if op == "predict":
            key = request.get("model")
            if key not in models:
                raise RequestError(f"Unknown model {key!r}; loaded: {sorted(models)}")
            model = models[key]
            prob = predict(model, table_from_request(request))
            return {
                "ok": True,
                "prob": prob,
                "threshold": float(model.meta["threshold"]),
                "model": model.meta["model"],
                "selection": model.meta["selection"],
            }
        raise RequestError(f"Unknown op {op!r}")
    except RequestError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # the server outlives any one request
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}


def serve(lines: Iterable[str], out: TextIO, models: dict[str, LoadedModel]) -> None:
    """Answer each input line until the input closes."""
    for line in lines:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            answer: dict[str, Any] = {"ok": False, "error": f"Not JSON: {e}"}
        else:
            answer = handle_request(request, models)
        out.write(json.dumps(answer, allow_nan=False) + "\n")
        out.flush()
