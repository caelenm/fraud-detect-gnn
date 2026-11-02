"""The model worker: its protocol as plain logic, and the dashboard's client
against a real subprocess running that protocol with an invented model.

The real model round trip, in df-analyze's environment, is in
test_export_model_dfa.py.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time

import numpy as np
import pytest
from synthetic import fake_model_a_metadata

from fraud_detect.dashboard.worker import ModelWorker, WorkerError
from fraud_detect.models.model_serving import (
    LoadedModel,
    RequestError,
    handle_request,
    predict,
    serve,
    table_from_request,
)

COLUMNS = ["own_f00", "own_f01"]
PREPARED = ["own_f00", "own_f01", "own_f02"]


class FakeEstimator:
    """Invented: P(fraud) is the first input column. Records what it was given."""

    def __init__(self, output=None, error=None) -> None:
        self.seen = None
        self.output = output
        self.error = error

    def predict_proba(self, table):
        self.seen = table
        if self.error:
            raise self.error
        if self.output is not None:
            return self.output
        p = table.iloc[:, 0].to_numpy()
        return np.column_stack([1 - p, p])


def model(estimator=None) -> LoadedModel:
    meta = fake_model_a_metadata()
    meta["threshold"] = 0.3
    meta["feature_set"] = "pilot_m1_own"
    return LoadedModel("k", meta, estimator or FakeEstimator(), ["old lightgbm"])


def ask(rows, columns=COLUMNS, key="k", estimator=None):
    return handle_request(
        {"op": "predict", "model": key, "columns": columns, "rows": rows},
        {"k": model(estimator)},
    )


def test_info_describes_each_model():
    answer = handle_request({"op": "info"}, {"k": model()})
    assert answer["ok"]
    info = answer["models"]["k"]
    assert info["model"] == "lgbm" and info["threshold"] == 0.3
    assert info["feature_set"] == "pilot_m1_own" and info["warnings"] == ["old lightgbm"]


def test_predict_returns_one_probability_per_row():
    answer = ask([[0.2, 9.0], [0.7, 9.0]])
    assert answer["ok"] and answer["prob"] == pytest.approx([0.2, 0.7])
    assert answer["threshold"] == 0.3 and answer["model"] == "lgbm"


def test_prepared_columns_are_cut_down_to_the_model_columns_in_order():
    estimator = FakeEstimator()
    answer = ask([[0.2, 5.0, 7.0]], columns=PREPARED, estimator=estimator)
    assert answer["ok"]
    assert list(estimator.seen.columns) == COLUMNS


@pytest.mark.parametrize(
    "columns",
    [
        ["own_f01", "own_f00"],  # reordered: the silent failure this guards against
        ["own_f00"],
        ["own_f00", "own_f01", "own_f99"],
        ["own_f02", "own_f01", "own_f00"],
    ],
)
def test_columns_that_do_not_match_are_refused_never_reordered(columns):
    estimator = FakeEstimator()
    answer = ask([[0.1] * len(columns)], columns=columns, estimator=estimator)
    assert not answer["ok"] and "columns" in answer["error"]
    assert estimator.seen is None


@pytest.mark.parametrize(
    "request_",
    [
        {"op": "predict", "model": "k", "columns": COLUMNS},
        {"op": "predict", "model": "k", "rows": [[1, 2]]},
        {"op": "predict", "model": "k", "columns": [], "rows": [[1, 2]]},
        {"op": "predict", "model": "k", "columns": COLUMNS, "rows": []},
        {"op": "predict", "model": "k", "columns": COLUMNS, "rows": [[0.1]]},
        {"op": "predict", "model": "k", "columns": COLUMNS, "rows": ["ab"]},
        {"op": "predict", "model": "k", "columns": COLUMNS, "rows": [["a", "b"]]},
        {"op": "predict", "model": "other", "columns": COLUMNS, "rows": [[0.1, 0.2]]},
        {"op": "nope"},
        {},
        ["not", "an", "object"],
        "text",
    ],
)
def test_a_bad_request_gets_an_error_answer(request_):
    answer = handle_request(request_, {"k": model()})
    assert answer["ok"] is False and answer["error"]


def test_table_from_request_makes_floats():
    table = table_from_request({"columns": COLUMNS, "rows": [[1, 2]]})
    assert table.dtypes.tolist() == [float, float]
    with pytest.raises(RequestError):
        table_from_request({"columns": COLUMNS, "rows": [[1]]})


@pytest.mark.parametrize(
    "output",
    [
        np.array([0.5, 0.5]),  # one dimension
        np.array([[0.5, 0.5, 0.0]]),  # three classes
        np.array([[0.5, 0.5], [0.5, 0.5]]),  # two rows for one
        np.array([[-0.5, 1.5]]),
        np.array([[0.5, np.nan]]),
        np.array([[0.5, np.inf]]),
    ],
)
def test_an_impossible_model_output_is_an_error(output):
    answer = ask([[0.2, 0.3]], estimator=FakeEstimator(output=output))
    assert answer["ok"] is False


def test_a_model_that_raises_does_not_stop_the_server():
    answer = ask([[0.2, 0.3]], estimator=FakeEstimator(error=RuntimeError("boom")))
    assert answer["ok"] is False and "boom" in answer["error"]


def test_predict_checks_rows_against_columns_directly():
    table = table_from_request({"columns": COLUMNS, "rows": [[0.4, 0.0]]})
    assert predict(model(), table) == pytest.approx([0.4])


def test_serve_answers_every_line_and_survives_bad_ones():
    lines = [
        "not json\n",
        "\n",
        json.dumps({"op": "info"}) + "\n",
        json.dumps(
            {"op": "predict", "model": "k", "columns": COLUMNS, "rows": [[0.25, 1.0]]}
        )
        + "\n",
    ]
    out = io.StringIO()
    serve(lines, out, {"k": model()})
    answers = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(answers) == 3  # the blank line is not a request
    assert answers[0]["ok"] is False and "Not JSON" in answers[0]["error"]
    assert answers[1]["ok"] and answers[2]["prob"] == pytest.approx([0.25])


# ---- The client, against a real subprocess --------------------------------
FAKE_WORKER = """
import json, os, pathlib, sys, time
import numpy as np
from fraud_detect.models.model_serving import LoadedModel, serve

mode, marker = sys.argv[1], pathlib.Path(sys.argv[2])

class Model:
    def predict_proba(self, table):
        if mode == "die_once" and not marker.exists():
            marker.write_text("died")
            os._exit(1)
        if mode == "die_always":
            os._exit(1)
        if mode == "hang_once" and not marker.exists():
            marker.write_text("hung")
            time.sleep(60)
        if mode == "slow":
            time.sleep(0.05)
        p = table.iloc[:, 0].to_numpy()
        return np.column_stack([1 - p, p])

meta = {"model": "fake", "selection": "none", "feature_set": "pilot_m1_own",
        "dataset": "fake", "threshold": 0.5, "input_columns": ["a"],
        "prepared_columns": ["a"]}
if mode == "garbage":
    sys.stdin.readline()
    print("this is not json", flush=True)
    sys.stdin.readline()
elif mode == "exit_now":
    sys.exit(3)
elif mode == "wrong_length":
    sys.stdin.readline()
    print(json.dumps({"ok": True, "models": {}}), flush=True)
    sys.stdin.readline()
    print(json.dumps({"ok": True, "prob": [0.1, 0.2]}), flush=True)
    sys.stdin.readline()
else:
    serve(sys.stdin, sys.stdout, {"k": LoadedModel("k", meta, Model())})
"""


def worker(tmp_path, mode: str, **kwargs) -> ModelWorker:
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, "utf-8")
    command = [sys.executable, str(script), mode, str(tmp_path / "marker")]
    kwargs.setdefault("start_timeout", 60)
    kwargs.setdefault("request_timeout", 30)
    return ModelWorker(command, log_path=tmp_path / "logs" / "worker.log", **kwargs)


def test_client_round_trip(tmp_path):
    w = worker(tmp_path, "normal")
    try:
        info = w.start()
        assert info["models"]["k"]["model"] == "fake" and w.alive()
        assert w.predict("k", ["a"], [[0.1], [0.9]]) == pytest.approx([0.1, 0.9])
        assert w.predict("k", ["a"], [[0.5]]) == pytest.approx([0.5])
        assert w.restarts == 0
        # Exact floats survive the trip: the verified check depends on it.
        value = 0.1234567890123456
        assert w.predict("k", ["a"], [[value]]) == [value]
    finally:
        w.close()
    assert not w.alive()
    assert (tmp_path / "logs" / "worker.log").is_file()


def test_large_requests_do_not_deadlock(tmp_path):
    w = worker(tmp_path, "normal")
    try:
        w.start()
        rows = [[float(i) / 40000] for i in range(40000)]
        prob = w.predict("k", ["a"], rows)
        assert len(prob) == 40000 and prob[123] == pytest.approx(123 / 40000)
    finally:
        w.close()


def test_a_refused_request_raises_and_the_worker_stays_up(tmp_path):
    w = worker(tmp_path, "normal")
    try:
        w.start()
        with pytest.raises(WorkerError, match="refused"):
            w.predict("nope", ["a"], [[0.1]])
        assert w.alive() and w.restarts == 0
        assert w.predict("k", ["a"], [[0.2]]) == pytest.approx([0.2])
    finally:
        w.close()


def test_a_worker_that_dies_is_restarted_once(tmp_path):
    w = worker(tmp_path, "die_once")
    try:
        w.start()
        assert w.predict("k", ["a"], [[0.3]]) == pytest.approx([0.3])
        assert w.restarts == 1 and (tmp_path / "marker").exists()
    finally:
        w.close()


def test_a_worker_that_keeps_dying_gives_an_error(tmp_path):
    w = worker(tmp_path, "die_always")
    try:
        w.start()
        with pytest.raises(WorkerError):
            w.predict("k", ["a"], [[0.3]])
        assert w.restarts == 1
        with pytest.raises(WorkerError):  # no second restart
            w.predict("k", ["a"], [[0.3]])
        assert w.restarts == 1
    finally:
        w.close()


def test_a_hung_worker_times_out_and_cannot_answer_the_next_request(tmp_path):
    w = worker(tmp_path, "hang_once", request_timeout=2)
    try:
        w.start()
        # The hung process is ended, so its late answer can never be taken for
        # the answer to the retried request.
        assert w.predict("k", ["a"], [[0.4]]) == pytest.approx([0.4])
        assert w.restarts == 1
    finally:
        w.close()


def test_output_that_is_not_json_is_an_error(tmp_path):
    w = worker(tmp_path, "garbage")
    try:
        with pytest.raises(WorkerError, match="not JSON"):
            w.start()
    finally:
        w.close()


def test_a_worker_that_cannot_start_gives_an_error(tmp_path):
    with pytest.raises(WorkerError):
        worker(tmp_path, "exit_now").start()
    missing = ModelWorker([str(tmp_path / "no_such_program")], start_timeout=5)
    with pytest.raises(WorkerError, match="Could not start"):
        missing.start()


def test_an_answer_of_the_wrong_length_is_refused(tmp_path):
    w = worker(tmp_path, "wrong_length")
    try:
        w.start()
        with pytest.raises(WorkerError, match="got 2"):
            w.predict("k", ["a"], [[0.1]])
    finally:
        w.close()


def test_requests_from_several_threads_get_their_own_answers(tmp_path):
    w = worker(tmp_path, "slow")
    results: dict[int, float] = {}

    def one(i: int) -> None:
        results[i] = w.predict("k", ["a"], [[i / 10]])[0]

    try:
        w.start()
        threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert results == pytest.approx({i: i / 10 for i in range(8)})
    finally:
        w.close()


def test_close_ends_the_process_promptly(tmp_path):
    w = worker(tmp_path, "normal")
    w.start()
    started = time.monotonic()
    w.close()
    assert not w.alive() and time.monotonic() - started < 6
    w.close()  # closing twice is fine
    with pytest.raises(WorkerError, match="not running"):
        w._exchange({"op": "info"}, 1)
