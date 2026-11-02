"""The dashboard's side of the model worker (scripts/dfa/serve_model.py).

df-analyze's environment is never imported here: the worker is a subprocess, and
the two talk in JSON lines (models/model_serving.py). This class takes any
command, so the tests run it against a fake worker.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, TextIO

START_TIMEOUT_S = 300.0  # uv may have to build df-analyze's environment
REQUEST_TIMEOUT_S = 120.0  # a whole test set at once


class WorkerError(RuntimeError):
    """The worker is not running, did not answer, or refused a request."""


class ModelWorker:
    """One worker process. Requests are served one at a time; if the process
    dies it is restarted once, and after that the error is the caller's to show."""

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        start_timeout: float = START_TIMEOUT_S,
        request_timeout: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self.command = command
        self.env = env
        self.log_path = log_path
        self.start_timeout = start_timeout
        self.request_timeout = request_timeout
        self.restarts = 0
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._log: TextIO | None = None
        self._lock = threading.Lock()

    def start(self) -> dict[str, Any]:
        """Start the process and wait until it answers; returns its `info`."""
        with self._lock:
            return self._start()

    def _start(self) -> dict[str, Any]:
        self._stop()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("a", encoding="utf-8")
        self._lines = queue.Queue()
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log if self._log else subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                env=self.env,
            )
        except OSError as e:
            raise WorkerError(f"Could not start the model worker: {e}") from e
        threading.Thread(
            target=self._read, args=(self._process.stdout, self._lines), daemon=True
        ).start()
        return self._exchange({"op": "info"}, self.start_timeout)

    @staticmethod
    def _read(stream: Any, lines: queue.Queue[str | None]) -> None:
        for line in stream:
            lines.put(line)
        lines.put(None)  # the process closed its stdout

    def _exchange(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.poll() is not None:
            raise WorkerError("The model worker is not running")
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise WorkerError(f"The model worker stopped: {e}") from e
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as e:
            # A late answer would be read as the next request's, so end the process.
            process.kill()
            process.wait()
            raise WorkerError(f"The model worker did not answer in {timeout:g} s") from e
        if line is None:
            raise WorkerError("The model worker stopped without answering")
        try:
            answer = json.loads(line)
        except json.JSONDecodeError as e:
            raise WorkerError(
                f"The model worker's answer is not JSON: {line[:200]}"
            ) from e
        if not isinstance(answer, dict) or not answer.get("ok"):
            error = answer.get("error") if isinstance(answer, dict) else answer
            raise WorkerError(f"The model worker refused the request: {error}")
        return answer

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        """One request. A refused request raises straight away; a worker that
        died is restarted once and the request tried again."""
        with self._lock:
            try:
                return self._exchange(request, self.request_timeout)
            except WorkerError:
                if self.alive() or self.restarts >= 1:
                    raise
            self.restarts += 1
            self._start()
            return self._exchange(request, self.request_timeout)

    def predict(
        self, key: str, columns: list[str], rows: list[list[float]]
    ) -> list[float]:
        answer = self.request(
            {"op": "predict", "model": key, "columns": columns, "rows": rows}
        )
        prob = answer["prob"]
        if len(prob) != len(rows):
            raise WorkerError(
                f"Asked for {len(rows)} rows, got {len(prob)} probabilities"
            )
        return prob

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()  # the worker exits when stdin closes
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._log is not None:
            self._log.close()
            self._log = None

    def close(self) -> None:
        with self._lock:
            self._stop()
