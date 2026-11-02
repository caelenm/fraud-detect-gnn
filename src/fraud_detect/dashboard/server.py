"""The dashboard's web server: standard library only, 127.0.0.1 only.

Not a pipeline stage: it spans feature sets and datasets, adds no stage outputs
and writes nothing but the worker's log (under outputs/). The aggregate sections
read the committed results database, so they work on a fresh clone. The classify
section reads row-level outputs, so the server never binds anything but the
loopback address, refuses requests addressed to any other host, and sends a
Content-Security-Policy that stops the page from reaching anything else.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

from fraud_detect.config import REPO_ROOT, df_analyze_dir
from fraud_detect.dashboard import checks as checks_module
from fraud_detect.dashboard import classify as cl
from fraud_detect.dashboard import data, narrative
from fraud_detect.dashboard import registry as reg
from fraud_detect.dashboard.worker import ModelWorker, WorkerError
from fraud_detect.external import (
    ExternalToolError,
    check_df_analyze_dir,
    df_analyze_command,
)
from fraud_detect.runlog import git_commit

HOST = "127.0.0.1"
DEFAULT_PORT = 8555
STATIC_DIR = Path(__file__).with_name("static")
REPORT_CSS = Path(__file__).parent.parent / "report" / "report.css"
STATIC_FILES = {
    "/": (STATIC_DIR / "index.html", "text/html; charset=utf-8"),
    "/app.js": (STATIC_DIR / "app.js", "text/javascript; charset=utf-8"),
    "/dashboard.css": (STATIC_DIR / "dashboard.css", "text/css; charset=utf-8"),
    "/report.css": (REPORT_CSS, "text/css; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; "
    "style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
MAX_BODY = 4096
WORKER_LOG = "dashboard/worker.log"
DATASET_LABELS = {
    "amazon": "Amazon (users; main)",
    "yelpchi": "YelpChi (reviews; validation)",
}
LOADING, READY, UNAVAILABLE = "loading", "ready", "unavailable"


class BadRequest(ValueError):
    """The request is wrong; the message goes back with a 400."""


def clean(value: Any) -> Any:
    """Plain JSON values: numpy types become Python ones and NaN becomes null,
    so a missing number can never reach the page as a number."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class PanelState:
    key: str
    dataset: str
    variant: str
    state: str = LOADING
    message: str = ""
    target: cl.Target | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "dataset": self.dataset,
            "variant": self.variant,
            "variant_label": reg.variant_label(self.variant, self.dataset),
            "state": self.state,
            "message": self.message,
        }
        if self.target is not None:
            out["target"] = self.target.status()
        return out


class App:
    """Everything the handlers answer with. The aggregate numbers are computed
    once at start-up: the database does not change while the server runs."""

    def __init__(
        self,
        config: dict[str, Any],
        db_dir: Path,
        use_model: bool = True,
        worker_factory: Any = None,
    ) -> None:
        self.config = config
        self.db_dir = db_dir
        self.use_model = use_model
        self.entries = data.load_entries(db_dir)
        self.runs = data.runs_table(self.entries)
        all_rows = data.all_contrasts(self.entries)
        self.contrasts = data.current_contrasts(all_rows)
        self.checks = checks_module.run_all(
            db_dir, self.entries, self.runs, all_rows, self.contrasts
        )
        duplicates = data.duplicate_runs(self.runs)
        if duplicates:  # a mean over a run counted twice would be silently wrong
            raise data.DashboardDataError(
                f"Runs recorded more than once; fix with --supersedes: {duplicates}"
            )
        self.datasets = sorted(
            set(self.runs["dataset"].dropna()), key=lambda d: (d != "amazon", d)
        )
        self.overview = data.overview(self.entries, self.runs)
        self.failed = data.failed_entries(self.entries)
        self.cross = data.cross_dataset(self.runs, self.contrasts)
        self.commit = git_commit(REPO_ROOT)
        self.worker_factory = worker_factory
        self.worker: ModelWorker | None = None
        self.worker_error = ""
        self.panels: dict[str, PanelState] = {}
        for dataset, variants in cl.TARGET_VARIANTS.items():
            for variant in variants:
                key = cl.target_key(dataset, variant)
                self.panels[key] = PanelState(key, dataset, variant)
        self._lock = threading.Lock()

    # ---- Aggregate answers ------------------------------------------------
    def variants_of(self, dataset: str) -> list[str]:
        have = set(self.runs[self.runs["dataset"] == dataset]["variant"])
        return [v for v in reg.VARIANTS if v in have]

    def check_dataset(self, dataset: Any) -> str:
        if dataset not in self.datasets:
            raise BadRequest(f"Unknown dataset {dataset!r}; choose from {self.datasets}")
        return str(dataset)

    def check_variant(self, dataset: str, variant: Any) -> str:
        options = self.variants_of(dataset)
        if variant not in options:
            raise BadRequest(
                f"{dataset} has no variant {variant!r}; choose from {options}"
            )
        return str(variant)

    def api_datasets(self) -> dict[str, Any]:
        failed_checks = [c.name for c in self.checks if not c.ok]
        return {
            "datasets": [
                {
                    "name": d,
                    "label": DATASET_LABELS.get(d, d),
                    "variants": [
                        {"name": v, "label": reg.variant_label(v, d)}
                        for v in self.variants_of(d)
                    ],
                }
                for d in self.datasets
            ],
            "budgets": [{"name": b, "label": reg.BUDGET_LABELS[b]} for b in reg.BUDGETS],
            "default_variant": reg.ALL,
            "default_budget": reg.PILOT,
            "commit": self.commit,
            "overview": self.overview,
            "general_caveats": narrative.general_caveats(self.overview, self.failed),
            "verdict_rule": narrative.VERDICT_RULE,
            "label_column_note": narrative.label_column_note(),
            "not_run": list(narrative.NOT_RUN),
            "demo_notes": narrative.demo_notes(),
            "failed_entries": self.failed,
            "failed_checks": failed_checks,
            "n_checks": len(self.checks),
            "use_model": self.use_model,
        }

    def api_methods(self, dataset: Any, budget: Any, variant: Any) -> dict[str, Any]:
        dataset = self.check_dataset(dataset)
        variant = self.check_variant(dataset, variant)
        if budget not in reg.BUDGETS:
            raise BadRequest(f"Unknown budget {budget!r}; choose from {reg.BUDGETS}")
        rows = data.setup_summary(self.runs, dataset, budget, variant)
        best = data.best_setup(self.runs, dataset, variant, self.contrasts)
        no19 = (
            data.setup_summary(self.runs, dataset, reg.PILOT, reg.NO19)
            if dataset == "amazon"
            else []
        )
        pilot_all = data.setup_summary(self.runs, dataset, reg.PILOT, reg.ALL)
        default_check = data.default_settings_check(self.entries, dataset)
        return {
            "dataset": dataset,
            "budget": budget,
            "variant": variant,
            "variant_label": reg.variant_label(variant, dataset),
            "rows": rows,
            "families": list(reg.FAMILIES),
            "best": best,
            "caveats": narrative.dataset_caveats(
                dataset, variant, pilot_all, no19, default_check
            ),
            "has_label_column": any(r["label_derived_column"] for r in rows),
            "coverage": data.coverage(
                self.runs, dataset, variant, list(self.config["feature_sets"])
            ),
        }

    def api_contrasts(self, dataset: Any, variant: Any) -> dict[str, Any]:
        dataset = self.check_dataset(dataset)
        variant = self.check_variant(dataset, variant)
        return {
            "dataset": dataset,
            "variant": variant,
            "questions": data.key_questions(self.contrasts, dataset, variant),
            "families": data.contrasts_for(self.contrasts, dataset, variant),
            "default_check": data.default_settings_check(self.entries, dataset),
            "tuned_vs_default": data.tuned_vs_default_cv(self.entries, dataset),
        }

    def api_cross(self) -> dict[str, Any]:
        return self.cross

    def api_best(self) -> dict[str, Any]:
        out = {}
        for dataset in self.datasets:
            for variant in self.variants_of(dataset):
                out[cl.target_key(dataset, variant)] = data.best_setup(
                    self.runs, dataset, variant, self.contrasts
                )
        return out

    def api_checks(self) -> dict[str, Any]:
        return {"checks": [c.as_dict() for c in self.checks]}

    # ---- Classify panel ---------------------------------------------------
    def api_status(self) -> dict[str, Any]:
        with self._lock:
            panels = [p.as_dict() for p in self.panels.values()]
        return {
            "panels": panels,
            "use_model": self.use_model,
            "worker_error": self.worker_error,
        }

    def _headline(self, feature_set: str, dataset: str) -> dict[str, Any] | None:
        usable = data.usable_runs(self.runs)
        rows = usable[
            (usable["feature_set"] == feature_set)
            & (usable["dataset"] == dataset)
            & (usable["split_repeat"] == cl.SPLIT)
        ]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            "test_pr_auc": row["test_pr_auc"],
            "threshold": row["threshold"],
            "n_test": row["n_test"],
        }

    def load_panels(self) -> None:
        """Load every target, then start the worker and verify each model on its
        whole test set. Runs in the background, so the aggregate sections are up
        at once. A target that cannot be loaded says why; the others still work."""
        graphs: dict[str, Any] = {}
        for panel in self.panels.values():
            best = data.best_setup(
                self.runs, panel.dataset, panel.variant, self.contrasts
            )
            if best is None:
                self._set_panel(panel, UNAVAILABLE, "No recorded run to choose from.")
                continue
            try:
                target = cl.load_target(
                    self.config,
                    panel.dataset,
                    panel.variant,
                    best,
                    self._headline(best["feature_set"], panel.dataset),
                    graphs,
                )
            except cl.TargetUnavailable as e:
                self._set_panel(panel, UNAVAILABLE, str(e))
                continue
            except Exception as e:  # one broken panel must not take the others down
                self._set_panel(panel, UNAVAILABLE, f"{type(e).__name__}: {e}"[:500])
                continue
            if not self.use_model and target.prepared is not None:
                target.note = "Started with --no-model, so these are saved predictions."
            with self._lock:
                panel.target = target
                panel.state = LOADING if self._wants_live(target) else READY
                panel.message = "" if panel.state == READY else "Starting the model."
        self._start_live()

    def _wants_live(self, target: cl.Target) -> bool:
        return self.use_model and target.prepared is not None

    def _set_panel(self, panel: PanelState, state: str, message: str) -> None:
        with self._lock:
            panel.state, panel.message = state, message

    def _start_live(self) -> None:
        live = [
            p
            for p in self.panels.values()
            if p.target is not None and self._wants_live(p.target)
        ]
        if not live:
            return
        try:
            self.worker = self._make_worker(live)
            self.worker.start()
        except (WorkerError, ExternalToolError, OSError) as e:
            self.worker_error = str(e)
            for panel in live:
                self._note_saved(panel, f"The model worker did not start: {e}")
            return
        for panel in live:
            try:
                assert panel.target is not None
                cl.verify_live(panel.target, self.worker)
            except (WorkerError, cl.ClassifyError) as e:
                self._note_saved(panel, f"Could not verify the running model: {e}")
                continue
            self._set_panel(panel, READY, "")

    def _note_saved(self, panel: PanelState, reason: str) -> None:
        with self._lock:
            assert panel.target is not None
            panel.target.mode = cl.SAVED
            panel.target.note = reason + " Showing saved predictions instead."
            panel.state, panel.message = READY, ""

    def _make_worker(self, panels: list[PanelState]) -> ModelWorker:
        pairs = [(p.key, p.target.model_dir) for p in panels if p.target is not None]
        if self.worker_factory is not None:
            return self.worker_factory(pairs)
        dfa = df_analyze_dir(self.config)
        script = str(REPO_ROOT / "scripts" / "dfa" / "serve_model.py")
        check_df_analyze_dir(
            dfa, "df-analyze.py", self.config["df_analyze"].get("commit")
        )
        args = [a for key, path in pairs for a in ("--model", f"{key}={path}")]
        command, env = df_analyze_command(
            dfa, script, args, self.config["df_analyze"].get("python")
        )
        env["DF_ANALYZE_DIR"] = str(dfa)
        outputs = Path(self.config["paths"]["outputs_dir"])
        outputs = outputs if outputs.is_absolute() else REPO_ROOT / outputs
        return ModelWorker(command, env=env, log_path=outputs / WORKER_LOG)

    def api_classify(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise BadRequest("The request body must be a JSON object")
        draw = body.get("draw", "any")
        if draw not in cl.DRAWS:
            raise BadRequest(f"Unknown draw {draw!r}; use one of {cl.DRAWS}")
        panel = self.panels.get(str(body.get("target")))
        if panel is None:
            raise BadRequest(f"Unknown target {body.get('target')!r}")
        seed = body.get("seed")
        with self._lock:
            state, message, target = panel.state, panel.message, panel.target
        if state != READY or target is None:
            raise BadRequest(message or f"{panel.key} is not ready yet")
        try:
            return cl.classify(target, self.worker, draw, seed)
        except cl.ClassifyError as e:
            raise BadRequest(str(e)) from e

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "fraud-detect-dashboard"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # requests hold nothing worth a log, and a log could hold node IDs

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(clean(payload), allow_nan=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _host_allowed(self) -> bool:
            """Only requests addressed to this machine: a web page elsewhere must
            not be able to reach row-level data by resolving its name to 127.0.0.1."""
            port = self.server.server_address[1]
            return self.headers.get("Host", "") in (
                f"127.0.0.1:{port}",
                f"localhost:{port}",
                f"[::1]:{port}",
            )

        def _origin_allowed(self) -> bool:
            origin = self.headers.get("Origin")
            port = self.server.server_address[1]
            return origin is None or origin in (
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
            )

        def _answer(self, make: Any) -> None:
            try:
                self._json(200, make())
            except (BadRequest, data.DashboardDataError) as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # shown on the page; the server keeps running
                self._json(500, {"error": f"{type(e).__name__}: {e}"[:500]})

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_allowed():
                self._json(403, {"error": "This server only answers on localhost"})
                return
            url = urlsplit(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            routes = {
                "/api/datasets": app.api_datasets,
                "/api/methods": lambda: app.api_methods(
                    query.get("ds"),
                    query.get("budget", reg.PILOT),
                    query.get("variant", reg.ALL),
                ),
                "/api/contrasts": lambda: app.api_contrasts(
                    query.get("ds"), query.get("variant", reg.ALL)
                ),
                "/api/cross": app.api_cross,
                "/api/best": app.api_best,
                "/api/checks": app.api_checks,
                "/api/status": app.api_status,
            }
            if url.path in routes:
                self._answer(routes[url.path])
            elif url.path in STATIC_FILES:  # a fixed list, so no path can escape it
                path, content_type = STATIC_FILES[url.path]
                self._send(200, path.read_bytes(), content_type)
            else:
                self._json(404, {"error": f"No such page: {url.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_allowed() or not self._origin_allowed():
                self._json(403, {"error": "This server only answers on localhost"})
                return
            if urlsplit(self.path).path != "/api/classify":
                self._json(404, {"error": "No such endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                self.close_connection = True
                self._json(400, {"error": f"The body must be at most {MAX_BODY} bytes"})
                return
            if "application/json" not in self.headers.get("Content-Type", ""):
                self.rfile.read(length)
                self._json(415, {"error": "Send application/json"})
                return
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"error": "The body is not JSON"})
                return
            self._answer(lambda: app.api_classify(body))

    return Handler


def make_server(app: App, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Always the loopback address; there is deliberately no option to change it."""
    return ThreadingHTTPServer((HOST, port), make_handler(app))
