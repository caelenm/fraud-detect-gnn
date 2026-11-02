"""The local results dashboard: every method, per dataset and across them.

    uv run scripts/dashboard.py                # http://127.0.0.1:8555
    uv run scripts/dashboard.py --no-model     # no model worker, e.g. a fresh clone
    uv run scripts/dashboard.py --check        # the sanity checks, then exit

The aggregate sections read results/runs/, so they work without any outputs. The
classify panel needs the runs' outputs and df-analyze's clone. See
docs/DASHBOARD.md. Row-level data is shown only on this machine: the server binds
127.0.0.1, and nothing here is ever to be committed or hosted.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from fraud_detect.config import DEFAULT_CONFIG, REPO_ROOT, ConfigError, load_config
from fraud_detect.dashboard import data
from fraud_detect.dashboard.server import DEFAULT_PORT, HOST, App, make_server

DEFAULT_DB = REPO_ROOT / "results" / "runs"


def print_checks(app: App) -> bool:
    for check in app.checks:
        print(("PASS  " if check.ok else "FAIL  ") + f"{check.name}: {check.detail}")
        for problem in check.problems[:8]:
            print(f"      {problem}")
    failed = sum(not c.ok for c in app.checks)
    print(f"{len(app.checks) - failed} of {len(app.checks)} checks passed.")
    return failed == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="do not start the model worker; saved predictions if outputs exist",
    )
    parser.add_argument("--check", action="store_true", help="run the checks and exit")
    args = parser.parse_args(argv)
    try:
        app = App(load_config(args.config), args.db, use_model=not args.no_model)
    except (ConfigError, data.DashboardDataError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    passed = print_checks(app)
    if args.check:
        return 0 if passed else 1
    if not passed:
        print("Some checks failed; the page shows them at the top.", file=sys.stderr)
    try:
        server = make_server(app, args.port)
    except OSError as e:
        print(f"error: cannot listen on {HOST}:{args.port}: {e}", file=sys.stderr)
        return 2
    threading.Thread(target=app.load_panels, daemon=True).start()
    print(f"Dashboard: http://{HOST}:{server.server_address[1]} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
