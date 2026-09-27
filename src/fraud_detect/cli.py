"""Command-line entry point shared by run.py and scripts/NN_*.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fraud_detect.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    apply_overrides,
    get_paths,
    load_config,
)
from fraud_detect.pipeline import STAGE_NAMES, STAGES, Context, StageError, get_stage
from fraud_detect.runlog import new_run_dir, set_seeds, write_run_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fraud-detect pipeline stages in order.",
        epilog="Stages: " + ", ".join(STAGE_NAMES),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--from", dest="start", choices=STAGE_NAMES, help="first stage")
    parser.add_argument("--to", dest="end", choices=STAGE_NAMES, help="last stage")
    parser.add_argument("--only", choices=STAGE_NAMES, help="run just this stage")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun stages even if their outputs already exist",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value for this run, e.g. "
        "--set df_analyze.htune_trials=10 (repeatable; recorded in the run log)",
    )
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    return parser


def select_stages(start: str | None, end: str | None, only: str | None) -> list[str]:
    if only:
        if start or end:
            raise ValueError("--only cannot be combined with --from/--to")
        return [only]
    i = STAGE_NAMES.index(start) if start else 0
    j = STAGE_NAMES.index(end) if end else len(STAGE_NAMES) - 1
    if i > j:
        raise ValueError(f"--from {start} comes after --to {end}")
    return STAGE_NAMES[i : j + 1]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for n, stage in enumerate(STAGES, start=1):
            print(f"{n:>2}. {stage.name:<18} {stage.description}")
        return 0
    try:
        names = select_stages(args.start, args.end, args.only)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        config = apply_overrides(load_config(args.config), args.overrides)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    paths = get_paths(config)
    for d in (paths.interim_dir, paths.processed_dir, paths.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    set_seeds(int(config["seed"]))
    run_dir = new_run_dir(paths.runs_dir)
    write_run_info(run_dir, config, names, REPO_ROOT, argv)
    ctx = Context(config=config, paths=paths, run_dir=run_dir)
    print(f"Run log: {run_dir}")

    force = args.force
    for name in names:
        stage = get_stage(name)
        outputs = stage.outputs(paths)
        if not force and all(o.exists() for o in outputs):
            print(f"== {name}: outputs exist, skipping (use --force to rerun)")
            continue
        print(f"== {name}: {stage.description}", flush=True)
        try:
            stage.run(ctx)
        except StageError as e:
            print(f"\nStage '{name}' stopped:\n{e}", file=sys.stderr)
            return 2
        # Inputs of later stages just changed, so they must rerun too.
        force = True
    print("Done.")
    return 0
