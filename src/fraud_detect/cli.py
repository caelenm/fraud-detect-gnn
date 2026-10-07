"""Command-line entry point shared by run.py and scripts/NN_*.py."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from fraud_detect.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ConfigError,
    Paths,
    apply_overrides,
    get_paths,
    load_config,
    with_dataset,
)
from fraud_detect.models import repeats
from fraud_detect.pipeline import (
    MODEL_A_SUMMARY,
    STAGE_NAMES,
    STAGES,
    Context,
    StageError,
    get_stage,
)
from fraud_detect.runlog import new_run_dir, set_seeds, write_run_info
from fraud_detect.runstate import (
    RunState,
    atomic_write_json,
    atomic_write_text,
    config_fingerprint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fraud-detect pipeline stages in order for one dataset.",
        epilog="Stages: " + ", ".join(STAGE_NAMES),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset",
        help="dataset to run: yelpchi or amazon (default: `dataset:` in the config)",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run: run the stages it had not finished, "
        "keeping finished work inside the interrupted stage",
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


def plan_stages(names: list[str], force: bool, paths: Paths) -> list[str]:
    """Stages that will run: from the first one that is forced or has missing
    outputs, through the end (later stages rerun because their inputs change)."""
    for i, name in enumerate(names):
        if force or not all(o.exists() for o in get_stage(name).outputs(paths)):
            return names[i:]
    return []


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for n, stage in enumerate(STAGES, start=1):
            print(f"{n:>2}. {stage.name:<18} {stage.description}")
        return 0
    if args.resume and (args.start or args.end or args.only or args.force):
        print("error: --resume cannot be combined with --from/--to/--only/--force",
              file=sys.stderr)  # fmt: skip
        return 2
    try:
        names = select_stages(args.start, args.end, args.only)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        config = apply_overrides(load_config(args.config), args.overrides)
        config = with_dataset(config, args.dataset)
    except (ValueError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # split.n_repeats > 1: run once per independently seeded split, unless one
    # repeat was chosen with --set split.repeat=k.
    n_repeats = int(config["split"]["n_repeats"])
    pinned = any(o.startswith("split.repeat=") for o in args.overrides)
    to_run = [config["split"]["repeat"]] if pinned else list(range(n_repeats))
    for repeat in to_run:
        cfg = copy.deepcopy(config)
        cfg["split"]["repeat"] = int(repeat)
        if n_repeats > 1:
            print(f"\n######## Split repeat {repeat} of 0..{n_repeats - 1}", flush=True)
        code = run_pipeline(cfg, names, args, argv, several=len(to_run) > 1)
        if code != 0:
            return code
    if len(to_run) > 1:
        write_repeat_summary(config, n_repeats)
    return 0


def run_pipeline(
    config: dict,
    names: list[str],
    args: argparse.Namespace,
    argv: list[str] | None,
    several: bool = False,
) -> int:
    """Run the planned stages for one dataset, split repeat and feature set.
    With `several` (a loop over repeats), --resume continues the repeats that
    were interrupted and runs the others normally."""
    paths = get_paths(config)
    for d in (paths.processed_dir, paths.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {config['dataset']} · feature set: {config['feature_set']}")

    fingerprint = config_fingerprint(config)
    state = RunState.load(paths.outputs_dir)
    resume = args.resume and state is not None
    if args.resume and state is None and not several:
        print("Nothing to resume: no interrupted run was found.")
        return 0
    if resume:
        assert state is not None
        if state.config_sha256 != fingerprint:
            print(
                "error: the config changed since the interrupted run started, so it "
                "cannot be resumed safely. Start it again with --force "
                f"(it had these stages left: {', '.join(state.pending)}).",
                file=sys.stderr,
            )
            return 2
        planned = list(state.pending)
    else:
        if state is not None and not args.force:
            print(
                f"error: an interrupted {config['dataset']} run has unfinished stages "
                f"({', '.join(state.pending)}). Continue it with `uv run run.py "
                f"--dataset {config['dataset']} --resume`, or start over with --force.",
                file=sys.stderr,
            )
            return 2
        planned = plan_stages(names, args.force, paths)
        if planned:
            command = list(sys.argv if argv is None else argv)
            state = RunState(
                pending=list(planned), config_sha256=fingerprint, command=command
            )
            state.save(paths.outputs_dir)

    set_seeds(int(config["seed"]))
    run_dir = new_run_dir(paths.runs_dir)
    write_run_info(run_dir, config, planned, REPO_ROOT, argv)
    print(f"Run log: {run_dir}")
    for name in names:
        if name not in planned:
            print(f"== {name}: outputs exist, skipping (use --force to rerun)")
    if not planned:
        print("Done.")
        return 0
    print(f"Stages to run: {', '.join(planned)}", flush=True)

    run_start = time.monotonic()
    for n, name in enumerate(planned, start=1):
        stage = get_stage(name)
        # Only the first stage of a resumed run was interrupted mid-way.
        ctx = Context(config=config, paths=paths, run_dir=run_dir,
                      resume=resume and n == 1, force=args.force)  # fmt: skip
        resumed = " (resuming)" if ctx.resume else ""
        print(f"\n== [{n}/{len(planned)}] {name}{resumed}: {stage.description}",
              flush=True)  # fmt: skip
        stage_start = time.monotonic()
        try:
            stage.run(ctx)
        except StageError as e:
            print(f"\nStage '{name}' stopped:\n{e}", file=sys.stderr)
            print(resume_hint(config), file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print(f"\n\nInterrupted during '{name}'. Every stage before it is saved.")
            print(resume_hint(config))
            return 130
        assert state is not None
        state.mark_done(name, paths.outputs_dir)
        took = _duration(time.monotonic() - stage_start)
        total = _duration(time.monotonic() - run_start)
        print(f"== {name} done in {took} (total {total})", flush=True)
    print("\nDone.")
    return 0


def write_repeat_summary(config: dict, n_repeats: int) -> None:
    """Model A's mean ± SD over the split repeats, once every repeat has a
    finished report; written next to repeat 0's feature-set outputs."""
    found = []
    for repeat in range(n_repeats):
        cfg = copy.deepcopy(config)
        cfg["split"]["repeat"] = repeat
        path = get_paths(cfg).model_reports_dir / MODEL_A_SUMMARY
        if not path.is_file():
            print(f"Repeat summary skipped: {path} does not exist yet.")
            return
        found.append(json.loads(path.read_text(encoding="utf-8")))
    summary = repeats.summarize(found)
    base = copy.deepcopy(config)
    base["split"]["repeat"] = 0
    out = get_paths(base).feature_set_dir
    text = repeats.markdown(summary, config["dataset"], config["feature_set"])
    atomic_write_json(out / "repeats_summary.json", summary)
    atomic_write_text(out / "repeats_summary.md", text)
    print(f"\n{text}\nSaved {out / 'repeats_summary.md'}")


def resume_hint(config: dict) -> str:
    return f"Continue later with: uv run run.py --dataset {config['dataset']} --resume"
