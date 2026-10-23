"""The group's results database (results/runs/): record, list and export runs.

Record a finished pipeline run (fills the dataset, feature set, headline
numbers, results table, start time, command and settings from what the run
saved):

    uv run scripts/results_db.py add \\
        --from-run outputs/amazon/pilot_m1_own \\
        --run-log outputs/amazon/runs/20261019T140501123456Z \\
        --device rtx4060-laptop --description "Pilot m1, split 0, 10 trials"

Record anything else (a probe, a comparison, a check) from a summary CSV:

    uv run scripts/results_db.py add --kind probe --csv outputs/amazon/probe/x.csv \\
        --dataset amazon --started-utc 2026-10-21T14:05:00Z \\
        --finished-utc 2026-10-21T14:20:00Z \\
        --device macbook-m2 --description "Probe: PCA size sweep"

Then commit the new folder: git add results/ && git commit.

    uv run scripts/results_db.py list [--dataset amazon] [--device ...] [--tag ...]
    uv run scripts/results_db.py show <run_id or unique prefix>
    uv run scripts/results_db.py export [--out outputs/results_db]
    uv run scripts/results_db.py check

See results/README.md and src/fraud_detect/results_db.py for the format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fraud_detect import results_db as db
from fraud_detect.config import REPO_ROOT
from fraud_detect.runstate import atomic_write_csv

DEFAULT_DB = REPO_ROOT / "results" / "runs"
DEFAULT_EXPORT = REPO_ROOT / "outputs" / "results_db"


def _merge(*sources: dict) -> dict:
    """Later sources win; `settings` dicts are merged key by key."""
    out: dict = {}
    for source in sources:
        for key, value in source.items():
            if value is None:
                continue
            if key == "settings" and isinstance(out.get(key), dict):
                out[key] = {**out[key], **value}
            else:
                out[key] = value
    return out


def cmd_add(args: argparse.Namespace) -> int:
    run_fields, run_table = {}, None
    if args.from_run:
        run_fields, run_table = db.from_pipeline_run(args.from_run)
        run_fields["source"] = _relative(args.from_run)
    log_fields = db.from_run_log(args.run_log) if args.run_log else {}
    cli_fields = {
        "dataset": args.dataset,
        "feature_set": args.feature_set,
        "split_repeat": args.split_repeat,
        "seed": args.seed,
        "started_utc": args.started_utc,
        "finished_utc": args.finished_utc,
        "duration_min": args.duration_min,
        "command": args.command,
        "tags": args.tag or None,
        "notes": args.notes,
        "supersedes": args.supersedes,
    }
    fields = _merge(run_fields, log_fields, cli_fields)
    fields.pop("stages", None)
    if args.csv:
        table = pd.read_csv(args.csv)
        fields.setdefault("source", _relative(args.csv))
    elif run_table is not None:
        table = run_table
    else:
        raise db.ResultsDBError("Give the results table: --csv or --from-run")
    started = fields.pop("started_utc", None)
    if not started:
        raise db.ResultsDBError(
            "Give the start time: --run-log (the run's outputs/<dataset>/runs/<stamp> "
            "folder) or --started-utc"
        )
    meta = db.build_meta(
        description=args.description,
        device=args.device,
        started_utc=started,
        kind=args.kind,
        status=args.status,
        root=REPO_ROOT,
        **fields,
    )
    if args.dry_run:
        db.check_meta(meta)
        db.check_results_table(table)
        print(json.dumps(meta, indent=2, default=str))
        print(f"\n(dry run) results table: {len(table)} rows x {len(table.columns)} cols")
        return 0
    entry = db.add_entry(args.db, meta, table)
    print(f"Recorded {entry}")
    if meta.get("git_dirty"):
        print("note: the working tree had uncommitted changes; say so in --notes")
    print(f"Next: git add {_relative(entry)} && git commit -m 'Results: ...'")
    return 0


def _relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _filtered(args: argparse.Namespace) -> pd.DataFrame:
    table = db.index_table(args.db)
    for column, value in (("dataset", args.dataset), ("device", args.device),
                          ("feature_set", args.feature_set)):  # fmt: skip
        if value:
            table = table[table[column] == value]
    if args.tag:
        table = table[table["tags"].str.split(", ").apply(lambda t: args.tag in t)]
    return table


def cmd_list(args: argparse.Namespace) -> int:
    table = _filtered(args)
    if table.empty:
        print(f"No entries in {args.db}")
        return 0
    shown = ["run_id", "started_utc", "duration_min", "device", "feature_set",
             "status", "headline.test_pr_auc", "description"]  # fmt: skip
    shown = [c for c in shown if c in table.columns]
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(table[shown].to_string(index=False))
    return 0


def _find(db_dir: Path, prefix: str) -> Path:
    matches = [d for d in db.entry_dirs(db_dir) if d.name.startswith(prefix)]
    if len(matches) != 1:
        raise db.ResultsDBError(
            f"{len(matches)} entries start with {prefix!r}; give more of the run_id"
        )
    return matches[0]


def cmd_show(args: argparse.Namespace) -> int:
    entry = _find(args.db, args.run_id)
    print(json.dumps(db.load_meta(entry), indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.read_csv(entry / db.RESULTS).to_string(index=False))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    index, results = db.index_table(args.db), db.combined_results(args.db)
    atomic_write_csv(args.out / "index.csv", index)
    atomic_write_csv(args.out / "all_results.csv", results)
    print(f"Wrote {args.out / 'index.csv'} ({len(index)} entries) and "
          f"{args.out / 'all_results.csv'} ({len(results)} rows)")  # fmt: skip
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    problems = db.check_db(args.db)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    n = len(db.entry_dirs(args.db))
    print(f"{n} entries, {len(problems)} with problems")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command_name", required=True)

    add = sub.add_parser("add", help="record one run")
    add.add_argument(
        "--description",
        required=True,
        help=f"one line, at most {db.MAX_DESCRIPTION} characters",
    )
    add.add_argument(
        "--device",
        required=True,
        help="short label for the machine, e.g. rtx4060-laptop "
        "(OS, CPU and GPU are detected automatically)",
    )
    add.add_argument(
        "--from-run",
        type=Path,
        help="a finished outputs/<dataset>[_repeat<k>]/<feature_set>/ folder",
    )
    add.add_argument(
        "--run-log",
        type=Path,
        help="the run's outputs/<dataset>/runs/<stamp>/ folder "
        "(start time, command, git commit, settings)",
    )
    add.add_argument("--csv", type=Path, help="results table (aggregate rows only)")
    add.add_argument("--kind", choices=db.KINDS, default="pipeline")
    add.add_argument("--status", choices=db.STATUSES, default="ok")
    add.add_argument("--dataset")
    add.add_argument("--feature-set")
    add.add_argument("--split-repeat", type=int)
    add.add_argument("--seed", type=int)
    add.add_argument("--started-utc", help="e.g. 2026-10-21T14:05:00Z")
    add.add_argument("--finished-utc", help="e.g. 2026-10-21T16:10:00Z")
    add.add_argument("--duration-min", type=float, help="wall-clock minutes")
    add.add_argument("--command", help="the command that produced the results")
    add.add_argument("--tag", action="append", help="repeatable, e.g. --tag pilot")
    add.add_argument("--notes", help="anything else worth knowing")
    add.add_argument("--supersedes", help="run_id of an entry this one corrects")
    add.add_argument("--dry-run", action="store_true", help="print, write nothing")
    add.set_defaults(func=cmd_add)

    listing = sub.add_parser("list", help="list entries, oldest first")
    listing.add_argument("--dataset")
    listing.add_argument("--device")
    listing.add_argument("--feature-set")
    listing.add_argument("--tag")
    listing.set_defaults(func=cmd_list)

    export = sub.add_parser(
        "export", help="write index.csv and all_results.csv (never committed)"
    )
    export.add_argument("--out", type=Path, default=DEFAULT_EXPORT)
    export.set_defaults(func=cmd_export)

    show = sub.add_parser("show", help="print one entry")
    show.add_argument("run_id", help="run_id or a unique prefix of it")
    show.set_defaults(func=cmd_show)

    check = sub.add_parser("check", help="validate every entry")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (db.ResultsDBError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
