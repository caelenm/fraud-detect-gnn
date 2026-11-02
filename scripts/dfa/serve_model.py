"""Keep exported Model As loaded and answer prediction requests over stdin.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT. The dashboard starts it
(src/fraud_detect/dashboard/worker.py) with one --model per model:

    uv run --directory "$DF_ANALYZE_DIR" --python '>=3.13.11,<3.14' \
        python <repo>/scripts/dfa/serve_model.py \
        --model amazon-all=<repo>/outputs/amazon/<feature set>/model

The protocol is in src/fraud_detect/models/model_serving.py. It exits when
stdin closes. Libraries may print to stdout, including from compiled code, which
would corrupt the protocol, so the real stdout is kept for answers and file
descriptor 1 is pointed at stderr.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from model_io import load_model_a, read_model_meta, setup_paths

setup_paths()

from fraud_detect.models.model_serving import LoadedModel, serve  # noqa: E402


def parse_model(text: str) -> tuple[str, Path]:
    key, sep, path = text.partition("=")
    if not sep or not key or not path:
        raise argparse.ArgumentTypeError(f"{text!r} is not KEY=MODEL_DIR")
    return key, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        type=parse_model,
        action="append",
        required=True,
        metavar="KEY=MODEL_DIR",
        help="a folder with model_a.json",
    )
    args = parser.parse_args()
    answers = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    models = {}
    for key, model_dir in args.model:
        meta, warnings = read_model_meta(model_dir)
        for warning in warnings:
            print(f"WARNING: {key}: {warning}", file=sys.stderr)
        models[key] = LoadedModel(key, meta, load_model_a(model_dir, meta), warnings)
        print(f"Loaded {key}: {meta['model']} ({meta['selection']})", file=sys.stderr)
    serve(sys.stdin, answers, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
