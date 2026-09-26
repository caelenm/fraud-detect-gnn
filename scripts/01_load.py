"""Stage 1: run only the 'load' stage (same as: uv run run.py --only load)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "load", *sys.argv[1:]]))
