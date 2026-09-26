"""Stage 4: run only the 'split' stage (same as: uv run run.py --only split)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "split", *sys.argv[1:]]))
