"""Stage 2: run only the 'label' stage (same as: uv run run.py --only label)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "label", *sys.argv[1:]]))
