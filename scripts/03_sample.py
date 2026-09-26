"""Stage 3: run only the 'sample' stage (same as: uv run run.py --only sample)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "sample", *sys.argv[1:]]))
