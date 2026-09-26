"""Stage 7: run only the 'features' stage (same as: uv run run.py --only features)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "features", *sys.argv[1:]]))
