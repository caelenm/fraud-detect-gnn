"""Stage 5: run only the 'embed' stage (same as: uv run run.py --only embed)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "embed", *sys.argv[1:]]))
