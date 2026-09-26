"""Stage 6: run only the 'pca' stage (same as: uv run run.py --only pca)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "pca", *sys.argv[1:]]))
