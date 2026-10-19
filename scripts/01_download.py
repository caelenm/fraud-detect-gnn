"""Stage 1: run only the 'download' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only download
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "download", *sys.argv[1:]]))
