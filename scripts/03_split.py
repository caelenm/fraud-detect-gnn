"""Stage 3: run only the 'split' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only split
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "split", *sys.argv[1:]]))
