"""Stage 2: run only the 'load' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only load
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "load", *sys.argv[1:]]))
