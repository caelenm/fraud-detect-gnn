"""Stage 8: run only the 'select_model' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only select_model
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "select_model", *sys.argv[1:]]))
