"""Stage 4: run only the 'features' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only features
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "features", *sys.argv[1:]]))
