"""Stage 6: run only the 'neighbour_agg' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only neighbour_agg
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "neighbour_agg", *sys.argv[1:]]))
