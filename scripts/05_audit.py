"""Stage 5: run only the 'audit' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only audit
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "audit", *sys.argv[1:]]))
