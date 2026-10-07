"""Stage 10: run only the 'web_report' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only web_report
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "web_report", *sys.argv[1:]]))
