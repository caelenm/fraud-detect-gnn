"""Stage 9: run only the 'df_analyze_report' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only df_analyze_report
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "df_analyze_report", *sys.argv[1:]]))
