"""Stage 10: run only the 'df_analyze_report' stage.

Same as: uv run run.py --only df_analyze_report
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "df_analyze_report", *sys.argv[1:]]))
