"""Stage 8: run only the 'df_analyze_input' stage
(same as: uv run run.py --only df_analyze_input)."""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "df_analyze_input", *sys.argv[1:]]))
