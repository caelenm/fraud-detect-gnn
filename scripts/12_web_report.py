"""Stage 12: run only the 'web_report' stage.

Same as: uv run run.py --only web_report
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "web_report", *sys.argv[1:]]))
