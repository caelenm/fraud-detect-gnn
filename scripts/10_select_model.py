"""Stage 10: run only the 'select_model' stage.

Same as: uv run run.py --only select_model
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "select_model", *sys.argv[1:]]))
