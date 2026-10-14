"""Stage 11: run only the 'export_model' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only export_model
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "export_model", *sys.argv[1:]]))
