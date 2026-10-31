"""Stage 8: run only the 'gnn_oof' stage (Model 6's gnnsup block) for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only gnn_oof
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "gnn_oof", *sys.argv[1:]]))
