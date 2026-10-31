"""Stage 15: run only the 'gnn_classify' stage (GNN classifier rungs) for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only gnn_classify
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "gnn_classify", *sys.argv[1:]]))
