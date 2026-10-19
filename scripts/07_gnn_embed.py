"""Stage 7: run only the 'gnn_embed' stage for one dataset.

Same as: uv run run.py --dataset <yelpchi|amazon> --only gnn_embed
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "gnn_embed", *sys.argv[1:]]))
