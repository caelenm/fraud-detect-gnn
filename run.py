"""Run the whole pipeline, or part of it, in order.

    uv run run.py --list              # show stages
    uv run run.py                     # run every stage, skipping finished ones
    uv run run.py --to split          # stop after a stage
    uv run run.py --only embed        # run a single stage
    uv run run.py --from pca --force  # rerun from a stage onwards

New stages are added to fraud_detect.pipeline.STAGES as they are implemented.
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
