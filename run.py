"""Run the whole pipeline, or part of it, in order, for one dataset.

    uv run run.py --list                          # show stages
    uv run run.py --dataset amazon                # all stages, skipping finished ones
    uv run run.py --dataset amazon --to split     # stop after a stage
    uv run run.py --dataset yelpchi --only audit  # run a single stage
    uv run run.py --dataset yelpchi --from features --force  # rerun from a stage

New stages are added to fraud_detect.pipeline.STAGES as they are implemented.
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
