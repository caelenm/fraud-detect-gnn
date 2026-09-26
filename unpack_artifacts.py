"""Unpack a bundle made by pack_artifacts.py into this repository.

Put the fraud_artifacts_<time>.tgz file next to this script, then:

    uv run unpack_artifacts.py                 # newest bundle in this folder
    uv run unpack_artifacts.py path/to/x.tgz   # a bundle somewhere else
    uv run unpack_artifacts.py --force         # overwrite files that differ

Every file is checksum-verified and goes back to where it was on the machine
that made the bundle, so `uv run run.py` then skips the finished stages.
"""

import sys

from fraud_detect.artifacts import unpack_main

if __name__ == "__main__":
    sys.exit(unpack_main(sys.argv[1:]))
