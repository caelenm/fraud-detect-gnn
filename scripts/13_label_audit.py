"""Stage 13: run only the 'label_audit' stage.

Same as: uv run run.py --only label_audit
"""

import sys

from fraud_detect.cli import main

if __name__ == "__main__":
    sys.exit(main(["--only", "label_audit", *sys.argv[1:]]))
