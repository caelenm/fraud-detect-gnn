"""Bundle one dataset's finished pipeline outputs (node table, split, features,
df-analyze results, reports) into one .tgz so group members can skip the long
stages.

    uv run pack_artifacts.py --dataset amazon     # fraud_artifacts_amazon_<time>.tgz
    uv run pack_artifacts.py --dataset amazon -o share/x.tgz

Share the bundle outside git (e.g. OneDrive or Teams): it contains row-level
data and is too large for GitHub. Group members unpack it with
unpack_artifacts.py. See "Sharing outputs with the group" in README.md.
"""

import sys

from fraud_detect.artifacts import pack_main

if __name__ == "__main__":
    sys.exit(pack_main(sys.argv[1:]))
