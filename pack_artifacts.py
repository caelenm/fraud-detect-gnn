"""Bundle finished pipeline outputs (embeddings, features, df-analyze results)
into one .tgz so group members can skip the long stages.

    uv run pack_artifacts.py                   # writes fraud_artifacts_<time>.tgz here
    uv run pack_artifacts.py -o share/x.tgz    # choose where to write it

Share the bundle outside git (e.g. OneDrive or Teams): it contains complaint
narratives and is too large for GitHub. Group members unpack it with
unpack_artifacts.py. See "Sharing outputs with the group" in README.md.
"""

import sys

from fraud_detect.artifacts import pack_main

if __name__ == "__main__":
    sys.exit(pack_main(sys.argv[1:]))
