#!/usr/bin/env bash
# Download YelpChi and Amazon (the CARE-GNN .mat files) into data/raw/care_gnn/.
#
# The work is done by the pipeline's `download` stage (Python, so it behaves
# the same on Linux, macOS and WSL): it fetches each zip from the pinned
# CARE-GNN commit, verifies its SHA-256, unzips it and verifies the .mat.
# Safe to rerun; a file with a different checksum is never overwritten.
#
# Usage:
#   ./download_dataset.sh            # both datasets
#   ./download_dataset.sh amazon     # one dataset
#
# The data is NOT committed to this repository.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

datasets=("$@")
if [[ ${#datasets[@]} -eq 0 ]]; then
  datasets=(yelpchi amazon)
fi
for dataset in "${datasets[@]}"; do
  uv run run.py --dataset "$dataset" --only download
done
