#!/usr/bin/env bash
# Download the CFPB Consumer Complaint Database Narratives Archive files used
# in this project (complaints received May 2018 through August 2023).
#
# Source: https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/
#
# Usage:
#   ./download_dataset.sh              # download and extract into data/raw/
#   ./download_dataset.sh --no-extract # download only
#
# Data is public domain but is NOT committed to this repository.

set -euo pipefail

BASE_URL="https://files.consumerfinance.gov/f/documents"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/raw"

FILES=(
  "CCDB_Export_2_May_2018_through_April_2021"
  "CCDB_Export_3_May_2021_through_October_2022"
  "CCDB_Export_4_November_2022_through_August_2023"
)

EXTRACT=1
if [[ "${1:-}" == "--no-extract" ]]; then
  EXTRACT=0
fi

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

download() {
  local url="$1" out="$2"
  if command -v wget >/dev/null 2>&1; then
    wget --continue --show-progress -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl --fail --location --continue-at - -o "$out" "$url"
  else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
  fi
}

for name in "${FILES[@]}"; do
  zip="${name}.zip"

  if [[ -f "$zip" ]] && unzip -tq "$zip" >/dev/null 2>&1; then
    echo "Already downloaded: $zip"
  else
    echo "Downloading: $zip"
    download "${BASE_URL}/${zip}" "$zip"
    if ! unzip -tq "$zip" >/dev/null 2>&1; then
      echo "Error: $zip is corrupt or incomplete. Delete it and rerun." >&2
      exit 1
    fi
  fi

  if [[ "$EXTRACT" -eq 1 ]]; then
    echo "Extracting: $zip"
    unzip -oq "$zip" -d "$name"
  fi
done

echo
echo "Done. Files are in: $DEST_DIR"
echo "Check that the extracted files contain a populated complaint narrative column."
