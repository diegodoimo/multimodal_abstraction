#!/bin/bash
# Download the two SentEval probing tasks of the linguistic-probing experiment (odd_man_out = SOMO,
# coordination_inversion = CoordInv) from the SentEval repository into data/probing (or $1).
#   bash scripts/probing/download_senteval.sh [target_dir]
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
DEST="${1:-$REPO/data/probing}"
URL=https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing
mkdir -p "$DEST"
for f in odd_man_out.txt coordination_inversion.txt; do
    if [ -s "$DEST/$f" ]; then echo "present: $DEST/$f"; continue; fi
    curl -fsSL "$URL/$f" -o "$DEST/$f.part" && mv "$DEST/$f.part" "$DEST/$f"
    echo "downloaded: $DEST/$f ($(wc -l < "$DEST/$f") lines)"
done
