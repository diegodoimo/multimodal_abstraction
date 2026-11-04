#!/bin/bash
# Download the SugarCrepe benchmark into SUGARCREPE_DATA_ROOT (default representations/datasets/sugar-crepe,
# the path used by slurm/compositionality.sbatch):
#   data/    the 7 JSON annotation files of github.com/RAIVNLab/sugar-crepe
#   val2017/ the COCO 2017 validation images (about 1 GB, 5,000 images)
#   bash scripts/finetuning/download_sugarcrepe.sh

REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
DEST="${SUGARCREPE_DATA_ROOT:-$REPO/representations/datasets/sugar-crepe}"
DATA_DIR="$DEST/data"
IMG_DIR="$DEST/val2017"

echo "SugarCrepe -> $DEST"
mkdir -p "$DATA_DIR" "$IMG_DIR"

# JSON annotation files
BASE_URL="https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data"
CATEGORIES=(add_obj add_att replace_obj replace_att replace_rel swap_obj swap_att)

for cat in "${CATEGORIES[@]}"; do
    out="$DATA_DIR/$cat.json"
    if [ ! -f "$out" ]; then
        wget -q -O "$out" "$BASE_URL/$cat.json" || { echo "Failed to download $cat.json"; exit 1; }
    fi
done

# COCO val2017 images (skipped when the directory already holds them)
if [ "$(ls -A "$IMG_DIR" 2>/dev/null | wc -l)" -le 100 ]; then
    COCO_ZIP="/tmp/val2017_sugarcrepe_$$.zip"
    wget -O "$COCO_ZIP" "http://images.cocodataset.org/zips/val2017.zip" \
        || { echo "Failed to download COCO val2017"; rm -f "$COCO_ZIP"; exit 1; }
    unzip -q "$COCO_ZIP" -d "$DEST/"
    rm -f "$COCO_ZIP"
fi

echo "JSON files: $(ls "$DATA_DIR"/*.json 2>/dev/null | wc -l) / ${#CATEGORIES[@]}, images: $(ls "$IMG_DIR" 2>/dev/null | wc -l)"
