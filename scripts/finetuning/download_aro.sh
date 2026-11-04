#!/bin/bash
# Download the ARO annotations and the images of the ARO order sub-tasks (COCO val2014 + Flickr30k) into
# ARO_DATA_ROOT (default: representations/datasets/vision-language-models-are-bows, the path used by
# slurm/compositionality.sbatch). The ARO dataset code is vendored in third_party/vision-language-models-are-bows.
#   bash scripts/finetuning/download_aro.sh [annotations|coco|flickr30k|all]
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
python "$REPO/scripts/finetuning/download_aro_images.py" \
    --aro-data-root "${ARO_DATA_ROOT:-$REPO/representations/datasets/vision-language-models-are-bows}" \
    --dataset "${1:-all}"
