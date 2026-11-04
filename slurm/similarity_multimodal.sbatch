#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=similarity
#SBATCH --output=slurm/logs/%x-%j.out
# Text-vs-multimodal representation similarity (information imbalance, linear CKA,
# neighborhood overlap) on the five VQA datasets (Fig. 4 and the geometry appendices).
#
#   sbatch --export=ALL,MODEL=lmms-lab/LLaVA-OneVision-1.5-4B-Base slurm/similarity_multimodal.sbatch
#   EXTRA : extra flags for scripts/geometry/similarity_multimodal.py, e.g. "--force-metrics-only"
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
EXTRA=${EXTRA:-}
# dataset | caption source | extra flags
DATASETS=(
    "l11p/VQAv2-COCO-restval-ICQ|human|--dataset-split split01"
    "l11p/COCOQA-restval-ICQA|human|--dataset-split split01"
    "l11p/GQA-val-balanced-noyesno-captioned|synthetic|"
    "l11p/MMBench_EN-dev-captioned|synthetic|--open-ended"
    "l11p/SEED-Bench-images-captioned|synthetic|--open-ended"
)
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset caption flags <<< "$entry"
    python scripts/geometry/similarity_multimodal.py \
        --model-name-or-path "$MODEL" --dataset-path-or-name "$dataset" --caption-source "$caption" $flags \
        --results-dir "$RESULTS/similarity" --representations-dir "$REPRESENTATIONS" \
        --save-representations $EXTRA
done
