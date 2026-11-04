#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=coco_retrieval
#SBATCH --output=slurm/logs/%x-%j.out
# COCO image/caption representations on the LLaVA detail_23k subset and retrieval metrics (appendix figure).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b slurm/coco_retrieval.sbatch
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
for prompt in --image_only --text_only; do
    python scripts/probing/extract_coco_embeddings.py \
        --model_name_or_path "$MODEL" --image_dir "$COCO_TRAIN2014" \
        --out_dir "$REPRESENTATIONS/coco" --target_layer res2 $prompt ${DETAIL_JSON:+--detail_json $DETAIL_JSON}
done
case "$MODEL" in   # folder names used by scripts/probing/extract_coco_embeddings.py
    llava-hf/llava-1.5-7b-hf) MODEL_DIR=llava_7b_hf ;;
    llava-hf/llava-1.5-13b-hf) MODEL_DIR=llava_13b_hf ;;
    lmms-lab/LLaVA-OneVision-1.5-4B-Instruct) MODEL_DIR=onevision_4b_hf ;;
    lmms-lab/LLaVA-OneVision-1.5-8B-Instruct) MODEL_DIR=onevision_8b_hf ;;
    OpenGVLab/InternVL2-8B) MODEL_DIR=internvl_8b_hf ;;
    nyu-visionx/cambrian-8b) MODEL_DIR=cambrian_8b ;;
    *) MODEL_DIR=$(basename "$MODEL") ;;
esac
python scripts/probing/retrieval.py --model "$MODEL_DIR" --representations-root "$REPRESENTATIONS/coco" \
    --suffix1 imgavg --suffix2 textavg --out-dir "$RESULTS/retrieval"
