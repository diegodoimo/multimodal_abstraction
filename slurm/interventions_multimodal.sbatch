#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=interv_mm
#SBATCH --output=slurm/logs/%x-%j.out
# Layer-skipping / cross-modal attention-knockout sweeps on multimodal benchmarks
# (Fig. 2 and the "Interventions: additional results" appendix).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b,MODE=image,SCHEDULE=progressive slurm/interventions_multimodal.sbatch
#   MODE     : image (skip the image tokens in the layer window) | cross_modal (knock out image->text attention)
#   SCHEDULE : progressive (skip layers [0, w) for w = 0, 2, 4, ...) | sliding (window of 5 layers moved in steps of 2)
#   TASKS    : space-separated lmms-eval tasks (default: the six of the paper)
#   LIMIT    : samples per task (default 2000)
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
MODE=${MODE:-image}
SCHEDULE=${SCHEDULE:-progressive}
TASKS=${TASKS:-"gqa mmbench_en_dev mmbench_cn_dev mme seedbench vqav2_val"}
LIMIT=${LIMIT:-2000}
if [ "$SCHEDULE" = "sliding" ]; then
    SWEEP="--window_size 5 --evaluate_full_sliding_window"
else
    SWEEP="--evaluate_full_progressive_ablation"
fi
python scripts/interventions/multimodal.py \
    --model_path "$MODEL" --tasks $TASKS --ablation_mode "$MODE" \
    --limit "$LIMIT" --max_new_tokens 64 --interval 2 $SWEEP \
    --output_dir "$RESULTS/interventions/multimodal"
