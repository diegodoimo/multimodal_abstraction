#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=interv_text
#SBATCH --output=slurm/logs/%x-%j.out
# Layer-skipping sweeps on text-only benchmarks (appendix "Interventions: all datasets (text benchmarks)").
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b,SCHEDULE=progressive slurm/interventions_text.sbatch
#   SCHEDULE : progressive (skip layers [0, w), w = 0, 2, 4, ...) | sliding (window of 4 layers, step 2)
#   EXTRA    : extra flags for scripts/interventions/text.py, e.g. "--limit 500"
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
SCHEDULE=${SCHEDULE:-progressive}
TASKS=${TASKS:-"arc_easy commonsense_qa hellaswag mmlu piqa winogrande"}
if [ "$SCHEDULE" = "sliding" ]; then
    SWEEP="--window_size 4 --evaluate_full_sliding_window"
else
    SWEEP="--evaluate_full_progressive_ablation"
fi
python scripts/interventions/text.py \
    --model_path "$MODEL" --tasks $TASKS --interval 2 $SWEEP ${EXTRA:-} \
    --output_dir "$RESULTS/interventions/text"
