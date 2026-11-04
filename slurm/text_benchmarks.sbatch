#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=text_bench
#SBATCH --output=slurm/logs/%x-%j.out
# Text-only benchmarks of one full model (appendix "Text benchmarks after localized fine-tuning"): the same
# evaluation as the text-intervention sweeps (lm-eval, zero-shot, the model's chat template, full sets).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b_6_17 slurm/text_benchmarks.sbatch
#   TASKS : space-separated lm-eval tasks (default: the six of the paper)
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
TASKS=${TASKS:-"arc_easy commonsense_qa hellaswag mmlu piqa winogrande"}
python scripts/interventions/text.py --model_path "$MODEL" --tasks $TASKS --full_model \
    --output_dir "$RESULTS/text_benchmarks"
