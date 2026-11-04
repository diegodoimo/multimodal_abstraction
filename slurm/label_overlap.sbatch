#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=label_overlap
#SBATCH --output=slurm/logs/%x-%j.out
# Label overlap of last-token representations on 10-way image-classification MCQ prompts
# (Fig. 3 left and the label-overlap appendices).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b slurm/label_overlap.sbatch
#   EXTRA : extra flags for scripts/probing/label_overlap.py, e.g. "--metrics-only --force-recompute-metrics"
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
EXTRA=${EXTRA:-}
# dataset | split | prompt mode (random 10 classes: generic_mcq_filter; 10 most populated: generic_mcq_top_filter)
# Override with DATASETS="entry entry ..." to run a subset.
DEFAULT_DATASETS="timm/mini-imagenet|validation|mcq_nocat_filter ethz/food101|validation|generic_mcq_filter tanganke/sun397|test|generic_mcq_filter flwrlabs/caltech101|train|generic_mcq_top_filter dpdl-benchmark/oxford_flowers102|test|generic_mcq_top_filter torch-uncertainty/Places365|train|generic_mcq_filter tanganke/dtd|train|generic_mcq_filter"
for entry in ${DATASETS:-$DEFAULT_DATASETS}; do
    IFS='|' read -r dataset split mode <<< "$entry"
    python scripts/probing/label_overlap.py \
        --model-name "$MODEL" --dataset-name "$dataset" --split "$split" --prompt-mode "$mode" --n-choices 10 \
        --guide-text "Answer with the option's letter from the given choices directly." \
        --tokens-pooling-method last --metrics label_overlap accuracy --class-fraction 0.3 \
        --results-dir "$RESULTS/label_overlap" --representations-dir "$REPRESENTATIONS" \
        --save-representations --save-prompts $EXTRA
done
