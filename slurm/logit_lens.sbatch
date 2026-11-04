#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=logit_lens
#SBATCH --output=slurm/logs/%x-%j.out
# Logit lens and linear probes on VQA datasets (Fig. 3 right and the probing appendices).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b slurm/logit_lens.sbatch
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
python scripts/probing/logit_lens.py --model-name-or-path "$MODEL" --results-dir "$RESULTS/logit_lens" --batch-size 1
