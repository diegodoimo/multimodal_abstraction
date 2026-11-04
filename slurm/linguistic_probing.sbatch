#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=ling_probing
#SBATCH --output=slurm/logs/%x-%j.out
# SentEval linguistic probing (CoordInv, SOMO) of the residual stream (appendix figure).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b slurm/linguistic_probing.sbatch
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
python scripts/probing/linguistic_probing.py \
    --model_name_or_path "$MODEL" --task_name paper --target_layer res2 \
    --out_dir "$RESULTS/linguistic_probing" --batch_size 16 --n_rep 2 --subset_size 5000
