#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=imbalance_planes
#SBATCH --output=slurm/logs/%x-%j.out
# Information-imbalance planes of one checkpoint (Fig. 4, appendix) from the cached representations
# written by slurm/similarity_multimodal.sbatch (run that first for the same checkpoint).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b slurm/imbalance_planes.sbatch
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
python scripts/geometry/imbalance_planes.py --model-name-or-path "$MODEL" \
    --representations-dir "$REPRESENTATIONS" --results-dir "$RESULTS/imbalance_planes" ${EXTRA:-}
