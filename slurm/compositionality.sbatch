#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=compositionality
#SBATCH --output=slurm/logs/%x-%j.out
# ARO and SugarCrepe compositionality benchmarks (Fig. 5 table, Fig. 6 and the radar appendix).
# Data: bash scripts/finetuning/download_aro.sh and bash scripts/finetuning/download_sugarcrepe.sh (see README).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b_6_17 slurm/compositionality.sbatch
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
OUT_MODEL=$(basename "$MODEL")
EXTRA=""
if [ "${NO_IMAGE:-0}" = "1" ]; then OUT_MODEL="${OUT_MODEL}_noimage"; EXTRA="--no-image"; fi
python scripts/finetuning/compositionality.py --model-name-or-path "$MODEL" --benchmark all --eval-method mcq_logits \
    --aro-data-root "${ARO_DATA_ROOT:-$REPRESENTATIONS/datasets/vision-language-models-are-bows}" \
    --sugarcrepe-data-root "${SUGARCREPE_DATA_ROOT:-$REPRESENTATIONS/datasets/sugar-crepe}" \
    --output-dir "$RESULTS/benchmarks/$OUT_MODEL/compositionality" $EXTRA
