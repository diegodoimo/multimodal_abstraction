#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --job-name=benchmarks
#SBATCH --output=slurm/logs/%x-%j.out
# lmms-eval benchmarks of one model (Fig. 5 table and sliding-window plot, Fig. 6 and the radar appendix).
#
#   sbatch --export=ALL,MODEL=lbasile/llava_7b_6_17 slurm/benchmarks.sbatch
#   NO_IMAGE=1 : evaluate the same prompts without the image (image-reliance baselines)
#   TASKS      : space-separated lmms-eval tasks (default: all tasks of the paper)
#   LIMIT      : optional cap on the number of samples per task
#   MMBench needs an OpenAI-compatible judge: export OPENAI_API_KEY, OPENAI_API_URL, MODEL_VERSION.
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:?set MODEL}
TASKS=${TASKS:-"scienceqa_img mme pope gqa seedbench mmbench_en_dev mmbench_cn_dev vqav2_val cv_bench naturalbench mmmu_val chartqa mmstar mmmu_pro_standard realworldqa countbench coco2017_cap_val flickr30k"}
OUT_MODEL=$(basename "$MODEL")
EXTRA=""
if [ "${NO_IMAGE:-0}" = "1" ]; then OUT_MODEL="${OUT_MODEL}_noimage"; EXTRA="--no-image"; fi
if [ -n "${LIMIT:-}" ]; then EXTRA="$EXTRA --limit $LIMIT"; fi
for task in $TASKS; do
    python scripts/finetuning/benchmarks.py --model-name-or-path "$MODEL" --task-name "$task" --batch-size 1 \
        --output-dir "$RESULTS/benchmarks/$OUT_MODEL/$task" $EXTRA
done
