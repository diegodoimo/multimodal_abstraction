#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:H100:3
#SBATCH --job-name=llama90b
#SBATCH --output=slurm/logs/%x-%j.out
# Llama-3.2-90B-Vision-Instruct experiments (appendix "Extension to multimodal cross-attention").
#
#   sbatch --export=ALL,JOB=ablation_input  slurm/llama90b.sbatch   # progressive cross-attention knockout from the input
#   sbatch --export=ALL,JOB=ablation_output slurm/llama90b.sbatch   # ... from the output
#   sbatch --export=ALL,JOB=window          slurm/llama90b.sbatch   # keep / remove cross-attention in layers [10, 65)
#   sbatch --export=ALL,JOB=similarity      slurm/llama90b.sbatch   # information imbalance between text and image prompts
#   MODEL: local path or HF id of meta-llama/Llama-3.2-90B-Vision-Instruct
#   TASKS: lmms-eval tasks (default: the six of the ablation figure; the twelve of Fig. 6's Llama panel for JOB=window)
#   LIMIT: samples per task (default: 2000 for the ablations, 5000 for the window, as in the paper)
#   EXTRA: extra flags for the similarity job, e.g. "--force-metrics-only"
# submit from the repository root: sbatch ... slurm/<name>.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
MODEL=${MODEL:-meta-llama/Llama-3.2-90B-Vision-Instruct}
JOB=${JOB:?set JOB}
SWEEP_TASKS="gqa mmbench_en_dev mmbench_cn_dev mme seedbench vqav2_val"
WINDOW_TASKS="chartqa countbench cv_bench gqa mmbench_cn_dev mmbench_en_dev mme mmmu_val naturalbench scienceqa_img seedbench vqav2_val"
SYSTEM="You are an AI assistant evaluating images for academic research. Answer every question concisely based solely on what you observe in the image. Do not refuse based on identity or content."
case "$JOB" in
  ablation_input)
    python scripts/interventions/multimodal.py --model_path "$MODEL" --tasks ${TASKS:-$SWEEP_TASKS} --ablation_mode cross_modal \
        --limit ${LIMIT:-2000} --max_new_tokens 64 --interval 5 --evaluate_full_progressive_ablation \
        --llama_system_instruction "$SYSTEM" --output_dir "$RESULTS/interventions/llama90b" ;;
  ablation_output)
    python scripts/interventions/multimodal.py --model_path "$MODEL" --tasks ${TASKS:-$SWEEP_TASKS} --ablation_mode cross_modal \
        --limit ${LIMIT:-2000} --max_new_tokens 64 --interval 5 --evaluate_full_reverse_progressive_ablation \
        --llama_system_instruction "$SYSTEM" --output_dir "$RESULTS/interventions/llama90b" ;;
  window)
    python scripts/interventions/multimodal.py --model_path "$MODEL" --tasks ${TASKS:-$WINDOW_TASKS} --ablation_mode cross_modal \
        --limit ${LIMIT:-5000} --max_new_tokens 64 --evaluate_complementary_intervals --start_layer 10 --end_layer 65 \
        --llama_system_instruction "$SYSTEM" --output_dir "$RESULTS/interventions/llama90b" ;;
  similarity)
    python scripts/geometry/similarity_llama_vision.py --model-path "$MODEL" \
        --results-dir "$RESULTS/similarity/llama90b" --representations-dir "$REPRESENTATIONS/representations" \
        --similarity-measures information_imbalance information_imbalance_reversed --save-representations ${EXTRA:-} \
        --dataset-runs "l11p/GQA-val-balanced-noyesno-captioned|synthetic|||false|true" \
                       "l11p/MMBench_EN-dev-captioned|synthetic|||false|true" \
                       "l11p/VQAv2-COCO-restval-ICQ|human|split01||false|true" ;;
  *) echo "unknown JOB=$JOB"; exit 1 ;;
esac
