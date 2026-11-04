# Sourced by every launcher in slurm/. Adapt the paths to your machine.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

ENV_DIR=${MA_ENV:-$REPO/env}                                   # conda env created by setup_env.sh
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}            # Hugging Face cache (models + datasets)
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export RESULTS=${MA_RESULTS:-$REPO/results}                    # experiment outputs (the figure scripts read here)
export REPRESENTATIONS=${MA_REPRESENTATIONS:-$REPO/representations}   # large activation caches
export COCO_TRAIN2014=${COCO_TRAIN2014:-$HOME/data/mscoco/train2014}  # COCO images for the retrieval experiment

source "$(conda info --base)/bin/activate" "$ENV_DIR"
module load cuda/12.8 2>/dev/null || true
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lmms-eval needs a writable node-local dataset cache
export LMMS_EVAL_DATASETS_CACHE=${LMMS_EVAL_DATASETS_CACHE:-/local_scratch/lmms_eval_datasets_$USER}
export LMMS_EVAL_HOME=${LMMS_EVAL_HOME:-$HF_HOME/lmms_eval_cache}
cd "$REPO"
