#!/bin/bash
# Creates the conda environment used for all experiments (mirrors the
# reference environment: Python 3.12, torch 2.8.0 + CUDA 12.8, flash-attn 2.8.3,
# lmms-eval 0.7.1 and lm-eval 0.4.9.2 from pinned git commits).
#
#   bash setup_env.sh                    # creates ./env
#   NO_FLASH_ATTN=1 bash setup_env.sh    # same, without flash-attn (only InternVL2-8B uses it)
#   source ./env/bin/activate
set -euo pipefail
ENV_DIR="${1:-$(dirname "$(readlink -f "$0")")/env}"

conda create -y -p "$ENV_DIR" python=3.12
source "$(conda info --base)/bin/activate" "$ENV_DIR"

pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if [ "${NO_FLASH_ATTN:-0}" != "1" ]; then
    pip install flash-attn==2.8.3 --no-build-isolation
fi
pip install -r "$(dirname "$(readlink -f "$0")")/requirements.txt"

# Evaluation harnesses, pinned to the commits used for the paper.
pip install "lm_eval[hf] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9"
pip install "lmms_eval @ git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@88b23e2bfa16a1edbc16e9e238ed82130b3a4f56"
python -m spacy download en_core_web_sm

echo "Environment ready in $ENV_DIR"
