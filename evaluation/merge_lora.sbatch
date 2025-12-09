#!/bin/bash
#SBATCH --job-name=merge-lora
#SBATCH --account=large-sc-2
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -e

PROJECT_DIR="/users/mneuwinger/scratch/apertus-finetuning-lsai/evaluation"
VENV_DIR="${PROJECT_DIR}/venv-eval"

# Base model (Apertus 8B Instruct)
BASE_MODEL="${BASE_MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"

# LoRA adapter checkpoint
ADAPTER_PATH="${ADAPTER_PATH:-/users/mneuwinger/scratch/apertus-finetuning-lsai/finetuning/Apertus-FT/output/apertus_lora_r64_ctx8k/checkpoint-1870}"

# Output path for merged model
OUTPUT_PATH="${OUTPUT_PATH:-/users/mneuwinger/scratch/apertus-finetuning-lsai/evaluation/merged_model}"

export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
mkdir -p "$HF_HOME"
mkdir -p "${PROJECT_DIR}/logs"

export PYTHONUNBUFFERED=1

srun --environment="${PROJECT_DIR}/evaluation.toml" bash -c "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd ${PROJECT_DIR}

    source ${VENV_DIR}/bin/activate

    pip install peft --quiet

    python merge_lora.py \
        --base-model ${BASE_MODEL} \
        --adapter ${ADAPTER_PATH} \
        --output ${OUTPUT_PATH}
"
