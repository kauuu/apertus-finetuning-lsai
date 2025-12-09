#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=lasi-project-apertus
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

set -e

PROJECT_DIR="/users/mneuwinger/scratch/apertus-finetuning-lsai/evaluation"

export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/.cache/triton"
mkdir -p "$HF_HOME"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "${PROJECT_DIR}/logs"

export PYTHONUNBUFFERED=1

srun --environment="${PROJECT_DIR}/evaluation.toml" bash -c "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd ${PROJECT_DIR}

    chmod +x setup_eval_venv.sh
    ./setup_eval_venv.sh
"
