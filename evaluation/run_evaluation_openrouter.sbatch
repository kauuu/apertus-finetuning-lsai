#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=slds-eval-debug-openrouter
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="/users/$USER/apertus-finetuning-lsai/evaluation"
VENV_DIR="${PROJECT_DIR}/venv-eval"

# MODEL_PATH="${MODEL_PATH:-/users/jmirlach/scratch/apertus-finetuning-lsai/evaluation/merged_model}"
MODEL_PATH="${MODEL_PATH:-swiss-ai/Apertus-70B-Instruct-2509}"

JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-deepseek/deepseek-v3.2}"
export OPENROUTER_API_KEY="ADD-OPENROUTER-API-KEY"


# Ensure the OpenRouter API key is available before continuing.
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY to your OpenRouter token before running.}"

export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/.cache/triton"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "${PROJECT_DIR}/logs"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Point LiteLLM (used by the judge) at OpenRouter’s OpenAI-compatible endpoint.
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_MODEL="openai/${JUDGE_MODEL_ID}"

srun --environment="${PROJECT_DIR}/evaluation.toml" bash -lc "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd ${PROJECT_DIR}

    source ${VENV_DIR}/bin/activate
    # Let venv see container-wide vllm
    export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:\$PYTHONPATH

    echo '=== Running full evaluation. Judge requests will go to OpenRouter ==='
    echo 'Model under test: ${MODEL_PATH}'
    echo 'Judge model: ${JUDGE_MODEL}'

    unset SSL_CERT_FILE

    export LITELLM_CACHE_DIR="${PROJECT_DIR}/.litellm_cache/${SLURM_JOB_ID:-$$}"
    mkdir -p "\${LITELLM_CACHE_DIR}"

    python evaluate.py \
        --model ${MODEL_PATH} \
        --decision-language de,fr,it \
        --headnote-language de,fr,it \
        --no-one-shot \
        --tensor-parallel-size ${SLURM_GPUS_ON_NODE:-4}

    echo 'Done!'
"
