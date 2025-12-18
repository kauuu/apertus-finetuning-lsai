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
# MODEL_PATH="${MODEL_PATH:-swiss-ai/Apertus-70B-Instruct-2509}"
MODEL_PATH="${MODEL_PATH:-swiss-ai/Apertus-8B-Instruct-2509}"

USE_ONE_SHOT=1  # Set to 1 for one-shot prompting, 0 for zero-shot
if [[ "$USE_ONE_SHOT" -eq 1 ]]; then
    ONE_SHOT_FLAG="--one-shot"
    SHOT_MODE="one-shot"
else
    ONE_SHOT_FLAG="--no-one-shot"
    SHOT_MODE="zero-shot"
fi

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

# Tune judge throughput (no eval logic changes; just parallelism/verbosity/caching).
export SLDS_JUDGE_CONCURRENT_REQUESTS="${SLDS_JUDGE_CONCURRENT_REQUESTS:-20}"
export SLDS_JUDGE_MAX_TOKENS="${SLDS_JUDGE_MAX_TOKENS:-1024}"
export SLDS_JUDGE_CACHE="${SLDS_JUDGE_CACHE:-0}"

export PROJECT_DIR VENV_DIR MODEL_PATH ONE_SHOT_FLAG SHOT_MODE JUDGE_MODEL

srun --environment="${PROJECT_DIR}/evaluation.toml" bash -lc '
    set -euo pipefail

    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

    RUN_DIR="${SLURM_TMPDIR:-/tmp/${USER:-$(id -un)}/lighteval_${SLURM_JOB_ID:-$$}}"
    mkdir -p "${RUN_DIR}"
    cd "${RUN_DIR}"

    source "${VENV_DIR}/bin/activate"
    # Let venv see container-wide vllm
    export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:${PYTHONPATH:-}"

    echo "=== Running full evaluation. Judge requests will go to OpenRouter ==="
    echo "Model under test: ${MODEL_PATH}"
    echo "Judge model: ${JUDGE_MODEL}"
    echo "Prompting mode: ${SHOT_MODE}"

    unset SSL_CERT_FILE

    export LITELLM_CACHE_DIR="${RUN_DIR}/.litellm_cache"
    mkdir -p "${LITELLM_CACHE_DIR}"

    MODEL_SAFE="${MODEL_PATH//\//_}"
    MODEL_SAFE="${MODEL_SAFE//:/_}"
    RESULTS_DIR="${PROJECT_DIR}/results/${SLURM_JOB_ID:-$$}_${MODEL_SAFE}_${SHOT_MODE}"
    mkdir -p "${RESULTS_DIR}"

    python "${PROJECT_DIR}/evaluate.py" \
        --model "${MODEL_PATH}" \
        --decision-language de,fr,it \
        --headnote-language de,fr,it \
        ${ONE_SHOT_FLAG} \
        --tensor-parallel-size "${SLURM_GPUS_ON_NODE:-4}" \
        --output-dir "${RESULTS_DIR}"

    echo "Done!"
'
