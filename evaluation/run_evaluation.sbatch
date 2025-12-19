#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=slds-eval
#SBATCH --output=logs/eval_aprtus_base_%j.out
#SBATCH --error=logs/eval_aprtus_base_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00

set -e

PROJECT_DIR="/users/mneuwinger/scratch/apertus-finetuning-lsai/evaluation"
VENV_DIR="${PROJECT_DIR}/venv-eval"

# Decide which model to use to compare
# MODEL_PATH="${MODEL_PATH:-/users/mneuwinger/scratch/apertus-finetuning-lsai/evaluation/merged_model}"
MODEL_PATH="${MODEL_PATH:-swiss-ai/Apertus-8B-Instruct-2509}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/.cache/triton"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "${PROJECT_DIR}/logs"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

srun --environment="${PROJECT_DIR}/evaluation.toml" bash -lc "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd ${PROJECT_DIR}

    source ${VENV_DIR}/bin/activate
    # Let venv see container-wide vllm
    export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:\$PYTHONPATH

    echo '=== Starting judge model on GPUs 1,2 (TP=2), port 8001 ==='
    CUDA_VISIBLE_DEVICES=1,2 python -m vllm.entrypoints.openai.api_server \
        --model ${JUDGE_MODEL_ID} \
        --max-model-len 16000 \
        --enable-prefix-caching \
        --seed 2025 \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size 2 \
        --port 8001 &
    JUDGE_MODEL_PID=\$!

    echo 'Waiting for judge model...'
    for i in \$(seq 1 300); do
        if curl -s http://0.0.0.0:8001/health > /dev/null 2>&1; then
            echo 'Judge model ready!'
            break
        fi
        if [ \$i -eq 300 ]; then
            echo 'Judge model timeout'
            kill \$JUDGE_MODEL_PID 2>/devnull || true
            exit 1
        fi
        sleep 1
    done

    echo '=== Judge ready! Running full evaluation (de,fr,it) with in-process vLLM ==='

    unset SSL_CERT_FILE
    export OPENAI_API_BASE=\"http://0.0.0.0:8001/v1\"
    export OPENAI_API_KEY=\"EMPTY\"
    export JUDGE_MODEL=\"openai/${JUDGE_MODEL_ID}\"

    CUDA_VISIBLE_DEVICES=0 python evaluate.py \
        --model ${MODEL_PATH} \
        --decision-language de,fr,it \
        --headnote-language de,fr,it \
        --no-one-shot

    echo 'Stopping servers...'
    kill \$JUDGE_MODEL_PID 2>/dev/null || true
    wait \$JUDGE_MODEL_PID 2>/dev/null || true
    echo 'Done!'
"
