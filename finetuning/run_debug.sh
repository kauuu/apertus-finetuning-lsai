#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=apertus-70b-debug
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --no-requeue

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

set -x

PROJECT_DIR=$(pwd)
cd "${PROJECT_DIR}"
echo "Working directory: ${PROJECT_DIR}"

# Set HuggingFace cache directory
export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
mkdir -p "$HF_HOME"

export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/.cache/triton"
mkdir -p "$TRITON_CACHE_DIR"

# Source your cluster environment
source ${PROJECT_DIR}/apertus_finetuning.toml

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=6010

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-$SLURM_GPUS_PER_NODE}
NUM_NODES=${SLURM_NNODES:-1}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# Run debug.py using accelerate in distributed mode
srun --cpu-bind=none --environment="${PROJECT_DIR}/apertus_finetuning.toml" bash -c "
    cd ${PROJECT_DIR}
    source venv-apertus/bin/activate
    python debug.py
"

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
