#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=apertus-lora-zero3
#SBATCH --time=04:00:00
#SBATCH --nodes=4
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

# Must be set before sbatch is called - SLURM copies the script to a temp directory
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
mkdir -p "$HF_HOME"

export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/.cache/triton"
mkdir -p "$TRITON_CACHE_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-6010}
export MASTER_ADDR MASTER_PORT
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-$SLURM_GPUS_PER_NODE}
NUM_NODES=${SLURM_NNODES:-1}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

ACCELERATE_CONFIG="configs/zero3_multinode.yaml"

srun --environment="${PROJECT_DIR}/apertus_finetuning.toml" bash -c "
    cd ${PROJECT_DIR}
    source venv-apertus/bin/activate
    accelerate launch --config_file ${ACCELERATE_CONFIG} \\
        --num_processes ${WORLD_SIZE} --num_machines ${NUM_NODES} \\
        --machine_rank \${SLURM_PROCID} --main_process_ip ${MASTER_ADDR} \\
        --main_process_port ${MASTER_PORT} \\
        sft_train.py --config configs/sft_lora.yaml
"

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
