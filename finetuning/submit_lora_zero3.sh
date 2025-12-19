#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=apertus-lora-zero3
#SBATCH --time=12:00:00
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

# 1. FIX PATH: Use PWD to avoid the double-path glitch you saw earlier
PROJECT_DIR=$(pwd)
cd "${PROJECT_DIR}"
echo "Working directory: ${PROJECT_DIR}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. FIX STORAGE: Redirect ALL caching/logging to Scratch (Bypasses full Home quota)
# --------------------------------------------------------------------------
SCRATCH_DIR="/iopsstor/scratch/cscs/$USER"

export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.cache/triton"
export WANDB_DIR="${SCRATCH_DIR}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/wandb_cache"
export WANDB_CONFIG_DIR="${SCRATCH_DIR}/wandb_config"

# CHANGE THIS IF YOU WANT TO LOG TO YOUR WANDB
export WANDB_API_KEY="API-KEY"
export WANDB_ENTITY=kauuu-eth-zurich
export WANDB_PROJECT=apertus-finetuning
unset WANDB_TEAM
export WANDB_MODE=offline # runs offline mode to reduce unnecessary latency

# Create directories if they don't exist
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# IMPORTANT: To fix the .netrc error, export your API key here directly.
# Uncomment the line below and paste your key from https://wandb.ai/authorize
# export WANDB_API_KEY="paste_your_key_here"
# --------------------------------------------------------------------------

# ---- NCCL FIX FOR SLINGSHOT ----
# NCCL configuration - disable OFI, use Socket transport
unset FI_PROVIDER
unset NCCL_NET_OFI_PROVIDER
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=INFO

# Disable the OFI plugin entirely
export NCCL_NET_PLUGIN=""
export NCCL_IB_DISABLE=1


MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-6010}
export MASTER_ADDR MASTER_PORT

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-$SLURM_GPUS_PER_NODE}
NUM_NODES=${SLURM_NNODES:-1}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

ACCELERATE_CONFIG="configs/zero3_multinode.yaml"

# 4. FIX BINDING: Added --cpu-bind=none to allow Accelerate to manage processes
srun --cpu-bind=none --environment="${PROJECT_DIR}/apertus_finetuning.toml" bash -c "
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
