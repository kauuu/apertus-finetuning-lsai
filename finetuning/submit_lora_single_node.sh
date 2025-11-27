#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=apertus-lora
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
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
# Get the submission directory (where sbatch was called from)
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set HuggingFace cache directory
export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
mkdir -p "$HF_HOME"

export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/.cache/triton"
mkdir -p "$TRITON_CACHE_DIR"
export WANDB_API_KEY="0d3c026f963e57e51a98ca9949542dc92a1c2276"

# Run training with Container Engine (use absolute path to EDF)
srun --environment="${PROJECT_DIR}/apertus_finetuning.toml" bash -c "
    cd ${PROJECT_DIR}
    source venv-apertus/bin/activate
    pip install datasets
    pip install -r requirements.txt
    python sft_train.py --config configs/sft_lora.yaml
"

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
