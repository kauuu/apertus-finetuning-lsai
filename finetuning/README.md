# Apertus Fine-tuning Guide

## Overview
This repository provides tools for fine-tuning large language models using both full fine-tuning and LoRA (Low-Rank Adaptation) methods with distributed training support.

## Quick Start

### 1. Environment Setup
```bash
# Request a compute node with the fine-tuning environment
srun --environment=/users/$USER/scratch/apertus-finetuning-lsai/finetuning/apertus_finetuning.toml [your resource requirements]

# Set up the Python virtual environment
./setup_venv.sh

# Exit the compute node after setup
exit
```

### 2. Configuration

Modify the configuration files in ./configs/ to set:
- Hyperparameters (learning rate, batch size, epochs, etc.)
- Dataset path and preprocessing options
- Base model selection
- Training method (LoRA or full fine-tuning)

### 3. Submit Training Jobs
Choose the appropriate script based on your needs:

| Script | Method | Hardware | Description |
|--------|--------|----------|-------------|
| `./submit_full_zero3.sh` | Full fine-tuning | Multi-node | Full parameter training with FSDP (ZeRO-3) |
| `./submit_lora_single_node.sh` | LoRA | Single node | Parameter-efficient fine-tuning on single node |
| `./submit_lora_zero3.sh` | LoRA | Multi-node | LoRA with Fully Sharded Data Parallelism |

Example:
```bash
sbatch ./submit_lora_single_node.sh
```

## Post-Training Operations

### Merging LoRA Adapters
After LoRA training, merge the adapter with the base model (ensure you are on the compute node with the right environment):

```bash
# Set your Hugging Face API token
export HF_API_TOKEN="your_api_token_here"

# Merge and push to Hugging Face Hub
./run_merge_push.sh
```

### Pushing Full Fine-tuned Models

For fully fine-tuned models, push directly to Hugging Face (ensure you are on the compute node with the right environment):

```bash
export HF_API_TOKEN="your_api_token_here"
./push_only.sh
```

