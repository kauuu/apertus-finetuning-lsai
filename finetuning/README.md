# Our README for LORA Finetuning for Apertus Models

## For Swiss legal dataset

### LORA hyperparameters

We selected our LoRA configuration based on findings from "LoRA Without Regret" (Schulman et al., 2025) and "Practical LoRA Research" (Kirkby & O'Neill, 2025).

    Rank (r=64): The SLDS dataset contains ~60k training examples. Kirkby & O'Neill found that low ranks (r≤8) begin to saturate (stop learning efficiently) around 30k examples. To accommodate the full dataset size and the high entropy of legal reasoning tasks, we selected r=64 to ensure the adapter has sufficient capacity to model the data distribution without bottlenecking.

    Target Modules (All-Linear): Schulman et al. demonstrated that applying LoRA only to attention layers significantly underperforms. We apply adapters to all linear layers (Attn + MLP) to ensure the model matches Full Fine-Tuning performance.

    Learning Rate (2e-4): We adhere to the "10x Rule" proposed by Schulman et al., which observed that the optimal learning rate for LoRA is consistently one order of magnitude higher than that of full fine-tuning for the same model architecture.

    Batch Size (32): To avoid the "large batch penalty" observed in LoRA training dynamics (Schulman et al.), we maintain a moderate effective batch size of 32 rather than scaling up to the hardware limit.


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

### Debug to look at results

You can run `sbatch ./run_debug.sh` (modify the params in the `python` command with the bash file accordingly) to run a specific model on SLDS and visualise the output. 
