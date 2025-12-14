#!/bin/bash

# =====================================
# Set all cache directories to scratch
# =====================================

# Hugging Face cache
export HF_HOME=/iopsstor/scratch/cscs/$USER/.cache/huggingface

# Triton cache (for DeepSpeed/transformers)
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/.triton
mkdir -p $TRITON_CACHE_DIR

# DO NOT FORGET TO export HF_API_TOKEN, the Hugging Face API token, before running this

source venv-apertus/bin/activate # Run setup_venv if you already haven't

# Optional: show info
echo "HF_HOME=$HF_HOME"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

# =====================================
# Step 1: Merge LoRA model
# =====================================
echo "=== Running merge_lora.py ==="
# python merge_lora.py \
#     --lora_dir /users/kkarthikeyan/scratch/apertus-finetuning-lsai/finetuning/Apertus-FT/output/apertus_70b_lora_r64_ctx8k/checkpoint-117

# =====================================
# Step 2: Push merged model to Hugging Face Hub
# =====================================
echo "=== Running push_model.py ==="
python push_model.py

echo "=== Done! ==="
