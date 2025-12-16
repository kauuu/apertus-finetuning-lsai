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

echo "=== Running push_model.py ==="
python push_model.py
echo "=== Done! ==="