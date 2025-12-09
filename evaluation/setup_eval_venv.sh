#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv-eval"

echo "Creating virtual environment with system site packages..."
python3 -m venv --system-site-packages "${VENV_DIR}"

echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing lighteval from git (base install only, no Asian tokenizers)..."
pip install "lighteval @ git+https://github.com/huggingface/lighteval"

echo "Installing multilingual dependencies (without Asian language spacy models)..."
# Install multilingual deps individually to avoid spacy[ja,ko,th] which pulls in sudachipy
pip install stanza
pip install spacy>=3.8.0  # base spacy without ja/ko/th extras
pip install jieba  # Chinese tokenizer
pip install pyvi   # Vietnamese tokenizer

echo "Installing XIELU (CUDA-fused xIELU)..."
export CUDA_HOME=/usr/local/cuda
pip install git+https://github.com/nickjbrowning/XIELU --no-build-isolation --no-deps

echo "Installing evaluation requirements..."
pip install -r "${SCRIPT_DIR}/requirements_local.txt"

echo "✅ Evaluation virtual environment setup complete!"
echo "Location: ${VENV_DIR}"
