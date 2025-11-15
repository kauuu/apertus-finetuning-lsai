#!/bin/bash
set -e

echo "Creating virtual environment with system site packages..."
python -m venv --system-site-packages venv-apertus

echo "Activating virtual environment..."
source venv-apertus/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

echo "Installing apertus-finetuning requirements..."
pip install -r requirements.txt

echo "✅ Virtual environment setup complete!"
echo "Location: $(pwd)/venv-apertus"
