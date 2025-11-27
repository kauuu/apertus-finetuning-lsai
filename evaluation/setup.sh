# venv for running vLLM API server
python -m venv ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install vLLM (and CUDA-compatible torch)
pip install vllm torch --index-url https://download.pytorch.org/whl/cu128
deactivate

# venv for SLDS evaluation
python -m venv ~/.venvs/slds-eval
source ~/.venvs/slds-eval/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install your dependencies using uv sync / pyproject.toml
uv sync
deactivate
