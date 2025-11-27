FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python and basic tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/vllm
ENV PATH="/opt/vllm/bin:$PATH"

# Upgrade pip and install vLLM (and its deps) into the venv
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "vllm==0.6.4"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]