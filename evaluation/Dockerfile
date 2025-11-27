FROM nvcr.io/nvidia/pytorch:24.11-py3

# Install OS dependencies
RUN apt-get update && apt-get install -y git vim curl build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir vllm

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
