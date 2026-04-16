# Containerfile for PyTorch Distributed DDP Benchmark
# Based on https://github.com/pytorch/pytorch/tree/main/benchmarks/distributed/ddp

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopenmpi-dev \
    iproute2 \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and install basic Python packages
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
# Using stable version - adjust version as needed
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional Python dependencies for benchmarks
RUN pip3 install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    tqdm \
    psutil \
    pyyaml

# Create working directory
WORKDIR /workspace

# Create directory structure and copy local benchmark files
RUN mkdir -p /workspace/benchmarks/distributed/ddp
COPY benchmark.py /workspace/benchmarks/distributed/ddp/benchmark.py
RUN chmod +x /workspace/benchmarks/distributed/ddp/benchmark.py

# Set working directory to the DDP benchmark location
WORKDIR /workspace/benchmarks/distributed/ddp

# Copy benchmark scripts to the container
COPY run_benchmark.sh /workspace/run_benchmark.sh
COPY test_single_gpu.sh /workspace/test_single_gpu.sh
COPY launch_runai.sh /workspace/launch_runai.sh

# Make scripts executable
RUN chmod +x /workspace/run_benchmark.sh /workspace/launch_runai.sh /workspace/test_single_gpu.sh

# Default command
CMD ["/workspace/run_benchmark.sh"]