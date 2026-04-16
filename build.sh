#!/bin/bash

# Build script for PyTorch DDP Benchmark

set -e

# Configuration
IMAGE_NAME="pytorch-ddp-benchmark"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building PyTorch DDP Benchmark container image..."

# Build the container image
podman build -t "${FULL_IMAGE_NAME}" .

echo ""
echo "Build complete! Image: ${FULL_IMAGE_NAME}"
echo ""
echo "Optional: Push to registry for run:ai cluster access:"
echo "  podman tag ${FULL_IMAGE_NAME} <your-registry>/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  podman push <your-registry>/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run the benchmark with run:ai:"
echo "  Single GPU test: runai submit single-gpu-test --image ${FULL_IMAGE_NAME} --gpu 1 --command '/workspace/test_single_gpu.sh'"
echo "  Multi-GPU: runai submit multi-gpu-test --image ${FULL_IMAGE_NAME} --gpu 4 -e WORLD_SIZE=4"
echo "  Multi-node distributed: runai submit distributed-test --image ${FULL_IMAGE_NAME} --gpu 8 --workers 2"
echo ""
echo "See README.md for more detailed usage instructions."