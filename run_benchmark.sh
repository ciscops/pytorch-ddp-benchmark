#!/bin/bash
set -e

# Default configuration
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND:-"nccl"}
BUCKET_SIZE=${BUCKET_SIZE:-25}
MODEL=${MODEL:-""}
JSON_OUTPUT=${JSON_OUTPUT:-""}

echo "==================================="
echo "PyTorch DDP Benchmark Configuration"
echo "==================================="
echo "World Size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Distributed Backend: $DISTRIBUTED_BACKEND"
echo "Bucket Size: ${BUCKET_SIZE}MB"
echo "Model: ${MODEL:-"All models (resnet50, resnet101, resnext50_32x4d, resnext101_32x8d)"}"
echo "JSON Output: ${JSON_OUTPUT:-"None"}"
echo "==================================="

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Build the command
CMD="python3 benchmark.py --rank $RANK --world-size $WORLD_SIZE --master-addr $MASTER_ADDR --master-port $MASTER_PORT --distributed-backend $DISTRIBUTED_BACKEND --bucket-size $BUCKET_SIZE"

if [ ! -z "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi

if [ ! -z "$JSON_OUTPUT" ]; then
    CMD="$CMD --json $JSON_OUTPUT"
fi

echo "Running command: $CMD"
echo "==================================="

# Execute the benchmark
exec $CMD