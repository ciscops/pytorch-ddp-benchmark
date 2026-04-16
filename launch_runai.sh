#!/bin/bash
set -e

# Run:AI launcher for DDP benchmark
# Usage: ./launch_runai.sh <gpus> <workers> [model]

GPUS=${1:-4}
WORKERS=${2:-1}
MODEL=${3:-""}
TOTAL_GPUS=$((GPUS * WORKERS))

echo "Launching run:ai DDP benchmark"
echo "GPUs per worker: $GPUS"
echo "Number of workers: $WORKERS"
echo "Total GPUs: $TOTAL_GPUS"
echo "Model: ${MODEL:-"All models"}"

# Build the run:ai command
CMD="runai submit pytorch-ddp-benchmark --image pytorch-ddp-benchmark --gpu $GPUS"

if [ $WORKERS -gt 1 ]; then
    CMD="$CMD --workers $WORKERS"
fi

CMD="$CMD -e WORLD_SIZE=$TOTAL_GPUS"

if [ ! -z "$MODEL" ]; then
    CMD="$CMD -e MODEL=$MODEL"
fi

echo "Command: $CMD"
echo "Execute this command to launch the benchmark"