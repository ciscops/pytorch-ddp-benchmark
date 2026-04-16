#!/bin/bash
set -e

echo "Testing single GPU benchmark..."
echo "This will run the benchmark on 1 GPU (no DDP)"

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK')"

# Run single GPU test
WORLD_SIZE=1 RANK=0 MODEL=resnet50 /workspace/run_benchmark.sh