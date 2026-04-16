# PyTorch Distributed DDP Benchmark - Run:AI Usage Examples

This document provides detailed examples of how to use the PyTorch DDP benchmark with run:ai.

## Prerequisites

- Run:AI CLI installed and configured (`runai config`)
- Access to a run:ai cluster with GPU-enabled nodes
- Benchmark image built and available in your registry

## Building the Image

```bash
./build.sh
```

## Usage Examples

### 1. Single GPU Benchmark (Quick Test)

Test that everything is working with a single GPU:

```bash
runai submit single-gpu-test \
  --image default-route-openshift-image-registry.apps.aipod1.ciscops.net/public-images/pytorch-ddp-benchmark:latest \
  --gpu 1 \
  --command '/workspace/test_single_gpu.sh'
```

### 2. Single Node, Multiple GPUs

Run DDP benchmark on multiple GPUs in one pod:

```bash
# For 4 GPUs on one node
runai submit multi-gpu-test \
  --image pytorch-ddp-benchmark \
  --gpu 4 \
  -e WORLD_SIZE=4 \
  -e RANK=0 \
  -e MODEL=resnet50
```

### 3. Specific Model Testing

Test a specific model with result export:

```bash
runai submit model-test \
  --image pytorch-ddp-benchmark \
  --gpu 2 \
  -e WORLD_SIZE=2 \
  -e MODEL=resnet101 \
  -e JSON_OUTPUT=/shared/results.json \
  --volume /shared-storage:/shared
```

### 4. Multi-Node Distributed Training

Run distributed benchmark across multiple nodes:

```bash
# Submit a multi-worker distributed job
runai submit distributed-benchmark \
  --image pytorch-ddp-benchmark \
  --gpu 8 \
  --workers 2 \
  -e WORLD_SIZE=8 \
  -e MODEL=resnext50_32x4d \
  -e MASTER_PORT=29500
```

### 5. Interactive Mode

Run container interactively for debugging:

```bash
runai submit interactive-debug \
  --image pytorch-ddp-benchmark \
  --gpu 1 \
  --interactive \
  --command '/bin/bash'
```

### 6. Save Benchmark Results

Save results to persistent storage:

```bash
runai submit benchmark-with-results \
  --image pytorch-ddp-benchmark \
  --gpu 4 \
  -e JSON_OUTPUT=/shared/benchmark_results.json \
  -e WORLD_SIZE=4 \
  --volume /shared-storage:/shared
```

### 7. Custom Configuration

Run with custom DDP bucket size and backend:

```bash
runai submit custom-config \
  --image pytorch-ddp-benchmark \
  --gpu 8 \
  --workers 2 \
  -e WORLD_SIZE=8 \
  -e DISTRIBUTED_BACKEND=nccl \
  -e BUCKET_SIZE=50 \
  -e MODEL=resnet50
```

### 8. Resource-Constrained Testing

Run with specific CPU and memory limits:

```bash
runai submit constrained-test \
  --image pytorch-ddp-benchmark \
  --gpu 2 \
  --cpu 8 \
  --memory 32G \
  -e WORLD_SIZE=2 \
  -e MODEL=resnet50
```

### 9. Priority and Scheduling

Run with high priority and node selection:

```bash
runai submit priority-benchmark \
  --image pytorch-ddp-benchmark \
  --gpu 4 \
  --priority high \
  --node-type gpu-v100 \
  -e WORLD_SIZE=4 \
  -e MODEL=resnext101_32x8d
```

### 10. Benchmark Comparison

Run benchmarks for different models and compare:

```bash
# Baseline
runai submit resnet50-baseline \
  --image pytorch-ddp-benchmark \
  --gpu 2 \
  -e MODEL=resnet50 \
  -e JSON_OUTPUT=/shared/resnet50_baseline.json \
  --volume /shared-storage:/shared

# Test
runai submit resnet101-test \
  --image pytorch-ddp-benchmark \
  --gpu 2 \
  -e MODEL=resnet101 \
  -e JSON_OUTPUT=/shared/resnet101_test.json \
  --volume /shared-storage:/shared

# Compare using the diff tool
runai submit compare-results \
  --image pytorch-ddp-benchmark \
  --gpu 0 \
  --volume /shared-storage:/shared \
  --command 'python3 /workspace/benchmarks/distributed/ddp/diff.py /shared/resnet50_baseline.json /shared/resnet101_test.json'
```

## Run:AI Specific Features

### Job Management

Monitor your jobs:

```bash
# List all jobs
runai list

# Get job details
runai describe job <job-name>

# View job logs
runai logs <job-name>

# Delete a job
runai delete job <job-name>
```

### Resource Monitoring

Monitor GPU usage:

```bash

# Monitor GPU utilization
runai top node

# Get cluster status
runai cluster status
```

### Preemption and Scheduling

Run with preemption settings:

```bash
runai submit benchmark-preemptible \
  --image pytorch-ddp-benchmark \
  --gpu 4 \
  --preemptible \
  -e WORLD_SIZE=4 \
  -e MODEL=resnet50
```

## Environment Variables Reference

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `WORLD_SIZE` | 1 | Total number of processes |
| `RANK` | 0 | Process rank (0 to WORLD_SIZE-1) |
| `MASTER_ADDR` | localhost | Master node IP address |
| `MASTER_PORT` | 29500 | Master node port |
| `DISTRIBUTED_BACKEND` | nccl | Communication backend (nccl/gloo) |
| `BUCKET_SIZE` | 25 | DDP bucket size in MB |
| `MODEL` | "" | Specific model to test |
| `JSON_OUTPUT` | "" | Path to save JSON results |

## Available Models

- `resnet50` - ResNet-50 (default batch size: 32)
- `resnet101` - ResNet-101 (default batch size: 32)
- `resnext50_32x4d` - ResNeXt-50 32x4d (default batch size: 32)
- `resnext101_32x8d` - ResNeXt-101 32x8d (default batch size: 32)

If no model is specified, all models will be benchmarked.

## Output Interpretation

The benchmark outputs performance metrics including:

- **sec/iter**: Seconds per training iteration
- **ex/sec**: Examples (images) processed per second
- **p50, p75, p90, p95**: Performance percentiles

Example output:

```text

Benchmark: resnet50 with batch size 32

                            sec/iter    ex/sec      sec/iter    ex/sec      sec/iter    ex/sec      sec/iter    ex/sec
   1 GPUs --   no ddp:  p50:  0.097s     329/s  p75:  0.097s     329/s  p90:  0.097s     329/s  p95:  0.097s     329/s
   1 GPUs --    1M/1G:  p50:  0.100s     319/s  p75:  0.100s     318/s  p90:  0.100s     318/s  p95:  0.100s     318/s
   2 GPUs --    1M/2G:  p50:  0.103s     310/s  p75:  0.103s     310/s  p90:  0.103s     310/s  p95:  0.103s     309/s
```

## Troubleshooting

### Common Issues

1. **GPU not available**: Check cluster GPU availability with `runai top node`
2. **Job stuck in pending**: Check resource quotas with `runai describe project`
3. **Network issues in multi-worker**: Run:ai handles networking automatically, check job logs
4. **Out of memory**: Reduce batch size or request more memory with `--memory` flag

### Debug Commands

```bash
# Check CUDA availability in a job
runai submit debug-cuda \
  --image pytorch-ddp-benchmark \
  --gpu 1 \
  --command 'python3 -c "import torch; print(torch.cuda.is_available())"'

# Check job logs
runai logs <job-name> --follow

# Get job status and events
runai describe job <job-name>

# List available GPU types
runai top node

# Check project quota
runai describe project
```

### Performance Tuning

1. **Use appropriate node types**: Specify `--node-type` for optimal hardware
2. **Set CPU/memory ratios**: Use `--cpu` and `--memory` for balanced resources  
3. **Monitor resource usage**: Use `runai top job <job-name>` to monitor utilization
4. **Use preemption wisely**: Set `--preemptible` for non-urgent jobs to save costs
