# PyTorch Distributed DDP Benchmark for Run:AI

This repository contains a run:ai deployment of the PyTorch Distributed Data Parallel (DDP) benchmark suite from the [official PyTorch repository](https://github.com/pytorch/pytorch/tree/main/benchmarks/distributed/ddp).

## What This Benchmark Does

- Runs distributed PyTorch training benchmarks using DistributedDataParallel (DDP)
- Tests multiple model architectures: ResNet-50, ResNet-101, ResNeXt-50, ResNeXt-101
- Measures training iteration time and throughput across different GPU configurations
- Supports both single-node multi-GPU and multi-node distributed setups
- Outputs performance metrics and can generate JSON reports for analysis
- Leverages run:ai for efficient GPU resource management and workload orchestration

## Quick Start

1. **Build the container image:**

   ```bash
   ./build.sh
   ```

2. **Optional: Push to registry (for multi-node clusters):**

   ```bash
   podman tag pytorch-ddp-benchmark:latest <your-registry>/pytorch-ddp-benchmark:latest
   podman push <your-registry>/pytorch-ddp-benchmark:latest
   ```

3. **Run a single GPU test:**

   ```bash
   runai submit single-gpu-test --image pytorch-ddp-benchmark:latest --gpu 1 --command '/workspace/test_single_gpu.sh'
   ```

4. **Run multi-GPU on single node:**

   ```bash
   runai submit multi-gpu-test --image pytorch-ddp-benchmark:latest --gpu 4 -e WORLD_SIZE=4
   ```

5. **Run distributed multi-node setup:**

   ```bash
   runai submit distributed-test --image pytorch-ddp-benchmark:latest --gpu 8 --workers 2 -e WORLD_SIZE=8
   ```

## Files in This Repository

- `Dockerfile` - Complete container definition with PyTorch, CUDA, and benchmark code (compatible with podman)
- `build.sh` - Simple script to build the image using run:ai
- `runai-job-template.yaml` - Customizable run:ai job template for advanced deployments
- `USAGE.md` - Comprehensive usage examples and configuration options
- `README.md` - This file

## What's Inside the Container

- **Base**: NVIDIA CUDA 12.1 with Ubuntu 22.04
- **PyTorch**: Latest stable version with CUDA support
- **Benchmark Code**: Full PyTorch DDP benchmark suite from the official repository
- **Run:AI Integration**: Configured for run:ai workload management
- **Models**: Pre-configured ResNet and ResNeXt models for testing

## Key Features

- 🚀 **Easy Setup**: Single run:ai command to launch benchmarks
- 🔧 **Configurable**: Environment variables for all benchmark parameters
- 📊 **Comprehensive Output**: Detailed performance metrics and JSON export
- 🌐 **Multi-Node Ready**: Run:ai handles distributed training orchestration
- 📋 **Multiple Models**: Tests various popular deep learning architectures
- ⚡ **GPU-Optimized**: Leverages run:ai for efficient GPU resource allocation
- 🎛️ **Resource Management**: Automatic scaling and resource optimization

## Prerequisites

- Podman installed (for building the container image)
- Run:AI CLI installed and configured
- Access to a run:ai cluster with GPU nodes
- CUDA-compatible GPU(s) in the cluster

## Performance Metrics

The benchmark measures:

- **Iteration Time**: Time per training step
- **Throughput**: Images processed per second
- **Scaling Efficiency**: Performance across different GPU counts
- **Statistical Analysis**: P50, P75, P90, P95 percentiles

## Use Cases

- **Performance Regression Testing**: Compare PyTorch versions or configurations
- **Hardware Evaluation**: Benchmark different GPU setups
- **Network Analysis**: Test distributed training across different network topologies
- **Scaling Studies**: Understand how performance scales with GPU count
- **Research Baselines**: Establish performance baselines for new techniques

## Example Output

```text
Benchmark: resnet50 with batch size 32

                            sec/iter    ex/sec      sec/iter    ex/sec
   1 GPUs --   no ddp:  p50:  0.097s     329/s  p75:  0.097s     329/s
   1 GPUs --    1M/1G:  p50:  0.100s     319/s  p75:  0.100s     318/s
   2 GPUs --    1M/2G:  p50:  0.103s     310/s  p75:  0.103s     310/s
   4 GPUs --    1M/4G:  p50:  0.103s     310/s  p75:  0.103s     310/s
   8 GPUs --    1M/8G:  p50:  0.104s     307/s  p75:  0.104s     307/s
```

## Advanced Usage

See [USAGE.md](USAGE.md) for detailed examples including:

- Multi-node distributed training strategies
- Run:AI job templates and resource management  
- Result analysis and comparison
- Custom model testing
- Performance tuning and monitoring
- Resource quotas and scheduling options

For advanced deployments, customize and use the provided job template:

```bash
runai submit -f runai-job-template.yaml
```

## Contributing

This container is based on the official PyTorch benchmark suite. For issues with the underlying benchmark code, please refer to the [PyTorch repository](https://github.com/pytorch/pytorch).

For container-specific improvements or issues, please open an issue in this repository.

## License

This project follows the same license as PyTorch (BSD-3-Clause). The benchmark code is from the official PyTorch repository.
