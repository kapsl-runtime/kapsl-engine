# kapsl-runtime

A high-performance AI model runtime environment written in Rust.

For full runtime documentation, see `FULL_DOCUMENTATION.md`.

## Prerequisites

- Rust 1.75+
- (Optional) CUDA Toolkit for NVIDIA GPU support

## Building

```bash
cargo build --release
```

## Runtime Release Artifacts

Standalone `kapsl` runtime installers are published by CI using:

- `.github/workflows/release-runtime-installers.yml`

Behavior:

- `workflow_dispatch`: builds and uploads runtime installer artifacts to the workflow run
- `release.published`: builds and attaches runtime installer artifacts to the GitHub Release

Installers are produced for Linux/macOS/Windows (`.deb`, `.pkg`, `.msi`) with SHA256 checksum files.

## Quick Start

### 1. Generate a Sample Package

We provide a helper script to download a sample ONNX model (MNIST) and package it correctly.

```bash
chmod +x scripts/packages/mnist/create_package.sh
./scripts/packages/mnist/create_package.sh
```

This will download the MNIST model and create `models/mnist/mnist.aimod`.

Note: all package creation scripts live under `scripts/packages/*/create_package.sh`.

### 2. Run the Runtime

Execute the CLI with the model package:

```bash
cargo run -p kapsl -- --model models/mnist/mnist.aimod
```

You can also specify a custom socket path for the IPC server:

```bash
cargo run -p kapsl -- --model models/mnist/mnist.aimod --socket /tmp/kapsl.sock
```

### 3. Run Lightweight HTTP Benchmark

Use the benchmark script to measure infer latency and throughput against `/api/models/:id/infer`.

```bash
# From kapsl-runtime/
python3 scripts/benchmark_http_infer.py --model-id 0 --requests 100 --warmup 10
```

Output includes `p50`, `p95`, and `throughput` (`req/s`).

### 4. Run Cross-Port Control Loop

Use the built-in control command to orchestrate model groups running on different runtime ports.
It polls health/model/system stats, computes pressure-based routing weights, and applies per-model
scaling policy templates by runtime profile.

```bash
# Example: control loop across two runtimes
cargo run -p kapsl -- control \
  --runtime gpu-latency=http://127.0.0.1:9095 \
  --runtime gpu-throughput=http://127.0.0.1:9096 \
  --runtime-profile gpu-latency=latency \
  --runtime-profile gpu-throughput=throughput \
  --auth-token "$KAPSL_API_TOKEN_ADMIN" \
  --weights-file ./runtime-control-weights.json
```

Notes:

- `--auth-token` or `--runtime-token NAME=TOKEN` should have writer/admin access when scaling updates are enabled.
- Use `--dry-run` to compute weights without posting scaling updates.
- The output weights snapshot file is intended for external gateways/load balancers.

## Package Structure

A `.aimod` package is a `tar.gz` archive containing:

- `metadata.json`: Package manifest
- `model.onnx`: The model file (or other supported format)

### Example `metadata.json`

```json
{
  "project_name": "mnist-demo",
  "framework": "onnx",
  "version": "1.0.0",
  "created_at": "2023-10-27T10:00:00Z",
  "model_file": "mnist.onnx",
  "hardware_requirements": {
    "preferred_provider": "cpu",
    "fallback_providers": ["cuda"]
  }
}
```

## Supported Backends

- **CPU**: Default fallback, works everywhere.
- **CUDA**: For NVIDIA GPUs.
- **TensorRT**: Optimized inference on NVIDIA GPUs (requires `tensorrt` provider in metadata).
- **Metal**: (Experimental) For Apple Silicon.
