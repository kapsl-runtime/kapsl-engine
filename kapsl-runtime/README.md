# kapsl-runtime

The Rust workspace for the `kapsl` runtime binary.

For full runtime documentation, see `FULL_DOCUMENTATION.md`.

## Prerequisites

- Rust 1.92.0
- (Optional) CUDA Toolkit for NVIDIA GPU support

## Layout

- `Cargo.toml`: runtime workspace manifest
- `crates/kapsl-cli/`: CLI, server orchestration, HTTP API, auth, metrics, RAG wiring, and runtime entry point
- `docs/`: runtime API and operations docs
- `ui/`: embedded web dashboard assets
- `patches/`: active third-party patches used by this workspace

Shared Rust crates such as `kapsl-core`, `kapsl-backends`, `kapsl-llm`,
`kapsl-scheduler`, `kapsl-ipc`, and `kapsl-shm` live in `kapsl-sdk`.

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

### 1. Run the Runtime

Execute the CLI with an `.aimod` model package:

```bash
cargo run -p kapsl -- --model /path/to/model.aimod
```

You can also specify a custom socket path for the IPC server:

```bash
cargo run -p kapsl -- --model /path/to/model.aimod --socket /tmp/kapsl.sock
```

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
