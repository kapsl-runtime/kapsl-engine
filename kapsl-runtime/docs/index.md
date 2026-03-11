# kapsl-runtime

`kapsl-runtime` is a high-performance AI model inference server written in Rust. It loads packaged `.aimod` model artifacts, serves them over multiple transports, and exposes a REST API, a web dashboard, and Prometheus metrics.

## What it does

- **Loads and serves packaged models** — `.aimod` archives containing ONNX, GGUF, SafeTensors, or PyTorch weights
- **Multiple transports** — Unix socket, TCP, shared memory, and hybrid IPC
- **REST HTTP API** — model management, inference, RAG, extensions, auth
- **Web dashboard** — browser UI for monitoring, model control, and extension management
- **Python client** — `kapsl-sdk` Python package for connecting from Python applications
- **Extension system** — installable connectors for RAG data sources (S3, Azure Blob, etc.)
- **Prometheus metrics** — throughput, queue depth, latency histograms

## Supported backends

| Backend | Hardware |
|---------|----------|
| ONNX Runtime | CPU (all platforms) |
| TensorRT | NVIDIA GPU |
| Metal | Apple Silicon (experimental) |
| ROCm | AMD GPU |
| DirectML | Windows GPU |

## Supported model formats

`.aimod` packages can contain models in: ONNX (`.onnx`), GGUF (`.gguf`), SafeTensors (`.safetensors`), PyTorch (`.pt`, `.pth`), TensorFlow SavedModel (`.pb`).

## Navigation

| Page | Description |
|------|-------------|
| [Deployment](./deployment.md) | Build, install, and run the runtime |
| [Model Packaging](./model-packaging.md) | Create and manage `.aimod` packages |
| [HTTP API](./http-api.md) | REST API reference |
| [Authentication](./authentication.md) | Token roles, API keys, access control |
| [Extensions & RAG](./extensions.md) | Extension connectors and RAG ingestion |
| [Web Dashboard](./web-dashboard.md) | Using the browser interface |
| [Configuration](./configuration.md) | CLI flags and environment variables |
