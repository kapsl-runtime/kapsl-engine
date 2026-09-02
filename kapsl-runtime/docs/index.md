# kapsl-runtime

`kapsl-runtime` is a high-performance AI model inference server written in Rust. It loads packaged `.aimod` model artifacts, serves them over multiple transports, and exposes a REST API, a web dashboard, and Prometheus metrics.

## What it does

- **Loads and serves packaged models** — `.aimod` archives containing ONNX, GGUF, or SafeTensors weights
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
| llama.cpp | CPU or NVIDIA GPU, for GGUF generation packages |
| Managed vLLM | NVIDIA GPU, for explicit SafeTensors causal-LM generation packages |
| TensorRT | NVIDIA GPU |
| Metal | Apple Silicon (experimental) |
| ROCm | AMD GPU |
| DirectML | Windows GPU |

## Supported model formats

`.aimod` packages can contain models in ONNX (`.onnx`), GGUF (`.gguf`), or
SafeTensors (`.safetensors`) format. Raw PyTorch (`.pt`, `.pth`) and TensorFlow
(`.pb`) weights are rejected rather than being misrouted into ONNX Runtime;
export them to a supported serving format first.

## Navigation

| Page | Description |
|------|-------------|
| [Deployment](./deployment.md) | Build, install, and run the runtime |
| [Lazy Backend Packs](./backend-packs.md) | Signed backend resolution, cache management, and offline bundles |
| [Model Packaging](./model-packaging.md) | Create and manage `.aimod` packages |
| [HTTP API](./http-api.md) | REST API reference |
| [Authentication](./authentication.md) | Token roles, API keys, access control |
| [Extensions & RAG](./extensions.md) | Extension connectors and RAG ingestion |
| [Web Dashboard](./web-dashboard.md) | Using the browser interface |
| [Configuration](./configuration.md) | CLI flags and environment variables |
| [Backend-neutral Removal Inventory](./backend-neutral-removal-inventory.md) | Classified migration inventory and dependency gates |
