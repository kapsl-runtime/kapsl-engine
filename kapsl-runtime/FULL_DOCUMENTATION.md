# kapsl-runtime Full Documentation

Last updated: 2026-03-05  
Primary runtime binary: `kapsl` (`kapsl-runtime/crates/kapsl-cli`)

## 1. What `kapsl-runtime` Is

`kapsl-runtime` is a Rust-based model runtime and packaging system for `.aimod` model artifacts.

Core capabilities:

- Load and serve one or more packaged models.
- Support multiple transports: Unix socket / TCP / shared memory / hybrid.
- Expose HTTP API, web UI, and Prometheus metrics.
- Build, push, and pull `.aimod` packages.
- Support extension-driven RAG ingestion and retrieval.
- Provide Python bindings (`kapsl_runtime`) for socket, SHM, and hybrid clients.

## 2. Repository Layout

Top-level runtime paths:

- `kapsl-runtime/Cargo.toml`: workspace manifest.
- `kapsl-runtime/crates/kapsl-cli`: `kapsl` CLI and HTTP server.
- `kapsl-runtime/crates/kapsl-core`: package loader, model registry, scaling policy types.
- `kapsl-runtime/crates/kapsl-engine-api`: shared engine/request/response types.
- `kapsl-runtime/crates/kapsl-backends`: backend selection and provider factory.
- `kapsl-runtime/crates/kapsl-scheduler`: scheduler queues and priority routing.
- `kapsl-runtime/crates/kapsl-ipc`: socket/TCP protocol server.
- `kapsl-runtime/crates/kapsl-shm`: shared memory transport primitives.
- `kapsl-runtime/crates/kapsl-transport`: common transport abstractions.
- `kapsl-runtime/crates/kapsl-pyo3`: Python extension module (`kapsl_runtime`).
- `kapsl-runtime/crates/kapsl-llm`: LLM backend and RAG prompt composition.
- `kapsl-runtime/crates/kapsl-rag`: extension runtime, doc store, vector store.
- `kapsl-runtime/crates/kapsl-rag-sdk`: connector manifest/protocol types.
- `kapsl-runtime/scripts`: tests, validation, and helper scripts.
- `kapsl-runtime/scripts/packages`: package creation scripts for sample models.
- `kapsl-runtime/ui`: web dashboard static files.

## 3. Build and Run

### 3.1 Prerequisites

- Rust 1.75+.
- Python 3.8+ (for helper scripts and PyO3 builds).
- Optional GPU/runtime dependencies depending on provider:
  - CUDA/TensorRT (NVIDIA),
  - CoreML/Metal (macOS),
  - ROCm,
  - DirectML.
- `ffmpeg` for video media inference payloads.

### 3.2 Build

From `kapsl-runtime/`:

```bash
cargo build --release
```

### 3.3 Quick Start

Create sample package:

```bash
./scripts/packages/mnist/create_package.sh
```

Run runtime:

```bash
cargo run -p kapsl -- --model models/mnist/mnist.aimod
```

Defaults:

- Transport: `socket`
- Socket path (Unix): `/tmp/kapsl.sock`
- HTTP bind: `127.0.0.1`
- HTTP/API/UI/metrics port: `9095`
- TCP transport port (if used): `9096`

## 4. `kapsl` CLI

`kapsl` supports subcommands and legacy direct run invocation:

- `kapsl run ...`
- `kapsl build ...`
- `kapsl push ...`
- `kapsl pull ...`
- `kapsl --model ...` (legacy compatibility for `run`)

### 4.1 Run Options

Key run flags:

- `--model <PATH>` (repeatable): `.aimod` package paths.
- `--transport <socket|tcp|shm|hybrid|auto>` (default `socket`).
- `--socket <PATH>` (default `/tmp/kapsl.sock` on Unix).
- `--bind <IP>` and `--port <PORT>` for TCP server (`127.0.0.1:9096` default).
- `--metrics-port <PORT>` (also HTTP API/UI bind port, default `9095`).
- `--http-bind <IP>` (default `127.0.0.1`).
- `--batch-size` (default `4`).
- `--scheduler-queue-size` (default `256`).
- `--scheduler-max-micro-batch` (default `4`).
- `--scheduler-queue-delay-ms` (default `2`).
- `--performance-profile <standard|balanced|throughput|latency>`.
- `--topology` and `--tp-degree` for parallel topology hints.

### 4.2 Queue Overflow Policy

The scheduler uses an internal `WorkQueue` per GPU worker with a fixed capacity set by `--scheduler-queue-size`. When the queue is full, the runtime applies one of three overflow policies:

| Policy | Behaviour |
| --- | --- |
| `block` (default) | Caller waits until a slot is free. |
| `drop_newest` | The incoming request is immediately rejected with an overload error. |
| `drop_oldest` | The oldest queued request is evicted (and receives an overload error); the new request is enqueued. |

The policy is not yet a CLI flag; it is set programmatically via `Scheduler::with_queue_overflow_policy`. High-priority (latency-critical) requests always use the high-priority queue; throughput-class requests use the low-priority queue and may be micro-batched.

Performance profile tuning (applies only when related flags are not explicitly passed):

- `balanced`: batch `8`, transport `hybrid`, queue size `512`, micro-batch `batch_size`, delay `3ms`.
- `throughput`: batch `16`, transport `hybrid`, queue size `2048`, micro-batch `batch_size`, delay `6ms`.
- `latency`: batch `1`, transport `socket`, queue size `128`, micro-batch `1`, delay `0ms`.

### 4.2 Build Command

`kapsl build` defaults to using the current directory as the build context (equivalent to `kapsl build .`).

Common ways to build:

1. Build from a model file:

```bash
kapsl build ./model.onnx --output ./model.aimod
```

1. Build from a context directory:

```bash
kapsl build ./models/my-model-context
```

1. Build from inside the context directory:

```bash
cd ./models/my-model-context
kapsl build
```

Context mode behavior:

- Reads optional `metadata.json` in context.
- Auto-finds model file (`.onnx`, `.gguf`, `.safetensors`, `.pt`, `.pth`, `.pb`) when unambiguous.
- Includes context files in archive, excluding generated `metadata.json` and `.aimod` outputs.

### 4.3 Push / Pull Commands

Push:

```bash
kapsl push acme/model:prod ./model.aimod
```

Or from inside a directory with a single `.aimod`:

```bash
cd ./models/mnist
kapsl push acme/mnist:prod
```

Pull:

```bash
kapsl pull acme/model:prod --destination-dir ./models
```

Target format:

- Required for push and pull: `<repo_name>/<model>:<label>`
- Examples: `alice/kokoro:latest`, `team-a/recommender:v1`

Default remote behavior:

- Uses `https://api.kapsl.net/v1` by default.
- If remote URL resolves to the legacy placeholder URL, mirrors artifacts to a local directory (configurable env var).
- If `remote_url` is overridden, uses HTTP PUT/GET remote backend.
- If `remote_url` starts with `oci://`, uses ORAS to push/pull the `.aimod` as an OCI artifact.

## 5. `.aimod` Package Format

A `.aimod` package is a `tar.gz` archive that includes at minimum:

- `metadata.json`
- model file referenced by `metadata.model_file`

### 5.1 Manifest Schema (`metadata.json`)

Canonical fields loaded by runtime:

- `project_name: string`
- `framework: string` (for example `onnx`, `llm`)
- `version: string`
- `created_at: string`
- `model_file: string`
- `metadata: object|null` (optional free-form metadata)
- `hardware_requirements: object` (optional, defaults applied)

`hardware_requirements` supports:

- `preferred_provider`
- `fallback_providers`
- `min_memory_mb`
- `min_vram_mb`
- `min_cuda_version`
- `required_precision`
- `optimized_for`
- `graph_optimization_level` (`disable|basic|extended|all` or numeric aliases)
- `device_id`
- `strategy`

### 5.2 Sample Manifest

```json
{
  "project_name": "mnist-demo",
  "framework": "onnx",
  "version": "1.0.0",
  "created_at": "2026-02-11T00:00:00Z",
  "model_file": "mnist.onnx",
  "hardware_requirements": {
    "preferred_provider": "cpu",
    "fallback_providers": ["cuda"],
    "graph_optimization_level": "all",
    "device_id": 0,
    "strategy": "pool"
  }
}
```

## 6. Transport Modes

Supported run transports:

- `socket`: Unix domain socket / named pipe server.
- `tcp`: network socket server.
- `shm`: shared-memory queue transport.
- `hybrid`: socket control plane + SHM data plane.
- `auto`: prefers SHM if available, otherwise socket.

Low-level IPC op codes:

- `1`: infer
- `2`: infer stream
- `3`: metrics (reserved in protocol)
- `4`: hybrid infer

Streaming status codes:

- `2`: stream chunk
- `3`: stream end

## 7. HTTP Server and API

The HTTP server hosts:

- Web UI: `/`
- UI assets: `/ui/*`
- Metrics: `/metrics`
- REST API: `/api/*`

### 7.1 Authentication and Access Model

Roles:

- `reader`
- `writer`
- `admin`

Behavior:

- If auth is enabled, `/api` and `/metrics` require `Authorization: Bearer <token-or-api-key>`.
- If auth is disabled, these endpoints are loopback-only.
- Metrics endpoint is admin-scoped.

Scope rules for API keys:

- Empty key scopes are treated as unrestricted role-based keys (backward-compatible behavior).
- Supported scope semantics include read/write/admin and wildcard variants.

Auth store:

- Default path: `~/.kapsl/auth-store.json` (or `%USERPROFILE%\.kapsl\auth-store.json` on Windows).
- Overridable by env var.

### 7.2 HTTP Bind Safety

By default runtime refuses non-loopback HTTP bind unless:

- `KAPSL_ALLOW_INSECURE_HTTP=1` (or legacy alias).

This is a guard against exposing plaintext API directly.

### 7.3 Endpoint Groups by Required Role

Reader endpoints:

- `GET /api/models`
- `GET /api/models/:id`
- `GET /api/health`
- `GET /api/hardware`
- `GET /api/system/stats`
- `POST /api/models/:id/infer`
- `GET /api/models/:id/scaling`
- `POST /api/rag/query`

Writer endpoints:

- `GET /api/extensions`
- `GET /api/extensions/marketplace?q=...`
- `POST /api/extensions/install`
- `POST /api/extensions/:id/uninstall`
- `POST /api/extensions/:id/config`
- `GET /api/extensions/:id/config?workspace_id=...`
- `POST /api/extensions/:id/launch`
- `POST /api/extensions/:id/sync`

Admin endpoints:

- `POST /api/engine/package`
- `POST /api/engine/push`
- `POST /api/engine/pull`
- `POST /api/models/start`
- `POST /api/models/:id/stop`
- `POST /api/models/:id/remove`
- `POST /api/models/:id/scaling`
- `GET /api/auth/roles`
- `POST /api/auth/roles`
- `GET /api/auth/access/status`
- `GET /api/auth/access/roles`
- `GET /api/auth/access/users`
- `POST /api/auth/access/users`
- `PATCH /api/auth/access/users/:id`
- `GET /api/auth/access/keys?user_id=...`
- `POST /api/auth/access/users/:id/keys`
- `POST /api/auth/access/keys/:id/revoke`
- `GET /metrics` (admin)

## 8. Inference Payloads

### 8.1 Tensor JSON Payload (`POST /api/models/:id/infer`)

Tensor mode passes through `InferenceRequest`:

```json
{
  "input": {
    "shape": [1, 1, 28, 28],
    "dtype": "float32",
    "data": [0, 0, 0, 63]
  },
  "additional_inputs": [],
  "session_id": "session-1",
  "metadata": {
    "request_id": "req-1",
    "timeout_ms": 1000,
    "priority": 1,
    "force_cpu": false
  }
}
```

You can also send tensor bytes as base64 (faster to parse than large JSON byte arrays):

```json
{
  "input": {
    "shape": [1, 1, 28, 28],
    "dtype": "float32",
    "data_base64": "AAAAPw=="
  }
}
```

Notes:

- Use exactly one of `input.data` or `input.data_base64` (or alias `input.base64`).
- `data` remains supported for backward compatibility.

`dtype` values:

- `float32`, `float64`, `float16`
- `int32`, `int64`
- `uint8`
- `string` (UTF-8 bytes in `data`)

### 8.2 Media Payload (Image/Video Base64)

Media adapter accepts:

```json
{
  "media": {
    "kind": "image",
    "mime_type": "image/png",
    "base64": "..."
  },
  "tensor_options": {
    "target_width": 224,
    "target_height": 224,
    "layout": "nchw",
    "channels": "rgb",
    "dtype": "float32",
    "normalize": "zero_to_one",
    "frame_count": 1,
    "frame_stride": 1
  },
  "additional_media_inputs": [],
  "additional_inputs": []
}
```

Notes:

- `base64` alias maps to `data_base64`.
- `kind` can be omitted; kind is auto-detected via MIME/data URI.
- Video path uses `ffmpeg` and supports `frame_count`, `frame_stride`, `start_time_ms`, `end_time_ms`.
- Additional named media tensors are supported via `additional_media_inputs`.
- Inline media preprocessing can be disabled via env var.

### 8.3 RAG Augmentation in Infer

`infer` payload may include `rag` options:

```json
{
  "input": {
    "shape": [1, 12],
    "dtype": "string",
    "data": [72, 101, 108, 108, 111]
  },
  "rag": {
    "workspace_id": "ws-1",
    "top_k": 4,
    "min_score": 0.1,
    "max_context_tokens": 768
  }
}
```

Important:

- RAG augmentation currently requires string infer input (`dtype: "string"`).
- `rag.workspace_id` is required when `rag` is present.

### 8.4 Streaming Inference

- IPC/socket protocol supports streaming op (`OP_INFER_STREAM`).
- Python `KapslClient.infer_stream(...)` consumes streaming frames.
- HTTP `POST /api/models/:id/infer` is currently synchronous (single response).

## 9. Model Lifecycle and Scaling API

Start model:

- `POST /api/models/start`
- body:
  - `model_path` (required)
  - `model_id` (optional)
  - `topology` (default `data-parallel`)
  - `tp_degree` (default `1`)

Stop/remove:

- `POST /api/models/:id/stop`
- `POST /api/models/:id/remove`

Scaling policy:

- `GET /api/models/:id/scaling`
- `POST /api/models/:id/scaling`

Policy schema:

- `min_replicas`
- `max_replicas`
- `target_queue_depth`
- `scale_down_threshold`
- `cooldown_seconds`

Defaults:

- `min_replicas=1`
- `max_replicas=4`
- `target_queue_depth=5`
- `scale_down_threshold=2`
- `cooldown_seconds=300`

## 10. Extensions and RAG Sync

Extension registry roots:

- extensions install root (default `extensions/`)
- workspace config root (default `extensions-config/`)

Manifest files accepted in extension folder:

- `rag-extension.toml` or `rag-extension.json`

Core extension operations:

- Discover/install/uninstall extensions.
- Set/get workspace connector config.
- Launch connector runtime (WASM or sidecar).
- Sync connector deltas and documents into local RAG store.

RAG local storage defaults:

- root: `rag-data/`
- docs: `rag-data/docs/...`
- vectors: `rag-data/vectors.sqlite3`

`/api/extensions/:id/sync` performs:

- connector `Sync`
- connector `FetchDocument` for upserts
- deletes/upserts into doc store + vector store
- returns counts (`upserted_docs`, `deleted_docs`, `chunks_upserted`, `next_cursor`, etc.)

## 11. Python Bindings (`kapsl_runtime`)

Module: `kapsl_runtime` (built from `crates/kapsl-pyo3`).

Classes:

- `KapslClient(endpoint: str | None = None, *, protocol: str | None = None, host: str | None = None, port: int | None = None, socket_path: str | None = None, pipe_name: str | None = None, max_pool_size: int = 8)`
  - explicit protocol values: `socket`, `tcp`, `pipe`
  - defaults when omitted:
    - no args: local socket/pipe (`/tmp/kapsl.sock` on Unix, `\\.\pipe\kapsl` on Windows)
    - `protocol="tcp"` with no host/port: `127.0.0.1:9096`
    - host-only or port-only (without endpoint): missing part is defaulted (`127.0.0.1` or `9096`)
    - `protocol="pipe"` with no `pipe_name`/`endpoint` (Windows): `\\.\pipe\kapsl`
  - endpoint examples (backward compatible):
    - Unix socket path: `/tmp/kapsl.sock`
    - Explicit Unix URI: `unix:///tmp/kapsl.sock`
    - TCP endpoint: `tcp://127.0.0.1:9096`
    - Windows named pipe: `\\.\pipe\kapsl` or `pipe://kapsl`
  - explicit protocol examples:
    - `KapslClient(protocol="tcp", host="127.0.0.1", port=9096)`
    - `KapslClient(protocol="socket", socket_path="/tmp/kapsl.sock")`
  - `infer(model_id, shape, dtype, data) -> bytes`
  - `infer_stream(model_id, shape, dtype, data) -> iterator[bytes]`
  - `protocol() -> str` (resolved protocol)
  - `endpoint() -> str` (resolved endpoint URI)
- `KapslShmClient(shm_name: str)`
  - `infer(shape, dtype, data) -> bytes`
- `KapslHybridClient(shm_name: str, socket_path: str)`
  - `infer(shape, dtype, data) -> bytes`

Build (example):

```bash
cd crates/kapsl-pyo3
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

## 12. Environment Variables

Primary runtime env vars:

- `KAPSL_API_TOKEN`
- `KAPSL_API_TOKEN_READER`
- `KAPSL_API_TOKEN_WRITER`
- `KAPSL_API_TOKEN_ADMIN`
- `KAPSL_AUTH_STORE_PATH`
- `KAPSL_ALLOW_INSECURE_HTTP`
- `KAPSL_PROVIDER_POLICY` (`fastest` or `manifest`)
- `KAPSL_LLM_ISOLATE_PROCESS`
- `KAPSL_LOG_SENSITIVE_IDS` (set truthy to disable request/session redaction)
- `KAPSL_RAG_STORAGE_ROOT`
- `KAPSL_EXTENSIONS_ROOT`
- `KAPSL_EXT_CONFIG_ROOT`
- `KAPSL_EXTENSION_MARKETPLACE_URL`
- `KAPSL_REMOTE_URL`
- `KAPSL_REMOTE_PLACEHOLDER_URL`
- `KAPSL_REMOTE_PLACEHOLDER_DIR`
- `KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS`
- `KAPSL_INFER_ADAPTERS` (optional adapters; e.g. `echo_tensor` when built with feature)

Model cache / disk-check env vars (read by `kapsl-core` loader):

- `KAPSL_MODEL_CACHE_DIR` (or `KAPSL_LITE_MODEL_CACHE_DIR`): override the model cache root directory (default: `.kapsl-model-cache/` next to the `.aimod` file).
- `KAPSL_MODEL_CACHE_MAX_BYTES` / `KAPSL_MODEL_CACHE_MAX_MIB`: cap the total size of the model cache; the loader will evict least-recently-used entries to stay within limit.
- `KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES` / `KAPSL_MODEL_CACHE_RESERVED_FREE_MIB`: minimum free disk space that must remain after a model cache copy; the loader will evict LRU entries until the constraint is satisfied (or fail with `InsufficientDiskSpace`).
- `KAPSL_PACKAGE_TMP_DIR` (or `KAPSL_LITE_PACKAGE_TMP_DIR`): override the temporary directory used when unpacking an `.aimod` archive.

All above support legacy `KAPSL_*` aliases in runtime code.

## 13. Example API Calls

Health:

```bash
curl http://127.0.0.1:9095/api/health
```

List models:

```bash
curl http://127.0.0.1:9095/api/models
```

Tensor infer:

```bash
curl -X POST http://127.0.0.1:9095/api/models/0/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "shape": [1,1,28,28],
      "dtype": "float32",
      "data": [0,0,0,63]
    }
  }'
```

Package build through API:

```bash
curl -X POST http://127.0.0.1:9095/api/engine/package \
  -H 'Content-Type: application/json' \
  -d '{
    "model_path": "/absolute/path/to/model.onnx",
    "output_path": "/absolute/path/to/model.aimod"
  }'
```

## 14. Testing and Benchmarks

Useful scripts:

- `scripts/benchmark_http_infer.py`: lightweight HTTP infer latency/throughput benchmark.
- `scripts/validate_all_features.py`: API/metrics/dashboard validation script.
- `scripts/test_deployment.py`: deployment-oriented smoke tests.
- `scripts/test_inference_llama_gpt.py`: LLM inference script.
- `benchmarks/benchmark_production.sh`: multi-client production scenario benchmark.
- `benchmarks/*.py`: additional transport and client benchmark experiments.

### kapsl vs vLLM Qwen benchmark

A one-command head-to-head throughput/latency comparison script lives at:

```text
engine/kapsl-benchmarks/run_kapsl_vs_vllm_qwen.sh
```

The script starts a tuned kapsl instance, waits for vLLM to be ready on its own port, runs throughput/latency sweeps against both, and prints a concise summary table.
See `engine/kapsl-benchmarks/README.md` for reproducible usage and common options.

## 15. Troubleshooting

Common issues:

- `Model not found`: verify `model_id` and that model is active in `/api/models`.
- Non-loopback bind rejected: set `KAPSL_ALLOW_INSECURE_HTTP=1` only behind trusted TLS/network controls.
- Video infer fails: ensure `ffmpeg` is installed and available in `PATH`.
- Media infer rejected: verify tensor shape/dtype expectations and preprocessing options.
- Push/pull placeholder error: ensure pushed artifact exists in placeholder mirror dir.
- Unauthorized/forbidden API: verify role token/api key and required route scope.
- `InsufficientDiskSpace` on load: the loader could not free enough cache space. Either set `KAPSL_MODEL_CACHE_MAX_BYTES`/`KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES` to a larger threshold, point `KAPSL_MODEL_CACHE_DIR` at a volume with more space, or remove stale entries from the cache directory manually.

## 16. Known Constraints

- HTTP infer is synchronous; token streaming is available on IPC protocol/Python stream client path.
- RAG embeddings are currently lightweight hash-based embeddings (not external embedding model-backed).
- Video media path depends on host `ffmpeg`.
- Some helper scripts in `scripts/` are exploratory and may not match the latest wire protocol exactly.

---

For starter usage, see `kapsl-runtime/README.md`.  
For implementation-level details, inspect `kapsl-runtime/crates/kapsl-cli/src/main.rs`.
