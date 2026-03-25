# kapsl-runtime Full Documentation

Last updated: 2026-03-25
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
- `kapsl login ...`
- `kapsl control ...`
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
- `--performance-profile <auto|standard|balanced|throughput|latency>` (default `auto`).
- `--topology` and `--tp-degree` for parallel topology hints.
- `--state-dir <PATH>`: root directory for runtime state (rag-data, extensions, extensions-config, auth-store.json). Overrides `KAPSL_RAG_STORAGE_ROOT`, `KAPSL_EXTENSIONS_ROOT`, `KAPSL_EXT_CONFIG_ROOT`, and `KAPSL_AUTH_STORE_PATH` when set.
- `--shm-size-mb <MIB>` (default `256`): size of the shared memory segment when using `shm` or `hybrid` transport. Also read from `KAPSL_SHM_SIZE_MB`.

ONNX runtime tuning flags:

- `--onnx-memory-pattern <BOOL>`: enable/disable ONNX memory pattern optimization.
- `--onnx-disable-cpu-mem-arena <BOOL>`: disable CPU memory arena allocator.
- `--onnx-session-buckets <N>`: number of dynamic-shape session buckets.
- `--onnx-bucket-dim-granularity <N>`: granularity for bucket dimension rounding.
- `--onnx-bucket-max-dims <N>`: maximum dynamic dimensions tracked per bucket.
- `--onnx-peak-concurrency-hint <N>`: hint for peak concurrent ONNX session usage.
- `--onnx-model-tuning <SPEC>` (repeatable): per-model ONNX tuning overrides. Format: `<model_id|*>:key=value[,key=value...]`.

### 4.2 Queue Overflow Policy

The scheduler uses an internal `WorkQueue` per GPU worker with a fixed capacity set by `--scheduler-queue-size`. When the queue is full, the runtime applies one of three overflow policies:

| Policy | Behaviour |
| --- | --- |
| `block` (default) | Caller waits until a slot is free. |
| `drop_newest` | The incoming request is immediately rejected with an overload error. |
| `drop_oldest` | The oldest queued request is evicted (and receives an overload error); the new request is enqueued. |

The policy is not yet a CLI flag; it is set programmatically via `Scheduler::with_queue_overflow_policy`. High-priority (latency-critical) requests always use the high-priority queue; throughput-class requests use the low-priority queue and may be micro-batched.

Performance profile tuning (applies only when related flags are not explicitly passed):

- `auto` (default): automatically selects parameters based on model size and system resources.
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

OCI pull with digest ref:

```bash
kapsl pull acme/model:prod --ref sha256:abc123... --destination-dir ./models
```

Target format:

- Required for push and pull: `<repo_name>/<model>:<label>`
- Examples: `alice/kokoro:latest`, `team-a/recommender:v1`

Default remote behavior:

- Uses `https://api.kapsl.net/v1` by default.
- If remote URL resolves to the legacy placeholder URL, mirrors artifacts to a local directory (configurable env var).
- If `remote_url` is overridden, uses HTTP PUT/GET remote backend.
- If `remote_url` starts with `oci://`, uses ORAS to push/pull the `.aimod` as an OCI artifact (MIME type `application/vnd.kapsl.aimod.v1`). Requires the `oras` binary (path configurable via `KAPSL_ORAS_BIN`). Credentials use `KAPSL_OCI_USERNAME` and `KAPSL_OCI_PASSWORD`.

### 4.4 Login Command

`kapsl login` authenticates with the Kapsl registry via OAuth.

Flags:

- `--provider <github|google>` (default `github`).
- `--callback-host <HOST>` (default `127.0.0.1`).
- `--callback-port <PORT>` (default `0` = ephemeral).
- `--timeout-seconds <SECONDS>` (default `180`).
- `--no-browser`: skip automatic browser launch.
- `--device-code`: use device code flow (headless/SSH environments; GitHub only).

Credentials are stored in `~/.kapsl/tokens.json` (path overridable via `KAPSL_REMOTE_TOKEN_STORE_PATH`).

### 4.5 Control Command

`kapsl control` runs a multi-runtime control loop that orchestrates cross-runtime weight distribution and scaling policy based on observed pressure and GPU utilization.

Required flags:

- `--runtime <NAME=URL>` (repeatable): registers a named runtime endpoint (e.g. `primary=http://127.0.0.1:9095`).

Optional flags:

- `--runtime-profile <NAME=PROFILE>`: per-runtime performance profile override (`latency|balanced|throughput`).
- `--runtime-token <NAME=TOKEN>`: per-runtime auth token.
- `--auth-token <TOKEN>`: default auth token used when no per-runtime token is set.
- `--memory-budget-bytes <NAME=BYTES>`: per-runtime memory budget constraint.
- `--interval-seconds <N>` (default `5`): control loop polling interval.
- `--timeout-ms <N>` (default `1500`): per-request timeout for health/stats checks.
- `--queue-target <N>` (default `10`): desired queue depth per runtime.
- `--high-pressure-score <F>` (default `0.85`): pressure score threshold for high-pressure state.
- `--low-pressure-score <F>` (default `0.45`): pressure score threshold for low-pressure state.
- `--hot-gpu-utilization <F>` (default `0.92`): GPU utilization ratio above which a runtime is considered hot.
- `--hot-memory-utilization <F>` (default `0.90`): memory utilization ratio above which a runtime is considered hot.
- `--overload-window-seconds <N>` (default `30`): sliding window for overload detection.
- `--hot-window-seconds <N>` (default `20`): sliding window for hot-GPU detection.
- `--unhealthy-hold-seconds <N>` (default `30`): time before an unhealthy runtime is reconsidered.
- `--weight-step <F>` (default `0.10`): increment/decrement per weight adjustment cycle.
- `--weight-floor <F>` (default `0.05`): minimum weight assigned to any active runtime.
- `--overload-shift-fraction <F>` (default `0.20`): fraction of traffic shifted away on overload.
- `--dry-run`: compute and log weight decisions without applying them.
- `--weights-file <PATH>` (default `runtime-control-weights.json`): file where current weight assignments are persisted.

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

Unauthenticated endpoints:

- `POST /api/auth/login`: probe auth status or validate a token. Request body: `{"token": "<optional>"}`. Response: `{"authenticated": bool, "auth_enabled": bool, "legacy_auth_enabled": bool, "role": "...", "scopes": [...], "mode": "...", "access": {"read": bool, "write": bool, "admin": bool}}`.

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
- `KAPSL_LLM_ALLOW_SCHEDULER_MICROBATCH`: allow the scheduler to micro-batch LLM requests.
- `KAPSL_LOG_SENSITIVE_IDS` (set truthy to disable request/session redaction)
- `KAPSL_RAG_STORAGE_ROOT`
- `KAPSL_EXTENSIONS_ROOT`
- `KAPSL_EXT_CONFIG_ROOT`
- `KAPSL_EXTENSION_MARKETPLACE_URL`
- `KAPSL_REMOTE_URL`
- `KAPSL_REMOTE_PLACEHOLDER_URL`
- `KAPSL_REMOTE_PLACEHOLDER_DIR`
- `KAPSL_REMOTE_TOKEN_STORE_PATH`: path to the OAuth token store file (default `~/.kapsl/tokens.json`).
- `KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS`
- `KAPSL_INFER_ADAPTERS` (optional adapters; e.g. `echo_tensor` when built with feature)
- `KAPSL_SHM_SIZE_MB`: size of the shared memory segment in MiB (default `256`).
- `KAPSL_SCHEDULER_QUEUE_OVERFLOW_POLICY` (legacy alias `KAPSL_LITE_INGRESS_BACKPRESSURE`): sets the queue overflow policy (`block|drop_newest|drop_oldest`).

OCI / ORAS env vars:

- `KAPSL_ORAS_BIN`: path to the `oras` binary used for OCI push/pull (defaults to `oras` on `PATH`).
- `KAPSL_OCI_USERNAME`: username for OCI registry authentication.
- `KAPSL_OCI_PASSWORD`: password for OCI registry authentication.

Inter-model relay env vars:

- `KAPSL_INTER_MODEL_ROUTES` (legacy alias `KAPSL_LITE_INTER_MODEL_ROUTES`): JSON-encoded map of source model ID to list of destination model IDs for automatic inter-model relay.
- `KAPSL_INTER_MODEL_RELAY_MIN_INTERVAL_MS` (legacy alias `KAPSL_LITE_INTER_MODEL_RELAY_MIN_INTERVAL_MS`): minimum interval in ms between relay publishes (default `2000`).

Server pressure env vars:

- `KAPSL_SERVER_PRESSURE_MEMORY_CONSERVE_PCT`: memory usage % above which runtime enters conserve pressure state.
- `KAPSL_SERVER_PRESSURE_MEMORY_EMERGENCY_PCT`: memory usage % above which runtime enters emergency pressure state.
- `KAPSL_SERVER_PRESSURE_GPU_UTIL_CONSERVE_PCT`: GPU utilization % threshold for conserve state.
- `KAPSL_SERVER_PRESSURE_GPU_UTIL_EMERGENCY_PCT`: GPU utilization % threshold for emergency state.
- `KAPSL_SERVER_PRESSURE_GPU_MEM_CONSERVE_PCT`: GPU memory usage % threshold for conserve state.
- `KAPSL_SERVER_PRESSURE_GPU_MEM_EMERGENCY_PCT`: GPU memory usage % threshold for emergency state.
- `KAPSL_SERVER_PRESSURE_CONSERVE_MAX_NEW_TOKENS`: max new tokens allowed per request in conserve state.
- `KAPSL_SERVER_PRESSURE_EMERGENCY_MAX_NEW_TOKENS`: max new tokens allowed per request in emergency state.

Model cache / disk-check env vars (read by `kapsl-core` loader):

- `KAPSL_MODEL_CACHE_DIR` (or `KAPSL_LITE_MODEL_CACHE_DIR`): override the model cache root directory (default: `.kapsl-model-cache/` next to the `.aimod` file).
- `KAPSL_MODEL_CACHE_MAX_BYTES` / `KAPSL_MODEL_CACHE_MAX_MIB`: cap the total size of the model cache; the loader will evict least-recently-used entries to stay within limit.
- `KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES` / `KAPSL_MODEL_CACHE_RESERVED_FREE_MIB`: minimum free disk space that must remain after a model cache copy; the loader will evict LRU entries until the constraint is satisfied (or fail with `InsufficientDiskSpace`).
- `KAPSL_PACKAGE_TMP_DIR` (or `KAPSL_LITE_PACKAGE_TMP_DIR`): override the temporary directory used when unpacking an `.aimod` archive.

All above support legacy `KAPSL_*` aliases in runtime code.

## 13. Inter-Model Relay

The runtime supports routing inference outputs from one model as inputs to one or more downstream models within the same process.

Configuration is via `KAPSL_INTER_MODEL_ROUTES` (or legacy `KAPSL_LITE_INTER_MODEL_ROUTES`): a JSON map from source model ID to a list of destination model IDs.

Example:

```json
{"encoder": ["decoder-a", "decoder-b"]}
```

Relay sessions are prefixed with `relay/` internally. The relay publishes at most once every `KAPSL_INTER_MODEL_RELAY_MIN_INTERVAL_MS` (default `2000 ms`).

## 13a. Runtime Pressure Management

The runtime monitors system resource utilization and adjusts inference behavior when under pressure. There are three pressure states:

| State | Meaning |
| --- | --- |
| `Normal` | No constraints applied. |
| `Conserve` | Token generation is capped to `KAPSL_SERVER_PRESSURE_CONSERVE_MAX_NEW_TOKENS`. |
| `Emergency` | Token generation is capped to `KAPSL_SERVER_PRESSURE_EMERGENCY_MAX_NEW_TOKENS`. |

Thresholds are configured via env vars (see Section 12). The runtime evaluates pressure state continuously using `evaluate_runtime_pressure_state()`.

## 13b. LLM Shared KV Cache

For multi-model deployments with LLM backends, the runtime coordinates a shared KV cache block pool across all LLM engine instances on the same device.

Key details:

- Each KV block is 2 MB.
- Block size is 16 tokens.
- The `SharedKvState` coordinates device-level block allocator pools and a `GlobalKvScheduler`.
- Each LLM engine attaches to the shared pool on startup and detaches on shutdown.
- Cross-model token-budget coordination prevents any single engine from monopolising device memory.

## 14. Example API Calls

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

## 15. Testing and Benchmarks

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

## 16. Troubleshooting

Common issues:

- `Model not found`: verify `model_id` and that model is active in `/api/models`.
- Non-loopback bind rejected: set `KAPSL_ALLOW_INSECURE_HTTP=1` only behind trusted TLS/network controls.
- Video infer fails: ensure `ffmpeg` is installed and available in `PATH`.
- Media infer rejected: verify tensor shape/dtype expectations and preprocessing options.
- Push/pull placeholder error: ensure pushed artifact exists in placeholder mirror dir.
- Unauthorized/forbidden API: verify role token/api key and required route scope.
- `InsufficientDiskSpace` on load: the loader could not free enough cache space. Either set `KAPSL_MODEL_CACHE_MAX_BYTES`/`KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES` to a larger threshold, point `KAPSL_MODEL_CACHE_DIR` at a volume with more space, or remove stale entries from the cache directory manually.
- OCI push/pull fails: ensure `oras` is on `PATH` (or set `KAPSL_ORAS_BIN`), and that `KAPSL_OCI_USERNAME`/`KAPSL_OCI_PASSWORD` are set correctly.
- `kapsl control` not shifting weights: check that each `--runtime` URL is reachable and that `--auth-token` / `--runtime-token` are set if auth is enabled. Use `--dry-run` to validate decisions without applying them.

## 17. Known Constraints

- HTTP infer is synchronous; token streaming is available on IPC protocol/Python stream client path.
- RAG embeddings are currently lightweight hash-based embeddings (not external embedding model-backed).
- Video media path depends on host `ffmpeg`.
- Some helper scripts in `scripts/` are exploratory and may not match the latest wire protocol exactly.

---

For starter usage, see `kapsl-runtime/README.md`.  
For implementation-level details, inspect `kapsl-runtime/crates/kapsl-cli/src/main.rs`.
