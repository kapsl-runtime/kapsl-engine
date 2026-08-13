# Configuration

The runtime is configured via CLI flags and environment variables. Environment variables take precedence over defaults; CLI flags take precedence over environment variables where both apply.

## CLI flags

```
kapsl [OPTIONS] --model <PATH>

Options:
  --model <PATH>              Path to an .aimod file (repeatable)
  --transport <TRANSPORT>     socket | tcp | shm | hybrid (default: socket)
  --socket-path <PATH>        Unix socket path (default: /tmp/kapsl.sock)
  --tcp-port <PORT>           IPC TCP port (default: 9096)
  --http-host <HOST>          HTTP bind host (default: 127.0.0.1)
  --http-port <PORT>          HTTP/API/UI port (default: 9095)
  --provider <BACKEND>        cpu | cuda | tensorrt | metal | rocm | directml
  --admin-token <TOKEN>       Set admin token at startup
  --log-level <LEVEL>         trace | debug | info | warn | error (default: info)
```

## Environment variables

### Authentication

| Variable | Description |
|----------|-------------|
| `KAPSL_API_TOKEN_READER` | Reader role token |
| `KAPSL_API_TOKEN_WRITER` | Writer role token |
| `KAPSL_API_TOKEN_ADMIN` | Admin role token |
| `KAPSL_AUTH_STORE_PATH` | Path to the auth store JSON (default: `~/.kapsl/auth-store.json`) |

### Network and security

| Variable | Description |
|----------|-------------|
| `KAPSL_ALLOW_INSECURE_HTTP` | Set to `1` to allow binding HTTP to non-loopback addresses |

### Model cache and storage

| Variable | Description |
|----------|-------------|
| `KAPSL_MODEL_CACHE_DIR` | Cache root directory (default: `.kapsl-model-cache/` next to `.aimod`) |
| `KAPSL_MODEL_CACHE_MAX_MIB` | Maximum cache size in MiB; LRU eviction enforced |
| `KAPSL_MODEL_CACHE_MAX_BYTES` | Maximum cache size in bytes |
| `KAPSL_MODEL_CACHE_RESERVED_FREE_MIB` | Minimum free disk to maintain after cache operations |
| `KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES` | Same in bytes |
| `KAPSL_PACKAGE_TMP_DIR` | Temp directory for unpacking `.aimod` archives |

### Extensions and RAG

| Variable | Description |
|----------|-------------|
| `KAPSL_EXTENSIONS_ROOT` | Directory where extensions are installed |
| `KAPSL_EXT_CONFIG_ROOT` | Directory for extension configuration |
| `KAPSL_RAG_STORAGE_ROOT` | Vector store data directory |
| `KAPSL_EXTENSION_MARKETPLACE_URL` | Override the marketplace API endpoint |

### Remote registry

| Variable | Description |
|----------|-------------|
| `KAPSL_REMOTE_URL` | Default remote registry URL |
| `KAPSL_REMOTE_TOKEN` | Bearer token for remote push/pull |
| `KAPSL_REMOTE_TOKEN_STORE_PATH` | Path to the OAuth token store |

### Backend and inference

| Variable | Description |
|----------|-------------|
| `KAPSL_PROVIDER_POLICY` | `fastest` (auto-select fastest backend) or `manifest` (use manifest-specified backend) |
| `KAPSL_LLM_ISOLATE_PROCESS` | Set to `1` to run LLM backends in a subprocess for isolation |
| `KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS` | Disable automatic image/video-to-tensor preprocessing |

### GPU memory and shared KV

The stable CUDA application profile enables `gguf-cuda-shared-kv` and defaults
the physical device pool to automatic sizing. Before any backend or ORT session
is constructed, Kapsl plans the startup models and chooses, per used CUDA
device:

```text
declared       = min(physical VRAM, configured process/device cap)
safe budget    = min(declared, live free VRAM) - max(10% of declared, 512 MiB)
pool capacity  = align_down(safe budget - known external weights - unpooled reserve, 2 MiB)
```

The result must also be at least the planned pooled ONNX weight footprint,
including configured session concurrency; otherwise preflight fails before an
ORT session is created. Remaining pooled capacity is available to ORT
workspaces and paged KV.

The default unpooled reserve is 20% of the safe budget with a 1 GiB floor and
one-third cap. It protects room for backend scratch, native-KV compatibility
fallback, and later model additions. The automatic pool minimum is 256 MiB.
The backing allocation is process-lifetime and cannot grow or shrink; later
model starts still pass admission only while the retained headroom is
available. When the process starts without models, implicit automatic pool
creation remains deferred until the first pooled model targets that device.

| Variable | Description |
|----------|-------------|
| `KAPSL_GPU_DEVICE_POOL_MODE[_N]` | `auto`, `fixed`, or `off`. Per-device values win. Omitted means automatic in the stable shared-KV CUDA profile, fixed when bytes are supplied, and off in other profiles. |
| `KAPSL_GPU_DEVICE_POOL_BYTES[_N]` | Exact positive fixed backing size; supports byte, `k`, `m`, and `g` suffixes. Never silently resized. |
| `KAPSL_GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES[_N]` | Automatic-mode reserve for scratch/fallback/later models. Uses the same suffixes, permits zero, and is never silently reduced. |
| `CUDA_DEVICE_MEMORY_LIMIT[_N]` | Strict process/device VRAM ceiling used by automatic sizing; malformed selected values fail startup. |
| `KAPSL_GPU_MEMORY_LIMIT_MB` | Process VRAM ceiling in MiB when no CUDA-specific ceiling is set. |
| `KAPSL_GGUF_DISABLE_SHARED_KV` | Set to `1` to force llama.cpp native KV for GGUF diagnosis or rollback. |
| `KAPSL_GPU_DEVICE_POOL_DISABLED` | Internal process override. It has highest precedence and keeps admission accounting active without a physical pool. |
| `KAPSL_ISOLATED_WORKER_GPU_POOL` | `true` attests that each isolated worker owns an exclusive GPU/MIG boundary. |

Implicit automatic allocation failure logs a sizing breakdown and continues
without a runtime-owned pool. Explicit `auto` and every `fixed` configuration
fail fast. Isolated workers disable inherited pooling unless an exclusive
GPU/MIG attestation or explicit process memory boundary is present. Pool mode,
bytes, and reserve settings alone never qualify as that boundary. An isolated
model targeting a device also suppresses a new implicit parent pool there, so
the parent cannot reserve most VRAM before the child loads; explicit operator
pool modes retain their stated behavior.
Mode and byte settings are one configuration contract: a global byte override
still conflicts with a per-device `auto` or `off` mode instead of being
silently ignored.

`/metrics` samples the live allocator at scrape time. Pool-wide series are
`kapsl_gpu_device_pool_allocated_bytes`, `_live_allocations`, `_free_bytes`,
`_free_ranges`, `_largest_free_range_bytes`, and `_fragmentation_ratio`, each
labelled by `device`. Per-owner usage, guaranteed/max quota, admission state,
and immediately allocatable bytes use the same prefix plus an `owner` label
(`onnx`, `gguf_kv:<model-id>`, or `native_kv:<model-id>`). Unloaded owner rows
are removed on the next scrape. `kapsl_device_memory_pooled_bytes` remains the
fixed backing capacity, not current usage.

### Observability

| Variable | Description |
|----------|-------------|
| `KAPSL_LOG_SENSITIVE_IDS` | Set to `1` to include request/session IDs in logs (off by default for privacy) |

## Example: production deployment

```bash
export KAPSL_API_TOKEN_ADMIN="$(openssl rand -hex 32)"
export KAPSL_API_TOKEN_WRITER="$(openssl rand -hex 32)"
export KAPSL_API_TOKEN_READER="$(openssl rand -hex 32)"
export KAPSL_ALLOW_INSECURE_HTTP=1          # TLS handled upstream by nginx
export KAPSL_MODEL_CACHE_MAX_MIB=10240      # 10 GiB cache
export KAPSL_MODEL_CACHE_RESERVED_FREE_MIB=2048  # Keep 2 GiB free
export KAPSL_EXTENSIONS_ROOT=/var/lib/kapsl/extensions
export KAPSL_RAG_STORAGE_ROOT=/var/lib/kapsl/rag

kapsl \
  --model /opt/models/primary.aimod \
  --http-host 0.0.0.0 \
  --http-port 9095
```

## Example: local development

```bash
# No auth, default socket, loopback HTTP
kapsl --model ./models/test.aimod
```

The dashboard is accessible at `http://127.0.0.1:9095` without a token.
