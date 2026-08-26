# Configuration

The runtime is configured via CLI flags and environment variables. Environment variables take precedence over defaults; CLI flags take precedence over environment variables where both apply.

## CLI flags

```
kapsl run [OPTIONS] [MODEL_OR_BUNDLE]...

Options:
  --model <PATH>              Backward-compatible .aimod path (repeatable)
  --offline                   Disable backend network access
  --transport <TRANSPORT>     socket | tcp | shm | hybrid (default: socket)
  --socket-path <PATH>        Unix socket path (default: /tmp/kapsl.sock)
  --kv-control-socket <PATH>  Local socket for versioned external KV participants
  --kv-control-lease-ttl-ms <MILLISECONDS>
                              Maximum heartbeat lease TTL (default: 30000)
  --kv-shared-pool-profile <PROFILE>
                              Exact conformance-tested adapter profile (repeatable)
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
| `KAPSL_BACKEND_CACHE_DIR` | Override the runtime-versioned lazy backend cache root |
| `KAPSL_BUNDLE_CACHE_DIR` | Override the verified offline-bundle extraction cache |
| `KAPSL_OFFLINE` | Set to `1` to forbid backend-index and artifact network access |
| `KAPSL_LAZY_BACKENDS` | Set to `0` to require lazy backends to be preinstalled |
| `KAPSL_LAZY_ONNX_PACKS` | Linux x86_64 beta switch for signed `onnx/cpu`, `onnx/cuda12`, and `onnx/tensorrt10` packs; set to `0` for the eager compatibility layout |
| `KAPSL_LAZY_LLAMA_CPP_PACKS` | Linux x86_64 beta switch for signed `llama-cpp/cpu` and `llama-cpp/cuda12` native packs; the CPU profile defaults on only when no eager GGUF backend is compiled |
| `KAPSL_LLAMA_CPP_ALLOW_NATIVE_KV` | Set to `1` to permit only a signed CUDA pack whose `kv_mode` is `native`; the shared-pool pack does not use this rollback override |
| `KAPSL_BACKEND_INDEX_URL` | Override the signed backend index URL |
| `KAPSL_BACKEND_INDEX_PATH` | Use a local signed index and adjacent `.sig` file |
| `KAPSL_BACKEND_PUBLIC_KEYS` | Additional trusted raw Ed25519 public keys for development or controlled rotation |

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
| `KAPSL_PROVIDER_MEMORY_LIMITS` | Hard limits for non-CPU/non-CUDA provider domains as `provider[:device]=size` entries, e.g. `metal=8g,directml:0=6g`. Exact-device entries override provider-wide values. |
| `KAPSL_GGUF_DISABLE_SHARED_KV` | Set to `1` to force llama.cpp native KV for GGUF diagnosis or rollback. |
| `KAPSL_VLLM_PYTHON` | Development override pointing at the exact certified managed-vLLM Python executable. Packaged installs discover the verified lazy cache and legacy beside-binary bundle automatically. |
| `KAPSL_VLLM_BUNDLE` | Development/package-layout override pointing at a bundle root containing `bin/python`. |
| `KAPSL_VLLM_BRIDGE_MODE` | Managed OpenAI bridge rollout mode: `wire` enables the typed byte relay; `async-translated` (the default) and `legacy` retain response translation for rollback. |
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

Every live backend is resampled on the two-second runtime monitor cadence.
Changes to backend-owned host, CUDA, Metal/CoreML, DirectML, ROCm, or custom
provider allocations atomically resize the owning memory lease. If physical
usage has already crossed a hard limit, Kapsl retains the over-limit observed
value in the authority snapshot, closes further admission, and enters the
normal pressure policy rather than reverting to a stale reservation.

### Serving backend policy

The package field `metadata.serving.backend` selects a deployment target, not
a CUDA execution provider. Use `kapsl backend-plan <MODEL>` to resolve it for
the current host. GGUF selects llama.cpp; an explicitly policy-tagged
SafeTensors causal-LM generation package may select managed vLLM on CUDA; all
other `auto` cases retain the built-in backend factory. A built-in SafeTensors
decision requires a binary compiled with the `native` feature; otherwise the
load is rejected before ONNX Runtime is constructed. Packages without the
field retain their legacy selection, subject to the same backend-availability
check.

For a vLLM package, the normal command is:

```bash
kapsl run ./qwen.aimod
```

Kapsl creates a private KV control socket, installs the certified shared-pool
profile, starts vLLM on an ephemeral loopback port, waits for readiness, routes
the existing Kapsl and OpenAI-compatible APIs to it, supervises bounded
restarts, and stops the complete vLLM process group when the model is unloaded
or Kapsl exits. The vLLM endpoint is an implementation detail and is not
exposed as a second public server.

This path requires a Linux `gpu-device-pool` build. On the first eligible run,
Kapsl performs preliminary GPU admission and then installs the certified
managed-vLLM pack from the signed backend index if it is not cached. The CUDA
installer no longer downloads this multi-gigabyte pack by default; pass
`--prefetch-backends vllm` to retain eager installation. Kapsl validates the
complete binary tuple before starting vLLM:
Python `3.12.3`, PyTorch `2.13.0+cu130`, torchvision `0.28.0+cu130`, torchaudio
`2.11.0+cu130`, CUDA runtime `13.0`, vLLM
`0.26.1rc1.dev1130+g2ec6f0d71`, and `kapsl-vllm-connector` `0.6.0` with profile
`vllm-v1-packed-cuda-ipc/flash-attn`. A missing or different bundle fails
closed; Kapsl never falls back to ONNX Runtime, native SafeTensors, or another
attention implementation. Source/development builds can point to the same
certified environment with `KAPSL_VLLM_PYTHON=/path/to/venv/bin/python`.
See [Lazy Backend Packs](./backend-packs.md) for cache, offline, and trust
configuration.

Optional package-level settings live under `metadata.serving.vllm`:

```yaml
metadata:
  serving:
    backend: vllm
    vllm:
      max_model_len: 4096
      kv_cache:
        mode: legacy_fraction
        gpu_memory_utilization: 0.5
      startup_timeout_seconds: 300
```

The compatibility defaults are `1024`, a `0.5` legacy fraction, and `300`
seconds respectively. The deprecated top-level `gpu_memory_utilization` field
is still accepted for existing packages but conflicts with `kv_cache` when
both are present.

The manifest parser also validates the future `kv_cache.mode: auto` and
`kv_cache.mode: fixed` shapes. This release deliberately rejects either exact
mode before downloading a backend or starting a child: exact launch remains
fail-closed until the certified runtime geometry provider and provisional
`MemoryAuthority` grant handoff are installed. It never falls back to `0.5`
after an operator requested an exact policy. Managed vLLM currently uses one
selected CUDA device per Kapsl replica. Use `CUDA_VISIBLE_DEVICES` to select or
isolate GPUs.

`KAPSL_VLLM_BRIDGE_MODE=wire` enables the protocol-native OpenAI path after
authentication, model resolution, pressure admission, priority selection, and
session scoping. The route normalizes the model alias once, forwards the JSON
body to the private managed process, and relays vLLM's status, allowlisted
headers, JSON body, or SSE bytes without reconstructing completion events.
Client authorization is never forwarded. Leave the variable unset to retain
the translated bridge while canary and semantic tests are running.

### External KV participants

`--kv-control-socket <PATH>` enables the versioned local control plane used by
independently managed out-of-process KV participants. Managed vLLM configures
this listener automatically; users do not need this flag for
`kapsl run --model`. When supplied explicitly, the path must be absolute, its
parent directory must already exist, and it must differ from the inference
`--socket`. On Unix, the runtime creates it with mode `0600` and refuses to
replace a non-socket path or an active listener.

```bash
install -d -m 0700 /run/kapsl
kapsl run \
  --kv-control-socket /run/kapsl/kv-control.sock \
  --kv-control-lease-ttl-ms 30000 \
  --kv-shared-pool-profile \
    'kapsl-vllm-connector,0.6.0,<vllm-version>,vllm-v1-packed-cuda-ipc/flash-attn'
```

The profile flag is required only for `shared_pool` and is repeatable. Its four
comma-separated fields are the exact adapter ID, adapter version, backend
version, and compatibility profile emitted by a conformance-tested adapter.
Do not add a tuple merely to make registration pass: it is the deployment
allowlist for builds whose backend-native attention write/read probes passed.
The participant declares this tuple in registration, so a mismatch is rejected
before the provisioner allocates or exports a device region.
An empty allowlist still permits opaque `kv_connected` participants but rejects
every external `shared_pool` registration.

The vLLM tuple must come from the opt-in **vLLM Shared-Pool Conformance** GPU
workflow. That job uses an exact vLLM wheel and SDK ref, invokes the production
CUDA IPC allocator seam on every requested rank, and tests native
FlashAttention writes, causal reads, guards, reuse, exhaustion, and synchronized
detach. It uploads a JSON report on every run but creates the plain-text
allowlist value only after every gate passes. Its temporary runtime necessarily
uses the candidate tuple as a local provisional allowlist; this permits the
test allocation and is not itself evidence of conformance.

Opaque `kv_connected` registrations use backend-owned KV. Every advertised
cache pool must name a bounded physical host, CUDA, or provider domain.
Reservations enter the same `MemoryAuthority` as built-in engines; admission
is rejected before backend allocation when the domain budget is unavailable or
exhausted. CUDA domains require a build with the CUDA memory authority
(`gpu-device-pool`) enabled.

ABI 1.2 defines two provisioned, runtime-owned `shared_pool` allocation modes.
`runtime_leased` publishes epoch/generation-checked block handles, zeros blocks
before assignment, requires synchronized release, and quarantines an unfenced
expiry. `participant_managed` exports the whole isolated backing while leaving
block-index selection to the backend; Kapsl still grants aggregate request
capacity, but those leases contain no physical block handles.

ABI 1.3 adds a fail-closed activation lifecycle. Registration only provisions
the isolated bindings. Every backend worker must then report its exact
epoch-bound binding, shard, adapter profile, imported byte size, and bounded
cache-layer views. The runtime accepts those attachments only when their exact
profile is allowlisted and all bindings have distinct expected ranks. An
explicit activation succeeds only after every receipt binding is attached;
request reservations are rejected before that point. Detach requires no live
leases plus backend-synchronized completion, and the backing is released only
after the coordinator lock is dropped.

On Linux builds with `gpu-device-pool`, enabling the control socket
automatically installs the CUDA IPC provisioner. It synchronously allocates and
zeros a dedicated exportable region for each participant/pool/device, charges
that physical region once through `MemoryAuthority`, and never exports the
runtime's general CUDA allocator slab. Other builds retain opaque support and
reject external `shared_pool` registration rather than accepting an
unprovisioned data plane.

Participants heartbeat active leases. A requested TTL may be shorter than the
runtime maximum but never longer. When heartbeats stop, an opaque lease returns
its capacity to the authority. Blocks from an expired shared-pool lease are
quarantined because timeout alone does not prove that another process has
stopped using the mapping. The listener is supervised: if it exits unexpectedly
while configured, the runtime fails rather than silently continuing in
unmanaged mode.

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
