# Kapsl CLI User Guide

The `kapsl` CLI helps you:

- run model packages
- add models to a running runtime without restarting
- list models loaded in a running runtime
- remove models from a running runtime
- build `.aimod` packages
- push packages to a remote backend
- pull packages from a remote backend

## Prerequisites

- Rust 1.75+
- A built `kapsl` binary (from this repository)

Build from source:

```bash
cd kapsl-runtime
cargo build --release -p kapsl
```

Run with `cargo run` during development:

```bash
cargo run -p kapsl -- --help
```

Or install the pre-built binary:

```bash
curl -fsSL https://downloads.kapsl.net/install.sh | sh
```

## Command Overview

```bash
kapsl [OPTIONS] [COMMAND]
```

Commands:

- `run`: run the runtime server
- `add-model`: add model(s) to an already-running runtime
- `list`: list models loaded in an already-running runtime
- `remove-model`: unload and unregister a model from a running runtime
- `build`: build a `.aimod` package
- `bundle`: create a verified model-plus-backend artifact for offline use
- `backend-plan`: inspect backend, download, and admission decisions
- `backend`: ensure, list, or prune the signed backend cache
- `push`: upload a `.aimod` package
- `pull`: download a `.aimod` package

Compatibility mode:

```bash
kapsl --model models/mnist/mnist.aimod
```

This is equivalent to:

```bash
kapsl run models/mnist/mnist.aimod
```

## 1) Install

### Pre-built binary (recommended)

```bash
curl -fsSL https://downloads.kapsl.net/install.sh | sh
```

Installs to `~/.local/bin/kapsl`. If that directory is not on your `PATH`, the script will print the export line to add to your shell profile.

Install a specific version:

```bash
KAPSL_VERSION=0.1.13 curl -fsSL https://downloads.kapsl.net/install.sh | sh
```

Install to a custom directory:

```bash
KAPSL_INSTALL_DIR=/usr/local/bin curl -fsSL https://downloads.kapsl.net/install.sh | sh
```

Test the script locally (without hitting the real server):

```bash
# Serve a local staging directory
cd /tmp/kapsl-test-serve && python3 -m http.server 8787

# In another terminal — override the base URL
KAPSL_BASE_URL=http://127.0.0.1:8787 KAPSL_INSTALL_DIR=/tmp/kapsl-out sh install.sh
```

### Build from source

```bash
cd kapsl-runtime
cargo build --release -p kapsl
```

## 2) Run Models (`kapsl run`)

Run one or more `.aimod` packages:

```bash
kapsl run --model models/mnist/mnist.aimod
```

Run multiple packages:

```bash
kapsl run models/mnist/mnist.aimod models/squeezenet/squeezenet.aimod
```

The repeatable `--model` option remains supported for backward compatibility.
For an offline host, run a bundle prepared on a connected machine:

```bash
kapsl bundle model.aimod --output model.kapsl-bundle
kapsl run model.kapsl-bundle
```

On Linux x86_64, ONNX models resolve one signed `cpu`, `cuda12`, or
`tensorrt10` native pack. Resolution happens before download, declared
fallbacks are honored exactly, and TensorRT requires an explicit package
declaration. Inspect or prefetch the decision with:

```bash
kapsl backend-plan model.aimod
kapsl backend ensure model.aimod
```

GGUF models use the same flow for signed `llama-cpp/cpu` or
`llama-cpp/cuda12` native packs. The shared library stays in the Kapsl process
behind a versioned C ABI. The portable core enables lazy CPU packs; the stable
CUDA build continues to use its certified eager shared-KV implementation while
the lazy CUDA shared-pool adapter is certified.

Useful run options:

- `--transport <socket|tcp|shm|hybrid|auto>` (default: `socket`)
- `--socket /tmp/kapsl.sock`
- `--kv-control-socket /run/kapsl/kv-control.sock` (enables the local external-KV participant control plane)
- `--kv-control-lease-ttl-ms 30000` (maximum heartbeat TTL for external KV leases)
- `--bind 127.0.0.1`
- `--port 9096`
- `--http-bind 127.0.0.1`
- `--metrics-port 9095`
- `--state-dir <dir>` (namespaces rag-data, extensions, extensions-config, auth-store.json)
- `--performance-profile <standard|balanced|throughput|latency>`

`hybrid` means Unix socket plus shared-memory tensor transfer; it does not open
a TCP listener. SHM and `auto` use the live model registry, so models added at
runtime are immediately addressable through SHM.

External backends such as vLLM use a separate versioned KV control socket. It
is disabled by default and never shares the inference socket. When enabled,
opaque backend reservations join Kapsl's process-wide memory authority before
the backend allocates KV. Linux CUDA builds can also provision an isolated
CUDA IPC backing for a backend that explicitly negotiates `shared_pool`; the
backend's attention tensors must directly alias that allocation. See the
[runtime configuration](../kapsl-runtime/docs/configuration.md#external-kv-participants)
for placement, TTL, CUDA-build, and socket-permission requirements.

Example with TCP transport:

```bash
export KAPSL_TCP_AUTH_TOKEN="replace-with-a-dedicated-token"
kapsl run \
  --model models/mnist/mnist.aimod \
  --transport tcp \
  --bind 0.0.0.0 \
  --port 9096
```

Loopback TCP does not require a token. Non-loopback TCP does, and the protocol
is still plaintext; use it only on a trusted network or through a TLS tunnel.

The stable CUDA application profile uses `gguf-cuda-shared-kv`. The explicit
`gguf-cuda` feature remains available as the llama.cpp native-KV rollback profile.
In that stable profile, the process-owned physical CUDA pool now defaults to
model-aware automatic sizing. Kapsl plans all startup packages before opening
backend sessions, subtracts known GGUF/native weights, retains unpooled room
for scratch, native-KV fallback, and later model additions, then creates one
immutable aligned backing allocation per CUDA device that needs it. The sizing
ceiling is also clamped to the configured device limit and live free VRAM.
Pooled ONNX weight copies, including configured session concurrency, form a
hard minimum checked before ORT session construction.

Pool controls accept an optional `_<device_id>` suffix, which takes precedence:

- `KAPSL_GPU_DEVICE_POOL_MODE=auto|fixed|off` selects the policy.
- `KAPSL_GPU_DEVICE_POOL_BYTES=8g` is an exact fixed-size override and implies
  `fixed` when mode is omitted. It is never silently reduced.
- `KAPSL_GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES=2g` replaces the automatic
  scratch/fallback/add-model reserve; zero is valid and an explicit reserve is
  never silently reduced.
- `CUDA_DEVICE_MEMORY_LIMIT[_N]` or `KAPSL_GPU_MEMORY_LIMIT_MB` bounds the
  process-visible VRAM used by automatic sizing and admission.
- `KAPSL_PROVIDER_MEMORY_LIMITS=metal=8g,directml:0=6g` sets hard limits for
  non-CUDA provider/device domains. Exact-device entries override the
  provider-wide fallback.

An unset runtime with no startup models does not reserve every detected GPU;
implicit automatic sizing is deferred until the first pooled model targets a
device. Mode and byte settings are validated together: for example, a global
fixed byte override conflicts with a per-device `auto` or `off` mode rather
than being silently ignored.

The default automatic reserve is 20% of the safe device budget with a 1 GiB
floor and one-third cap. A separate driver safety band of at least 512 MiB (or
10% of declared VRAM) remains outside that budget. Automatic pools have a
256 MiB minimum and 2 MiB alignment. If implicit automatic sizing cannot make
a viable allocation, Kapsl logs the decision and continues without a
runtime-owned pool; explicit `auto` and `fixed` configurations fail fast.

Physical CUDA pooling is process-owned. Isolated model workers disable their
local pool by default, even when pool settings are inherited from the parent.
Enable it only when every worker has an exclusive GPU/MIG slice or an explicit
`CUDA_DEVICE_MEMORY_LIMIT[_N]`/`KAPSL_GPU_MEMORY_LIMIT_MB` quota;
`KAPSL_ISOLATED_WORKER_GPU_POOL=true` is an operator attestation of an
exclusive boundary. Pool mode, size, or reserve alone is not an isolation
boundary. A planned isolated model suppresses implicit parent-pool creation on
its target device; only an explicit operator pool policy overrides that choice.

Prometheus scrapes expose current pool allocation, free-range, and
fragmentation state under `kapsl_gpu_device_pool_*`, plus per-owner usage,
quota, admission, and allocatable-byte gauges. The existing
`kapsl_device_memory_pooled_bytes` gauge is the immutable backing capacity;
it is not live usage.

The runtime also resamples each live backend's cross-domain memory report every
two seconds and reconciles growth, shrink, compaction, and migration into its
authority lease. Rejected over-limit growth remains visible as observed usage,
so admission and pressure decisions cannot fall back to stale planned bytes.

Example tuned for low latency:

```bash
kapsl run \
  --model models/mnist/mnist.aimod \
  --performance-profile latency
```

## 3) Add Models to a Running Runtime (`kapsl add-model`)

Add one or more models to a runtime that is already running, without stopping or restarting it.

```bash
kapsl add-model --model ./model.aimod
```

Add multiple models at once:

```bash
kapsl add-model \
  --model ./model1.aimod \
  --model ./model2.aimod
```

Target a non-default HTTP port:

```bash
kapsl add-model --model ./model.aimod --http-port 9100
```

Authenticated runtime:

```bash
kapsl add-model --model ./model.aimod --auth-token "$KAPSL_API_TOKEN_ADMIN"
```

Full URL override (e.g. remote host):

```bash
kapsl add-model --model ./model.aimod --http-url http://192.168.1.10:9095
```

Options:

- `--model <PATH>` — path to `.aimod` package (repeat for each model, required)
- `--http-port <PORT>` — HTTP API port of the running runtime (default: `9095`)
- `--http-host <HOST>` — HTTP bind address of the running runtime (default: `127.0.0.1`)
- `--http-url <URL>` — full base URL, overrides `--http-host` and `--http-port`
- `--auth-token <TOKEN>` — bearer token for authenticated runtimes
- `--topology <TOPOLOGY>` — mesh topology for added models (default: `data-parallel`)
- `--tp-degree <N>` — tensor parallelism degree (default: `1`)
- `--timeout-ms <MS>` — per-request timeout when contacting the runtime API (default: `30000`)

The command sends `POST /api/models/start` for each model. The runtime loads it asynchronously and returns the assigned `model_id`. All transport, port, and scheduler configuration of the running instance is preserved.

## 4) List Models in a Running Runtime (`kapsl list`)

List the models loaded in the local runtime:

```bash
kapsl list
```

The default table includes each model's ID, name, version, format, device,
status, and health. To target another runtime or one with API authentication:

```bash
kapsl list \
  --http-url http://192.168.1.10:9095 \
  --auth-token "$KAPSL_API_TOKEN_READER"
```

Print the complete `GET /api/models` response for scripts:

```bash
kapsl list --json
```

Options:

- `--http-port <PORT>` — HTTP API port of the running runtime (default: `9095`)
- `--http-host <HOST>` — runtime host (default: `127.0.0.1`)
- `--http-url <URL>` — full base URL, overrides `--http-host` and `--http-port`
- `--auth-token <TOKEN>` — bearer token for authenticated runtimes
- `--json` — print the complete API response as formatted JSON
- `--timeout-ms <MS>` — request timeout (default: `30000`)

## 5) Remove a Model from a Running Runtime (`kapsl remove-model`)

Use `kapsl list` to find the model ID, then remove it:

```bash
kapsl list
kapsl remove-model 2
```

Target a remote or authenticated runtime with the same connection options:

```bash
kapsl remove-model 2 \
  --http-url http://192.168.1.10:9095 \
  --auth-token "$KAPSL_API_TOKEN_ADMIN"
```

The command calls `POST /api/models/:id/remove`. The running engine stops the
model and its replicas, releases their runtime resources, and unregisters them.
The source `.aimod` package remains on disk and can be loaded again later.

Options:

- `<MODEL_ID>` — numeric ID shown by `kapsl list` (required)
- `--http-port <PORT>` — HTTP API port of the running runtime (default: `9095`)
- `--http-host <HOST>` — runtime host (default: `127.0.0.1`)
- `--http-url <URL>` — full base URL, overrides `--http-host` and `--http-port`
- `--auth-token <TOKEN>` — admin bearer token for authenticated runtimes
- `--timeout-ms <MS>` — request timeout (default: `30000`)

## 6) Build Packages (`kapsl build`)

You can build in two modes.

### A) Build from a model file

```bash
kapsl build ./model.onnx --output ./model.aimod
```

Optional overrides:

```bash
kapsl build \
  ./model.gguf \
  --output ./my-llm.aimod \
  --project-name my-llm \
  --framework llm \
  --version 1.2.0
```

Add metadata JSON:

```bash
kapsl build \
  ./model.onnx \
  --output ./model.aimod \
  --metadata-json '{"team":"inference","tier":"prod"}'
```

### B) Build from a context directory

```bash
kapsl build ./models/gpt-llm
```

Context mode is useful when your model directory includes extra files (tokenizer/config/etc.).

Or from inside the context directory:

```bash
cd ./models/gpt-llm
kapsl build
```

## 7) Push Packages (`kapsl push`)

Push target format:
- Required: `<repo_name>/<model>:<label>`
- Example: `alice/mnist:prod`

Upload a package:

```bash
kapsl push alice/model:prod ./model.aimod
```

Or from inside a directory with a single `.aimod` file:

```bash
cd ./models/mnist
kapsl push alice/mnist:prod
```

Override remote URL:

```bash
kapsl push \
  alice/model:prod \
  ./model.aimod \
  --remote-url https://my-registry.example.com/v1
```

If the remote backend requires auth, pass a token:

```bash
kapsl push \
  alice/model:prod \
  ./model.aimod \
  --remote-url https://my-registry.example.com/v1 \
  --remote-token "$JWT_TOKEN"
```

Or sign in once and reuse saved credentials:

```bash
kapsl login --remote-url https://my-registry.example.com/v1
kapsl push alice/model:prod ./model.aimod --remote-url https://my-registry.example.com/v1
```

After first successful login, `kapsl login` reuses the last remote URL automatically.

For headless/SSH sessions, use device-code login:

```bash
kapsl login --remote-url https://my-registry.example.com/v1 --device-code
```

In SSH sessions, plain `kapsl login` automatically prefers device-code flow (GitHub).

If no token is configured and the remote returns `401`, `kapsl push`/`kapsl pull` will automatically start browser login and retry once.

## 8) Pull Packages (`kapsl pull`)

Pull by target:

```bash
kapsl pull alice/mnist:prod --destination-dir ./models
```

Or pull into the current directory:

```bash
cd ./models
kapsl pull alice/mnist:prod
```

Pull from a custom remote URL:

```bash
kapsl pull \
  alice/mnist:prod \
  --destination-dir ./models \
  --remote-url https://my-registry.example.com/v1
```

Authenticated pull:

```bash
kapsl pull \
  alice/mnist:prod \
  --destination-dir ./models \
  --remote-url https://my-registry.example.com/v1 \
  --remote-token "$JWT_TOKEN"
```

If you already ran `kapsl login`, you can omit `--remote-token`.

## Common Workflows

### Workflow A: Build and run locally

```bash
kapsl build ./model.onnx --output ./model.aimod
kapsl run --model ./model.aimod
```

### Workflow B: Build, push, and pull on another machine

Machine A:

```bash
kapsl build ./model.onnx --output ./model.aimod
kapsl push alice/model:prod ./model.aimod --remote-url https://my-registry.example.com/v1
```

Machine B:

```bash
kapsl pull alice/model:prod --destination-dir ./models --remote-url https://my-registry.example.com/v1
kapsl run --model ./models/model.aimod
```

## Notes on Default Remote Behavior

If you do not pass `--remote-url`, `push`/`pull` use `https://api.kapsl.net/v1` by default.

- This is useful when you want to use the shared hosted backend without passing `--remote-url` explicitly.
- For production sharing across users/machines, use a dedicated remote URL via `--remote-url` (or set `KAPSL_REMOTE_URL`).
- For authenticated remotes, use `--remote-token` or set `KAPSL_REMOTE_TOKEN`.
- If you install the CLI with Cargo, prefer `cargo install --path crates/kapsl-cli --locked` so the binary matches the checked-in lockfile and dependency feature set.

## Authentication (API)

When API auth is enabled, requests to `/api/*` and `/metrics` require a bearer token.

Common environment variables:

- `KAPSL_API_TOKEN_READER`
- `KAPSL_API_TOKEN_WRITER`
- `KAPSL_API_TOKEN_ADMIN`

Example:

```bash
export KAPSL_API_TOKEN_ADMIN="your-token"

curl http://127.0.0.1:9095/api/models \
  -H "Authorization: Bearer $KAPSL_API_TOKEN_ADMIN"
```

If auth is disabled, API access is loopback-only by default.

## Extensions

Install an extension from Kapsl Hub into a running local engine:

```bash
kapsl extension install connector.echo
```

If runtime API authentication is enabled:

```bash
kapsl extension install connector.echo --auth-token "$KAPSL_API_TOKEN_ADMIN"
```

Use runtime API endpoints for the rest of the extension lifecycle tasks.

List installed extensions:

```bash
curl http://127.0.0.1:9095/api/extensions \
  -H "Authorization: Bearer $KAPSL_API_TOKEN_ADMIN"
```

Install from a local extension directory:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/install \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN_ADMIN" \
  -d '{"path":"./extensions/my-extension"}'
```

Set extension config for a workspace:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.echo/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN_ADMIN" \
  -d '{"workspace_id":"default","config":{"api_key":"...","project":"..."}}'
```

Launch and sync:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.echo/launch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN_ADMIN" \
  -d '{"workspace_id":"default"}'

curl -X POST http://127.0.0.1:9095/api/extensions/connector.echo/sync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN_ADMIN" \
  -d '{"workspace_id":"default"}'
```

## Troubleshooting

### `Model not found` when running infer requests

Check loaded models:

```bash
curl http://127.0.0.1:9095/api/models
```

Use the correct model ID in API requests.

### `push` says package does not exist

Make sure you pass a valid `.aimod` file path and the file ends with `.aimod`.

### API not reachable

Check the HTTP bind/port values passed to `run`:

```bash
kapsl run --model ./model.aimod --http-bind 127.0.0.1 --metrics-port 9095
```

## Quick Reference

```bash
# install
curl -fsSL https://downloads.kapsl.net/install.sh | sh

# run
kapsl run --model <path-to-kapsl>

# add model to running runtime
kapsl add-model --model <path-to-kapsl> [--http-port <port>] [--auth-token <token>]

# list models in running runtime
kapsl list [--http-url <url>] [--auth-token <token>] [--json]

# unload and unregister model from running runtime
kapsl remove-model <model-id> [--http-url <url>] [--auth-token <admin-token>]

# build
kapsl build <path-to-model-file> --output <output.aimod>
kapsl build [<context-dir>]

# push
kapsl push <repo>/<model>:<label> [<path-to-kapsl>] [--remote-url <url>]

# pull
kapsl pull <repo>/<model>:<label> [--destination-dir <dir>] [--remote-url <url>]
```
