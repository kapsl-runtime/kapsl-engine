# Deployment

## Prerequisites

- Rust 1.75 or later
- Python 3.8+ (optional — only needed for helper scripts and building `kapsl-sdk`)
- `ffmpeg` (optional — required for video/audio inference payloads)
- GPU drivers and SDKs only if using a non-CPU backend:
  - A compatible NVIDIA display driver on Windows; the provider packs include
    the required CUDA 12, cuDNN 9, and TensorRT 10 user-space runtime libraries
  - A compatible NVIDIA driver on Linux. Signed lazy ONNX packs include their
    CUDA 12/cuDNN 9 or TensorRT 10 user-space dependency closure, while the
    temporary eager CUDA installer retains its merged GGUF compatibility
    libraries. The driver must support CUDA 13.0 when serving through vLLM.
  - Xcode command line tools for Metal (macOS)

## Runtime and accelerator packages

The default installer contains the portable Kapsl runtime and ONNX Runtime core
libraries. The Linux CUDA installer is a separate self-contained archive so CPU,
DirectML, and Apple Silicon installations stay small.

Install the default runtime:

```bash
curl -fsSL https://downloads.kapsl.net/install.sh | sh
```

Install the Linux x86_64 CUDA 12 runtime:

```bash
curl -fsSL https://downloads.kapsl.net/install-cuda.sh | sh
```

The command currently downloads the CUDA-compiled GGUF compatibility runtime.
It does not download managed vLLM, ONNX accelerator packs, or portable
llama.cpp CPU packs up front. The first eligible model run performs preliminary
memory admission and installs the exact signed `vllm/cu130-flash-attn`,
`onnx/cuda12`, `onnx/tensorrt10`, or `llama-cpp/cpu` pack in the
runtime-versioned cache. Subsequent runs reuse it. Use
`--prefetch-backends vllm` for the temporary eager vLLM compatibility flow.

After packaging an explicit SafeTensors causal-LM deployment for vLLM, the
serving command remains the same as every other Kapsl model:

```bash
kapsl run ./model.aimod
```

For a no-network deployment, create and transfer one file:

```bash
kapsl bundle ./model.aimod --output ./model.kapsl-bundle
kapsl run ./model.kapsl-bundle
```

See [Lazy Backend Packs](./backend-packs.md) for cross-target and multi-model
bundles.

The stable CUDA runtime uses Kapsl's paged shared-KV path for supported GGUF
architectures. Models rejected by its compatibility policy use llama.cpp's
native KV path instead. Set `KAPSL_GGUF_DISABLE_SHARED_KV=1` to force that path
for diagnosis or rollback. Source builds can instead select the explicit
`gguf-cuda` feature to exclude shared-KV entirely.

Native llama.cpp packs expose a stable C ABI and remain in-process, so they do
not add an inference child process, tensor IPC, or a second CUDA context. The
shared-pool `llama-cpp/cuda12` candidate obtains its KV storage and device block
table from Kapsl core callbacks and is certified against the eager CUDA
reference. A separately signed native-KV pack remains a guarded rollback; set
`KAPSL_LLAMA_CPP_ALLOW_NATIVE_KV=1` only when intentionally selecting that
signed native mode.

The same stable profile automatically sizes one process-owned CUDA backing
pool on each device used by a pooled model. Startup packages are planned first,
so external GGUF/native weights and conservative unpooled scratch/fallback
headroom are removed before the immutable pool size is selected. Use
`KAPSL_GPU_DEVICE_POOL_MODE=off` to opt out, `auto` to make failure strict, or
`fixed` together with `KAPSL_GPU_DEVICE_POOL_BYTES[_N]` for an exact operator
allocation. See the GPU memory section in `docs/configuration.md` for sizing
and isolated-worker behavior.

Install CUDA 12 and TensorRT 10 packs:

```bash
curl -fsSL https://downloads.kapsl.net/install.sh |
  sh -s -- --accelerator tensorrt
```

On Windows PowerShell, use the installer matching the required runtime:

```powershell
# Core runtime
irm https://downloads.kapsl.net/install.ps1 | iex

# Core runtime with CUDA 12
irm https://downloads.kapsl.net/install-cuda.ps1 | iex

# Core runtime with CUDA 12 and TensorRT 10
irm https://downloads.kapsl.net/install-tensorrt.ps1 | iex
```

Add acceleration to an existing Windows installation:

```powershell
kapsl provider install cuda12
kapsl provider install tensorrt10
```

The TensorRT command installs CUDA 12 first when needed. For a system-wide MSI
installation under `C:\Program Files`, run the command from an Administrator
PowerShell. A saved copy of the general installer also accepts explicit parameters:

```powershell
.\install.ps1 -Accelerator cuda
.\install.ps1 -Accelerator tensorrt
```

The latest beta has equivalent entry points:

```bash
curl -fsSL https://downloads.kapsl.net/install-beta.sh | sh
# Explicit CUDA override:
curl -fsSL https://downloads.kapsl.net/install-beta-cuda.sh | sh
```

The generic beta installer selects the CUDA build on Linux x86_64 when
`nvidia-smi` confirms a working NVIDIA driver; otherwise it selects the
portable build. `KAPSL_ACCELERATOR` or `--accelerator` overrides detection.

```powershell
irm https://downloads.kapsl.net/install-beta.ps1 | iex
irm https://downloads.kapsl.net/install-beta-cuda.ps1 | iex
irm https://downloads.kapsl.net/install-beta-tensorrt.ps1 | iex
```

Windows provider packs and the signed Linux ONNX backend packs contain their
calculated user-space dependency closures. Legacy standalone Linux provider
archives and the merged CUDA installer remain available during rollout. macOS
uses system Metal/CoreML frameworks and does not require an accelerator pack.

## Docker images

Docker images follow the same modular split:

```bash
# Small, multi-architecture CPU image; also published as :latest
docker pull ghcr.io/kapsl-runtime/kapsl-engine:latest-cpu

# Linux amd64 with the merged GGUF + ONNX CUDA 12 runtime
docker pull ghcr.io/kapsl-runtime/kapsl-engine:latest-cuda

# Linux amd64 with the merged CUDA runtime and TensorRT 10 add-on
docker pull ghcr.io/kapsl-runtime/kapsl-engine:latest-tensorrt
```

Run NVIDIA images with the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/models" \
  -p 9095:9095 \
  -e KAPSL_ALLOW_INSECURE_HTTP=1 \
  ghcr.io/kapsl-runtime/kapsl-engine:latest-cuda \
  run --model /models/model.aimod --http-bind 0.0.0.0
```

Release-specific tags use `<kapsl-version>-cpu`, `<kapsl-version>-cuda`, and
`<kapsl-version>-tensorrt`. The unqualified `latest` tag always points to the
CPU image so pulling Kapsl never downloads NVIDIA or TensorRT libraries
implicitly. Stable releases update `latest`, `latest-cpu`, `latest-cuda`, and
`latest-tensorrt`; beta releases instead update `beta`, `beta-cpu`,
`beta-cuda`, and `beta-tensorrt`.

## Build from source

```bash
git clone https://github.com/kapsl-runtime/kapsl-engine
cd kapsl-engine/kapsl-runtime

./scripts/build-with-embedded-ui.sh --release
```

The compiled binary is at `target/release/kapsl`. The wrapper invalidates the
embedded dashboard before building, which also makes source deployments safe
after timestamp-preserving transfers such as `rsync -a`.

## Quick start

### 1. Package a model

```bash
# Built-in helper for MNIST (for testing)
./scripts/packages/mnist/create_package.sh
```

Or package your own ONNX model:

```bash
./target/release/kapsl package \
  --model /path/to/model.onnx \
  --output /path/to/model.aimod
```

### 2. Start the runtime

```bash
./target/release/kapsl --model /path/to/model.aimod
```

Defaults:

| Setting | Default |
|---------|---------|
| Transport | Unix socket |
| Socket path (Unix) | `/tmp/kapsl.sock` |
| TCP port (IPC) | `9096` |
| HTTP host | `127.0.0.1` |
| HTTP/API/UI port | `9095` |

### 3. Verify it is running

```bash
curl http://127.0.0.1:9095/api/health
```

```json
{"status": "ok"}
```

## Common startup options

```bash
# Load multiple models at startup
kapsl \
  --model models/model_a.aimod \
  --model models/model_b.aimod

# Bind HTTP to a specific interface
kapsl --model model.aimod --http-host 0.0.0.0 --http-port 8080

# Use TCP transport instead of Unix socket
kapsl --model model.aimod --transport tcp --tcp-port 9096

# Set an admin token at startup
kapsl --model model.aimod --admin-token my-secret-admin-token

# Enable a specific backend
kapsl --model model.aimod --provider cuda
```

> **Security note**: By default the runtime refuses to bind HTTP to a non-loopback address. Set `KAPSL_ALLOW_INSECURE_HTTP=1` or use a reverse proxy with TLS for production deployments.

## Production checklist

- [ ] Run behind a reverse proxy (nginx, Caddy) with TLS termination
- [ ] Set `KAPSL_ALLOW_INSECURE_HTTP=1` only if TLS is handled upstream
- [ ] Configure API authentication (see [Authentication](./authentication.md))
- [ ] Set `KAPSL_MODEL_CACHE_MAX_MIB` to cap disk usage
- [ ] Set `KAPSL_MODEL_CACHE_RESERVED_FREE_MIB` to maintain free disk headroom
- [ ] Configure a Prometheus scrape target at `http://<host>:9095/metrics`

## Running as a systemd service

```ini
[Unit]
Description=kapsl-runtime inference server
After=network.target

[Service]
ExecStart=/usr/local/bin/kapsl --model /opt/models/model.aimod
Restart=on-failure
Environment=KAPSL_ALLOW_INSECURE_HTTP=1
Environment=KAPSL_API_TOKEN_ADMIN=your-admin-token

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now kapsl
```

## Upgrading

The runtime loads `.aimod` packages dynamically at runtime via the management API. A running server can load new models or unload old ones without restart:

```bash
# Start a new model without downtime
curl -X POST http://127.0.0.1:9095/api/models/start \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/new_model.aimod"}'
```
