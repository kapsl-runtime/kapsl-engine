# Deployment

## Prerequisites

- Rust 1.75 or later
- Python 3.8+ (optional — only needed for helper scripts and building `kapsl-sdk`)
- `ffmpeg` (optional — required for video/audio inference payloads)
- GPU drivers and SDKs only if using a non-CPU backend:
  - A compatible NVIDIA display driver on Windows; the provider packs include
    the required CUDA 12, cuDNN 9, and TensorRT 10 user-space runtime libraries
  - A compatible NVIDIA driver on Linux; the CUDA installer includes the CUDA
    12, cuDNN 9, and NCCL user-space libraries. Bare-metal TensorRT installs
    additionally require compatible TensorRT 10 system libraries
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

That single archive contains the CUDA-compiled GGUF runtime and the ONNX CUDA
execution provider, plus their user-space CUDA dependencies. It requires only a
compatible host NVIDIA driver, like a Triton GPU image.

The stable CUDA runtime uses Kapsl's paged shared-KV path for supported GGUF
architectures. Models rejected by its compatibility policy use llama.cpp's
native KV path instead. Set `KAPSL_GGUF_DISABLE_SHARED_KV=1` to force that path
for diagnosis or rollback. Source builds can instead select the explicit
`gguf-cuda` feature to exclude shared-KV entirely.

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
curl -fsSL https://downloads.kapsl.net/install-beta-cuda.sh | sh
```

```powershell
irm https://downloads.kapsl.net/install-beta.ps1 | iex
irm https://downloads.kapsl.net/install-beta-cuda.ps1 | iex
irm https://downloads.kapsl.net/install-beta-tensorrt.ps1 | iex
```

Windows provider packs contain the calculated NVIDIA DLL dependency closure.
The standalone Linux provider packs remain available for existing portable
installations and require compatible NVIDIA system runtime libraries. The merged
Linux CUDA installer does not require that extra provider step. macOS uses system
Metal/CoreML frameworks and does not require an accelerator pack.

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

cargo build --release
```

The compiled binary is at `target/release/kapsl`.

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
