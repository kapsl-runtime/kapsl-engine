# Deployment

## Prerequisites

- Rust 1.75 or later
- Python 3.8+ (optional — only needed for helper scripts and building `kapsl-sdk`)
- `ffmpeg` (optional — required for video/audio inference payloads)
- GPU drivers and SDKs only if using a non-CPU backend:
  - CUDA 11+ / TensorRT 8+ for NVIDIA
  - Xcode command line tools for Metal (macOS)

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
