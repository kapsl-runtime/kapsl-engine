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
| `KAPSL_API_TOKEN` | Shared admin token (legacy fallback) |
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
| `KAPSL_REMOTE_PLACEHOLDER_URL` | Placeholder URL displayed in the dashboard |
| `KAPSL_REMOTE_PLACEHOLDER_DIR` | Default remote directory shown in the dashboard |

### Backend and inference

| Variable | Description |
|----------|-------------|
| `KAPSL_PROVIDER_POLICY` | `fastest` (auto-select fastest backend) or `manifest` (use manifest-specified backend) |
| `KAPSL_LLM_ISOLATE_PROCESS` | Set to `1` to run LLM backends in a subprocess for isolation |
| `KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS` | Disable automatic image/video-to-tensor preprocessing |

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
