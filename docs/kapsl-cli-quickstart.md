# Kapsl CLI Quickstart

`kapsl` is the command-line tool for packaging and serving AI models with Kapsl Runtime.

Use it to:

- run `.aimod` model packages
- build `.aimod` packages from model files
- push/pull packages from a remote backend

## Install / Build

From this repository:

```bash
cd kapsl-runtime
cargo build --release -p kapsl
```

Run with Cargo in dev:

```bash
cargo run -p kapsl -- --help
```

## 60-Second Start

```bash
cd kapsl-runtime

# 1) Create sample model package
./scripts/packages/mnist/create_package.sh

# 2) Start runtime
cargo run -p kapsl -- --model models/mnist/mnist.aimod
```

Runtime endpoints (defaults):

- API: `http://127.0.0.1:9095/api`
- Metrics: `http://127.0.0.1:9095/metrics`

## Core Commands

Run model package:

```bash
kapsl run --model ./model.aimod
```

Build package from model file:

```bash
kapsl build ./model.onnx --output ./model.aimod
```

Build from context directory:

```bash
kapsl build ./models/my-model-context
```

Or from inside the context directory:

```bash
cd ./models/my-model-context
kapsl build
```

Push package:

```bash
kapsl push alice/model:prod ./model.aimod
```

Authenticated push:

```bash
kapsl login --remote-url https://your-registry.example.com/v1
kapsl push alice/model:prod ./model.aimod --remote-url https://your-registry.example.com/v1
```

After first successful login, `kapsl login` reuses the last remote URL automatically.

Headless/SSH login (no localhost callback required):

```bash
kapsl login --remote-url https://your-registry.example.com/v1 --device-code
```

In SSH sessions, plain `kapsl login` automatically prefers device-code flow (GitHub).

If no token is available, `kapsl push`/`kapsl pull` will automatically open browser login on a 401 response and retry once.

Or from inside a directory with a single `.aimod` file:

```bash
cd ./models/mnist
kapsl push alice/mnist:prod
```

Pull package:

```bash
kapsl pull alice/model:prod --destination-dir ./models
```

Authenticated pull:

```bash
kapsl pull alice/model:prod --destination-dir ./models --remote-url https://your-registry.example.com/v1
```

Or pull into the current directory:

```bash
cd ./models
kapsl pull alice/model:prod
```

## Most Useful Flags

- `--transport <socket|tcp|shm|hybrid|auto>`
- `--http-bind <ip>`
- `--metrics-port <port>`
- `--state-dir <dir>`
- `--performance-profile <standard|balanced|throughput|latency>`

Example:

```bash
kapsl run \
  --model ./model.aimod \
  --transport hybrid \
  --performance-profile balanced
```

## Typical Team Workflow

Producer machine:

```bash
kapsl login --remote-url https://your-registry.example.com/v1
kapsl build ./model.onnx --output ./model.aimod
kapsl push alice/model:prod ./model.aimod --remote-url https://your-registry.example.com/v1
```

Consumer machine:

```bash
kapsl login --remote-url https://your-registry.example.com/v1
kapsl pull alice/model:prod --destination-dir ./models --remote-url https://your-registry.example.com/v1
kapsl run --model ./models/model.aimod
```

OCI registry alternative (requires `oras`):

```bash
kapsl push alice/model:prod ./model.aimod --remote-url oci://ghcr.io
kapsl pull alice/model:prod --destination-dir ./models --remote-url oci://ghcr.io
```

## Authentication (API)

When API auth is enabled, calls to `/api/*` and `/metrics` must include a bearer token.

Common env vars:

- `KAPSL_API_TOKEN` (shared fallback token)
- `KAPSL_API_TOKEN_READER`
- `KAPSL_API_TOKEN_WRITER`
- `KAPSL_API_TOKEN_ADMIN`

Example:

```bash
export KAPSL_API_TOKEN="your-token"

curl http://127.0.0.1:9095/api/models \
  -H "Authorization: Bearer $KAPSL_API_TOKEN"
```

If auth is disabled, API is loopback-only by default.

## Extensions (via Runtime API)

`kapsl` CLI does not have dedicated extension subcommands.  
Use the runtime API endpoints for extension lifecycle operations.

List installed extensions:

```bash
curl http://127.0.0.1:9095/api/extensions \
  -H "Authorization: Bearer $KAPSL_API_TOKEN"
```

Install from local directory:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/install \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN" \
  -d '{"path":"./extensions/my-extension"}'
```

Set workspace config:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.echo/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN" \
  -d '{"workspace_id":"default","config":{"api_key":"...","project":"..."}}'
```

Launch connector and sync data into local RAG index:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.echo/launch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN" \
  -d '{"workspace_id":"default"}'

curl -X POST http://127.0.0.1:9095/api/extensions/connector.echo/sync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN" \
  -d '{"workspace_id":"default"}'
```

## Need Full Docs?

For complete command coverage and advanced options, see:

- `docs/kapsl-cli-user-guide.md`
