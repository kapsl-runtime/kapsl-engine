# Kapsl CLI User Guide

The `kapsl` CLI helps you:

- run model packages
- add models to a running runtime without restarting
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
- `build`: build a `.aimod` package
- `push`: upload a `.aimod` package
- `pull`: download a `.aimod` package

Compatibility mode:

```bash
kapsl --model models/mnist/mnist.aimod
```

This is equivalent to:

```bash
kapsl run --model models/mnist/mnist.aimod
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
kapsl run \
  --model models/mnist/mnist.aimod \
  --model models/squeezenet/squeezenet.aimod
```

Useful run options:

- `--transport <socket|tcp|shm|hybrid|auto>` (default: `socket`)
- `--socket /tmp/kapsl.sock`
- `--bind 127.0.0.1`
- `--port 9096`
- `--http-bind 127.0.0.1`
- `--metrics-port 9095`
- `--state-dir <dir>` (namespaces rag-data, extensions, extensions-config, auth-store.json)
- `--performance-profile <standard|balanced|throughput|latency>`

Example with TCP transport:

```bash
kapsl run \
  --model models/mnist/mnist.aimod \
  --transport tcp \
  --bind 0.0.0.0 \
  --port 9096
```

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
kapsl add-model --model ./model.aimod --auth-token "$KAPSL_API_TOKEN"
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

## 4) Build Packages (`kapsl build`)

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

## 5) Push Packages (`kapsl push`)

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

### OCI Remote (ORAS)

If `--remote-url` starts with `oci://`, `kapsl` uses ORAS to push/pull the `.aimod` as an OCI artifact.

Prerequisites:
- Install `oras` and make it available on `PATH` (or set `KAPSL_ORAS_BIN`).

Push to an OCI registry:

```bash
kapsl push alice/mnist:prod ./mnist.aimod --remote-url oci://ghcr.io
```

Optional CI auth (otherwise use `oras login` / docker credential store):
- `KAPSL_OCI_USERNAME`
- `KAPSL_OCI_PASSWORD`

## 6) Pull Packages (`kapsl pull`)

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

Pull from an OCI registry:

```bash
kapsl pull alice/mnist:prod --destination-dir ./models --remote-url oci://ghcr.io
```

Pull by OCI manifest digest (reproducible):

```bash
kapsl pull \
  alice/mnist:prod \
  --destination-dir ./models \
  --remote-url oci://ghcr.io \
  --ref sha256:<manifestDigest>
```

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
- For authenticated remotes, use `--remote-token` or set `KAPSL_REMOTE_TOKEN` (`KAPSL_REMOTE_TOKEN` legacy).
- If you install the CLI with Cargo, prefer `cargo install --path crates/kapsl-cli --locked` so the binary matches the checked-in lockfile and dependency feature set.

## Authentication (API)

When API auth is enabled, requests to `/api/*` and `/metrics` require a bearer token.

Common environment variables:

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

If auth is disabled, API access is loopback-only by default.

## Extensions (via Runtime API)

`kapsl` CLI does not expose dedicated extension subcommands today.
Use runtime API endpoints for extension lifecycle tasks.

List installed extensions:

```bash
curl http://127.0.0.1:9095/api/extensions \
  -H "Authorization: Bearer $KAPSL_API_TOKEN"
```

Install from a local extension directory:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/install \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN" \
  -d '{"path":"./extensions/my-extension"}'
```

Set extension config for a workspace:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.echo/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KAPSL_API_TOKEN" \
  -d '{"workspace_id":"default","config":{"api_key":"...","project":"..."}}'
```

Launch and sync:

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

# build
kapsl build <path-to-model-file> --output <output.aimod>
kapsl build [<context-dir>]

# push
kapsl push <repo>/<model>:<label> [<path-to-kapsl>] [--remote-url <url>]

# pull
kapsl pull <repo>/<model>:<label> [--destination-dir <dir>] [--remote-url <url>]
```
