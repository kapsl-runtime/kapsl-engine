# kapsl-engine

`kapsl-engine` is the runtime repository in the Kapsl split-repo layout.

This repo owns the `kapsl` runtime CLI and local inference server.

The runtime serves an OpenAI-compatible API at `/v1` alongside its native
`/api` routes, so existing OpenAI clients work by changing only their base URL.
See [`docs/openai-compatible-api.md`](docs/openai-compatible-api.md).

Online runs resolve, verify, and cache a required lazy backend automatically:

```bash
curl -fsSL https://downloads.kapsl.net/install-beta.sh | sh
kapsl run model.aimod
```

Managed vLLM and Linux x86_64 ONNX CPU/CUDA 12/TensorRT 10 and llama.cpp
CPU/CUDA 12 profiles use the signed lazy cache. The ORT CPU candidate is built
from an exact out-of-tree integrations commit and is distinguished from the
legacy accelerator bundles by its signed standard-ABI marker. Embedded ORT
remains the default CPU rollback during parity certification. Provider
fallback remains package-controlled, and TensorRT is selected only when the
model contract explicitly permits it. The portable core lazily loads llama.cpp
CPU; the certified eager CUDA shared-KV profile remains the default during
CUDA pack certification.

For a no-network host, prepare one verified bundle on a connected machine and
run it directly after transfer:

```bash
kapsl bundle model.aimod --output model.kapsl-bundle
kapsl run model.kapsl-bundle
```

Release engineers can build the same bundle from an already-downloaded,
signed release directory with `--backend-artifacts-dir`. Kapsl still verifies
the signed index entry, archive size, SHA-256 digest, and Ed25519 signature;
the option does not enable `file://` artifacts in ordinary runtime startup.

See [Lazy Backend Packs](kapsl-runtime/docs/backend-packs.md) for the trust,
cache, administration, and cross-target bundle model.

Shared Rust libraries are maintained in [`kapsl-sdk`](https://github.com/kapsl-runtime/kapsl-sdk).
The runtime binary depends on those crates through normal Cargo dependencies.

## Repository Layout

- `kapsl-runtime/`: main Rust workspace for the runtime binary
- `kapsl-runtime/crates/kapsl-cli/`: `kapsl` CLI, server orchestration, HTTP API, and runtime entry point
- `kapsl-runtime/ui/`: embedded web dashboard assets
- `kapsl-runtime/docs/`: runtime-specific user and API docs
- `kapsl-runtime/patches/`: active third-party crate patches used only by this workspace
- `installers/`: source scripts published at the stable and beta installer URLs
- `docker/`: Dockerfiles for CPU and CUDA images
- `docs/`: runtime-specific documentation

## Architecture Boundary

`kapsl-engine` should stay thin around product runtime concerns:

- CLI commands, installer/runtime packaging, and runtime process startup
- HTTP API, embedded dashboard, auth, metrics, and operational control loops
- Wiring shared SDK crates into a runnable local inference server

Reusable Rust libraries, client bindings, transports, schedulers, backend abstractions,
RAG primitives, and Python packaging belong in `kapsl-sdk`.

## Requirements

- Rust `1.98.0` (pinned by `rust-toolchain.toml`); the runtime workspace MSRV
  remains Rust `1.92.0`
- platform build tools for your target OS
- optional accelerator toolchains depending on which runtime backends you enable

## Local Development

Build the runtime:

```bash
cargo build --manifest-path kapsl-runtime/Cargo.toml -p kapsl
```

Run the runtime:

```bash
cargo run --manifest-path kapsl-runtime/Cargo.toml -p kapsl -- --help
```

Run workspace checks:

```bash
cargo check --manifest-path kapsl-runtime/Cargo.toml --workspace
```

When developing `kapsl-engine` and `kapsl-sdk` together, prefer a local Cargo
override in your own checkout instead of committing local paths to this repo.

## Release Flow

Runtime installers are built by GitHub Actions from:

- `.github/workflows/release-runtime-installers.yml`

Supported outputs:

- Linux: `.deb`
- macOS: `.pkg`
- Windows: `.msi`

macOS and Windows signing are optional in CI. If the Apple or Windows
certificate secrets are not configured, the workflow still produces
unsigned installers instead of failing.

Backend-pack publication is fail-closed and requires both
`KAPSL_BACKEND_SIGNING_KEY_B64` (a base64-encoded Ed25519 PEM private key) and
`KAPSL_BACKEND_PUBLIC_KEYS` (one or more raw 32-byte public keys in base64).
The latter is embedded in every runtime binary; the release job proves the key
pair matches before it signs or uploads `backend-index.json`.

Publishing flow:

1. Create a version tag such as `v0.1.1`
2. Push the tag to GitHub
3. The installer workflow builds platform installers and uploads them to the matching GitHub Release

Example:

```bash
git tag v0.1.1
git push origin v0.1.1
```

`workflow_dispatch` remains available for manual test runs without creating a release tag.

## Related Repositories

- [kapsl-sdk](https://github.com/kapsl-runtime/kapsl-sdk) — shared Rust crates and Python package
- [kapsl-extensions](https://github.com/kapsl-runtime/kapsl-extensions) — runtime extensions (RAG connectors, prompt transformer)
- [kapsl-benchmarks](https://github.com/kapsl-runtime/kapsl-benchmarks) — benchmarks and inference test harnesses
- [kapsl-lite](https://github.com/kapsl-runtime/kapsl-lite) — lightweight runtime distribution
- [kapsl-desktop](https://github.com/kapsl-runtime/kapsl-desktop) — desktop application and bundled installer flow
- [penasys](https://github.com/kapsl-runtime/penasys) — Go backend and Kubernetes deployment assets
