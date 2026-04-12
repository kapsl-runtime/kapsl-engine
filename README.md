# kapsl-engine

`kapsl-engine` is the runtime repository in the Kapsl split-repo layout.

This repo owns the `kapsl` runtime CLI and local inference server.

Shared Rust libraries are maintained in [`kapsl-sdk`](https://github.com/kapsl-runtime/kapsl-sdk). This repo consumes those shared crates as Git dependencies.

## Repository Layout

- `kapsl-runtime/`: main Rust workspace for the runtime binary
- `docker/`: Dockerfiles for CPU and CUDA images
- `docs/`: runtime-specific documentation

## Requirements

- Rust `1.92.0`
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
