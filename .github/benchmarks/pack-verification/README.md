# Installed-pack checksum profiling

This manual, CPU-only benchmark compiles the engine's actual checksum module.
It compares the previous serial verification loop with bounded parallel reads
using identical SHA-256 code and files. It does not load adapters or use a GPU.

Supply a JSON object mapping relative filenames to their expected SHA-256
digests. For an installed pack, this is its signed manifest's `files` map:

```sh
cargo run --release --locked \
  --manifest-path .github/benchmarks/pack-verification/Cargo.toml -- \
  /path/to/installed/pack /path/to/checksums.json 10 > measurements.json
```

The tool warms both routes, then records 10 ABBA blocks (40 samples) by default.
Every sample must validate every required checksum. The output includes raw
timings, medians and the candidate/reference ratio, without a performance gate.
Run it without overlapping builds or inference workloads. File-cache state,
CPU allocation and storage bandwidth affect the result.

This isolates verification cost. Signature checks, adapter loading, inference
and end-to-end startup qualification remain the engine's responsibility. This
benchmark does not replace or change the release's 1.5x startup threshold.
It times the SHA-256 `files` map only; packs that sign a BLAKE3 map
(`files_blake3`) are verified with BLAKE3 by the engine.

## Correctness tests

The same crate runs the checksum module's unit tests (SHA-256 and BLAKE3,
tampering, missing files, read errors, worker panics and the per-file profiler)
without building the engine, ORT or any backend runtime. CI runs them on Linux,
macOS and Windows:

```sh
cargo test --locked --manifest-path .github/benchmarks/pack-verification/Cargo.toml
```

## Per-file timings in a real launch

Set `KAPSL_PACK_VERIFICATION_PROFILING=1` on the engine. Each validation then
logs one `KAPSL_PACK_VERIFICATION_PROFILE` JSON record with per-file bytes and
open, read and hash wall times. These are diagnostics, not qualification
measurements; leave the variable unset for qualification runs.
