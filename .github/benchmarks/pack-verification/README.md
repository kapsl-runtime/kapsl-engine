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
