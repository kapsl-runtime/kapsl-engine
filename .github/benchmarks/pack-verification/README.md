# Installed pack verification diagnostics

This CPU-only harness compiles the production checksum module with the engine's
locked SHA-256 dependency. Correctness tests do not require ORT, a GPU, or an
engine build:

```sh
cargo test --locked --manifest-path .github/benchmarks/pack-verification/Cargo.toml
```

For a manual per-file breakdown, supply an installed pack directory and a JSON
object mapping every signed relative file path to its SHA-256 digest:

```sh
cargo run --release --locked --manifest-path .github/benchmarks/pack-verification/Cargo.toml -- --diagnostic PACK_ROOT CHECKSUMS_JSON
```

The diagnostic performs exactly one validation without a preliminary warmup. It
reports bytes actually read, read and hash wall time, elapsed time, worker index,
and completion offset. Sum file elapsed times by worker and divide by the whole
validation wall time to estimate worker occupancy. This includes preemption;
it is not CPU utilization. Reads include page-cache hits and scheduling delay;
they are not physical disk time. Capture CPU quota/affinity, process CPU time,
faults and block I/O separately. Record whether inputs were recently read; a
second run is a warm-cache diagnostic, not an independent cold-start sample.
The observer adds clocks and per-file bookkeeping; never use these timings as
qualification results. A subset of runtime files does not represent full-pack
verification or launch-to-model-ready time.

Set `KAPSL_PACK_VERIFICATION_PROFILING=1` for the same per-file measurements during
an actual engine launch. One `Pack verification diagnostic` JSON record is logged
after each completed validation. Existing signed-manifest, path, symlink and
entrypoint checks still precede file verification. Missing files, read failures,
and worker panics cannot become successful verification. Invalid/missing metadata
can stop validation before any file observations; read errors propagate normally.

Leave the variable unset for qualification. The default path takes no per-chunk
clock readings and collects no diagnostic records. It retains the four-worker
limit, largest-first scheduling, 1 MiB buffers, and full content verification on
every load. No verification results are cached or reused.

Without `--diagnostic`, the existing serial-versus-parallel ABBA harness retains
its warmup, block count, and measurements. Neither mode replaces the unchanged
1.5x startup and 0.80x throughput qualification gates.

The original comparison command remains:

```sh
cargo run --release --locked --manifest-path .github/benchmarks/pack-verification/Cargo.toml -- PACK_ROOT CHECKSUMS_JSON 10 > measurements.json
```

It warms both routes, then runs 10 ABBA blocks (40 samples) by default and emits
raw times, medians, and the parallel/serial ratio. Stop builds, inference and
compression before manual timings. Signature checks, adapter loading and total
startup remain the engine's responsibility.
