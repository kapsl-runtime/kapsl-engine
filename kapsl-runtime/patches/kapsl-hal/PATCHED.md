# Patched kapsl-hal

- Upstream: kapsl-hal 0.3.2 from crates.io, copied as published (minus `.cargo_vcs_info.json`, `Cargo.toml.orig`, `.cargo-ok`).
- Change: `src/device/info.rs` (~L725), `.map(|(maj, min)| format!("{maj}.{min}"))` became `.map(|cc| format!("{}.{}", cc.major, cc.minor))`.
- Why: nvml-wrapper 0.10 returns a `CudaComputeCapability { major, minor }` struct from `cuda_compute_capability()`, not a tuple, so the `nvml` feature did not compile. `gpu-device-pool` enables it.
- Remove once kapsl-hal 0.3.3 is published and kapsl-cli pins it.
