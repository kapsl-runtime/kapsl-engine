//! Host resource probes used by automatic sizing policies.

use std::path::PathBuf;
use sysinfo::System;

/// Return the largest readable model file size in MiB.
pub(crate) fn largest_model_size_mb(model_paths: &[PathBuf]) -> u64 {
    model_paths
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok().map(|metadata| metadata.len()))
        .max()
        .map(|bytes| bytes / (1024 * 1024))
        .unwrap_or(0)
}

/// Return currently available system memory in MiB.
pub(crate) fn available_ram_mb() -> u64 {
    let mut system = System::new();
    system.refresh_memory();
    system.available_memory() / (1024 * 1024)
}

/// Return the host's logical CPU count, falling back to one core.
pub(crate) fn logical_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
}
