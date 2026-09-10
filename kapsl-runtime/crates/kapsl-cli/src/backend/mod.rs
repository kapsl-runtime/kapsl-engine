//! Backend selection, signed pack management, and offline bundle support.

use super::*;

mod bundle;
mod llama_cpp;
mod manager;
mod native;
mod onnx;
mod selection;

pub(crate) use bundle::*;
pub(crate) use llama_cpp::*;
pub(crate) use manager::*;
pub(crate) use native::*;
pub(crate) use onnx::*;
pub(crate) use selection::*;

/// `DeviceInfo::total_memory` is reported in KiB, while pack manifests and
/// memory-governance snapshots use bytes.
pub(crate) fn guarded_host_memory_bytes(total_memory_kib: u64, guard_percent: u64) -> u64 {
    total_memory_kib
        .saturating_mul(1024)
        .saturating_mul(100_u64.saturating_sub(guard_percent))
        .saturating_div(100)
}

#[cfg(test)]
mod tests {
    use super::guarded_host_memory_bytes;

    #[test]
    fn guarded_host_memory_converts_kib_to_bytes() {
        assert_eq!(
            guarded_host_memory_bytes(16 * 1024 * 1024, 10),
            16 * 1024 * 1024 * 1024 * 9 / 10
        );
    }
}
