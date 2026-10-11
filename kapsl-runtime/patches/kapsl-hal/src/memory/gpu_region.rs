//! Physical GPU region capabilities and observations.
//!
//! Admission, workload ownership, participant protocols and scheduler readiness
//! belong to the caller. These types describe backing, never budget grants.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRegionKind {
    Arena,
    Ipc,
    Vmm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRegionExport {
    None,
    CudaIpc,
    PosixFileDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRegionCapabilities {
    pub local_suballocation: bool,
    pub export: GpuRegionExport,
    pub stable_addresses: bool,
    pub explicit_resize: bool,
}

impl GpuRegionKind {
    pub const fn capabilities(self) -> GpuRegionCapabilities {
        GpuRegionCapabilities {
            local_suballocation: matches!(self, Self::Arena),
            export: match self {
                Self::Arena => GpuRegionExport::None,
                Self::Ipc => GpuRegionExport::CudaIpc,
                Self::Vmm => GpuRegionExport::PosixFileDescriptor,
            },
            stable_addresses: true,
            explicit_resize: matches!(self, Self::Vmm),
        }
    }
}

/// Physical state of one region. Exported `ready_bytes` is cleared, accessible
/// backing in the owning context; it says nothing about workers or schedulers.
/// Arenas leave it unknown because readiness belongs to individual allocation
/// contracts, including allocations retained after failed initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRegionSnapshot {
    pub device_id: usize,
    pub kind: GpuRegionKind,
    pub committed_bytes: usize,
    pub mapped_bytes: usize,
    pub ready_bytes: Option<usize>,
    pub virtual_reserved_bytes: usize,
    pub released: bool,
}

/// An observation handle owns no CUDA context, pointer or physical allocation.
/// Holding it must not delay physical release past the caller's budget lease.
pub trait GpuRegionObserver: Send + Sync {
    fn physical_snapshot(&self) -> GpuRegionSnapshot;
}

/// Shared physical contract. Allocation/export/resize operations stay on the
/// concrete region types so unsupported operations cannot be called by accident.
pub trait GpuRegion: Send + Sync {
    fn kind(&self) -> GpuRegionKind;

    fn capabilities(&self) -> GpuRegionCapabilities {
        self.kind().capabilities()
    }

    fn physical_snapshot(&self) -> GpuRegionSnapshot;
}

#[derive(Debug, Error)]
pub enum GpuRegionError {
    #[error("invalid GPU region request: {0}")]
    InvalidRequest(&'static str),
    #[error("GPU region has been released")]
    Released,
    #[error("GPU backing is not ready; initialization or cleanup is incomplete")]
    NotReady,
    #[error("GPU segment is stale or belongs to another region")]
    InvalidSegment,
    #[error("GPU region operation {operation} failed: {message}")]
    Driver {
        operation: &'static str,
        message: String,
    },
    #[error(
        "{operation}; rollback also failed: {cleanup}; retain the region and its budget charge"
    )]
    Rollback {
        operation: Box<GpuRegionError>,
        cleanup: Box<GpuRegionError>,
    },
}

#[cfg(feature = "cuda")]
pub(crate) fn driver_error(
    operation: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> GpuRegionError {
    move |error| GpuRegionError::Driver {
        operation,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_keep_local_backing_separate_from_isolated_exports() {
        let arena = GpuRegionKind::Arena.capabilities();
        let ipc = GpuRegionKind::Ipc.capabilities();
        let vmm = GpuRegionKind::Vmm.capabilities();
        assert!(arena.local_suballocation);
        assert_eq!(arena.export, GpuRegionExport::None);
        assert!(!arena.explicit_resize);
        assert!(!ipc.local_suballocation);
        assert_eq!(ipc.export, GpuRegionExport::CudaIpc);
        assert!(!ipc.explicit_resize);
        assert!(!vmm.local_suballocation);
        assert_eq!(vmm.export, GpuRegionExport::PosixFileDescriptor);
        assert!(vmm.stable_addresses && vmm.explicit_resize);
    }
}
