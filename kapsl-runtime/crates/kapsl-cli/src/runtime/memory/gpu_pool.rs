//! Per-device backing ownership, compatible-region selection and opaque leases.
//!
//! Capacity admission remains in MemoryAuthority. Registering backing transfers
//! ownership, not another budget charge. Allocation never creates or grows it.

use super::gpu_regions::{
    GpuRegionId, GpuRegionIsolation, GpuRegionKind, GpuRegionRegistry, GpuRegionSnapshot,
    GpuRegionSource, GpuRegionUsage,
};
use cudarc::driver::CudaDevice;
use kapsl_hal::gpu_arena::{GpuAllocation, PoolOwner, PoolWorkload};
use kapsl_hal::gpu_arena_region::GpuArenaRegion;
#[cfg(any(target_os = "linux", all(test, unix)))]
use kapsl_hal::{
    gpu_ipc_region::GpuIpcRegion, gpu_region::GpuRegion, gpu_vmm_region::GpuVmmRegion,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
pub(crate) enum ExportCapacity {
    Fixed { bytes: usize },
    Elastic { maximum_bytes: usize },
}

/// A local request reuses ready arena capacity. Export requests are explicit
/// capacity operations and always own dedicated backing for their isolation.
#[derive(Debug, Clone)]
pub(crate) enum GpuRegionRequest {
    Local {
        owner: PoolOwner,
        bytes: usize,
        alignment: usize,
    },
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    Exported {
        isolation: GpuRegionIsolation,
        capacity: ExportCapacity,
    },
}

impl GpuRegionRequest {
    fn kind(&self) -> Result<GpuRegionKind, String> {
        match self {
            Self::Local { alignment: 0, .. } => Err("region alignment must be nonzero".into()),
            Self::Local { .. } => Ok(GpuRegionKind::Arena),
            Self::Exported {
                isolation: GpuRegionIsolation::Local,
                ..
            } => Err("export requires participant isolation".into()),
            Self::Exported {
                capacity:
                    ExportCapacity::Fixed { bytes: 0 } | ExportCapacity::Elastic { maximum_bytes: 0 },
                ..
            } => Err("region capacity must be nonzero".into()),
            Self::Exported {
                capacity: ExportCapacity::Fixed { .. },
                ..
            } => Ok(GpuRegionKind::Ipc),
            Self::Exported {
                capacity: ExportCapacity::Elastic { .. },
                ..
            } => Ok(GpuRegionKind::Vmm),
        }
    }
}

trait AllocationStorage: Send + Sync {
    fn pointer(&self) -> usize;
    fn bytes(&self) -> usize;
    fn free_after_fence(&self) -> Result<(), String>;
}

trait RegionStorage: GpuRegionSource {
    fn kind(&self) -> GpuRegionKind;
    fn fits_local(&self, _owner: PoolOwner, _bytes: usize, _alignment: usize) -> bool {
        false
    }
    fn allocate(
        &self,
        _owner: PoolOwner,
        _bytes: usize,
        _alignment: usize,
    ) -> Result<Box<dyn AllocationStorage>, String> {
        Err("region does not support local allocations".into())
    }
    fn arena(&self) -> Option<&Arc<GpuArenaRegion>> {
        None
    }
    #[cfg(any(target_os = "linux", all(test, unix)))]
    fn ipc(&self) -> Option<&GpuIpcRegion> {
        None
    }
    #[cfg(any(target_os = "linux", all(test, unix)))]
    fn vmm(&self) -> Option<&GpuVmmRegion> {
        None
    }
    fn can_retire_arena(&self) -> bool {
        false
    }
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    fn release_export_after_fence(&self) -> Result<(), String> {
        Err("local arenas retire after their final view and allocation".into())
    }
}

struct RegionRecord {
    isolation: GpuRegionIsolation,
    storage: Box<dyn RegionStorage>,
}

impl GpuRegionSource for RegionRecord {
    fn region_usage(&self) -> Option<GpuRegionUsage> {
        self.storage.region_usage()
    }
}

pub(crate) struct GpuDevicePool {
    device_id: usize,
    device: Option<Arc<CudaDevice>>,
    registry: GpuRegionRegistry,
    regions: Mutex<BTreeMap<GpuRegionId, Arc<RegionRecord>>>,
    next_allocation: AtomicU64,
}

impl GpuDevicePool {
    pub(crate) fn new(device: Arc<CudaDevice>) -> Arc<Self> {
        Arc::new(Self {
            device_id: device.ordinal(),
            registry: GpuRegionRegistry::new(device.ordinal()),
            device: Some(device),
            regions: Mutex::new(BTreeMap::new()),
            next_allocation: AtomicU64::new(1),
        })
    }

    pub(crate) fn device(&self) -> &Arc<CudaDevice> {
        self.device
            .as_ref()
            .expect("production pool owns its CUDA context")
    }

    fn insert(
        self: &Arc<Self>,
        isolation: GpuRegionIsolation,
        storage: Box<dyn RegionStorage>,
    ) -> GpuRegionLease {
        let kind = storage.kind();
        let record = Arc::new(RegionRecord {
            isolation: isolation.clone(),
            storage,
        });
        let source: Arc<dyn GpuRegionSource> = record.clone();
        let id = self.registry.register(kind, isolation, &source);
        self.regions.lock().insert(id, record.clone());
        GpuRegionLease {
            pool: self.clone(),
            id,
            record,
            owner: None,
        }
    }

    pub(crate) fn install_arena(
        self: &Arc<Self>,
        arena: Arc<GpuArenaRegion>,
    ) -> Result<GpuRegionLease, String> {
        if arena.device().ordinal() != self.device_id {
            return Err("arena belongs to another CUDA device".into());
        }
        Ok(self.insert(
            GpuRegionIsolation::Local,
            Box::new(HalStorage::Arena(arena)),
        ))
    }

    pub(crate) fn acquire_region(
        self: &Arc<Self>,
        request: GpuRegionRequest,
    ) -> Result<GpuRegionLease, String> {
        let kind = request.kind()?;
        match request {
            GpuRegionRequest::Local {
                owner,
                bytes,
                alignment,
            } => {
                // Pin candidates, then release the catalog lock before taking
                // allocator/policy locks. Pins exclude simultaneous retirement.
                let candidates = self
                    .regions
                    .lock()
                    .iter()
                    .map(|(id, record)| (*id, record.clone()))
                    .collect::<Vec<_>>();
                for (id, record) in candidates {
                    if record.isolation == GpuRegionIsolation::Local
                        && record.storage.kind() == kind
                        && record.storage.fits_local(owner, bytes, alignment)
                    {
                        return Ok(GpuRegionLease {
                            pool: self.clone(),
                            id,
                            record,
                            owner: Some(owner.workload()),
                        });
                    }
                }
                Err("no compatible arena has admitted, ready capacity; request a separate capacity change".into())
            }
            #[cfg(any(target_os = "linux", all(test, unix)))]
            GpuRegionRequest::Exported {
                isolation,
                capacity,
            } => {
                // Caller has admitted this explicit capacity operation. Never
                // satisfy an isolated export from logical holes in an arena.
                let device = self
                    .device
                    .as_ref()
                    .ok_or("export requires an explicit CUDA capacity operation")?;
                let storage = match capacity {
                    ExportCapacity::Fixed { bytes } => HalStorage::Ipc(
                        GpuIpcRegion::allocate(device.clone(), bytes).map_err(|e| e.to_string())?,
                    ),
                    ExportCapacity::Elastic { maximum_bytes } => HalStorage::Vmm(
                        GpuVmmRegion::reserve(device.clone(), maximum_bytes)
                            .map_err(|e| e.to_string())?,
                    ),
                };
                Ok(self.insert(isolation, Box::new(storage)))
            }
            #[cfg(not(any(target_os = "linux", all(test, unix))))]
            GpuRegionRequest::Exported { .. } => {
                Err("exported GPU regions require the Linux CUDA transport".into())
            }
        }
    }

    fn validate(&self, lease: &GpuRegionLease) -> Result<(), String> {
        if self.device_id != lease.pool.device_id || !std::ptr::eq(self, Arc::as_ptr(&lease.pool)) {
            return Err("region lease belongs to another device pool".into());
        }
        if !self
            .regions
            .lock()
            .get(&lease.id)
            .is_some_and(|current| Arc::ptr_eq(current, &lease.record))
        {
            return Err("region lease is stale or already retired".into());
        }
        Ok(())
    }

    pub(crate) fn allocate(
        self: &Arc<Self>,
        region: &GpuRegionLease,
        owner: PoolOwner,
        bytes: usize,
        alignment: usize,
    ) -> Result<GpuAllocationLease, String> {
        self.validate(region)?;
        if region
            .owner
            .is_some_and(|workload| workload != owner.workload())
        {
            return Err("allocation owner does not match its region lease".into());
        }
        if bytes == 0 || alignment == 0 {
            return Err("allocation size and alignment must be nonzero".into());
        }
        if region.record.isolation != GpuRegionIsolation::Local {
            return Err("exported regions are allocated by their participant".into());
        }
        let sequence = self
            .next_allocation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| "GPU allocation identity exhausted".to_string())?;
        let storage = region.record.storage.allocate(owner, bytes, alignment)?;
        Ok(GpuAllocationLease {
            region: region.clone(),
            owner,
            sequence,
            storage: Mutex::new(Some(storage)),
        })
    }

    /// Called with the manager's sole compatibility lease, after admissions and
    /// backend clients retire. Snapshot readers and other leases make it wait.
    pub(crate) fn retire_arena(&self, lease: &GpuRegionLease) -> Result<bool, String> {
        self.validate(lease)?;
        if lease.record.storage.kind() != GpuRegionKind::Arena {
            return Err("expected local arena".into());
        }
        Ok(self.registry.retire_if(lease.id, || {
            let mut regions = self.regions.lock();
            if Arc::strong_count(&lease.record) != 2 || !lease.record.storage.can_retire_arena() {
                return false;
            }
            regions.remove(&lease.id);
            true
        }))
    }

    pub(crate) fn snapshots(&self) -> Vec<GpuRegionSnapshot> {
        self.registry.snapshots()
    }
}

#[derive(Clone)]
pub(crate) struct GpuRegionLease {
    pool: Arc<GpuDevicePool>,
    id: GpuRegionId,
    record: Arc<RegionRecord>,
    owner: Option<PoolWorkload>,
}

impl GpuRegionLease {
    pub(crate) fn arena(&self) -> Result<&Arc<GpuArenaRegion>, String> {
        self.pool.validate(self)?;
        self.record
            .storage
            .arena()
            .ok_or_else(|| "region is not a local arena".into())
    }
    pub(crate) fn allocate(
        &self,
        owner: PoolOwner,
        bytes: usize,
        alignment: usize,
    ) -> Result<GpuAllocationLease, String> {
        self.pool.allocate(self, owner, bytes, alignment)
    }
    #[cfg(any(target_os = "linux", all(test, unix)))]
    pub(crate) fn ipc(&self) -> Result<&GpuIpcRegion, String> {
        self.pool.validate(self)?;
        self.record
            .storage
            .ipc()
            .ok_or_else(|| "region does not provide fixed IPC backing".into())
    }
    #[cfg(any(target_os = "linux", all(test, unix)))]
    pub(crate) fn vmm(&self) -> Result<&GpuVmmRegion, String> {
        self.pool.validate(self)?;
        self.record
            .storage
            .vmm()
            .ok_or_else(|| "region does not provide elastic VMM backing".into())
    }
    /// # Safety
    /// All local/remote users, exported descriptors and imported handles must
    /// be retired. The caller retains its grant until this succeeds.
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    pub(crate) unsafe fn release_export_after_fence(&self) -> Result<(), String> {
        // HAL release is idempotent even after the catalog entry was removed.
        self.record.storage.release_export_after_fence()?;
        self.pool.regions.lock().remove(&self.id);
        self.pool.registry.retire(self.id);
        Ok(())
    }
}

pub(crate) struct GpuAllocationLease {
    region: GpuRegionLease,
    owner: PoolOwner,
    sequence: u64,
    storage: Mutex<Option<Box<dyn AllocationStorage>>>,
}

impl GpuAllocationLease {
    pub(crate) fn pointer(&self) -> Result<usize, String> {
        self.region.pool.validate(&self.region)?;
        self.storage
            .lock()
            .as_ref()
            .map(|s| s.pointer())
            .ok_or_else(|| "allocation lease was released".into())
    }
    pub(crate) fn bytes(&self) -> usize {
        self.storage.lock().as_ref().map(|s| s.bytes()).unwrap_or(0)
    }
    /// # Safety
    /// Every stream that could reference the allocation must be fenced.
    pub(crate) unsafe fn release_after_fence(&self) -> Result<(), String> {
        self.region.pool.validate(&self.region)?;
        let mut storage = self.storage.lock();
        storage
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "allocation {} for {:?} was already released",
                    self.sequence, self.owner
                )
            })?
            .free_after_fence()?;
        storage.take();
        Ok(())
    }
}

impl Drop for GpuAllocationLease {
    fn drop(&mut self) {
        if self.storage.get_mut().is_some() {
            log::error!(
                "[gpu-pool] allocation {} for {:?} abandoned without a fence; extent retained",
                self.sequence,
                self.owner
            );
        }
    }
}

enum HalStorage {
    Arena(Arc<GpuArenaRegion>),
    #[cfg(any(target_os = "linux", all(test, unix)))]
    Ipc(GpuIpcRegion),
    #[cfg(any(target_os = "linux", all(test, unix)))]
    Vmm(GpuVmmRegion),
}

impl GpuRegionSource for HalStorage {
    fn region_usage(&self) -> Option<GpuRegionUsage> {
        match self {
            Self::Arena(arena) => arena.region_usage(),
            #[cfg(any(target_os = "linux", all(test, unix)))]
            other => {
                let snapshot = match other {
                    Self::Ipc(region) => region.physical_snapshot(),
                    Self::Vmm(region) => region.physical_snapshot(),
                    _ => unreachable!(),
                };
                exported_usage(snapshot)
            }
        }
    }
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn exported_usage(snapshot: kapsl_hal::gpu_region::GpuRegionSnapshot) -> Option<GpuRegionUsage> {
    (!snapshot.released).then(|| {
        GpuRegionUsage::exported(
            snapshot.committed_bytes,
            snapshot.mapped_bytes,
            snapshot.virtual_reserved_bytes,
        )
    })
}

impl RegionStorage for HalStorage {
    fn kind(&self) -> GpuRegionKind {
        match self {
            Self::Arena(_) => GpuRegionKind::Arena,
            #[cfg(any(target_os = "linux", all(test, unix)))]
            Self::Ipc(_) => GpuRegionKind::Ipc,
            #[cfg(any(target_os = "linux", all(test, unix)))]
            Self::Vmm(_) => GpuRegionKind::Vmm,
        }
    }
    fn fits_local(&self, owner: PoolOwner, bytes: usize, alignment: usize) -> bool {
        self.arena().is_some_and(|arena| {
            arena.is_owner_admitted(owner)
                && (bytes == 0 || arena.max_allocatable(owner, bytes, alignment) > 0)
        })
    }
    fn allocate(
        &self,
        owner: PoolOwner,
        bytes: usize,
        alignment: usize,
    ) -> Result<Box<dyn AllocationStorage>, String> {
        let arena = self.arena().ok_or("local allocations require an arena")?;
        if !arena.is_owner_admitted(owner) {
            return Err("allocation owner has no memory admission".into());
        }
        let allocation = arena
            .alloc(owner, bytes, alignment)
            .map_err(|e| e.to_string())?;
        let pointer = arena.allocation_ptr(&allocation) as usize;
        if pointer == 0 || !pointer.is_multiple_of(alignment) {
            arena.free(allocation).map_err(|e| e.to_string())?;
            return Err("arena cannot satisfy absolute pointer alignment".into());
        }
        Ok(Box::new(ArenaExtent {
            arena: arena.clone(),
            allocation,
        }))
    }
    fn arena(&self) -> Option<&Arc<GpuArenaRegion>> {
        match self {
            Self::Arena(arena) => Some(arena),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
    #[cfg(any(target_os = "linux", all(test, unix)))]
    fn ipc(&self) -> Option<&GpuIpcRegion> {
        match self {
            Self::Ipc(region) => Some(region),
            _ => None,
        }
    }
    #[cfg(any(target_os = "linux", all(test, unix)))]
    fn vmm(&self) -> Option<&GpuVmmRegion> {
        match self {
            Self::Vmm(region) => Some(region),
            _ => None,
        }
    }
    fn can_retire_arena(&self) -> bool {
        self.arena().is_some_and(|arena| {
            Arc::strong_count(arena) == 1 && arena.free_bytes() == arena.capacity_bytes()
        })
    }
    fn release_export_after_fence(&self) -> Result<(), String> {
        match self {
            Self::Arena(_) => {
                Err("arena release requires final compatibility-view retirement".into())
            }
            #[cfg(any(target_os = "linux", all(test, unix)))]
            Self::Ipc(region) => unsafe { region.release_after_fence() }.map_err(|e| e.to_string()),
            #[cfg(any(target_os = "linux", all(test, unix)))]
            Self::Vmm(region) => unsafe { region.release_after_fence() }.map_err(|e| e.to_string()),
        }
    }
}

struct ArenaExtent {
    arena: Arc<GpuArenaRegion>,
    allocation: GpuAllocation,
}
impl AllocationStorage for ArenaExtent {
    fn pointer(&self) -> usize {
        self.arena.allocation_ptr(&self.allocation) as usize
    }
    fn bytes(&self) -> usize {
        self.allocation.bytes()
    }
    fn free_after_fence(&self) -> Result<(), String> {
        self.arena
            .free(self.allocation.clone())
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
#[path = "gpu_pool_tests.rs"]
mod tests;
