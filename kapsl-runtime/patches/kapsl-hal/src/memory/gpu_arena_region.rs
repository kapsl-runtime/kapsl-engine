//! A fixed, local CUDA slab and its range allocator.
//!
//! `gpu_arena::GpuDevicePool` remains an alias for source compatibility with
//! existing ONNX callbacks, typed buffers and single-base-pointer KV views.

use super::*;
use crate::gpu_region::{GpuRegion, GpuRegionKind, GpuRegionSnapshot};
use std::cell::UnsafeCell;
use std::sync::atomic::AtomicU64;

static NEXT_DEVICE_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// One stable backing allocation and byte-range allocator for a CUDA device.
pub struct GpuArenaRegion {
    pool_id: u64,
    device: Arc<CudaDevice>,
    // Kept optional so Drop can release the cudaMallocAsync allocation before
    // trimming CUDA's default memory pool. CudaSlice::drop alone returns the
    // range to that pool, but the driver is free to retain the physical pages.
    storage: UnsafeCell<Option<CudaSlice<u8>>>,
    allocator: Mutex<AlignedRangeAllocator>,
    policy: Mutex<PoolPolicy>,
}

impl std::fmt::Debug for GpuArenaRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuArenaRegion")
            .field("capacity_bytes", &self.capacity_bytes())
            .field("free_bytes", &self.free_bytes())
            .finish()
    }
}

// SAFETY: the allocator assigns non-overlapping extents. Consumers may mutate
// only extents they own, and the backing CudaSlice remains pinned for the pool
// lifetime.
unsafe impl Send for GpuArenaRegion {}
unsafe impl Sync for GpuArenaRegion {}

impl GpuArenaRegion {
    pub fn new(device: Arc<CudaDevice>, capacity_bytes: usize) -> Result<Self, ArenaError> {
        if capacity_bytes == 0 {
            return Err(ArenaError::InvalidAllocationRequest);
        }
        let storage = device.alloc_zeros::<u8>(capacity_bytes)?;
        log::info!(
            "GPU device pool allocated: {} MiB",
            capacity_bytes / (1024 * 1024)
        );
        Ok(Self {
            pool_id: NEXT_DEVICE_POOL_ID.fetch_add(1, Ordering::Relaxed),
            device,
            storage: UnsafeCell::new(Some(storage)),
            allocator: Mutex::new(AlignedRangeAllocator::new(capacity_bytes)),
            policy: Mutex::new(PoolPolicy::new(capacity_bytes)),
        })
    }

    /// Reserve an extent, completing KV initialization before publishing it.
    /// If initialization fails, the extent remains charged and unavailable
    /// until pool destruction: a failed fence cannot authorize its reuse.
    pub fn alloc(
        &self,
        owner: PoolOwner,
        bytes: usize,
        alignment: usize,
    ) -> Result<GpuAllocation, ArenaError> {
        // Policy is always locked before allocator throughout this type.
        let mut policy = self.policy.lock().unwrap();
        let mut allocator = self.allocator.lock().unwrap();
        allocate_initialized(
            self.pool_id,
            &mut policy,
            &mut allocator,
            owner,
            bytes,
            alignment,
            |allocation| self.zero_allocation_sync(allocation),
        )
    }

    fn zero_allocation_sync(&self, allocation: &GpuAllocation) -> Result<(), ArenaError> {
        self.device.bind_to_thread()?;
        let device_ptr =
            self.allocation_ptr(allocation) as usize as cudarc::driver::sys::CUdeviceptr;
        // SAFETY: `allocation` is still live in this pool, its full byte range
        // is exclusively owned by the allocating caller, and an all-zero bit
        // pattern is valid for KV storage. Even cuMemsetD8 (without Async in
        // its name) can return before device memory is cleared. Submit to the
        // retained pool stream and explicitly wait before exposing the pointer
        // to a consumer that may execute on a different nonblocking stream.
        unsafe {
            result::memset_d8_async(device_ptr, 0, allocation.bytes(), *self.device.cu_stream())?
        };
        self.device.synchronize()?;
        Ok(())
    }

    pub fn free(&self, allocation: GpuAllocation) -> Result<(), ArenaError> {
        let mut policy = self.policy.lock().unwrap();
        let mut allocator = self.allocator.lock().unwrap();
        allocator.free(&allocation)?;
        policy.account_free(allocation.owner, allocation.bytes);
        Ok(())
    }

    pub fn set_owner_quota(
        &self,
        owner: PoolOwner,
        guaranteed_bytes: usize,
        max_bytes: usize,
    ) -> Result<(), ArenaError> {
        let mut policy = self.policy.lock().unwrap();
        let allocator = self.allocator.lock().unwrap();
        let workload = owner.workload();
        let previous = policy.quotas.get(&workload).copied();
        policy.set_quota(owner, guaranteed_bytes, max_bytes)?;
        if policy.unmet_reservations() > allocator.free_bytes() {
            if let Some(previous) = previous {
                policy.quotas.insert(workload, previous);
            } else {
                policy.quotas.remove(&workload);
            }
            return Err(ArenaError::QuotaExceeded {
                owner,
                requested: guaranteed_bytes,
                available: allocator.free_bytes(),
            });
        }
        Ok(())
    }

    pub fn set_owner_admitted(&self, owner: PoolOwner, admitted: bool) -> Result<(), ArenaError> {
        let mut policy = self.policy.lock().unwrap();
        let allocator = self.allocator.lock().unwrap();
        let workload = owner.workload();
        let usage = policy.workload_usage_bytes(workload);
        if !admitted && usage != 0 {
            return Err(ArenaError::OwnerInUse { owner, usage });
        }
        let was_admitted = policy.admitted.contains(&workload);
        let previous_owner = policy.admission_owners.get(&workload).copied();
        policy.set_admitted(owner, admitted);
        if admitted && policy.unmet_reservations() > allocator.free_bytes() {
            if was_admitted {
                policy.admitted.insert(workload);
                if let Some(previous_owner) = previous_owner {
                    policy.admission_owners.insert(workload, previous_owner);
                }
            } else {
                policy.admitted.remove(&workload);
                policy.admission_owners.remove(&workload);
            }
            return Err(ArenaError::QuotaExceeded {
                owner,
                requested: policy.quota(owner).guaranteed_bytes,
                available: allocator.free_bytes(),
            });
        }
        Ok(())
    }

    pub fn owner_usage_bytes(&self, owner: PoolOwner) -> usize {
        self.policy.lock().unwrap().usage_bytes(owner)
    }

    /// Observe workload admission without constructing a pool-wide snapshot.
    ///
    /// This takes only the policy lock and does not reserve memory or acquire
    /// an admission lease. Callers must keep their workload's admission alive
    /// throughout allocation and use, just as when checking [`Self::snapshot`].
    /// All allocation classes of the same backend/model/replica share admission.
    pub fn is_owner_admitted(&self, owner: PoolOwner) -> bool {
        self.policy.lock().unwrap().is_admitted(owner)
    }

    /// Aggregate bytes owned by all allocation classes for this model replica.
    pub fn workload_usage_bytes(&self, owner: PoolOwner) -> usize {
        self.policy
            .lock()
            .unwrap()
            .workload_usage_bytes(owner.workload())
    }

    pub fn owner_quota(&self, owner: PoolOwner) -> OwnerQuota {
        self.policy.lock().unwrap().quota(owner)
    }

    pub fn free_bytes(&self) -> usize {
        self.allocator.lock().unwrap().free_bytes()
    }

    /// Capture pool geometry, allocation, fragmentation, and per-owner policy
    /// state from one instant.
    ///
    /// Policy is locked before the allocator, matching all pool operations, so
    /// owner usage and live ranges cannot come from different mutations.
    pub fn snapshot(&self) -> GpuDevicePoolSnapshot {
        let policy = self.policy.lock().unwrap();
        let allocator = self.allocator.lock().unwrap();
        build_device_pool_snapshot(&policy, &allocator)
    }

    /// Maximum number of `unit_bytes` allocations currently possible for an
    /// owner, including quota/reservation checks and alignment fragmentation.
    pub fn max_allocatable(&self, owner: PoolOwner, unit_bytes: usize, alignment: usize) -> usize {
        if unit_bytes == 0 || alignment == 0 {
            return 0;
        }
        let policy = self.policy.lock().unwrap();
        let allocator = self.allocator.lock().unwrap();
        let quota_count = policy.available_for(owner, allocator.free_bytes()) / unit_bytes;
        quota_count.min(allocator.max_allocatable_units(unit_bytes, alignment))
    }

    pub fn base_ptr(&self) -> *mut std::ffi::c_void {
        let storage = unsafe { &*self.storage.get() }
            .as_ref()
            .expect("live GPU device pool storage");
        *storage.device_ptr() as *mut std::ffi::c_void
    }

    pub fn allocation_ptr(&self, allocation: &GpuAllocation) -> *mut std::ffi::c_void {
        debug_assert_eq!(allocation.pool_id, self.pool_id);
        (self.base_ptr() as usize + allocation.offset) as *mut std::ffi::c_void
    }

    pub fn capacity_bytes(&self) -> usize {
        let storage = unsafe { &*self.storage.get() }
            .as_ref()
            .expect("live GPU device pool storage");
        storage.len()
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub(super) fn f16_storage(&self) -> CudaView<'_, half::f16> {
        let storage = unsafe { &*self.storage.get() }
            .as_ref()
            .expect("live GPU device pool storage");
        // CUDA allocations are sufficiently aligned for f16; a trailing odd
        // byte, if any, is intentionally not exposed through the typed view.
        unsafe { storage.transmute(storage.len() / std::mem::size_of::<half::f16>()) }
            .expect("f16 view fits device pool")
    }

    /// # Safety
    /// The caller must write only extents allocated to it. Multiple mutable
    /// views may coexist because ownership is enforced by the range allocator.
    pub(super) unsafe fn f16_storage_mut(&self) -> CudaViewMut<'_, half::f16> {
        let storage = unsafe { &mut *self.storage.get() }
            .as_mut()
            .expect("live GPU device pool storage");
        let len = storage.len() / std::mem::size_of::<half::f16>();
        unsafe { storage.transmute_mut(len) }.expect("f16 view fits device pool")
    }
}

impl Drop for GpuArenaRegion {
    fn drop(&mut self) {
        let capacity = unsafe { &mut *self.storage.get() }
            .take()
            .map(|storage| {
                let capacity = storage.len();
                // On memory-pool-capable devices this enqueues cudaFreeAsync.
                drop(storage);
                capacity
            })
            .unwrap_or(0);

        if capacity == 0 {
            return;
        }
        if let Err(error) = self.device.synchronize() {
            log::warn!(
                "GPU device pool released {} bytes but could not synchronize before trimming CUDA's default memory pool: {}",
                capacity,
                error
            );
            return;
        }

        let memory_pools_supported = self
            .device
            .attribute(
                cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )
            .map(|supported| supported > 0)
            .unwrap_or(false);
        if !memory_pools_supported {
            log::info!(
                "GPU device pool backing released: {} MiB",
                capacity / (1024 * 1024)
            );
            return;
        }

        let trim_result = unsafe {
            use cudarc::driver::sys;

            let mut default_pool = std::ptr::null_mut();
            sys::lib()
                .cuDeviceGetDefaultMemPool(&mut default_pool, *self.device.cu_device())
                .result()
                .and_then(|()| sys::lib().cuMemPoolTrimTo(default_pool, 0).result())
        };
        match trim_result {
            Ok(()) => log::info!(
                "GPU device pool backing released and CUDA default memory pool trimmed: {} MiB",
                capacity / (1024 * 1024)
            ),
            Err(error) => log::warn!(
                "GPU device pool backing released, but CUDA default memory pool trim failed: {}",
                error
            ),
        }
    }
}

impl GpuRegion for GpuArenaRegion {
    fn kind(&self) -> GpuRegionKind {
        GpuRegionKind::Arena
    }

    fn physical_snapshot(&self) -> GpuRegionSnapshot {
        GpuRegionSnapshot {
            device_id: self.device.ordinal(),
            kind: self.kind(),
            committed_bytes: self.capacity_bytes(),
            mapped_bytes: self.capacity_bytes(),
            ready_bytes: None,
            virtual_reserved_bytes: 0,
            released: false,
        }
    }
}
