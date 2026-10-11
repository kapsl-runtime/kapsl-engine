//! Stable CUDA virtual ranges with explicitly managed physical segments.
//!
//! The caller owns budget grants, worker mappings, activation and retirement
//! acknowledgments. This module owns only the parent context's CUDA resources.

use crate::gpu_region::{
    driver_error, GpuRegion, GpuRegionError, GpuRegionKind, GpuRegionObserver, GpuRegionSnapshot,
};
use cudarc::driver::{result, sys, CudaDevice};
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_REGION_ID: AtomicU64 = AtomicU64::new(1);

/// Opaque identity, never reused when an offset is reclaimed and mapped again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuVmmSegmentId {
    region: u64,
    sequence: u64,
}

impl std::fmt::Display for GpuVmmSegmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.region, self.sequence)
    }
}

/// Backend-neutral byte geometry. A worker adapter supplies its own binding
/// IDs, block counts, generations and file-descriptor ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuVmmSegment {
    pub id: GpuVmmSegmentId,
    pub offset_bytes: usize,
    pub length_bytes: usize,
    pub mapped: bool,
    pub ready: bool,
}

/// A reserved virtual range with explicitly admitted physical growth.
/// Keep this object and its budget charge until fenced release succeeds.
pub struct GpuVmmRegion {
    inner: VmmRegion<CudaVmmDriver>,
}

impl GpuVmmRegion {
    pub fn allocation_granularity(device: &Arc<CudaDevice>) -> Result<usize, GpuRegionError> {
        let driver = CudaVmmDriver(device.clone());
        driver.bind()?;
        driver.granularity()
    }

    /// Reserve addresses only. No physical bytes are committed until `grow_to`.
    /// Splitting these steps keeps partial growth failures owned by the caller.
    pub fn reserve(device: Arc<CudaDevice>, virtual_bytes: usize) -> Result<Self, GpuRegionError> {
        Ok(Self {
            inner: VmmRegion::reserve(CudaVmmDriver(device), virtual_bytes)?,
        })
    }

    pub fn granularity(&self) -> usize {
        self.inner.granularity
    }

    /// Explicit capacity operation: create, map, clear, and fence a new segment.
    /// The returned segment is ready in this context only. A failed operation
    /// can retain physical handles; inspect the snapshot and retry cleanup
    /// before relinquishing the incremental budget grant.
    pub fn grow_to(&self, target_bytes: usize) -> Result<GpuVmmSegment, GpuRegionError> {
        self.inner.grow_to(target_bytes)
    }

    pub fn segments(&self) -> Vec<GpuVmmSegment> {
        self.inner
            .observation
            .state
            .lock()
            .unwrap()
            .segments
            .iter()
            .map(|s| s.descriptor)
            .collect()
    }

    pub fn tail_segments(&self, target_bytes: usize) -> Result<Vec<GpuVmmSegment>, GpuRegionError> {
        self.inner.tail_segments(target_bytes)
    }

    /// Largest whole-segment boundary at or below a proposed shrink target.
    /// Workload minimum capacity is enforced by the runtime, not HAL.
    pub fn shrink_boundary(&self, requested_bytes: usize) -> Result<usize, GpuRegionError> {
        self.inner.shrink_boundary(requested_bytes)
    }

    pub fn export_segment(&self, id: GpuVmmSegmentId) -> Result<OwnedFd, GpuRegionError> {
        self.inner.export_segment(id)
    }

    pub fn base_ptr(&self) -> Result<*mut std::ffi::c_void, GpuRegionError> {
        self.inner
            .ready_pointer()
            .map(|pointer| pointer as *mut std::ffi::c_void)
    }

    pub fn observer(&self) -> Arc<dyn GpuRegionObserver> {
        self.inner.observation.clone()
    }

    /// # Safety
    /// The retired tail must no longer be schedulable, referenced by local
    /// users, or mapped by any importer. All exported file descriptors and
    /// imported allocation handles for the tail must be closed. The caller must
    /// collect all required GPU fences and worker unmap acknowledgments first.
    pub unsafe fn release_tail_after_fence(
        &self,
        target_bytes: usize,
    ) -> Result<(), GpuRegionError> {
        self.inner.release_tail(target_bytes)
    }

    /// # Safety
    /// All users and importers of the complete region must be fenced and all
    /// exported file descriptors, imported mappings and allocation handles
    /// released. Local synchronization is not evidence of completion in another
    /// process.
    pub unsafe fn release_after_fence(&self) -> Result<(), GpuRegionError> {
        self.inner.release()
    }
}

impl GpuRegion for GpuVmmRegion {
    fn kind(&self) -> GpuRegionKind {
        GpuRegionKind::Vmm
    }
    fn physical_snapshot(&self) -> GpuRegionSnapshot {
        self.inner.observation.physical_snapshot()
    }
}

struct Segment {
    descriptor: GpuVmmSegment,
    handle: u64,
}

#[derive(Default)]
struct VmmState {
    segments: Vec<Segment>,
    next_sequence: u64,
    exported: bool,
    released: bool,
    release_requested: bool,
    cleanup_target: Option<usize>,
}

impl VmmState {
    fn end(&self) -> usize {
        self.segments
            .last()
            .map(|s| s.descriptor.offset_bytes + s.descriptor.length_bytes)
            .unwrap_or(0)
    }

    fn check_live(&self) -> Result<(), GpuRegionError> {
        if self.released {
            Err(GpuRegionError::Released)
        } else {
            Ok(())
        }
    }

    fn check_ready(&self) -> Result<(), GpuRegionError> {
        self.check_live()?;
        if self.release_requested
            || self.cleanup_target.is_some()
            || self.segments.iter().any(|s| !s.descriptor.ready)
        {
            Err(GpuRegionError::NotReady)
        } else {
            Ok(())
        }
    }

    fn validate_boundary(&self, target: usize) -> Result<(), GpuRegionError> {
        self.check_live()?;
        if target > self.end()
            || (target != 0
                && !self
                    .segments
                    .iter()
                    .any(|s| s.descriptor.offset_bytes + s.descriptor.length_bytes == target))
        {
            return Err(GpuRegionError::InvalidRequest(
                "tail target must be a committed segment boundary",
            ));
        }
        Ok(())
    }
}

struct VmmObservation {
    device_id: usize,
    virtual_bytes: usize,
    state: Mutex<VmmState>,
}

impl GpuRegionObserver for VmmObservation {
    fn physical_snapshot(&self) -> GpuRegionSnapshot {
        let state = self.state.lock().unwrap();
        let mut ready_bytes = 0;
        let mut snapshot = GpuRegionSnapshot {
            device_id: self.device_id,
            kind: GpuRegionKind::Vmm,
            committed_bytes: 0,
            mapped_bytes: 0,
            ready_bytes: Some(0),
            virtual_reserved_bytes: if state.released {
                0
            } else {
                self.virtual_bytes
            },
            released: state.released,
        };
        for segment in &state.segments {
            let descriptor = segment.descriptor;
            snapshot.committed_bytes += descriptor.length_bytes;
            if descriptor.mapped {
                snapshot.mapped_bytes += descriptor.length_bytes;
            }
            if descriptor.ready {
                ready_bytes += descriptor.length_bytes;
            }
        }
        snapshot.ready_bytes = Some(ready_bytes);
        snapshot
    }
}

trait VmmDriver: Send + Sync {
    fn device_id(&self) -> usize;
    fn bind(&self) -> Result<(), GpuRegionError>;
    fn granularity(&self) -> Result<usize, GpuRegionError>;
    fn reserve(&self, bytes: usize, alignment: usize) -> Result<u64, GpuRegionError>;
    fn create(&self, bytes: usize) -> Result<u64, GpuRegionError>;
    fn map(&self, pointer: u64, bytes: usize, handle: u64) -> Result<(), GpuRegionError>;
    fn initialize(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError>;
    fn synchronize(&self) -> Result<(), GpuRegionError>;
    fn unmap(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError>;
    fn release_handle(&self, handle: u64) -> Result<(), GpuRegionError>;
    fn free_address(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError>;
    fn export(&self, handle: u64) -> Result<OwnedFd, GpuRegionError>;
    fn retain_context_on_leak(&self);
}

struct VmmRegion<D: VmmDriver> {
    driver: D,
    region_id: u64,
    pointer: u64,
    granularity: usize,
    observation: Arc<VmmObservation>,
}

impl<D: VmmDriver> VmmRegion<D> {
    fn reserve(driver: D, virtual_bytes: usize) -> Result<Self, GpuRegionError> {
        if virtual_bytes == 0 {
            return Err(GpuRegionError::InvalidRequest(
                "virtual capacity must be nonzero",
            ));
        }
        driver.bind()?;
        let granularity = driver.granularity()?;
        if granularity == 0 || !virtual_bytes.is_multiple_of(granularity) {
            return Err(GpuRegionError::InvalidRequest(
                "virtual capacity must align to device granularity",
            ));
        }
        let region_id = NEXT_REGION_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| GpuRegionError::InvalidRequest("region identity exhausted"))?;
        let pointer = driver.reserve(virtual_bytes, granularity)?;
        let device_id = driver.device_id();
        Ok(Self {
            driver,
            region_id,
            pointer,
            granularity,
            observation: Arc::new(VmmObservation {
                device_id,
                virtual_bytes,
                state: Mutex::new(VmmState::default()),
            }),
        })
    }

    fn grow_to(&self, target: usize) -> Result<GpuVmmSegment, GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        state.check_ready()?;
        let previous = state.end();
        if target <= previous
            || target > self.observation.virtual_bytes
            || !target.is_multiple_of(self.granularity)
        {
            return Err(GpuRegionError::InvalidRequest(
                "growth target is outside or misaligned with virtual capacity",
            ));
        }
        let address = self
            .pointer
            .checked_add(previous as u64)
            .ok_or(GpuRegionError::InvalidRequest("VMM address overflow"))?;
        self.pointer
            .checked_add(target as u64)
            .ok_or(GpuRegionError::InvalidRequest("VMM address range overflow"))?;
        let sequence = state
            .next_sequence
            .checked_add(1)
            .ok_or(GpuRegionError::InvalidRequest("segment identity exhausted"))?;
        state.next_sequence = sequence;
        let length = target - previous;
        self.driver.bind()?;
        let handle = self.driver.create(length)?;
        state.segments.push(Segment {
            descriptor: GpuVmmSegment {
                id: GpuVmmSegmentId {
                    region: self.region_id,
                    sequence,
                },
                offset_bytes: previous,
                length_bytes: length,
                mapped: false,
                ready: false,
            },
            handle,
        });
        let initialized = (|| {
            self.driver.map(address, length, handle)?;
            state.segments.last_mut().unwrap().descriptor.mapped = true;
            self.driver.initialize(address, length)?;
            self.driver.synchronize()?;
            state.segments.last_mut().unwrap().descriptor.ready = true;
            Ok(())
        })();
        if let Err(operation) = initialized {
            return match self.cleanup_to(&mut state, previous) {
                Ok(()) => Err(operation),
                Err(cleanup) => Err(GpuRegionError::Rollback {
                    operation: Box::new(operation),
                    cleanup: Box::new(cleanup),
                }),
            };
        }
        Ok(state.segments.last().unwrap().descriptor)
    }

    fn tail_segments(&self, target: usize) -> Result<Vec<GpuVmmSegment>, GpuRegionError> {
        let state = self.observation.state.lock().unwrap();
        state.validate_boundary(target)?;
        Ok(state
            .segments
            .iter()
            .filter(|s| s.descriptor.offset_bytes >= target)
            .map(|s| s.descriptor)
            .collect())
    }

    fn shrink_boundary(&self, requested: usize) -> Result<usize, GpuRegionError> {
        let state = self.observation.state.lock().unwrap();
        state.check_ready()?;
        if requested > state.end() {
            return Err(GpuRegionError::InvalidRequest(
                "shrink request exceeds committed capacity",
            ));
        }
        Ok(state
            .segments
            .iter()
            .map(|s| s.descriptor.offset_bytes + s.descriptor.length_bytes)
            .filter(|end| *end <= requested)
            .max()
            .unwrap_or(0))
    }

    fn ready_pointer(&self) -> Result<u64, GpuRegionError> {
        let state = self.observation.state.lock().unwrap();
        state.check_ready()?;
        if state.segments.is_empty() {
            return Err(GpuRegionError::NotReady);
        }
        Ok(self.pointer)
    }

    fn export_segment(&self, id: GpuVmmSegmentId) -> Result<OwnedFd, GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        state.check_ready()?;
        let segment = state
            .segments
            .iter()
            .find(|s| s.descriptor.id == id)
            .ok_or(GpuRegionError::InvalidSegment)?;
        self.driver.bind()?;
        let fd = self.driver.export(segment.handle)?;
        state.exported = true;
        Ok(fd)
    }

    // On error, preserve the last successful mapping/handle state so retry
    // never repeats a completed unmap or loses a physical charge.
    fn cleanup_to(&self, state: &mut VmmState, target: usize) -> Result<(), GpuRegionError> {
        state.cleanup_target = Some(target);
        self.driver.synchronize()?;
        while state
            .segments
            .last()
            .is_some_and(|s| s.descriptor.offset_bytes >= target)
        {
            let segment = state.segments.last_mut().unwrap();
            if segment.descriptor.mapped {
                self.driver.unmap(
                    self.pointer + segment.descriptor.offset_bytes as u64,
                    segment.descriptor.length_bytes,
                )?;
                segment.descriptor.mapped = false;
                segment.descriptor.ready = false;
            }
            self.driver.release_handle(segment.handle)?;
            state.segments.pop();
        }
        state.cleanup_target = None;
        Ok(())
    }

    fn release_tail(&self, target: usize) -> Result<(), GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        state.validate_boundary(target)?;
        if state.release_requested {
            return Err(GpuRegionError::NotReady);
        }
        if state
            .cleanup_target
            .is_some_and(|pending| pending != target)
        {
            return Err(GpuRegionError::InvalidRequest(
                "retry cleanup at its original boundary",
            ));
        }
        self.driver.bind()?;
        self.cleanup_to(&mut state, target)
    }

    fn release(&self) -> Result<(), GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        if state.released {
            return Ok(());
        }
        state.release_requested = true;
        self.driver.bind()?;
        self.cleanup_to(&mut state, 0)?;
        self.driver
            .free_address(self.pointer, self.observation.virtual_bytes)?;
        state.released = true;
        Ok(())
    }
}

impl<D: VmmDriver> Drop for VmmRegion<D> {
    fn drop(&mut self) {
        let (released, exported) = {
            let state = self.observation.state.lock().unwrap();
            (state.released, state.exported)
        };
        if released {
            return;
        }
        if exported {
            self.driver.retain_context_on_leak();
            log::error!(
                "exported CUDA VMM region dropped without fenced release; backing retained"
            );
        } else if let Err(error) = self.release() {
            self.driver.retain_context_on_leak();
            log::error!("CUDA VMM region cleanup failed; backing retained: {error}");
        }
    }
}

struct CudaVmmDriver(Arc<CudaDevice>);

impl CudaVmmDriver {
    fn properties(&self) -> Result<sys::CUmemAllocationProp, GpuRegionError> {
        let id = i32::try_from(self.0.ordinal())
            .map_err(|_| GpuRegionError::InvalidRequest("device ordinal exceeds CUDA range"))?;
        Ok(sys::CUmemAllocationProp {
            type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes:
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id,
            },
            ..Default::default()
        })
    }
}

impl VmmDriver for CudaVmmDriver {
    fn device_id(&self) -> usize {
        self.0.ordinal()
    }
    fn bind(&self) -> Result<(), GpuRegionError> {
        self.0
            .bind_to_thread()
            .map_err(driver_error("bind VMM context"))
    }
    fn granularity(&self) -> Result<usize, GpuRegionError> {
        let properties = self.properties()?;
        let mut granularity = 0;
        unsafe {
            sys::lib()
                .cuMemGetAllocationGranularity(
                    &mut granularity,
                    &properties,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
                .result()
        }
        .map_err(driver_error("query VMM granularity"))?;
        if granularity == 0 {
            return Err(GpuRegionError::InvalidRequest(
                "CUDA returned zero granularity",
            ));
        }
        Ok(granularity)
    }
    fn reserve(&self, bytes: usize, alignment: usize) -> Result<u64, GpuRegionError> {
        let mut pointer = 0;
        unsafe {
            sys::lib()
                .cuMemAddressReserve(&mut pointer, bytes, alignment, 0, 0)
                .result()
        }
        .map_err(driver_error("reserve VMM addresses"))?;
        Ok(pointer)
    }
    fn create(&self, bytes: usize) -> Result<u64, GpuRegionError> {
        let mut handle = 0;
        let properties = self.properties()?;
        unsafe {
            sys::lib()
                .cuMemCreate(&mut handle, bytes, &properties, 0)
                .result()
        }
        .map_err(driver_error("create VMM backing"))?;
        Ok(handle)
    }
    fn map(&self, pointer: u64, bytes: usize, handle: u64) -> Result<(), GpuRegionError> {
        unsafe { sys::lib().cuMemMap(pointer, bytes, 0, handle, 0).result() }
            .map_err(driver_error("map VMM segment"))
    }
    fn initialize(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError> {
        let access = sys::CUmemAccessDesc {
            location: self.properties()?.location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        unsafe {
            sys::lib()
                .cuMemSetAccess(pointer, bytes, &access, 1)
                .result()
        }
        .map_err(driver_error("set VMM segment access"))?;
        unsafe { result::memset_d8_async(pointer, 0, bytes, *self.0.cu_stream()) }
            .map_err(driver_error("zero VMM segment"))
    }
    fn synchronize(&self) -> Result<(), GpuRegionError> {
        self.0
            .synchronize()
            .map_err(driver_error("synchronize VMM backing"))
    }
    fn unmap(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError> {
        unsafe { sys::lib().cuMemUnmap(pointer, bytes).result() }
            .map_err(driver_error("unmap VMM segment"))
    }
    fn release_handle(&self, handle: u64) -> Result<(), GpuRegionError> {
        unsafe { sys::lib().cuMemRelease(handle).result() }
            .map_err(driver_error("release VMM handle"))
    }
    fn free_address(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError> {
        unsafe { sys::lib().cuMemAddressFree(pointer, bytes).result() }
            .map_err(driver_error("free VMM addresses"))
    }
    fn export(&self, handle: u64) -> Result<OwnedFd, GpuRegionError> {
        let mut fd = -1i32;
        unsafe {
            sys::lib()
                .cuMemExportToShareableHandle(
                    (&mut fd as *mut i32).cast(),
                    handle,
                    sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                )
                .result()
        }
        .map_err(driver_error("export VMM handle"))?;
        if fd < 0 {
            return Err(GpuRegionError::InvalidRequest(
                "CUDA returned an invalid file descriptor",
            ));
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
    fn retain_context_on_leak(&self) {
        std::mem::forget(self.0.clone());
    }
}

#[cfg(test)]
#[path = "gpu_vmm_region_tests.rs"]
mod tests;
