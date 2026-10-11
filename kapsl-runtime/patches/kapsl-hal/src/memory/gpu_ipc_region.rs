//! Dedicated fixed CUDA IPC backing, without admission or participant policy.

use crate::gpu_region::{
    driver_error, GpuRegion, GpuRegionError, GpuRegionKind, GpuRegionObserver, GpuRegionSnapshot,
};
use cudarc::driver::{result, sys, CudaDevice};
use std::sync::{Arc, Mutex};

/// Raw CUDA IPC handle. Encoding it for a worker protocol belongs to the adapter.
#[derive(Clone, PartialEq, Eq)]
pub struct GpuIpcHandle([u8; 64]);

impl GpuIpcHandle {
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }
}

impl std::fmt::Debug for GpuIpcHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuIpcHandle").finish_non_exhaustive()
    }
}

/// Fixed, isolated backing. Allocate under an existing budget grant, initialize
/// it, then export. Retain both the region and grant until explicit release
/// succeeds. A failed initialization leaves the backing owned and observable.
pub struct GpuIpcRegion {
    inner: IpcRegion<CudaIpcDriver>,
}

impl GpuIpcRegion {
    /// Allocate uninitialized backing. Export and local access remain disabled
    /// until `initialize_zeroed` has completed its CUDA fence.
    pub fn allocate(device: Arc<CudaDevice>, bytes: usize) -> Result<Self, GpuRegionError> {
        Ok(Self {
            inner: IpcRegion::allocate(CudaIpcDriver(device), bytes)?,
        })
    }

    pub fn initialize_zeroed(&self) -> Result<(), GpuRegionError> {
        self.inner.initialize_zeroed()
    }

    pub fn export_handle(&self) -> Result<GpuIpcHandle, GpuRegionError> {
        self.inner.export_handle()
    }

    pub fn base_ptr(&self) -> Result<*mut std::ffi::c_void, GpuRegionError> {
        self.inner
            .ready_pointer()
            .map(|pointer| pointer as *mut std::ffi::c_void)
    }

    pub fn observer(&self) -> Arc<dyn GpuRegionObserver> {
        self.inner.observation.clone()
    }

    /// Clear a leased extent before assigning it to another owner.
    ///
    /// # Safety
    /// No process or stream may access this range until the call completes.
    /// The caller must establish all cross-stream and importer dependencies.
    pub unsafe fn zero_range_after_fence(
        &self,
        offset: usize,
        bytes: usize,
    ) -> Result<(), GpuRegionError> {
        self.inner.zero_range(offset, bytes)
    }

    /// # Safety
    /// All importers must have closed their mappings and all users of local
    /// pointers must be fenced. A local CUDA synchronization alone cannot
    /// establish that another process has finished using the backing.
    pub unsafe fn release_after_fence(&self) -> Result<(), GpuRegionError> {
        self.inner.release()
    }
}

impl GpuRegion for GpuIpcRegion {
    fn kind(&self) -> GpuRegionKind {
        GpuRegionKind::Ipc
    }
    fn physical_snapshot(&self) -> GpuRegionSnapshot {
        self.inner.observation.physical_snapshot()
    }
}

#[derive(Default)]
struct IpcState {
    initialized: bool,
    exported: bool,
    released: bool,
}

struct IpcObservation {
    device_id: usize,
    bytes: usize,
    state: Mutex<IpcState>,
}

impl GpuRegionObserver for IpcObservation {
    fn physical_snapshot(&self) -> GpuRegionSnapshot {
        let state = self.state.lock().unwrap();
        let committed = if state.released { 0 } else { self.bytes };
        GpuRegionSnapshot {
            device_id: self.device_id,
            kind: GpuRegionKind::Ipc,
            committed_bytes: committed,
            mapped_bytes: committed,
            ready_bytes: Some(if state.initialized { committed } else { 0 }),
            virtual_reserved_bytes: 0,
            released: state.released,
        }
    }
}

trait IpcDriver: Send + Sync {
    fn device_id(&self) -> usize;
    fn bind(&self) -> Result<(), GpuRegionError>;
    fn allocate(&self, bytes: usize) -> Result<u64, GpuRegionError>;
    fn zero(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError>;
    fn synchronize(&self) -> Result<(), GpuRegionError>;
    fn export(&self, pointer: u64) -> Result<GpuIpcHandle, GpuRegionError>;
    fn free(&self, pointer: u64) -> Result<(), GpuRegionError>;
    fn retain_context_on_leak(&self);
}

struct IpcRegion<D: IpcDriver> {
    driver: D,
    pointer: u64,
    observation: Arc<IpcObservation>,
}

impl<D: IpcDriver> IpcRegion<D> {
    fn allocate(driver: D, bytes: usize) -> Result<Self, GpuRegionError> {
        if bytes == 0 {
            return Err(GpuRegionError::InvalidRequest("capacity must be nonzero"));
        }
        driver.bind()?;
        let pointer = driver.allocate(bytes)?;
        let device_id = driver.device_id();
        Ok(Self {
            driver,
            pointer,
            observation: Arc::new(IpcObservation {
                device_id,
                bytes,
                state: Mutex::new(IpcState::default()),
            }),
        })
    }

    fn initialize_zeroed(&self) -> Result<(), GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        if state.released {
            return Err(GpuRegionError::Released);
        }
        if state.initialized {
            return Ok(());
        }
        if state.exported {
            return Err(GpuRegionError::NotReady);
        }
        self.driver.bind()?;
        self.driver.zero(self.pointer, self.observation.bytes)?;
        self.driver.synchronize()?;
        state.initialized = true;
        Ok(())
    }

    fn ready_pointer(&self) -> Result<u64, GpuRegionError> {
        let state = self.observation.state.lock().unwrap();
        if state.released {
            return Err(GpuRegionError::Released);
        }
        if !state.initialized {
            return Err(GpuRegionError::NotReady);
        }
        Ok(self.pointer)
    }

    fn export_handle(&self) -> Result<GpuIpcHandle, GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        if state.released {
            return Err(GpuRegionError::Released);
        }
        if !state.initialized {
            return Err(GpuRegionError::NotReady);
        }
        self.driver.bind()?;
        let handle = self.driver.export(self.pointer)?;
        state.exported = true;
        Ok(handle)
    }

    fn zero_range(&self, offset: usize, bytes: usize) -> Result<(), GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        if state.released {
            return Err(GpuRegionError::Released);
        }
        if !state.initialized {
            return Err(GpuRegionError::NotReady);
        }
        if bytes == 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > self.observation.bytes)
        {
            return Err(GpuRegionError::InvalidRequest(
                "zero range is outside backing",
            ));
        }
        let pointer = self
            .pointer
            .checked_add(offset as u64)
            .ok_or(GpuRegionError::InvalidRequest("device pointer overflow"))?;
        self.driver.bind()?;
        state.initialized = false;
        self.driver.zero(pointer, bytes)?;
        self.driver.synchronize()?;
        state.initialized = true;
        Ok(())
    }

    fn release(&self) -> Result<(), GpuRegionError> {
        let mut state = self.observation.state.lock().unwrap();
        if state.released {
            return Ok(());
        }
        self.driver.bind()?;
        self.driver.synchronize()?;
        self.driver.free(self.pointer)?;
        state.released = true;
        Ok(())
    }
}

impl<D: IpcDriver> Drop for IpcRegion<D> {
    fn drop(&mut self) {
        let (released, exported) = {
            let state = self.observation.state.lock().unwrap();
            (state.released, state.exported)
        };
        if released {
            return;
        }
        if exported {
            // Drop is not an importer fence. Preserve the context as well as
            // the allocation when a caller abandons exported backing.
            self.driver.retain_context_on_leak();
            log::error!(
                "exported CUDA IPC region dropped without fenced release; backing retained"
            );
        } else if let Err(error) = self.release() {
            self.driver.retain_context_on_leak();
            log::error!("CUDA IPC region cleanup failed; backing retained: {error}");
        }
    }
}

struct CudaIpcDriver(Arc<CudaDevice>);

impl IpcDriver for CudaIpcDriver {
    fn device_id(&self) -> usize {
        self.0.ordinal()
    }
    fn bind(&self) -> Result<(), GpuRegionError> {
        self.0
            .bind_to_thread()
            .map_err(driver_error("bind IPC context"))
    }
    fn allocate(&self, bytes: usize) -> Result<u64, GpuRegionError> {
        // IPC requires a legacy allocation, not cudaMallocAsync pool storage.
        unsafe { result::malloc_sync(bytes) }.map_err(driver_error("allocate IPC backing"))
    }
    fn zero(&self, pointer: u64, bytes: usize) -> Result<(), GpuRegionError> {
        unsafe { result::memset_d8_async(pointer, 0, bytes, *self.0.cu_stream()) }
            .map_err(driver_error("zero IPC backing"))
    }
    fn synchronize(&self) -> Result<(), GpuRegionError> {
        self.0
            .synchronize()
            .map_err(driver_error("synchronize IPC backing"))
    }
    fn export(&self, pointer: u64) -> Result<GpuIpcHandle, GpuRegionError> {
        let mut handle = sys::CUipcMemHandle::default();
        unsafe { sys::lib().cuIpcGetMemHandle(&mut handle, pointer).result() }
            .map_err(driver_error("export IPC handle"))?;
        Ok(GpuIpcHandle(handle.reserved.map(|byte| byte as u8)))
    }
    fn free(&self, pointer: u64) -> Result<(), GpuRegionError> {
        unsafe { result::free_sync(pointer) }.map_err(driver_error("free IPC backing"))
    }
    fn retain_context_on_leak(&self) {
        std::mem::forget(self.0.clone());
    }
}

#[cfg(test)]
#[path = "gpu_ipc_region_tests.rs"]
mod tests;
