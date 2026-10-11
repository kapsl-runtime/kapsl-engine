use super::*;
use std::collections::VecDeque;

#[derive(Default)]
struct State {
    calls: Vec<&'static str>,
    failures: VecDeque<&'static str>,
    live: bool,
    pending_zero: bool,
}

impl State {
    fn step(&mut self, operation: &'static str) -> Result<(), GpuRegionError> {
        self.calls.push(operation);
        if self.failures.front() == Some(&operation) {
            self.failures.pop_front();
            return Err(GpuRegionError::Driver {
                operation,
                message: "injected failure".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Driver(Arc<Mutex<State>>);

impl IpcDriver for Driver {
    fn device_id(&self) -> usize {
        3
    }
    fn bind(&self) -> Result<(), GpuRegionError> {
        self.0.lock().unwrap().step("bind")
    }
    fn allocate(&self, _: usize) -> Result<u64, GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("allocate")?;
        assert!(!state.live);
        state.live = true;
        Ok(4096)
    }
    fn zero(&self, _: u64, _: usize) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("zero")?;
        assert!(state.live);
        state.pending_zero = true;
        Ok(())
    }
    fn synchronize(&self) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("synchronize")?;
        state.pending_zero = false;
        Ok(())
    }
    fn export(&self, _: u64) -> Result<GpuIpcHandle, GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("export")?;
        assert!(state.live && !state.pending_zero);
        Ok(GpuIpcHandle([7; 64]))
    }
    fn free(&self, _: u64) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("free")?;
        assert!(state.live && !state.pending_zero);
        state.live = false;
        Ok(())
    }
    fn retain_context_on_leak(&self) {
        self.0.lock().unwrap().calls.push("retain_context");
    }
}

#[test]
fn initialization_must_complete_before_pointer_or_handle_publication() {
    let driver = Driver::default();
    let region = IpcRegion::allocate(driver.clone(), 64).unwrap();
    let observer = region.observation.clone();
    assert_eq!(observer.physical_snapshot().committed_bytes, 64);
    assert_eq!(observer.physical_snapshot().ready_bytes, Some(0));
    assert!(matches!(
        region.export_handle(),
        Err(GpuRegionError::NotReady)
    ));
    assert!(matches!(
        region.ready_pointer(),
        Err(GpuRegionError::NotReady)
    ));
    driver.0.lock().unwrap().failures.push_back("synchronize");
    assert!(region.initialize_zeroed().is_err());
    assert!(driver.0.lock().unwrap().pending_zero);
    assert_eq!(observer.physical_snapshot().ready_bytes, Some(0));
    assert_eq!(observer.physical_snapshot().committed_bytes, 64);
    assert!(matches!(
        region.export_handle(),
        Err(GpuRegionError::NotReady)
    ));
    region.initialize_zeroed().unwrap();
    assert_eq!(observer.physical_snapshot().ready_bytes, Some(64));
    assert_eq!(region.ready_pointer().unwrap(), 4096);
    assert_eq!(region.export_handle().unwrap().as_bytes(), &[7; 64]);
    // Reinitialization cannot clear memory that has been published to a worker.
    let before = driver.0.lock().unwrap().calls.len();
    region.initialize_zeroed().unwrap();
    assert_eq!(driver.0.lock().unwrap().calls.len(), before);
    region.release().unwrap();
    assert!(observer.physical_snapshot().released);
}

#[test]
fn failed_release_retains_backing_and_retry_is_idempotent() {
    let driver = Driver::default();
    let region = IpcRegion::allocate(driver.clone(), 64).unwrap();
    region.initialize_zeroed().unwrap();
    region.export_handle().unwrap();
    driver.0.lock().unwrap().failures.push_back("free");
    assert!(region.release().is_err());
    assert!(driver.0.lock().unwrap().live);
    assert_eq!(region.observation.physical_snapshot().committed_bytes, 64);
    region.release().unwrap();
    let calls = driver.0.lock().unwrap().calls.clone();
    region.release().unwrap();
    assert_eq!(driver.0.lock().unwrap().calls, calls);
    assert!(matches!(
        region.export_handle(),
        Err(GpuRegionError::Released)
    ));
    assert!(matches!(
        region.zero_range(0, 1),
        Err(GpuRegionError::Released)
    ));
}

#[test]
fn invalid_ranges_do_not_issue_driver_operations() {
    let driver = Driver::default();
    assert!(IpcRegion::allocate(driver.clone(), 0).is_err());
    assert!(driver.0.lock().unwrap().calls.is_empty());
    let region = IpcRegion::allocate(driver.clone(), 64).unwrap();
    region.initialize_zeroed().unwrap();
    let before = driver.0.lock().unwrap().calls.clone();
    for (offset, bytes) in [(0, 0), (64, 1), (usize::MAX, 2)] {
        assert!(region.zero_range(offset, bytes).is_err());
    }
    assert_eq!(driver.0.lock().unwrap().calls, before);
}

#[test]
fn observations_do_not_keep_unexported_backing_alive() {
    let driver = Driver::default();
    let region = IpcRegion::allocate(driver.clone(), 64).unwrap();
    let observer = region.observation.clone();
    drop(region);
    assert!(!driver.0.lock().unwrap().live);
    assert!(observer.physical_snapshot().released);
    assert_eq!(observer.physical_snapshot().committed_bytes, 0);
}

#[test]
fn dropping_an_exported_region_is_not_an_importer_fence() {
    let driver = Driver::default();
    let region = IpcRegion::allocate(driver.clone(), 64).unwrap();
    region.initialize_zeroed().unwrap();
    region.export_handle().unwrap();
    let observer = region.observation.clone();
    drop(region);
    let state = driver.0.lock().unwrap();
    assert!(state.live);
    assert!(state.calls.contains(&"retain_context"));
    assert!(!state.calls.contains(&"free"));
    assert_eq!(observer.physical_snapshot().committed_bytes, 64);
}

#[test]
fn failed_reassignment_zero_disables_further_exports() {
    let driver = Driver::default();
    let region = IpcRegion::allocate(driver.clone(), 64).unwrap();
    region.initialize_zeroed().unwrap();
    region.export_handle().unwrap();
    driver.0.lock().unwrap().failures.push_back("synchronize");
    assert!(region.zero_range(0, 16).is_err());
    assert_eq!(region.observation.physical_snapshot().ready_bytes, Some(0));
    assert!(matches!(
        region.export_handle(),
        Err(GpuRegionError::NotReady)
    ));
    assert!(matches!(
        region.initialize_zeroed(),
        Err(GpuRegionError::NotReady)
    ));
    region.release().unwrap();
}

#[test]
#[ignore = "requires CUDA hardware; no workers import the handle in this smoke test"]
fn hardware_ipc_initialization_export_and_release() {
    let device = CudaDevice::new_with_stream(0).unwrap();
    let region = GpuIpcRegion::allocate(device, 4096).unwrap();
    region.initialize_zeroed().unwrap();
    assert_eq!(region.physical_snapshot().ready_bytes, Some(4096));
    let observer = region.observer();
    assert_eq!(region.export_handle().unwrap().as_bytes().len(), 64);
    // SAFETY: no local pointers or imported mappings were created.
    unsafe {
        region.release_after_fence().unwrap();
    }
    assert!(observer.physical_snapshot().released);
}
