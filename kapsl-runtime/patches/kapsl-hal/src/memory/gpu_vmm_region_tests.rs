use super::*;
use std::collections::{BTreeMap, HashMap, VecDeque};

#[derive(Default)]
struct State {
    calls: Vec<&'static str>,
    failures: VecDeque<(&'static str, usize)>,
    handles: HashMap<u64, usize>,
    mappings: BTreeMap<u64, u64>,
    next_handle: u64,
    reserved: bool,
}

impl State {
    fn step(&mut self, operation: &'static str) -> Result<(), GpuRegionError> {
        self.calls.push(operation);
        if let Some((expected, skip)) = self.failures.front_mut() {
            if *expected == operation {
                if *skip == 0 {
                    self.failures.pop_front();
                    return Err(GpuRegionError::Driver {
                        operation,
                        message: "injected failure".into(),
                    });
                }
                *skip -= 1;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Driver(Arc<Mutex<State>>);

impl Driver {
    fn fail(&self, operation: &'static str, skip: usize) {
        self.0.lock().unwrap().failures.push_back((operation, skip));
    }
}

impl VmmDriver for Driver {
    fn device_id(&self) -> usize {
        5
    }
    fn bind(&self) -> Result<(), GpuRegionError> {
        self.0.lock().unwrap().step("bind")
    }
    fn granularity(&self) -> Result<usize, GpuRegionError> {
        Ok(64)
    }
    fn reserve(&self, _: usize, _: usize) -> Result<u64, GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("reserve")?;
        assert!(!state.reserved);
        state.reserved = true;
        Ok(4096)
    }
    fn create(&self, bytes: usize) -> Result<u64, GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("create")?;
        state.next_handle += 1;
        let handle = state.next_handle;
        state.handles.insert(handle, bytes);
        Ok(handle)
    }
    fn map(&self, pointer: u64, bytes: usize, handle: u64) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("map")?;
        assert_eq!(state.handles.get(&handle), Some(&bytes));
        assert!(state.mappings.insert(pointer, handle).is_none());
        Ok(())
    }
    fn initialize(&self, pointer: u64, _: usize) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        assert!(state.mappings.contains_key(&pointer));
        state.step("initialize")
    }
    fn synchronize(&self) -> Result<(), GpuRegionError> {
        self.0.lock().unwrap().step("synchronize")
    }
    fn unmap(&self, pointer: u64, _: usize) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("unmap")?;
        assert!(
            state.mappings.remove(&pointer).is_some(),
            "must not unmap twice"
        );
        Ok(())
    }
    fn release_handle(&self, handle: u64) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("release_handle")?;
        assert!(!state.mappings.values().any(|live| *live == handle));
        assert!(
            state.handles.remove(&handle).is_some(),
            "must not release twice"
        );
        Ok(())
    }
    fn free_address(&self, _: u64, _: usize) -> Result<(), GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("free_address")?;
        assert!(state.handles.is_empty() && state.mappings.is_empty());
        assert!(state.reserved);
        state.reserved = false;
        Ok(())
    }
    fn export(&self, handle: u64) -> Result<OwnedFd, GpuRegionError> {
        let mut state = self.0.lock().unwrap();
        state.step("export")?;
        assert!(state.handles.contains_key(&handle));
        Ok(std::fs::File::open("/dev/null").unwrap().into())
    }
    fn retain_context_on_leak(&self) {
        self.0.lock().unwrap().calls.push("retain_context");
    }
}

fn region() -> (VmmRegion<Driver>, Driver) {
    let driver = Driver::default();
    let region = VmmRegion::reserve(driver.clone(), 512).unwrap();
    (region, driver)
}

fn assert_physical(
    region: &VmmRegion<Driver>,
    driver: &Driver,
    committed: usize,
    mapped: usize,
    ready: usize,
) {
    let snapshot = region.observation.physical_snapshot();
    assert_eq!(snapshot.committed_bytes, committed);
    assert_eq!(snapshot.mapped_bytes, mapped);
    assert_eq!(snapshot.ready_bytes, Some(ready));
    let state = driver.0.lock().unwrap();
    assert_eq!(state.handles.values().sum::<usize>(), committed);
    assert_eq!(
        state
            .mappings
            .values()
            .map(|handle| state.handles[handle])
            .sum::<usize>(),
        mapped
    );
}

#[test]
fn reservation_growth_and_shrink_keep_one_base_and_distinct_physical_accounting() {
    let (region, driver) = region();
    assert_physical(&region, &driver, 0, 0, 0);
    assert_eq!(
        region
            .observation
            .physical_snapshot()
            .virtual_reserved_bytes,
        512
    );
    assert!(matches!(
        region.ready_pointer(),
        Err(GpuRegionError::NotReady)
    ));
    let first = region.grow_to(64).unwrap();
    let pointer = region.ready_pointer().unwrap();
    let tail = region.grow_to(192).unwrap();
    assert_eq!(tail.offset_bytes, 64);
    assert_eq!(tail.length_bytes, 128);
    assert_eq!(region.ready_pointer().unwrap(), pointer);
    assert_eq!(region.shrink_boundary(128).unwrap(), 64);
    assert_eq!(region.tail_segments(64).unwrap(), vec![tail]);
    assert_physical(&region, &driver, 192, 192, 192);
    region.release_tail(64).unwrap();
    assert_eq!(region.ready_pointer().unwrap(), pointer);
    assert_physical(&region, &driver, 64, 64, 64);
    assert_eq!(region.tail_segments(0).unwrap(), vec![first]);
}

#[test]
fn invalid_growth_and_shrink_do_not_mutate_backing() {
    let (region, driver) = region();
    region.grow_to(64).unwrap();
    region.grow_to(192).unwrap();
    let calls = driver.0.lock().unwrap().calls.clone();
    for target in [0, 128, 192, 513, usize::MAX] {
        assert!(region.grow_to(target).is_err());
    }
    for target in [1, 128, 193, usize::MAX] {
        assert!(region.release_tail(target).is_err());
    }
    assert_eq!(driver.0.lock().unwrap().calls, calls);
    assert_physical(&region, &driver, 192, 192, 192);
}

#[test]
fn mapping_failure_with_successful_rollback_does_not_commit_new_capacity() {
    let (region, driver) = region();
    region.grow_to(64).unwrap();
    driver.fail("map", 0);
    assert!(region.grow_to(192).is_err());
    assert_physical(&region, &driver, 64, 64, 64);
    region.grow_to(192).unwrap();
    assert_physical(&region, &driver, 192, 192, 192);
}

#[test]
fn failed_mapping_and_handle_cleanup_retain_unmapped_backing() {
    let (region, driver) = region();
    region.grow_to(64).unwrap();
    driver.fail("map", 0);
    driver.fail("release_handle", 0);
    assert!(matches!(
        region.grow_to(192),
        Err(GpuRegionError::Rollback { .. })
    ));
    assert_physical(&region, &driver, 192, 64, 64);
    assert!(matches!(region.grow_to(256), Err(GpuRegionError::NotReady)));
    let failed = region.tail_segments(64).unwrap()[0];
    assert!(matches!(
        region.export_segment(failed.id),
        Err(GpuRegionError::NotReady)
    ));
    region.release_tail(64).unwrap();
    assert_physical(&region, &driver, 64, 64, 64);
    region.grow_to(192).unwrap();
}

#[test]
fn failed_initialization_and_unmap_preserve_mapped_but_unready_backing() {
    let (region, driver) = region();
    region.grow_to(64).unwrap();
    driver.fail("initialize", 0);
    driver.fail("unmap", 0);
    assert!(matches!(
        region.grow_to(128),
        Err(GpuRegionError::Rollback { .. })
    ));
    assert_physical(&region, &driver, 128, 128, 64);
    region.release_tail(64).unwrap();
    assert_physical(&region, &driver, 64, 64, 64);
}

#[test]
fn failed_initialization_fence_and_rollback_fence_never_publish_ready_capacity() {
    let (region, driver) = region();
    driver.fail("synchronize", 0);
    driver.fail("synchronize", 0);
    assert!(matches!(
        region.grow_to(64),
        Err(GpuRegionError::Rollback { .. })
    ));
    assert_physical(&region, &driver, 64, 64, 0);
    assert!(matches!(
        region.ready_pointer(),
        Err(GpuRegionError::NotReady)
    ));
    region.release_tail(0).unwrap();
    assert_physical(&region, &driver, 0, 0, 0);
}

#[test]
fn partially_released_tail_can_retry_without_double_unmap_or_double_free() {
    let (region, driver) = region();
    region.grow_to(64).unwrap();
    region.grow_to(128).unwrap();
    region.grow_to(192).unwrap();
    driver.fail("release_handle", 1); // release the last handle, then fail the next
    assert!(region.release_tail(64).is_err());
    assert_physical(&region, &driver, 128, 64, 64);
    assert!(region.release_tail(0).is_err()); // retry the admitted boundary
    region.release_tail(64).unwrap();
    assert_physical(&region, &driver, 64, 64, 64);
}

#[test]
fn failed_address_release_keeps_reservation_visible_and_blocks_regrowth() {
    let (region, driver) = region();
    region.grow_to(64).unwrap();
    let observer = region.observation.clone();
    driver.fail("free_address", 0);
    assert!(region.release().is_err());
    assert_physical(&region, &driver, 0, 0, 0);
    assert_eq!(observer.physical_snapshot().virtual_reserved_bytes, 512);
    assert!(!observer.physical_snapshot().released);
    assert!(matches!(region.grow_to(64), Err(GpuRegionError::NotReady)));
    region.release().unwrap();
    let calls = driver.0.lock().unwrap().calls.clone();
    region.release().unwrap();
    assert_eq!(driver.0.lock().unwrap().calls, calls);
    assert!(observer.physical_snapshot().released);
    assert_eq!(observer.physical_snapshot().virtual_reserved_bytes, 0);
    assert!(matches!(region.grow_to(64), Err(GpuRegionError::Released)));
}

#[test]
fn stale_and_foreign_segment_ids_cannot_export_reused_offsets() {
    let (region, driver) = region();
    let original = region.grow_to(64).unwrap();
    region.release_tail(0).unwrap();
    let replacement = region.grow_to(64).unwrap();
    assert_ne!(original.id, replacement.id);
    assert_eq!(original.offset_bytes, replacement.offset_bytes);
    let other = VmmRegion::reserve(Driver::default(), 512).unwrap();
    let foreign = other.grow_to(64).unwrap();
    let calls = driver.0.lock().unwrap().calls.clone();
    for id in [original.id, foreign.id] {
        assert!(matches!(
            region.export_segment(id),
            Err(GpuRegionError::InvalidSegment)
        ));
    }
    assert_eq!(driver.0.lock().unwrap().calls, calls);
    drop(region.export_segment(replacement.id).unwrap());
    region.release().unwrap();
}

#[test]
fn dropping_exported_backing_preserves_context_and_all_handles() {
    let (region, driver) = region();
    let segment = region.grow_to(64).unwrap();
    drop(region.export_segment(segment.id).unwrap());
    let observer = region.observation.clone();
    drop(region);
    let state = driver.0.lock().unwrap();
    assert!(state.reserved);
    assert_eq!(state.handles.values().sum::<usize>(), 64);
    assert!(!state.calls.contains(&"unmap"));
    assert!(state.calls.contains(&"retain_context"));
    assert_eq!(observer.physical_snapshot().committed_bytes, 64);
}

#[test]
fn observers_do_not_retain_unexported_allocations() {
    let (region, driver) = region();
    region.grow_to(64).unwrap();
    let observer = region.observation.clone();
    drop(region);
    assert!(!driver.0.lock().unwrap().reserved);
    assert!(observer.physical_snapshot().released);
}

#[test]
#[ignore = "requires Linux CUDA VMM hardware; no remote mappings are created"]
fn hardware_vmm_growth_preserves_base_and_releases_physical_tail() {
    let device = CudaDevice::new_with_stream(0).unwrap();
    let granularity = GpuVmmRegion::allocation_granularity(&device).unwrap();
    let region = GpuVmmRegion::reserve(device, 4 * granularity).unwrap();
    let first = region.grow_to(granularity).unwrap();
    let pointer = region.base_ptr().unwrap();
    drop(region.export_segment(first.id).unwrap());
    region.grow_to(3 * granularity).unwrap();
    assert_eq!(region.base_ptr().unwrap(), pointer);
    // SAFETY: no importers or local kernels use this test region.
    unsafe {
        region.release_tail_after_fence(granularity).unwrap();
    }
    assert_eq!(region.physical_snapshot().committed_bytes, granularity);
    assert_eq!(region.base_ptr().unwrap(), pointer);
    unsafe {
        region.release_after_fence().unwrap();
    }
    assert!(region.physical_snapshot().released);
}
