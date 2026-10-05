use super::super::MemoryOwner;
use super::*;
use kapsl_hal::gpu_arena::PoolAllocationClass;

#[derive(Default)]
struct State {
    capacity: usize,
    next_id: u64,
    allocation: Option<(u64, usize)>,
    fail_free: bool,
    fail_release: bool,
    released: bool,
}

struct FakeRegion {
    kind: GpuRegionKind,
    state: Arc<Mutex<State>>,
}
struct FakeAllocation {
    state: Arc<Mutex<State>>,
    id: u64,
    bytes: usize,
}

fn owner() -> PoolOwner {
    PoolOwner::onnx(1, 2, PoolAllocationClass::TransientWorkspace)
}

impl GpuRegionSource for FakeRegion {
    fn region_usage(&self) -> Option<GpuRegionUsage> {
        let state = self.state.lock();
        if state.released {
            return None;
        }
        let mut result = GpuRegionUsage::exported(state.capacity, state.capacity, 0);
        if self.kind == GpuRegionKind::Arena {
            let used = state.allocation.map(|(_, bytes)| bytes).unwrap_or(0);
            result.logical_allocated_bytes = Some(used);
            result.reusable_bytes = Some(state.capacity - used);
            result.largest_free_range_bytes = result.reusable_bytes;
        }
        Some(result)
    }
}

impl RegionStorage for FakeRegion {
    fn kind(&self) -> GpuRegionKind {
        self.kind
    }
    fn fits_local(&self, candidate: PoolOwner, bytes: usize, alignment: usize) -> bool {
        let state = self.state.lock();
        self.kind == GpuRegionKind::Arena
            && candidate == owner()
            && alignment <= 4096
            && bytes <= state.capacity
            && (bytes == 0 || state.allocation.is_none())
            && !state.released
    }
    fn allocate(
        &self,
        candidate: PoolOwner,
        bytes: usize,
        alignment: usize,
    ) -> Result<Box<dyn AllocationStorage>, String> {
        if !self.fits_local(candidate, bytes, alignment) {
            return Err("no admitted ready extent".into());
        }
        let mut state = self.state.lock();
        state.next_id += 1;
        let id = state.next_id;
        state.allocation = Some((id, bytes));
        Ok(Box::new(FakeAllocation {
            state: self.state.clone(),
            id,
            bytes,
        }))
    }
    fn can_retire_arena(&self) -> bool {
        self.kind == GpuRegionKind::Arena && self.state.lock().allocation.is_none()
    }
    fn release_export_after_fence(&self) -> Result<(), String> {
        let mut state = self.state.lock();
        if state.fail_release {
            return Err("injected handle release failure".into());
        }
        state.released = true;
        Ok(())
    }
}

impl AllocationStorage for FakeAllocation {
    fn pointer(&self) -> usize {
        4096
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn free_after_fence(&self) -> Result<(), String> {
        let mut state = self.state.lock();
        if state.fail_free {
            return Err("injected free failure".into());
        }
        if state.allocation != Some((self.id, self.bytes)) {
            return Err("stale extent".into());
        }
        state.allocation = None;
        Ok(())
    }
}

fn pool(device_id: usize) -> Arc<GpuDevicePool> {
    Arc::new(GpuDevicePool {
        device_id,
        device: None,
        registry: GpuRegionRegistry::new(device_id),
        regions: Mutex::new(BTreeMap::new()),
        next_allocation: AtomicU64::new(1),
    })
}

fn arena(pool: &Arc<GpuDevicePool>, capacity: usize) -> (GpuRegionLease, Arc<Mutex<State>>) {
    let state = Arc::new(Mutex::new(State {
        capacity,
        ..State::default()
    }));
    (
        pool.insert(
            GpuRegionIsolation::Local,
            Box::new(FakeRegion {
                kind: GpuRegionKind::Arena,
                state: state.clone(),
            }),
        ),
        state,
    )
}

fn local(bytes: usize, alignment: usize) -> GpuRegionRequest {
    GpuRegionRequest::Local {
        owner: owner(),
        bytes,
        alignment,
    }
}

fn isolated() -> GpuRegionIsolation {
    GpuRegionIsolation::Participant {
        owner: MemoryOwner::new(4, 5),
        participant_id: "worker".into(),
        binding_id: "kv".into(),
        generation: 7,
    }
}

#[test]
fn arena_retirement_waits_for_a_telemetry_pin_then_releases() {
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    struct ObservedArena {
        entered: Arc<Barrier>,
        resume: Arc<Barrier>,
    }
    impl GpuRegionSource for ObservedArena {
        fn region_usage(&self) -> Option<GpuRegionUsage> {
            self.entered.wait();
            self.resume.wait();
            Some(GpuRegionUsage::exported(4096, 4096, 0))
        }
    }
    impl RegionStorage for ObservedArena {
        fn kind(&self) -> GpuRegionKind {
            GpuRegionKind::Arena
        }
        fn can_retire_arena(&self) -> bool {
            true
        }
    }

    let pool = pool(0);
    let entered = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let lease = pool.insert(
        GpuRegionIsolation::Local,
        Box::new(ObservedArena {
            entered: entered.clone(),
            resume: resume.clone(),
        }),
    );
    let sampler = pool.clone();
    let sample = std::thread::spawn(move || sampler.snapshots());
    entered.wait();

    let retiring = pool.clone();
    let (started, has_started) = mpsc::channel();
    let (done, completed) = mpsc::channel();
    let retire = std::thread::spawn(move || {
        started.send(()).unwrap();
        done.send(retiring.retire_arena(&lease)).unwrap();
    });
    has_started.recv().unwrap();
    assert!(completed.recv_timeout(Duration::from_millis(30)).is_err());
    resume.wait();
    sample.join().unwrap();
    assert_eq!(
        completed.recv_timeout(Duration::from_secs(2)).unwrap(),
        Ok(true)
    );
    retire.join().unwrap();
    assert!(pool.snapshots().is_empty());
}

#[test]
fn selection_uses_ready_compatible_arena_and_never_resizes_on_allocation() {
    let pool = pool(3);
    let (_small, small_state) = arena(&pool, 64);
    let (large, large_state) = arena(&pool, 256);
    let lease = pool.acquire_region(local(128, 64)).unwrap();
    assert_eq!(lease.id, large.id);
    let allocation = lease.allocate(owner(), 128, 64).unwrap();
    assert_eq!(allocation.bytes(), 128);
    assert!(pool.acquire_region(local(128, 64)).is_err());
    assert_eq!(small_state.lock().capacity, 64);
    assert_eq!(large_state.lock().capacity, 256);
    unsafe {
        allocation.release_after_fence().unwrap();
    }
}

#[test]
fn foreign_pool_and_owner_cannot_allocate_with_a_region_lease() {
    let a = pool(0);
    let b = pool(0);
    let (_arena, _) = arena(&a, 128);
    let lease = a.acquire_region(local(64, 1)).unwrap();
    assert!(b.allocate(&lease, owner(), 64, 1).is_err());
    let foreign = PoolOwner::onnx(2, 2, PoolAllocationClass::TransientWorkspace);
    assert!(a.allocate(&lease, foreign, 64, 1).is_err());
    assert!(a.acquire_region(local(1, 0)).is_err());
}

#[test]
fn stale_allocation_cannot_release_a_new_extent_at_the_same_address() {
    let pool = pool(0);
    let (region, state) = arena(&pool, 128);
    let old = region.allocate(owner(), 64, 1).unwrap();
    let pointer = old.pointer().unwrap();
    unsafe {
        old.release_after_fence().unwrap();
    }
    let new = region.allocate(owner(), 64, 1).unwrap();
    assert_eq!(new.pointer().unwrap(), pointer);
    assert_ne!(old.sequence, new.sequence);
    assert!(unsafe { old.release_after_fence() }.is_err());
    assert!(old.pointer().is_err());
    assert_eq!(state.lock().allocation.unwrap().1, 64);
    unsafe {
        new.release_after_fence().unwrap();
    }
}

#[test]
fn failed_free_keeps_the_extent_and_region_pinned_until_retry() {
    let pool = pool(0);
    let (region, state) = arena(&pool, 128);
    let allocation = region.allocate(owner(), 64, 1).unwrap();
    state.lock().fail_free = true;
    assert!(unsafe { allocation.release_after_fence() }.is_err());
    assert_eq!(allocation.bytes(), 64);
    assert!(!pool.retire_arena(&region).unwrap());
    assert_eq!(pool.snapshots()[0].usage.committed_bytes, 128);
    state.lock().fail_free = false;
    unsafe {
        allocation.release_after_fence().unwrap();
    }
    drop(allocation);
    assert!(pool.retire_arena(&region).unwrap());
    assert!(pool.validate(&region).is_err());
    assert!(pool.snapshots().is_empty());
}

#[test]
fn an_abandoned_allocation_retains_capacity_until_supervised_cleanup() {
    let pool = pool(0);
    let (region, state) = arena(&pool, 128);
    drop(region.allocate(owner(), 64, 1).unwrap());
    assert!(!pool.retire_arena(&region).unwrap());
    assert_eq!(state.lock().allocation.unwrap().1, 64);
    assert_eq!(pool.snapshots()[0].usage.reusable_bytes, Some(64));
}

#[test]
fn additional_region_leases_block_arena_retirement() {
    let pool = pool(0);
    let (region, _) = arena(&pool, 128);
    let view = pool.acquire_region(local(0, 1)).unwrap();
    assert!(!pool.retire_arena(&region).unwrap());
    drop(view);
    assert!(pool.retire_arena(&region).unwrap());
    let (replacement, _) = arena(&pool, 128);
    assert_ne!(replacement.id, region.id);
    assert!(region.allocate(owner(), 64, 1).is_err());
}

#[test]
fn logical_free_arena_capacity_does_not_satisfy_an_isolated_export() {
    let pool = pool(0);
    let (region, _) = arena(&pool, 4 * 1024);
    let allocation = region.allocate(owner(), 2 * 1024, 1).unwrap();
    unsafe {
        allocation.release_after_fence().unwrap();
    }
    for capacity in [
        ExportCapacity::Fixed { bytes: 2 * 1024 },
        ExportCapacity::Elastic {
            maximum_bytes: 8 * 1024,
        },
    ] {
        let result = pool.acquire_region(GpuRegionRequest::Exported {
            isolation: isolated(),
            capacity,
        });
        assert!(result.err().unwrap().contains("capacity operation"));
    }
    assert_eq!(pool.snapshots()[0].usage.committed_bytes, 4 * 1024);
    assert_eq!(pool.snapshots()[0].usage.reusable_bytes, Some(4 * 1024));
    assert!(pool
        .acquire_region(GpuRegionRequest::Exported {
            isolation: GpuRegionIsolation::Local,
            capacity: ExportCapacity::Fixed { bytes: 1024 }
        })
        .is_err());
}

#[test]
fn failed_export_release_keeps_backing_visible_and_retry_retires_it() {
    let pool = pool(0);
    let state = Arc::new(Mutex::new(State {
        capacity: 128,
        fail_release: true,
        ..State::default()
    }));
    let region = pool.insert(
        isolated(),
        Box::new(FakeRegion {
            kind: GpuRegionKind::Ipc,
            state: state.clone(),
        }),
    );
    assert!(region.allocate(owner(), 64, 1).is_err());
    assert!(unsafe { region.release_export_after_fence() }.is_err());
    assert_eq!(pool.snapshots()[0].usage.committed_bytes, 128);
    state.lock().fail_release = false;
    unsafe {
        region.release_export_after_fence().unwrap();
    }
    assert!(pool.snapshots().is_empty());
    unsafe {
        region.release_export_after_fence().unwrap();
    }
}

#[test]
fn hal_observations_preserve_unmapped_handles_and_unreleased_virtual_ranges() {
    use kapsl_hal::gpu_region::{GpuRegionKind as HalKind, GpuRegionSnapshot};
    let mut snapshot = GpuRegionSnapshot {
        device_id: 0,
        kind: HalKind::Vmm,
        committed_bytes: 3072,
        mapped_bytes: 1024,
        ready_bytes: Some(1024),
        virtual_reserved_bytes: 8192,
        released: false,
    };
    assert_eq!(
        exported_usage(snapshot),
        Some(GpuRegionUsage::exported(3072, 1024, 8192))
    );
    snapshot.mapped_bytes = 0;
    assert_eq!(exported_usage(snapshot).unwrap().committed_bytes, 3072);
    snapshot.committed_bytes = 0;
    assert_eq!(
        exported_usage(snapshot).unwrap().virtual_reserved_bytes,
        8192
    );
    snapshot.released = true;
    assert_eq!(exported_usage(snapshot), None);
}
