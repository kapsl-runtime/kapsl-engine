use super::super::MemoryOwner;
use super::*;
use kapsl_hal::gpu_arena::PoolAllocationClass;
use std::collections::HashMap;

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
        ordinal: device_id,
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

/// The admission contract of `HalStorage::Arena` without CUDA: a local request
/// needs its workload admitted (any allocation class of the same backend,
/// model and replica), and an allocation must leave every other admitted
/// workload's unmet guarantee free, as `PoolPolicy::available_for` does.
#[derive(Default)]
struct AdmissionState {
    capacity: usize,
    guaranteed: HashMap<PoolWorkload, usize>,
    usage: HashMap<PoolWorkload, usize>,
    live: HashMap<u64, (PoolWorkload, usize)>,
    next_id: u64,
}

impl AdmissionState {
    fn available_for(&self, workload: PoolWorkload) -> usize {
        let used: usize = self.usage.values().sum();
        let unmet_others: usize = self
            .guaranteed
            .iter()
            .filter(|(other, _)| **other != workload)
            .map(|(other, guaranteed)| {
                guaranteed.saturating_sub(self.usage.get(other).copied().unwrap_or(0))
            })
            .sum();
        self.capacity
            .saturating_sub(used)
            .saturating_sub(unmet_others)
    }
}

struct AdmissionArena {
    state: Arc<Mutex<AdmissionState>>,
}

struct AdmissionExtent {
    state: Arc<Mutex<AdmissionState>>,
    id: u64,
    bytes: usize,
}

/// What `DeviceMemoryManager::admit_pool` and `try_finish_release` do to an
/// arena's owners during load admission and unload.
#[derive(Clone)]
pub(crate) struct ArenaAdmissions(Arc<Mutex<AdmissionState>>);

impl ArenaAdmissions {
    pub(crate) fn admit(&self, owner: PoolOwner, guaranteed_bytes: usize) {
        self.0
            .lock()
            .guaranteed
            .insert(owner.workload(), guaranteed_bytes);
    }
}

impl GpuRegionSource for AdmissionArena {
    fn region_usage(&self) -> Option<GpuRegionUsage> {
        let state = self.state.lock();
        let used: usize = state.usage.values().sum();
        let mut usage = GpuRegionUsage::exported(state.capacity, state.capacity, 0);
        usage.logical_allocated_bytes = Some(used);
        usage.reusable_bytes = Some(state.capacity - used);
        usage.largest_free_range_bytes = usage.reusable_bytes;
        Some(usage)
    }
}

impl RegionStorage for AdmissionArena {
    fn kind(&self) -> GpuRegionKind {
        GpuRegionKind::Arena
    }
    fn fits_local(&self, owner: PoolOwner, bytes: usize, _alignment: usize) -> bool {
        let state = self.state.lock();
        state.guaranteed.contains_key(&owner.workload())
            && (bytes == 0 || state.available_for(owner.workload()) >= bytes)
    }
    fn allocate(
        &self,
        owner: PoolOwner,
        bytes: usize,
        _alignment: usize,
    ) -> Result<Box<dyn AllocationStorage>, String> {
        let mut state = self.state.lock();
        let workload = owner.workload();
        if !state.guaranteed.contains_key(&workload) {
            return Err("allocation owner has no memory admission".into());
        }
        let available = state.available_for(workload);
        if bytes > available {
            return Err(format!(
                "quota exceeded: requested {bytes}, available {available}"
            ));
        }
        state.next_id += 1;
        let id = state.next_id;
        state.live.insert(id, (workload, bytes));
        *state.usage.entry(workload).or_default() += bytes;
        Ok(Box::new(AdmissionExtent {
            state: self.state.clone(),
            id,
            bytes,
        }))
    }
    fn can_retire_arena(&self) -> bool {
        self.state.lock().live.is_empty()
    }
}

impl AllocationStorage for AdmissionExtent {
    fn pointer(&self) -> usize {
        4096 * self.id as usize
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn free_after_fence(&self) -> Result<(), String> {
        let mut state = self.state.lock();
        let (workload, bytes) = state.live.remove(&self.id).ok_or("stale extent")?;
        if let Some(usage) = state.usage.get_mut(&workload) {
            *usage = usage.saturating_sub(bytes);
        }
        Ok(())
    }
}

/// A device pool with one installed local arena that enforces admission as
/// the CUDA arena does. The lease is the manager's own (installation) lease.
pub(crate) fn pool_with_admission_arena(
    device_id: usize,
    capacity: usize,
) -> (Arc<GpuDevicePool>, GpuRegionLease, ArenaAdmissions) {
    let pool = pool(device_id);
    let state = Arc::new(Mutex::new(AdmissionState {
        capacity,
        ..AdmissionState::default()
    }));
    let lease = pool.insert(
        GpuRegionIsolation::Local,
        Box::new(AdmissionArena {
            state: state.clone(),
        }),
    );
    (pool, lease, ArenaAdmissions(state))
}

/// The v0.3.0 qualification on an RTX 3090: automatic sizing gave the pool
/// 1,669,332,992 bytes and ORT model 0 was admitted with a quarter of it
/// guaranteed. GGUF model 1 was not admitted yet when its backend asked.
pub(crate) const GATE_POOL_BYTES: usize = 1_669_332_992;
pub(crate) const GATE_ORT_GUARANTEED_BYTES: usize = GATE_POOL_BYTES / 4;
const GATE_REFUSAL: &str =
    "no compatible arena has admitted, ready capacity; request a separate capacity change";

#[test]
fn a_local_request_before_load_admission_is_refused_with_the_gate_error() {
    let (pool, _arena, admissions) = pool_with_admission_arena(0, GATE_POOL_BYTES);
    admissions.admit(
        PoolOwner::onnx(0, 0, PoolAllocationClass::PersistentWeights),
        GATE_ORT_GUARANTEED_BYTES,
    );
    let gguf_kv = PoolOwner::gguf(1, 0, PoolAllocationClass::KvCache);
    let refused = pool
        .acquire_region(GpuRegionRequest::Local {
            owner: gguf_kv,
            bytes: 0,
            alignment: 1,
        })
        .err()
        .expect("an unadmitted owner gets no lease, whatever the free capacity");
    assert_eq!(refused, GATE_REFUSAL);

    // Load admission admits the workload under its weights class; every class
    // of the same model/replica then shares it.
    admissions.admit(
        gguf_kv.with_class(PoolAllocationClass::PersistentWeights),
        GATE_POOL_BYTES / 4,
    );
    assert!(pool
        .acquire_region(GpuRegionRequest::Local {
            owner: gguf_kv,
            bytes: 0,
            alignment: 1,
        })
        .is_ok());
}

#[test]
fn a_construction_binding_leases_the_arena_only_after_load_admission() {
    let (pool, _arena, admissions) = pool_with_admission_arena(0, GATE_POOL_BYTES);
    let ort = PoolOwner::onnx(0, 0, PoolAllocationClass::PersistentWeights);
    admissions.admit(ort, GATE_ORT_GUARANTEED_BYTES);
    let gguf_kv = PoolOwner::gguf(1, 0, PoolAllocationClass::KvCache);

    // Backend construction: binds without an admission and without a lease.
    let binding = pool
        .bind_local_arena(gguf_kv)
        .expect("binding at construction must not require the admission that follows it");
    assert_eq!(binding.owner(), gguf_kv);
    let early = binding.acquire().err().expect("still no admission");
    assert!(early.contains(GATE_REFUSAL), "{early}");
    assert!(early.contains("Gguf"), "{early}");
    assert_eq!(pool.snapshots().len(), 1);

    // Load admission, then the pack creates its KV pool from load_model.
    admissions.admit(
        gguf_kv.with_class(PoolAllocationClass::PersistentWeights),
        GATE_POOL_BYTES / 4,
    );
    let lease = binding
        .acquire()
        .expect("admitted workload leases the arena");
    let table_owner = gguf_kv.with_class(PoolAllocationClass::BlockTable);
    let table = lease
        .allocate(table_owner, 64 * 1024, 4)
        .expect("block table fits");
    // Admission is unchanged: the ORT guarantee stays out of the GGUF's reach.
    let beyond = GATE_POOL_BYTES - GATE_ORT_GUARANTEED_BYTES;
    assert!(lease.allocate(gguf_kv, beyond, 1).is_err());
    // Another workload cannot allocate through this workload's lease.
    assert!(lease.allocate(ort, 64, 1).is_err());
    unsafe {
        table.release_after_fence().unwrap();
    }
}

#[test]
fn a_construction_binding_refuses_a_device_without_an_arena() {
    let pool = pool(0);
    let gguf_kv = PoolOwner::gguf(1, 0, PoolAllocationClass::KvCache);
    let error = pool
        .bind_local_arena(gguf_kv)
        .err()
        .expect("no pool, no binding");
    assert!(error.contains("no runtime-owned arena"), "{error}");

    // Exported participant backing is not a local arena either.
    let state = Arc::new(Mutex::new(State {
        capacity: 128,
        ..State::default()
    }));
    let export = pool.insert(
        isolated(),
        Box::new(FakeRegion {
            kind: GpuRegionKind::Ipc,
            state,
        }),
    );
    assert!(pool.bind_local_arena(gguf_kv).is_err());
    unsafe {
        export.release_export_after_fence().unwrap();
    }

    // A binding whose arena was reclaimed before admission is refused, not
    // redirected: the lease still needs an installed, admitted arena.
    let (pool, arena, admissions) = pool_with_admission_arena(0, GATE_POOL_BYTES);
    let binding = pool.bind_local_arena(gguf_kv).unwrap();
    assert!(pool.retire_arena(&arena).unwrap());
    admissions.admit(gguf_kv, GATE_POOL_BYTES / 4);
    assert!(binding.acquire().is_err());
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
