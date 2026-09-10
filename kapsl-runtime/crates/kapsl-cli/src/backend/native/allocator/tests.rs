use super::*;

type Owner = (u32, u32);

#[derive(Default)]
pub(crate) struct FakePool {
    state: Mutex<FakePoolState>,
}

#[derive(Default)]
struct FakePoolState {
    owners: HashMap<Owner, usize>,
    live: HashMap<usize, (Owner, usize)>,
    next: usize,
    syncs: usize,
    synchronized: HashSet<usize>,
    frees: usize,
    fail_sync: bool,
    fail_free: bool,
    panic_allocate: bool,
}

impl FakePool {
    pub(crate) fn host(
        self: &Arc<Self>,
        model: u32,
        replica: u32,
        quota: usize,
    ) -> NativeBackendHost {
        self.state
            .lock()
            .unwrap()
            .owners
            .insert((model, replica), quota);
        NativeBackendHost::with_allocator(
            Arc::new(FakeAllocator {
                pool: Arc::clone(self),
                owner: (model, replica),
            }),
            0,
            model,
            replica,
            true,
        )
    }

    pub(crate) fn bytes(&self, owner: Owner) -> usize {
        self.state
            .lock()
            .unwrap()
            .live
            .values()
            .filter(|(o, _)| *o == owner)
            .map(|(_, bytes)| bytes)
            .sum()
    }

    pub(crate) fn fail_sync(&self, fail: bool) {
        self.state.lock().unwrap().fail_sync = fail;
    }
}

struct FakeAllocator {
    pool: Arc<FakePool>,
    owner: Owner,
}
struct FakeStorage {
    pool: Arc<FakePool>,
    pointer: usize,
    bytes: usize,
}

impl DeviceAllocator for FakeAllocator {
    fn allocate(
        &self,
        _class: u32,
        bytes: usize,
        alignment: usize,
    ) -> Result<Box<dyn DeviceAllocation>, String> {
        let mut state = self.pool.state.lock().unwrap();
        if state.panic_allocate {
            drop(state);
            panic!("injected allocator panic");
        }
        let quota = *state
            .owners
            .get(&self.owner)
            .ok_or("fake owner has no admission")?;
        let used: usize = state
            .live
            .values()
            .filter(|(owner, _)| *owner == self.owner)
            .map(|(_, bytes)| bytes)
            .sum();
        let total: usize = state.live.values().map(|(_, bytes)| bytes).sum();
        if bytes > quota.saturating_sub(used) || bytes > 4096_usize.saturating_sub(total) {
            return Err("fake pool admission denied".into());
        }
        // No device or heap memory is exposed: adapter tests only use opaque
        // device handles. Deliberately recycle addresses when the pool empties.
        if state.live.is_empty() {
            state.next = 4096;
        }
        let pointer = state.next.next_multiple_of(alignment);
        state.next = pointer + bytes;
        state.synchronized.remove(&pointer);
        state.live.insert(pointer, (self.owner, bytes));
        Ok(Box::new(FakeStorage {
            pool: Arc::clone(&self.pool),
            pointer,
            bytes,
        }))
    }
    fn synchronize(&self) -> Result<(), String> {
        let mut state = self.pool.state.lock().unwrap();
        state.syncs += 1;
        if state.fail_sync {
            Err("injected synchronization failure".into())
        } else {
            state.synchronized = state.live.keys().copied().collect();
            Ok(())
        }
    }
}

impl DeviceAllocation for FakeStorage {
    fn pointer(&self) -> usize {
        self.pointer
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn free(&self) -> Result<(), String> {
        let mut state = self.pool.state.lock().unwrap();
        if state.fail_free {
            return Err("injected free failure".into());
        }
        assert!(
            state.synchronized.remove(&self.pointer),
            "engine must synchronize before each reuse"
        );
        state
            .live
            .remove(&self.pointer)
            .ok_or("invalid fake free")?;
        state.frees += 1;
        Ok(())
    }
}

fn request(kind: u32, id: u64, ids: &[u64]) -> KapslScopedDeviceAllocationRequestV1 {
    KapslScopedDeviceAllocationRequestV1::new(
        0,
        KAPSL_MEMORY_CUDA,
        KAPSL_ALLOCATION_CLASS_WORKSPACE,
        KapslDeviceAllocationScopeV1::new(kind, id, 7, 2, ids),
        128,
        64,
    )
}

fn allocate(
    host: &NativeBackendHost,
    request: &KapslScopedDeviceAllocationRequestV1,
) -> Result<KapslDeviceAllocationV1, i32> {
    let table = &host.table;
    let mut result = KapslDeviceAllocationV1::empty();
    let status = unsafe {
        table.allocate_device_scoped.unwrap()(table.base.user_data, request, &mut result)
    };
    if status == KAPSL_STATUS_OK {
        Ok(result)
    } else {
        assert_eq!(result.allocation_id, 0);
        assert!(result.device_ptr.is_null());
        Err(status)
    }
}

fn free(host: &NativeBackendHost, allocation: &KapslDeviceAllocationV1) -> i32 {
    unsafe { host.table.base.free_device.unwrap()(host.table.base.user_data, allocation) }
}

#[test]
fn extended_table_and_all_scopes_preserve_charge_until_matching_free() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    let extension = unsafe { KapslBackendHostScopedAllocatorV1::from_base(host.table()) }.unwrap();
    assert_eq!(
        extension.base.struct_size as usize,
        std::mem::size_of::<KapslBackendHostScopedAllocatorV1>()
    );
    assert_eq!(
        extension.scoped_allocator_version,
        KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION
    );
    assert!(extension.is_well_formed());
    let model_call = host.begin_model_call().unwrap();
    let model = allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 1, &[])).unwrap();
    let replica = allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REPLICA, 2, &[])).unwrap();
    drop(model_call);
    let call = host.begin_requests([(11, None), (12, None)]).unwrap();
    let single = allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 3, &[11])).unwrap();
    let batch = allocate(
        &host,
        &request(KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH, 4, &[11, 12]),
    )
    .unwrap();
    drop(call);
    assert_eq!(pool.bytes((7, 2)), 512);
    let report = host.actual_memory(MemoryReport::default());
    assert_eq!(report.allocations.len(), 4);
    assert!(report
        .allocations
        .iter()
        .any(|row| row.allocation_id.contains("scope-4-4/requests-[11, 12]")));
    for allocation in [model, replica, single, batch] {
        assert_eq!(free(&host, &allocation), KAPSL_STATUS_OK);
    }
    assert_eq!(pool.bytes((7, 2)), 0);
    assert!(host
        .actual_memory(MemoryReport::default())
        .allocations
        .is_empty());
}

#[test]
fn malformed_missing_and_foreign_scopes_never_reach_physical_allocator() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    drop(host.begin_model_call().unwrap());
    let _call = host.begin_requests([(11, None), (12, None)]).unwrap();
    let ids = [11, 12];
    let valid = request(KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH, 1, &ids);
    let mut invalid = Vec::new();
    macro_rules! invalid { ($field:ident $(.$sub:ident)?, $value:expr) => {{ let mut r = valid; r.$field $(.$sub)? = $value; invalid.push(r); }}; }
    invalid!(struct_size, 4);
    invalid!(scope.struct_size, 4);
    invalid!(device_id, 1);
    invalid!(memory_kind, KAPSL_MEMORY_HOST);
    invalid!(allocation_class, 99);
    invalid!(flags, 1);
    invalid!(reserved, 1);
    invalid!(bytes, 0);
    invalid!(bytes, u64::MAX);
    invalid!(alignment, 0);
    invalid!(alignment, 3);
    invalid!(scope.scope_kind, 99);
    invalid!(scope.scope_id, 0);
    invalid!(scope.model_id, 8);
    invalid!(scope.replica_id, 3);
    invalid!(scope.reserved, 1);
    invalid!(scope.request_count, 0);
    invalid!(scope.request_count, u32::MAX);
    invalid!(scope.request_ids, std::ptr::null());
    invalid!(scope.scope_kind, KAPSL_ALLOCATION_SCOPE_REQUEST);
    for r in invalid {
        assert!(allocate(&host, &r).is_err(), "accepted {r:?}");
    }
    for ids in [[0, 11], [11, 11], [11, 99]] {
        assert!(allocate(
            &host,
            &request(KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH, 2, &ids)
        )
        .is_err());
    }
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 3, &[])).is_err());
    assert_eq!(pool.bytes((7, 2)), 0);
    assert!(pool.state.lock().unwrap().live.is_empty());
}

#[test]
fn truncated_callback_objects_are_rejected_before_reading_the_tail() {
    #[repr(C, align(8))]
    struct Prefix(u32);
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    let prefix = Prefix(4);
    let mut result = KapslDeviceAllocationV1::empty();
    unsafe {
        assert_ne!(
            host.table.allocate_device_scoped.unwrap()(
                host.table.base.user_data,
                (&prefix as *const Prefix).cast(),
                &mut result
            ),
            KAPSL_STATUS_OK
        );
        assert_ne!(
            host.table.base.allocate_device.unwrap()(
                host.table.base.user_data,
                (&prefix as *const Prefix).cast(),
                &mut result
            ),
            KAPSL_STATUS_OK
        );
        assert_ne!(
            host.table.base.free_device.unwrap()(
                host.table.base.user_data,
                (&prefix as *const Prefix).cast()
            ),
            KAPSL_STATUS_OK
        );
        assert_ne!(
            host.table.allocate_device_scoped.unwrap()(
                host.table.base.user_data,
                std::ptr::null(),
                &mut result
            ),
            KAPSL_STATUS_OK
        );
        assert_ne!(
            host.table.allocate_device_scoped.unwrap()(
                std::ptr::null_mut(),
                std::ptr::null(),
                &mut result
            ),
            KAPSL_STATUS_OK
        );
        assert_ne!(
            host.table.allocate_device_scoped.unwrap()(
                host.table.base.user_data,
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            KAPSL_STATUS_OK
        );
    }
}

#[test]
fn scopes_cannot_rebind_retire_then_reappear_or_mix_concurrent_calls() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    drop(host.begin_model_call().unwrap());
    let first_call = host.begin_requests([(11, None)]).unwrap();
    let second_call = host.begin_requests([(12, None)]).unwrap();
    let allocation = allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 1, &[11])).unwrap();
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 1, &[12])).is_err());
    assert!(allocate(
        &host,
        &request(KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH, 2, &[11, 12])
    )
    .is_err());
    drop(first_call);
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 3, &[11])).is_err());
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 1, &[12])).is_err());
    drop(second_call);
    assert_eq!(pool.bytes((7, 2)), 128);
    assert_eq!(free(&host, &allocation), KAPSL_STATUS_OK);
    host.reclaim().unwrap();
    let _load = host.begin_model_call().unwrap();
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 1, &[])).is_err());
}

#[test]
fn cancelled_scopes_stop_allocating_but_remain_freeable() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    drop(host.begin_model_call().unwrap());
    let token = CancellationToken::new();
    let call = host
        .begin_requests([(11, Some(token.clone())), (12, None)])
        .unwrap();
    let allocation = allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 1, &[11])).unwrap();
    token.cancel();
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 2, &[11])).is_err());
    assert!(allocate(
        &host,
        &request(KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH, 3, &[11, 12])
    )
    .is_err());
    host.allocator.as_ref().unwrap().cancel(12);
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_REQUEST, 4, &[12])).is_err());
    assert_eq!(free(&host, &allocation), KAPSL_STATUS_OK);
    drop(call);
    host.reclaim().unwrap();
}

#[test]
fn admission_limits_cross_class_usage_and_failed_frees_stay_charged() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 128);
    let call = host.begin_model_call().unwrap();
    let allocation = allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 1, &[])).unwrap();
    let mut second = request(KAPSL_ALLOCATION_SCOPE_REPLICA, 2, &[]);
    second.allocation_class = KAPSL_ALLOCATION_CLASS_KV;
    assert!(allocate(&host, &second).is_err());
    pool.fail_sync(true);
    assert_ne!(free(&host, &allocation), KAPSL_STATUS_OK);
    assert_eq!(pool.bytes((7, 2)), 128);
    assert_eq!(host.live_allocations(), 1);
    pool.fail_sync(false);
    pool.state.lock().unwrap().fail_free = true;
    assert_ne!(free(&host, &allocation), KAPSL_STATUS_OK);
    assert_eq!(host.live_allocations(), 1);
    pool.state.lock().unwrap().fail_free = false;
    assert_eq!(free(&host, &allocation), KAPSL_STATUS_OK);
    pool.state.lock().unwrap().owners.remove(&(7, 2));
    assert!(allocate(&host, &second).is_err());
    drop(call);
}

#[test]
fn frees_validate_instance_identity_size_pointer_and_reused_addresses() {
    let pool = Arc::new(FakePool::default());
    let first = pool.host(7, 2, 1024);
    let second = pool.host(8, 2, 1024);
    let call = first.begin_model_call().unwrap();
    let allocation = allocate(&first, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 1, &[])).unwrap();
    assert_ne!(free(&second, &allocation), KAPSL_STATUS_OK);
    for invalid in [
        KapslDeviceAllocationV1 {
            device_ptr: std::ptr::null_mut(),
            ..allocation
        },
        KapslDeviceAllocationV1 {
            granted_bytes: 64,
            ..allocation
        },
        KapslDeviceAllocationV1 {
            reserved: 1,
            ..allocation
        },
    ] {
        assert_ne!(free(&first, &invalid), KAPSL_STATUS_OK);
    }
    assert_eq!(free(&first, &allocation), KAPSL_STATUS_OK);
    assert_ne!(free(&first, &allocation), KAPSL_STATUS_OK);
    drop(call);
    let call = second.begin_model_call().unwrap();
    let mut r = request(KAPSL_ALLOCATION_SCOPE_MODEL, 1, &[]);
    r.scope.model_id = 8;
    let replacement = allocate(&second, &r).unwrap();
    assert_eq!(replacement.device_ptr, allocation.device_ptr);
    assert_ne!(replacement.allocation_id, allocation.allocation_id);
    assert_ne!(free(&second, &allocation), KAPSL_STATUS_OK);
    drop(call);
    second.reclaim().unwrap();
    assert_eq!(pool.bytes((8, 2)), 0);
}

#[test]
fn scoped_adapter_cannot_use_legacy_callback_and_failed_cleanup_can_retry() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    let call = host.begin_model_call().unwrap();
    let legacy = KapslDeviceAllocationRequestV1 {
        struct_size: std::mem::size_of::<KapslDeviceAllocationRequestV1>() as u32,
        device_id: 0,
        memory_kind: KAPSL_MEMORY_CUDA,
        allocation_class: KAPSL_ALLOCATION_CLASS_WEIGHTS,
        model_id: 7,
        replica_id: 2,
        flags: 0,
        reserved: 0,
        bytes: 128,
        alignment: 64,
    };
    let mut out = KapslDeviceAllocationV1::empty();
    assert_ne!(
        unsafe {
            host.table.base.allocate_device.unwrap()(host.table.base.user_data, &legacy, &mut out)
        },
        KAPSL_STATUS_OK
    );
    allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 1, &[])).unwrap();
    assert!(host.reclaim().is_err());
    drop(call);
    pool.fail_sync(true);
    assert!(host.reclaim().is_err());
    assert_eq!(host.live_allocations(), 1);
    assert!(host.begin_model_call().is_err());
    pool.fail_sync(false);
    host.reclaim().unwrap();
    assert_eq!(pool.bytes((7, 2)), 0);
    assert!(allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 2, &[])).is_err());
}

#[test]
fn allocator_panics_do_not_unwind_across_the_abi() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    let call = host.begin_model_call().unwrap();
    pool.state.lock().unwrap().panic_allocate = true;
    assert_eq!(
        allocate(&host, &request(KAPSL_ALLOCATION_SCOPE_MODEL, 1, &[])).unwrap_err(),
        KAPSL_STATUS_PANIC
    );
    pool.state.lock().unwrap().panic_allocate = false;
    drop(call);
    host.reclaim().unwrap();
}

#[test]
fn retired_scope_ids_compact_without_accepting_reuse_or_order_assumptions() {
    let mut ledger = AllocationLedger::default();
    for id in [4, 2, 6, 1, 3, 5] {
        ledger.remember_scope_id(id).unwrap();
    }
    assert_eq!(ledger.used_scope_ids, BTreeMap::from([(1, 6)]));
    for id in 1..=6 {
        assert!(ledger.remember_scope_id(id).is_err());
    }
    ledger.remember_scope_id(u64::MAX).unwrap();
    assert!(ledger.remember_scope_id(u64::MAX).is_err());
}

#[test]
fn synchronize_callback_enforces_device_identity_and_propagates_failure() {
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    let synchronize = host.table.base.synchronize_device.unwrap();
    let context = host.table.base.user_data;
    unsafe {
        assert_eq!(synchronize(context, 1), KAPSL_STATUS_INVALID_ARGUMENT);
        assert_eq!(
            synchronize(std::ptr::null_mut(), 0),
            KAPSL_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(pool.state.lock().unwrap().syncs, 0);
        assert_eq!(synchronize(context, 0), KAPSL_STATUS_OK);
        pool.fail_sync(true);
        assert_eq!(synchronize(context, 0), KAPSL_STATUS_BACKEND_ERROR);
        pool.fail_sync(false);
    }
}

#[test]
fn host_without_a_device_pool_does_not_advertise_allocator_callbacks() {
    let host = NativeBackendHost::from_allocator(None);
    assert!(unsafe { KapslBackendHostScopedAllocatorV1::from_base(host.table()) }.is_none());
    assert!(host.table.base.allocate_device.is_none());
    assert!(host.table.base.free_device.is_none());
    assert!(host.table.base.synchronize_device.is_none());
}
