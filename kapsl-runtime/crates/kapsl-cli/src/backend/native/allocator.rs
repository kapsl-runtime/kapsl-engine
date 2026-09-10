//! Engine-owned allocation governance. Only the physical pool is replaceable
//! in host-only tests; ABI validation, attribution and lifecycle are shared.

use super::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};

// Allocation handles must not alias across instances, even if a pool later
// reuses the same address and size for another model or replica.
static NEXT_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);

pub(super) trait DeviceAllocation: Send {
    fn pointer(&self) -> usize;
    fn bytes(&self) -> usize;
    // Failure retains ownership and accounting so the caller can retry.
    fn free(&self) -> Result<(), String>;
}

pub(super) trait DeviceAllocator: Send + Sync {
    // Implementations must enforce the instance's admission and aggregate
    // quota across allocation classes, including competing pool owners.
    fn allocate(
        &self,
        class: u32,
        bytes: usize,
        alignment: usize,
    ) -> Result<Box<dyn DeviceAllocation>, String>;
    fn synchronize(&self) -> Result<(), String>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AllocationScope {
    kind: u32,
    id: u64,
    model_id: u32,
    replica_id: u32,
    request_ids: Vec<u64>,
    operation: u64,
}

struct LiveAllocation {
    storage: Box<dyn DeviceAllocation>,
    class: u32,
    scope: Arc<AllocationScope>,
}

struct ActiveRequest {
    operation: u64,
    cancellation: Option<CancellationToken>,
    cancelled: bool,
}

#[derive(Default)]
struct AllocationLedger {
    enabled: bool,
    cleanup_pending: bool,
    model_operation: Option<u64>,
    next_operation: u64,
    requests: HashMap<u64, ActiveRequest>,
    scopes: HashMap<u64, Arc<AllocationScope>>,
    // Inclusive intervals compact sequential retired scope IDs, without
    // keeping per-request metadata forever or permitting ID reuse on reload.
    used_scope_ids: BTreeMap<u64, u64>,
    allocations: HashMap<u64, LiveAllocation>,
}

impl AllocationLedger {
    fn next_operation(&mut self) -> Result<u64, String> {
        self.next_operation = self
            .next_operation
            .checked_add(1)
            .ok_or("native allocation operation IDs exhausted")?;
        Ok(self.next_operation)
    }

    fn remember_scope_id(&mut self, id: u64) -> Result<(), String> {
        let mut start = id;
        let mut end = id;
        if let Some((&left, &right)) = self.used_scope_ids.range(..=id).next_back() {
            if id <= right {
                return Err("native allocation scope ID was already retired".into());
            }
            if right.checked_add(1) == Some(id) {
                start = left;
                self.used_scope_ids.remove(&left);
            }
        }
        if let Some((&left, &right)) = self.used_scope_ids.range(id..).next() {
            if id.checked_add(1) == Some(left) {
                end = right;
                self.used_scope_ids.remove(&left);
            }
        }
        self.used_scope_ids.insert(start, end);
        Ok(())
    }
}

pub(super) struct GovernedDeviceHost {
    allocator: Arc<dyn DeviceAllocator>,
    device_id: u32,
    model_id: u32,
    replica_id: u32,
    require_scoped: bool,
    ledger: Mutex<AllocationLedger>,
}

pub(super) struct NativeAllocationCall<'a> {
    host: Option<&'a GovernedDeviceHost>,
    operation: u64,
}

impl Drop for NativeAllocationCall<'_> {
    fn drop(&mut self) {
        if let Some(host) = self.host {
            let mut ledger = host.ledger.lock().unwrap_or_else(|p| p.into_inner());
            ledger
                .requests
                .retain(|_, request| request.operation != self.operation);
            ledger
                .scopes
                .retain(|_, scope| scope.operation != self.operation);
            if ledger.model_operation == Some(self.operation) {
                ledger.model_operation = None;
            }
            // Request completion retires attribution authority, not memory.
            // Adapters may retain arenas until their matching free or unload.
        }
    }
}

pub(super) struct NativeBackendHost {
    table: Box<KapslBackendHostScopedAllocatorV1>,
    pub(super) allocator: Option<Arc<GovernedDeviceHost>>,
}

impl NativeBackendHost {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        resources: &RuntimeResources,
        backend: &str,
        device_id: usize,
        model_id: u32,
        replica_id: u32,
        require_governed: bool,
        require_scoped: bool,
    ) -> Result<Self, String> {
        let device_id = u32::try_from(device_id).map_err(|_| "device ID exceeds native ABI")?;
        if !require_governed {
            return Ok(Self::from_allocator(None));
        }
        #[cfg(feature = "gpu-device-pool")]
        {
            let pool = resources.device_pool(device_id as usize).ok_or_else(|| format!(
                "native {backend} pack requires governed device memory, but device {device_id} has no runtime-owned pool"
            ))?;
            let allocator = Arc::new(GpuAllocator {
                pool,
                backend: pool_backend(backend),
                model_id,
                replica_id,
            });
            Ok(Self::with_allocator(
                allocator,
                device_id,
                model_id,
                replica_id,
                require_scoped,
            ))
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            let _ = (
                resources,
                backend,
                device_id,
                model_id,
                replica_id,
                require_scoped,
            );
            Err("native pack requires governed device memory, but this runtime was built without GPU pool authority".into())
        }
    }

    #[cfg_attr(not(any(test, feature = "gpu-device-pool")), allow(dead_code))]
    pub(super) fn with_allocator(
        allocator: Arc<dyn DeviceAllocator>,
        device_id: u32,
        model_id: u32,
        replica_id: u32,
        require_scoped: bool,
    ) -> Self {
        Self::from_allocator(Some(Arc::new(GovernedDeviceHost {
            allocator,
            device_id,
            model_id,
            replica_id,
            require_scoped,
            ledger: Mutex::new(AllocationLedger::default()),
        })))
    }

    fn from_allocator(allocator: Option<Arc<GovernedDeviceHost>>) -> Self {
        let base = KapslBackendHostV1 {
            struct_size: std::mem::size_of::<KapslBackendHostV1>() as u32,
            abi_version: KAPSL_BACKEND_ABI_VERSION,
            user_data: allocator
                .as_ref()
                .map_or(std::ptr::null_mut(), |a| Arc::as_ptr(a) as *mut c_void),
            log: Some(host_log),
            allocate_device: allocator
                .as_ref()
                .map(|_| allocate_device as KapslDeviceAllocateFn),
            free_device: allocator.as_ref().map(|_| free_device as KapslDeviceFreeFn),
            synchronize_device: allocator
                .as_ref()
                .map(|_| synchronize_device as KapslDeviceSynchronizeFn),
        };
        let table = if allocator.is_some() {
            KapslBackendHostScopedAllocatorV1::new(base, allocate_device_scoped)
        } else {
            KapslBackendHostScopedAllocatorV1 {
                base,
                scoped_allocator_version: 0,
                reserved: 0,
                allocate_device_scoped: None,
            }
        };
        Self {
            table: Box::new(table),
            allocator,
        }
    }

    pub(super) fn table(&self) -> *const KapslBackendHostV1 {
        &self.table.base
    }

    pub(super) fn begin_model_call(&self) -> Result<NativeAllocationCall<'_>, String> {
        let Some(host) = self.allocator.as_deref() else {
            return Ok(NativeAllocationCall {
                host: None,
                operation: 0,
            });
        };
        let mut ledger = host.ledger.lock().unwrap_or_else(|p| p.into_inner());
        if ledger.model_operation.is_some() || !ledger.requests.is_empty() || ledger.cleanup_pending
        {
            return Err(
                "native model lifecycle started with outstanding calls or incomplete cleanup"
                    .into(),
            );
        }
        ledger.enabled = true;
        let operation = ledger.next_operation()?;
        ledger.model_operation = Some(operation);
        Ok(NativeAllocationCall {
            host: Some(host),
            operation,
        })
    }

    pub(super) fn begin_requests(
        &self,
        requests: impl IntoIterator<Item = (u64, Option<CancellationToken>)>,
    ) -> Result<NativeAllocationCall<'_>, EngineError> {
        let requests: Vec<_> = requests.into_iter().collect();
        if requests
            .iter()
            .any(|(_, token)| token.as_ref().is_some_and(CancellationToken::is_cancelled))
        {
            return Err(EngineError::cancelled(
                "native request was cancelled before allocation admission",
            ));
        }
        let Some(host) = self.allocator.as_deref() else {
            return Ok(NativeAllocationCall {
                host: None,
                operation: 0,
            });
        };
        let mut ledger = host.ledger.lock().unwrap_or_else(|p| p.into_inner());
        if !ledger.enabled || ledger.model_operation.is_some() {
            return Err(EngineError::backend(
                "native allocator is outside the loaded lifecycle",
            ));
        }
        let mut unique = HashSet::new();
        if requests.is_empty()
            || requests
                .iter()
                .any(|(id, _)| *id == 0 || !unique.insert(*id) || ledger.requests.contains_key(id))
        {
            return Err(EngineError::backend(
                "invalid native request ownership registration",
            ));
        }
        let operation = ledger.next_operation().map_err(EngineError::backend)?;
        for (id, cancellation) in requests {
            ledger.requests.insert(
                id,
                ActiveRequest {
                    operation,
                    cancellation,
                    cancelled: false,
                },
            );
        }
        Ok(NativeAllocationCall {
            host: Some(host),
            operation,
        })
    }

    pub(super) fn stop_allocating(&self) {
        if let Some(host) = &self.allocator {
            let mut ledger = host.ledger.lock().unwrap_or_else(|p| p.into_inner());
            ledger.enabled = false;
            ledger.cleanup_pending = true;
        }
    }

    pub(super) fn reclaim(&self) -> Result<(), String> {
        self.allocator
            .as_ref()
            .map_or(Ok(()), |host| host.reclaim())
    }

    pub(super) fn live_allocations(&self) -> usize {
        self.allocator.as_ref().map_or(0, |host| {
            host.ledger
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .allocations
                .len()
        })
    }

    pub(super) fn live_bytes(&self) -> usize {
        self.allocator.as_ref().map_or(0, |host| {
            host.ledger
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .allocations
                .values()
                .fold(0usize, |bytes, allocation| {
                    bytes.saturating_add(allocation.storage.bytes())
                })
        })
    }

    pub(super) fn actual_memory(&self, mut report: MemoryReport) -> MemoryReport {
        use kapsl_engine_api::{MemoryAllocationClass as Class, MemoryDomain};
        let Some(host) = &self.allocator else {
            return report;
        };
        let domain = MemoryDomain::Cuda {
            device_id: host.device_id as usize,
        };
        // Governed device bytes come from the engine's live ledger. Adapter
        // reports remain authoritative for its other memory domains.
        report
            .allocations
            .retain(|allocation| allocation.domain != domain);
        let ledger = host.ledger.lock().unwrap_or_else(|p| p.into_inner());
        let mut allocations: Vec<_> = ledger.allocations.iter().collect();
        allocations.sort_by_key(|(id, _)| **id);
        for (id, allocation) in allocations {
            let scope = &allocation.scope;
            let class = match allocation.class {
                KAPSL_ALLOCATION_CLASS_WEIGHTS => Class::PersistentWeights,
                KAPSL_ALLOCATION_CLASS_WORKSPACE => Class::TransientWorkspace,
                KAPSL_ALLOCATION_CLASS_KV => Class::KvCache,
                KAPSL_ALLOCATION_CLASS_REQUEST => Class::RequestTransient,
                _ => Class::ExternallyOwned,
            };
            report.extend(MemoryReport::runtime(
                format!("native/device-{}/model-{}/replica-{}/scope-{}-{}/requests-{:?}/allocation-{id}", host.device_id, scope.model_id, scope.replica_id, scope.kind, scope.id, scope.request_ids),
                domain.clone(), class, allocation.storage.bytes(),
            ));
        }
        report
    }
}

impl GovernedDeviceHost {
    pub(super) fn cancel(&self, id: u64) {
        if let Some(request) = self
            .ledger
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .requests
            .get_mut(&id)
        {
            request.cancelled = true;
        }
    }

    unsafe fn allocate_scoped(
        &self,
        request: KapslScopedDeviceAllocationRequestV1,
    ) -> Result<KapslDeviceAllocationV1, String> {
        if !request.is_well_formed() {
            return Err("malformed scoped device allocation request".into());
        }
        if request.device_id != self.device_id
            || request.scope.model_id != self.model_id
            || request.scope.replica_id != self.replica_id
        {
            return Err("device allocation ownership does not match its backend instance".into());
        }
        let mut ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        if !ledger.enabled {
            return Err("native allocator is closed for this lifecycle".into());
        }
        // Bound the borrowed array before dereferencing it, including on
        // 32-bit hosts. The adapter owns readable storage for this callback.
        if request.scope.request_count as usize > ledger.requests.len() {
            return Err("allocation references requests not owned by this instance".into());
        }
        if !request.scope.request_ids.is_null()
            && !(request.scope.request_ids as usize).is_multiple_of(std::mem::align_of::<u64>())
        {
            return Err("allocation request IDs are misaligned".into());
        }
        // SAFETY: cardinality/alignment checked; storage lifetime is the ABI caller's contract.
        let ids =
            unsafe { request.scope.request_ids() }.ok_or("missing allocation request ownership")?;
        let operation = if ids.is_empty() {
            ledger
                .model_operation
                .ok_or("model/replica allocations require an active model lifecycle call")?
        } else {
            let mut unique = HashSet::new();
            let mut operation = None;
            for id in ids {
                if *id == 0 || !unique.insert(*id) {
                    return Err("allocation request IDs must be non-zero and distinct".into());
                }
                let active = ledger
                    .requests
                    .get(id)
                    .ok_or("allocation request is not active in this instance")?;
                if active.cancelled
                    || active
                        .cancellation
                        .as_ref()
                        .is_some_and(CancellationToken::is_cancelled)
                {
                    return Err("allocation request has been cancelled".into());
                }
                if operation.is_some_and(|operation| operation != active.operation) {
                    return Err("allocation batch combines unrelated native dispatches".into());
                }
                operation = Some(active.operation);
            }
            operation.expect("nonempty validated request IDs")
        };
        let scope = AllocationScope {
            kind: request.scope.scope_kind,
            id: request.scope.scope_id,
            model_id: self.model_id,
            replica_id: self.replica_id,
            request_ids: ids.to_vec(),
            operation,
        };
        let scope = if let Some(existing) = ledger.scopes.get(&scope.id) {
            if **existing != scope {
                return Err("allocation scope ID was rebound to different ownership".into());
            }
            Arc::clone(existing)
        } else {
            ledger.remember_scope_id(scope.id)?;
            let scope = Arc::new(scope);
            ledger.scopes.insert(scope.id, Arc::clone(&scope));
            scope
        };
        self.allocate_locked(
            &mut ledger,
            scope,
            request.memory_kind,
            request.allocation_class,
            request.bytes,
            request.alignment,
        )
    }

    fn allocate_legacy(
        &self,
        request: KapslDeviceAllocationRequestV1,
    ) -> Result<KapslDeviceAllocationV1, String> {
        if self.require_scoped {
            return Err(
                "backend requires scoped allocation; the legacy callback cannot bypass ownership"
                    .into(),
            );
        }
        if request.reserved != 0
            || request.flags != 0
            || request.device_id != self.device_id
            || request.model_id != self.model_id
            || request.replica_id != self.replica_id
        {
            return Err("invalid legacy device allocation ownership or flags".into());
        }
        let mut ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        if !ledger.enabled {
            return Err("native allocator is closed for this lifecycle".into());
        }
        let (operation, ids) = if let Some(operation) = ledger.model_operation {
            (operation, Vec::new())
        } else {
            let mut operations = ledger.requests.values().map(|r| r.operation);
            let first = operations
                .next()
                .ok_or("legacy allocation has no active owner")?;
            if operations.any(|operation| operation != first) {
                return Err("legacy allocation has ambiguous request ownership".into());
            }
            if ledger.requests.values().any(|r| {
                r.cancelled
                    || r.cancellation
                        .as_ref()
                        .is_some_and(CancellationToken::is_cancelled)
            }) {
                return Err("legacy allocation request has been cancelled".into());
            }
            (first, ledger.requests.keys().copied().collect::<Vec<_>>())
        };
        let kind = match ids.len() {
            0 => KAPSL_ALLOCATION_SCOPE_REPLICA,
            1 => KAPSL_ALLOCATION_SCOPE_REQUEST,
            _ => KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH,
        };
        // Scope ID zero is diagnostic-only for the legacy ABI. It never enters
        // the adapter's scoped-ID namespace or relaxes the scoped contract.
        let scope = Arc::new(AllocationScope {
            kind,
            id: 0,
            model_id: self.model_id,
            replica_id: self.replica_id,
            request_ids: ids,
            operation,
        });
        self.allocate_locked(
            &mut ledger,
            scope,
            request.memory_kind,
            request.allocation_class,
            request.bytes,
            request.alignment,
        )
    }

    fn allocate_locked(
        &self,
        ledger: &mut AllocationLedger,
        scope: Arc<AllocationScope>,
        memory_kind: u32,
        class: u32,
        bytes: u64,
        alignment: u64,
    ) -> Result<KapslDeviceAllocationV1, String> {
        if memory_kind != KAPSL_MEMORY_CUDA
            || !matches!(
                class,
                KAPSL_ALLOCATION_CLASS_WEIGHTS
                    | KAPSL_ALLOCATION_CLASS_WORKSPACE
                    | KAPSL_ALLOCATION_CLASS_KV
                    | KAPSL_ALLOCATION_CLASS_REQUEST
                    | KAPSL_ALLOCATION_CLASS_OTHER
            )
        {
            return Err("unsupported device memory kind or allocation class".into());
        }
        let bytes = usize::try_from(bytes).map_err(|_| "allocation bytes exceed this platform")?;
        let alignment =
            usize::try_from(alignment).map_err(|_| "allocation alignment exceeds this platform")?;
        if bytes == 0 || !alignment.is_power_of_two() || bytes.checked_add(alignment - 1).is_none()
        {
            return Err("invalid device allocation size or alignment".into());
        }
        let id = NEXT_ALLOCATION_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "native allocation IDs exhausted")?;
        let storage = self
            .allocator
            .allocate(class, bytes, alignment)
            .map_err(|error| format!("device {} scope {scope:?}: {error}", self.device_id))?;
        let allocation = KapslDeviceAllocationV1 {
            struct_size: std::mem::size_of::<KapslDeviceAllocationV1>() as u32,
            reserved: 0,
            allocation_id: id,
            device_ptr: storage.pointer() as *mut c_void,
            granted_bytes: storage.bytes() as u64,
        };
        log::debug!(
            "native governed allocate id={id} device={} class={class} bytes={} scope={scope:?}",
            self.device_id,
            allocation.granted_bytes
        );
        ledger.allocations.insert(
            id,
            LiveAllocation {
                storage,
                class,
                scope,
            },
        );
        Ok(allocation)
    }

    fn free(&self, returned: KapslDeviceAllocationV1) -> Result<(), String> {
        if returned.reserved != 0 || returned.allocation_id == 0 {
            return Err("invalid device free identity".into());
        }
        let mut ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        let allocation = ledger
            .allocations
            .get(&returned.allocation_id)
            .ok_or("device free references an unknown allocation ID")?;
        if allocation.storage.pointer() != returned.device_ptr as usize
            || allocation.storage.bytes() as u64 != returned.granted_bytes
        {
            return Err("device free pointer or bytes do not match its allocation ID".into());
        }
        // A successful free makes the range immediately reusable by another
        // model. Never rely on an adapter's declaration that it synchronized.
        self.allocator.synchronize()?;
        allocation.storage.free()?;
        log::debug!(
            "native governed free id={} device={} class={} bytes={} scope={:?}",
            returned.allocation_id,
            self.device_id,
            allocation.class,
            returned.granted_bytes,
            allocation.scope
        );
        ledger.allocations.remove(&returned.allocation_id);
        Ok(())
    }

    fn reclaim(&self) -> Result<(), String> {
        let mut ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        ledger.enabled = false;
        ledger.cleanup_pending = true;
        if !ledger.requests.is_empty() || ledger.model_operation.is_some() {
            return Err("cannot reclaim memory while native calls are active".into());
        }
        if ledger.allocations.is_empty() {
            ledger.cleanup_pending = false;
            return Ok(());
        }
        self.allocator.synchronize()?;
        let ids: Vec<_> = ledger.allocations.keys().copied().collect();
        let mut failure = None;
        for id in ids {
            let allocation = &ledger.allocations[&id];
            log::warn!(
                "reclaim native allocation id={id} device={} class={} bytes={} scope={:?}",
                self.device_id,
                allocation.class,
                allocation.storage.bytes(),
                allocation.scope
            );
            match allocation.storage.free() {
                Ok(()) => {
                    ledger.allocations.remove(&id);
                }
                Err(error) => {
                    failure = Some(error);
                }
            }
        }
        ledger.cleanup_pending = failure.is_some();
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for GovernedDeviceHost {
    fn drop(&mut self) {
        if let Err(error) = self.reclaim() {
            log::error!("native allocation cleanup failed; unreclaimed ranges remain charged and unavailable: {error}");
            // Retain the pool and allocation tokens when synchronization or a
            // free failed. Reusing or dropping potentially busy storage would
            // violate isolation; process teardown is the terminal boundary.
            let ledger = self.ledger.get_mut().unwrap_or_else(|p| p.into_inner());
            for (_, allocation) in ledger.allocations.drain() {
                std::mem::forget(allocation);
            }
        }
    }
}

// Read only the size prefix until the caller proves a complete ABI object.
unsafe fn read_sized<T: Copy>(pointer: *const T) -> Result<T, String> {
    if pointer.is_null() || !(pointer as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err("null or misaligned native allocator argument".into());
    }
    // SAFETY: non-null input storage includes its u32 size prefix by ABI contract.
    if unsafe { pointer.cast::<u32>().read() } < std::mem::size_of::<T>() as u32 {
        return Err("truncated native allocator argument".into());
    }
    // SAFETY: the advertised size covers the complete object.
    Ok(unsafe { pointer.read() })
}

unsafe extern "C" fn allocate_device_scoped(
    user_data: *mut c_void,
    request: *const KapslScopedDeviceAllocationRequestV1,
    out: *mut KapslDeviceAllocationV1,
) -> i32 {
    allocation_callback(user_data, out, |host| {
        // SAFETY: pointers are borrowed for the synchronous host callback.
        unsafe { host.allocate_scoped(read_sized(request)?) }
    })
}

unsafe extern "C" fn allocate_device(
    user_data: *mut c_void,
    request: *const KapslDeviceAllocationRequestV1,
    out: *mut KapslDeviceAllocationV1,
) -> i32 {
    allocation_callback(user_data, out, |host| {
        // SAFETY: request remains borrowed for this callback.
        host.allocate_legacy(unsafe { read_sized(request)? })
    })
}

fn allocation_callback(
    user_data: *mut c_void,
    out: *mut KapslDeviceAllocationV1,
    allocate: impl FnOnce(&GovernedDeviceHost) -> Result<KapslDeviceAllocationV1, String>,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null()
            || out.is_null()
            || !(out as usize).is_multiple_of(std::mem::align_of::<KapslDeviceAllocationV1>())
        {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: caller supplies a writable output slot and the retained host context.
        unsafe {
            *out = KapslDeviceAllocationV1::empty();
        }
        let host = unsafe { &*user_data.cast::<GovernedDeviceHost>() };
        match allocate(host) {
            Ok(allocation) => {
                unsafe {
                    *out = allocation;
                }
                KAPSL_STATUS_OK
            }
            Err(error) => {
                log::error!(
                    "native governed allocation rejected device={} model={} replica={}: {error}",
                    host.device_id,
                    host.model_id,
                    host.replica_id
                );
                KAPSL_STATUS_BACKEND_ERROR
            }
        }
    }))
    .unwrap_or(KAPSL_STATUS_PANIC)
}

unsafe extern "C" fn free_device(
    user_data: *mut c_void,
    allocation: *const KapslDeviceAllocationV1,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        let host = unsafe { &*user_data.cast::<GovernedDeviceHost>() };
        let result = unsafe { read_sized(allocation) }.and_then(|allocation| host.free(allocation));
        match result {
            Ok(()) => KAPSL_STATUS_OK,
            Err(error) => {
                log::error!("native governed free rejected: {error}");
                KAPSL_STATUS_BACKEND_ERROR
            }
        }
    }))
    .unwrap_or(KAPSL_STATUS_PANIC)
}

unsafe extern "C" fn synchronize_device(user_data: *mut c_void, device_id: u32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        let host = unsafe { &*user_data.cast::<GovernedDeviceHost>() };
        if device_id != host.device_id {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        match host.allocator.synchronize() {
            Ok(()) => KAPSL_STATUS_OK,
            Err(error) => {
                log::error!("native device synchronization failed: {error}");
                KAPSL_STATUS_BACKEND_ERROR
            }
        }
    }))
    .unwrap_or(KAPSL_STATUS_PANIC)
}

#[cfg(feature = "gpu-device-pool")]
use kapsl_hal::gpu_arena::{
    GpuAllocation, GpuDevicePool, PoolAllocationClass, PoolBackend, PoolOwner,
};

#[cfg(feature = "gpu-device-pool")]
fn pool_backend(backend: &str) -> PoolBackend {
    match backend.trim().to_ascii_lowercase().as_str() {
        "onnx" | "ort" | "onnxruntime" => PoolBackend::Onnx,
        "llama.cpp" | "llama_cpp" | "llama-cpp" => PoolBackend::Gguf,
        _ => PoolBackend::Native,
    }
}

#[cfg(feature = "gpu-device-pool")]
struct GpuAllocator {
    pool: Arc<GpuDevicePool>,
    backend: PoolBackend,
    model_id: u32,
    replica_id: u32,
}

#[cfg(feature = "gpu-device-pool")]
struct GpuStorage {
    pool: Arc<GpuDevicePool>,
    allocation: GpuAllocation,
}

#[cfg(feature = "gpu-device-pool")]
impl DeviceAllocator for GpuAllocator {
    fn allocate(
        &self,
        class: u32,
        bytes: usize,
        alignment: usize,
    ) -> Result<Box<dyn DeviceAllocation>, String> {
        let class = match class {
            KAPSL_ALLOCATION_CLASS_WEIGHTS => PoolAllocationClass::PersistentWeights,
            KAPSL_ALLOCATION_CLASS_WORKSPACE => PoolAllocationClass::TransientWorkspace,
            KAPSL_ALLOCATION_CLASS_KV => PoolAllocationClass::KvCache,
            KAPSL_ALLOCATION_CLASS_REQUEST => PoolAllocationClass::RequestTransient,
            KAPSL_ALLOCATION_CLASS_OTHER => PoolAllocationClass::ExternallyOwned,
            _ => return Err("unknown device allocation class".into()),
        };
        let owner = PoolOwner::new(self.backend, self.model_id, self.replica_id, class);
        if !self
            .pool
            .snapshot()
            .owners
            .iter()
            .any(|entry| entry.owner.workload() == owner.workload() && entry.admitted)
        {
            return Err(format!(
                "native device owner {owner:?} has no engine memory admission"
            ));
        }
        // The engine holds this owner's memory lease through unload. The pool
        // atomically enforces its aggregate quota and other owners' guarantees.
        let allocation = self
            .pool
            .alloc(owner, bytes, alignment)
            .map_err(|e| e.to_string())?;
        // The HAL aligns offsets in its arena; validate the absolute address
        // as well before publishing an alignment promise through the ABI.
        let pointer = self.pool.allocation_ptr(&allocation) as usize;
        if pointer == 0 || !pointer.is_multiple_of(alignment) {
            self.pool.free(allocation).map_err(|e| e.to_string())?;
            return Err("device pool cannot satisfy the requested pointer alignment".into());
        }
        Ok(Box::new(GpuStorage {
            pool: Arc::clone(&self.pool),
            allocation,
        }))
    }
    fn synchronize(&self) -> Result<(), String> {
        self.pool
            .device()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        self.pool.device().synchronize().map_err(|e| e.to_string())
    }
}

#[cfg(feature = "gpu-device-pool")]
impl DeviceAllocation for GpuStorage {
    fn pointer(&self) -> usize {
        self.pool.allocation_ptr(&self.allocation) as usize
    }
    fn bytes(&self) -> usize {
        self.allocation.bytes()
    }
    fn free(&self) -> Result<(), String> {
        self.pool
            .free(self.allocation.clone())
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
pub(super) mod tests;
