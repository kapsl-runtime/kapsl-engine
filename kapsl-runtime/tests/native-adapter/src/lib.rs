//! A host-only ABI conformance fixture. Device addresses are opaque handles
//! supplied by the engine's fake pool; this library has no compute backend.
use kapsl_backend_abi::*;
use kapsl_engine_api::{EngineMetrics, MemoryAllocationClass, MemoryDomain, MemoryReport};
use serde_json::json;
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

pub const CAPABILITIES: u64 = KAPSL_BACKEND_CAP_CUDA
    | KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR
    | KAPSL_BACKEND_CAP_SCOPED_DEVICE_ALLOCATOR
    | KAPSL_BACKEND_CAP_MEMORY_REPORTING
    | KAPSL_BACKEND_CAP_BATCHING
    | KAPSL_BACKEND_CAP_STREAMING
    | KAPSL_BACKEND_CAP_CANCELLATION
    | KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE;

struct State {
    host: KapslBackendHostScopedAllocatorV1,
    device: u32,
    model: u32,
    replica: u32,
    mode: String,
    options: serde_json::Value,
    scope: AtomicU64,
    loaded: AtomicBool,
    allocations: Mutex<Vec<KapslDeviceAllocationV1>>,
    cancelled: Mutex<HashSet<u64>>,
    changed: Condvar,
    calls: AtomicU64,
    cancels: AtomicU64,
    unloads: AtomicU64,
    chunks: AtomicU64,
    last_batch: AtomicU64,
}

// The host retains the callback table/context through shutdown. All mutable
// state is atomic or mutex-protected, including concurrently accessed handles.
unsafe impl Send for State {}
unsafe impl Sync for State {}

unsafe fn state<'a>(handle: *mut c_void) -> &'a State {
    unsafe { &*handle.cast::<State>() }
}

fn output(bytes: Vec<u8>, out: *mut KapslOwnedBuffer) {
    let mut bytes = bytes;
    unsafe {
        *out = KapslOwnedBuffer {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        };
    }
    std::mem::forget(bytes);
}

unsafe extern "C" fn free_buffer(buffer: KapslOwnedBuffer) {
    if !buffer.ptr.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.capacity));
        }
    }
}

unsafe extern "C" fn describe(out: *mut KapslOwnedBuffer, _error: *mut KapslOwnedBuffer) -> i32 {
    output(
        serde_json::to_vec(&json!({
            "schema_version": KAPSL_BACKEND_DESCRIPTOR_SCHEMA_V1,
            "backend": "fake-native", "profiles": ["cuda12"],
            "backend_abi": KAPSL_BACKEND_ABI_VERSION,
            "wire_format": KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            "execution_mode": "native", "governed_device_memory": true,
            "formats": ["onnx"], "tasks": ["forward"]
        }))
        .unwrap(),
        out,
    );
    KAPSL_STATUS_OK
}

impl State {
    fn allocate(&self, kind: u32, requests: &[u64], bytes: u64) -> i32 {
        let scope = KapslDeviceAllocationScopeV1::new(
            kind,
            self.scope.fetch_add(1, Ordering::Relaxed),
            self.model,
            self.replica,
            requests,
        );
        let class = if requests.is_empty() {
            KAPSL_ALLOCATION_CLASS_WEIGHTS
        } else {
            KAPSL_ALLOCATION_CLASS_WORKSPACE
        };
        let request = KapslScopedDeviceAllocationRequestV1::new(
            self.device,
            KAPSL_MEMORY_CUDA,
            class,
            scope,
            bytes,
            64,
        );
        let mut allocation = KapslDeviceAllocationV1::empty();
        let status = unsafe {
            self.host.allocate_device_scoped.unwrap()(
                self.host.base.user_data,
                &request,
                &mut allocation,
            )
        };
        if status == KAPSL_STATUS_OK {
            self.allocations.lock().unwrap().push(allocation);
        }
        status
    }

    fn release_allocations(&self) -> i32 {
        let mut allocations = self.allocations.lock().unwrap();
        allocations.retain(|allocation| unsafe { self.host.base.free_device.unwrap()(self.host.base.user_data, allocation) } != KAPSL_STATUS_OK);
        if allocations.is_empty() {
            KAPSL_STATUS_OK
        } else {
            KAPSL_STATUS_BACKEND_ERROR
        }
    }

    fn wait_for_cancel(&self, request_id: u64) -> i32 {
        let guard = self.cancelled.lock().unwrap();
        let (guard, _) = self
            .changed
            .wait_timeout_while(guard, Duration::from_secs(5), |ids| {
                !ids.contains(&request_id)
            })
            .unwrap();
        if guard.contains(&request_id) {
            // Cancellation revokes new allocation authority before the
            // adapter callback runs, while existing allocations stay charged.
            if self.allocate(KAPSL_ALLOCATION_SCOPE_REQUEST, &[request_id], 64) == KAPSL_STATUS_OK {
                return KAPSL_STATUS_BACKEND_ERROR;
            }
            KAPSL_STATUS_CANCELLED
        } else {
            KAPSL_STATUS_BACKEND_ERROR
        }
    }
}

unsafe extern "C" fn initialize(
    config: *const KapslBackendConfigV1,
    handle: *mut *mut c_void,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    let config = unsafe { &*config };
    let Some(host) = (unsafe { KapslBackendHostScopedAllocatorV1::from_base(config.host) }) else {
        return KAPSL_STATUS_INCOMPATIBLE_ABI;
    };
    if config.require_governed_device_memory != 1
        || host.base.struct_size as usize
            != std::mem::size_of::<KapslBackendHostScopedAllocatorV1>()
        || host.base.user_data.is_null()
        || host.base.log.is_none()
    {
        return KAPSL_STATUS_INCOMPATIBLE_ABI;
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(unsafe { config.manifest_json.as_bytes() }.unwrap()).unwrap();
    let state = Box::new(State {
        host: *host,
        device: config.device_id,
        model: config.model_id,
        replica: config.replica_id,
        mode: manifest["project_name"].as_str().unwrap_or("normal").into(),
        options: serde_json::from_slice(unsafe { config.options_json.as_bytes() }.unwrap())
            .unwrap(),
        scope: AtomicU64::new(1),
        loaded: AtomicBool::new(false),
        allocations: Mutex::new(Vec::new()),
        cancelled: Mutex::new(HashSet::new()),
        changed: Condvar::new(),
        calls: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        unloads: AtomicU64::new(0),
        chunks: AtomicU64::new(0),
        last_batch: AtomicU64::new(0),
    });
    if state.mode.starts_with("fail-initialize") || state.mode == "initialize-allocates" {
        let status = state.allocate(KAPSL_ALLOCATION_SCOPE_MODEL, &[], 256);
        if status != KAPSL_STATUS_OK {
            return status;
        }
        if state.mode == "initialize-allocates" {
            unsafe {
                *handle = Box::into_raw(state).cast();
            }
            return KAPSL_STATUS_OK;
        }
        if state.mode != "fail-initialize-null" {
            unsafe {
                *handle = Box::into_raw(state).cast();
            }
        }
        return KAPSL_STATUS_BACKEND_ERROR;
    }
    unsafe {
        *handle = Box::into_raw(state).cast();
    }
    KAPSL_STATUS_OK
}

fn report_memory(state: &State, bytes: usize, out: *mut KapslOwnedBuffer) {
    output(
        serde_json::to_vec(&MemoryReport::runtime(
            "fake-memory",
            MemoryDomain::Cuda {
                device_id: state.device as usize,
            },
            MemoryAllocationClass::PersistentWeights,
            bytes,
        ))
        .unwrap(),
        out,
    );
}

unsafe extern "C" fn planned_memory(
    handle: *mut c_void,
    _path: KapslSlice,
    out: *mut KapslOwnedBuffer,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    report_memory(unsafe { state(handle) }, 256, out);
    KAPSL_STATUS_OK
}

unsafe extern "C" fn load(
    handle: *mut c_void,
    path: KapslSlice,
    error: *mut KapslOwnedBuffer,
) -> i32 {
    let state = unsafe { state(handle) };
    if state.mode != "host-memory" {
        let status = state.allocate(KAPSL_ALLOCATION_SCOPE_MODEL, &[], 256);
        if status != KAPSL_STATUS_OK {
            return status;
        }
    }
    if state.mode == "fail-load" || state.mode == "fail-load-cleanup" {
        return KAPSL_STATUS_BACKEND_ERROR;
    }
    if state.mode == "package-assets" {
        let path =
            std::path::Path::new(std::str::from_utf8(unsafe { path.as_bytes() }.unwrap()).unwrap());
        let root = path.parent().and_then(std::path::Path::parent).unwrap();
        let assets: &[(&str, &[u8])] = &[
            ("graphs/model.onnx", b"fake model"),
            ("vocab.json", b"root vocabulary"),
            ("graphs/vocab.json", b"graph vocabulary"),
            ("assets/config.json", b"model configuration"),
        ];
        for (relative, expected) in assets {
            if std::fs::read(root.join(relative)).ok().as_deref() != Some(*expected) {
                output(
                    format!("missing package asset: {relative}").into_bytes(),
                    error,
                );
                return KAPSL_STATUS_BACKEND_ERROR;
            }
        }
    }
    state.loaded.store(true, Ordering::Release);
    KAPSL_STATUS_OK
}

unsafe extern "C" fn planned_request_memory(
    handle: *mut c_void,
    _request: *const KapslInferenceRequestV1,
    out: *mut KapslOwnedBuffer,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    report_memory(unsafe { state(handle) }, 64, out);
    KAPSL_STATUS_OK
}

struct OwnedResult {
    tensor: KapslNamedTensorViewV1,
    shape: [i64; 1],
    data: [u8; 1],
}

fn result() -> KapslInferenceResultV1 {
    let mut owned = Box::new(OwnedResult {
        shape: [1],
        data: [42],
        tensor: KapslNamedTensorViewV1 {
            struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
            reserved: 0,
            name: KapslSlice::from_bytes(b"output"),
            tensor: KapslTensorViewV1 {
                struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
                dtype: KAPSL_DTYPE_U8,
                memory_kind: KAPSL_MEMORY_HOST,
                flags: KAPSL_TENSOR_FLAG_CONTIGUOUS,
                device_id: -1,
                rank: 1,
                shape: std::ptr::null(),
                strides: std::ptr::null(),
                data: std::ptr::null(),
                byte_len: 1,
            },
        },
    });
    owned.tensor.tensor.shape = owned.shape.as_ptr();
    owned.tensor.tensor.data = owned.data.as_ptr().cast();
    KapslInferenceResultV1 {
        struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
        output_count: 1,
        outputs: &owned.tensor,
        metadata_json: KapslSlice::empty(),
        owner_context: Box::into_raw(owned).cast(),
    }
}

unsafe extern "C" fn release(_handle: *mut c_void, result: *mut KapslInferenceResultV1) {
    let result = unsafe { &mut *result };
    if !result.owner_context.is_null() {
        unsafe {
            drop(Box::from_raw(result.owner_context.cast::<OwnedResult>()));
        }
    }
    *result = KapslInferenceResultV1::empty();
}

unsafe extern "C" fn infer(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    out: *mut KapslInferenceResultV1,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    let state = unsafe { state(handle) };
    let request = unsafe { &*request };
    if !state.loaded.load(Ordering::Acquire) {
        return KAPSL_STATUS_BACKEND_ERROR;
    }
    let status = state.allocate(KAPSL_ALLOCATION_SCOPE_REQUEST, &[request.request_id], 64);
    if status != KAPSL_STATUS_OK {
        return status;
    }
    state.calls.fetch_add(1, Ordering::Release);
    if state.mode == "wait-cancel" {
        return state.wait_for_cancel(request.request_id);
    }
    if state.mode == "fail-infer" {
        return KAPSL_STATUS_BACKEND_ERROR;
    }
    if state.mode == "invalid-scope" {
        let ids = [request.request_id];
        let scope = KapslDeviceAllocationScopeV1::new(
            KAPSL_ALLOCATION_SCOPE_REQUEST,
            900,
            state.model + 1,
            state.replica,
            &ids,
        );
        let request = KapslScopedDeviceAllocationRequestV1::new(
            state.device,
            KAPSL_MEMORY_CUDA,
            KAPSL_ALLOCATION_CLASS_WORKSPACE,
            scope,
            64,
            64,
        );
        let mut allocation = KapslDeviceAllocationV1::empty();
        if unsafe {
            state.host.allocate_device_scoped.unwrap()(
                state.host.base.user_data,
                &request,
                &mut allocation,
            )
        } == KAPSL_STATUS_OK
        {
            return KAPSL_STATUS_BACKEND_ERROR;
        }
    }
    unsafe {
        *out = result();
    }
    KAPSL_STATUS_OK
}

unsafe extern "C" fn infer_batch(
    handle: *mut c_void,
    batch: *const KapslInferenceBatchV1,
    out: *mut KapslInferenceBatchResultV1,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    let state = unsafe { state(handle) };
    let batch = unsafe { &*batch };
    let requests =
        unsafe { std::slice::from_raw_parts(batch.requests, batch.request_count as usize) };
    let ids: Vec<_> = requests.iter().map(|r| r.request_id).collect();
    let kind = if ids.len() == 1 {
        KAPSL_ALLOCATION_SCOPE_REQUEST
    } else {
        KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH
    };
    let status = state.allocate(kind, &ids, 64 * ids.len() as u64);
    if status != KAPSL_STATUS_OK {
        return status;
    }
    state.last_batch.store(ids.len() as u64, Ordering::Release);
    state.calls.fetch_add(1, Ordering::Release);
    if state.mode == "wait-cancel" {
        return state.wait_for_cancel(ids[0]);
    }
    let results: Vec<_> = requests.iter().map(|_| result()).collect();
    let owned = Box::new(results);
    unsafe {
        *out = KapslInferenceBatchResultV1 {
            struct_size: std::mem::size_of::<KapslInferenceBatchResultV1>() as u32,
            result_count: owned.len() as u32,
            results: owned.as_ptr(),
            owner_context: Box::into_raw(owned).cast(),
        };
    }
    KAPSL_STATUS_OK
}

unsafe extern "C" fn release_batch(handle: *mut c_void, result: *mut KapslInferenceBatchResultV1) {
    let result = unsafe { &mut *result };
    let mut owned =
        unsafe { Box::from_raw(result.owner_context.cast::<Vec<KapslInferenceResultV1>>()) };
    for result in owned.iter_mut() {
        unsafe {
            release(handle, result);
        }
    }
    *result = KapslInferenceBatchResultV1::empty();
}

unsafe extern "C" fn infer_stream(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    user: *mut c_void,
    on_chunk: Option<KapslBackendStreamChunkFn>,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    let state = unsafe { state(handle) };
    let id = unsafe { (*request).request_id };
    let status = state.allocate(KAPSL_ALLOCATION_SCOPE_REQUEST, &[id], 64);
    if status != KAPSL_STATUS_OK {
        return status;
    }
    state.calls.fetch_add(1, Ordering::Release);
    for _ in 0..2 {
        let mut result = result();
        let chunk_id = if state.mode == "stream-wrong-owner" {
            id + 1
        } else {
            id
        };
        let status = unsafe { on_chunk.unwrap()(user, chunk_id, &result) };
        unsafe {
            release(handle, &mut result);
        }
        if status != KAPSL_STATUS_OK {
            return status;
        }
        state.chunks.fetch_add(1, Ordering::Release);
        if state.mode == "stream-wait-cancel" {
            return state.wait_for_cancel(id);
        }
    }
    KAPSL_STATUS_OK
}

unsafe extern "C" fn cancel(handle: *mut c_void, id: u64) -> i32 {
    let state = unsafe { state(handle) };
    state.cancelled.lock().unwrap().insert(id);
    state.cancels.fetch_add(1, Ordering::Release);
    state.changed.notify_all();
    KAPSL_STATUS_OK
}

unsafe extern "C" fn actual_memory(
    handle: *mut c_void,
    out: *mut KapslOwnedBuffer,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    let state = unsafe { state(handle) };
    let mut report = MemoryReport::runtime(
        "fake-memory",
        MemoryDomain::Cuda {
            device_id: state.device as usize,
        },
        MemoryAllocationClass::PersistentWeights,
        state
            .allocations
            .lock()
            .unwrap()
            .iter()
            .map(|a| a.granted_bytes as usize)
            .sum(),
    );
    if matches!(state.mode.as_str(), "host-memory" | "mixed-memory")
        && state.loaded.load(Ordering::Acquire)
    {
        report.extend(MemoryReport::single(
            "host-session",
            MemoryDomain::Host,
            MemoryAllocationClass::ModelSession,
            1024,
        ));
        report.extend(MemoryReport::single(
            "host-weights",
            MemoryDomain::Host,
            MemoryAllocationClass::PersistentWeights,
            512,
        ));
    }
    output(serde_json::to_vec(&report).unwrap(), out);
    KAPSL_STATUS_OK
}

unsafe extern "C" fn metrics(
    handle: *mut c_void,
    out: *mut KapslOwnedBuffer,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    let state = unsafe { state(handle) };
    let mut metrics = EngineMetrics::new();
    metrics.throughput = state.calls.load(Ordering::Acquire) as f64;
    metrics.batch_size = state.last_batch.load(Ordering::Acquire) as usize;
    metrics.memory_usage = state
        .allocations
        .lock()
        .unwrap()
        .iter()
        .map(|a| a.granted_bytes as usize)
        .sum();
    if matches!(state.mode.as_str(), "host-memory" | "mixed-memory") {
        // Reproduce an adapter with complete actual_memory reporting and an
        // unset legacy metric, including when it uses the governed pool.
        metrics.memory_usage = 0;
    }
    output(serde_json::to_vec(&metrics).unwrap(), out);
    KAPSL_STATUS_OK
}

unsafe extern "C" fn model_info(
    handle: *mut c_void,
    out: *mut KapslOwnedBuffer,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    output(serde_json::to_vec(&json!({"input_names": ["input"], "output_names": ["output"], "input_shapes": [[1]], "output_shapes": [[1]], "input_dtypes": ["uint8"], "output_dtypes": ["uint8"], "framework": "fake", "options": unsafe { state(handle) }.options})).unwrap(), out);
    KAPSL_STATUS_OK
}

unsafe extern "C" fn batching(
    _handle: *mut c_void,
    out: *mut KapslOwnedBuffer,
    _error: *mut KapslOwnedBuffer,
) -> i32 {
    output(
        br#"{"mode":"request_coalescing","max_requests":4}"#.to_vec(),
        out,
    );
    KAPSL_STATUS_OK
}

unsafe extern "C" fn health(handle: *mut c_void, _error: *mut KapslOwnedBuffer) -> i32 {
    if unsafe { state(handle) }.loaded.load(Ordering::Acquire) {
        KAPSL_STATUS_OK
    } else {
        KAPSL_STATUS_BACKEND_ERROR
    }
}

unsafe extern "C" fn unload(handle: *mut c_void, _error: *mut KapslOwnedBuffer) -> i32 {
    let state = unsafe { state(handle) };
    let previous = state.unloads.fetch_add(1, Ordering::AcqRel);
    if state.mode == "fail-load-cleanup" || (state.mode == "fail-unload" && previous == 0) {
        return KAPSL_STATUS_BACKEND_ERROR;
    }
    if state.mode == "leak-unload" {
        state.allocations.lock().unwrap().clear();
    } else {
        let status = state.release_allocations();
        if status != KAPSL_STATUS_OK {
            return status;
        }
    }
    state.loaded.store(false, Ordering::Release);
    KAPSL_STATUS_OK
}

unsafe extern "C" fn shutdown(handle: *mut c_void) {
    let state = unsafe { Box::from_raw(handle.cast::<State>()) };
    state.release_allocations();
}

/// Inspect synchronization points without making a control/report call that
/// would intentionally wait for the engine's in-flight inference guard.
#[no_mangle]
pub unsafe extern "C" fn kapsl_test_probe_v1(handle: *mut c_void, counter: u32) -> u64 {
    let state = unsafe { state(handle) };
    match counter {
        0 => state.calls.load(Ordering::Acquire),
        1 => state.cancels.load(Ordering::Acquire),
        2 => state.unloads.load(Ordering::Acquire),
        _ => state.chunks.load(Ordering::Acquire),
    }
}

static API: KapslBackendApiV1 = KapslBackendApiV1 {
    magic: KAPSL_BACKEND_ENTRYPOINT_MAGIC,
    abi_version: KAPSL_BACKEND_ABI_VERSION,
    struct_size: std::mem::size_of::<KapslBackendApiV1>() as u32,
    wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
    capabilities: CAPABILITIES,
    describe: Some(describe),
    initialize: Some(initialize),
    planned_memory: Some(planned_memory),
    load_model: Some(load),
    planned_request_memory: Some(planned_request_memory),
    infer: Some(infer),
    infer_batch: Some(infer_batch),
    infer_stream: Some(infer_stream),
    cancel: Some(cancel),
    actual_memory: Some(actual_memory),
    metrics: Some(metrics),
    model_info: Some(model_info),
    kv_capabilities: None,
    kv_topology: None,
    batching_policy: Some(batching),
    health_check: Some(health),
    unload: Some(unload),
    shutdown: Some(shutdown),
    release_result: Some(release),
    release_batch_result: Some(release_batch),
    free_buffer: Some(free_buffer),
};

#[no_mangle]
pub extern "C" fn kapsl_backend_v1() -> *const KapslBackendApiV1 {
    &API
}
