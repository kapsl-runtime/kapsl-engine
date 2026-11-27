//! Pooled GGUF tasks (embed, rerank, classify) behind the backend-neutral
//! `kapsl_backend_v1` function table.
//!
//! Generation keeps the legacy `kapsl_llama_cpp_backend_v1` table, which is
//! built around llama.cpp's KV cache. Pooled tasks have no KV state, so they
//! use the standard tensor ABI instead: Kapsl core selects this table only
//! when the signed pack manifest declares the `kapsl-backend-v1` adapter
//! contract for the model's task. Both tables live in the same library, so a
//! process that serves a chat model and an embedding model loads llama.cpp
//! once.
//!
//! Only the CPU pack exports this table. Accelerator packs on the standard ABI
//! must route device memory through the governed allocator, and llama.cpp's
//! CUDA buffers do not support that yet.

use crate::cancellation::{
    spawn_monitor, ActiveCancellations, CancellationProbe, CancellationRegistration,
};
use crate::ffi::{clear_output, json_error, path_from_slice, with_ffi_error, write_output};
use crate::logger::HostLogger;
use kapsl_backend_abi::*;
use kapsl_engine_api::{
    BinaryTensorPacket, CancellationToken, Engine as _, EngineError, InferenceRequest,
    MemoryDomain, MemoryReport, NamedTensor, RequestMetadata, TensorDtype,
};
use kapsl_llm::{GgufPooledBackend, GgufPooledConfig};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

/// Signed backend identity (`backend` in the pack manifest).
const BACKEND: &str = "llama-cpp";
/// The only profile that exports this table.
const PROFILE: &str = "cpu";
/// Upper bound on tensor rank accepted from the host.
const MAX_TENSOR_RANK: usize = 32;

const CAPABILITIES: u64 = KAPSL_BACKEND_CAP_CPU
    | KAPSL_BACKEND_CAP_STREAMING
    | KAPSL_BACKEND_CAP_CANCELLATION
    | KAPSL_BACKEND_CAP_MEMORY_REPORTING
    | KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE;

/// Formats and tasks this table serves. They must equal the signed pack
/// manifest's `formats` and `tasks`; the host rejects any difference.
pub(crate) const FORMATS: [&str; 1] = ["gguf"];
pub(crate) const TASKS: [&str; 3] = ["classify", "embed", "rerank"];

struct PooledPack {
    runtime: tokio::runtime::Runtime,
    backend: RwLock<GgufPooledBackend>,
    cancellations: ActiveCancellations,
    monitor_stop: Arc<AtomicBool>,
    monitor: Option<std::thread::JoinHandle<()>>,
    logger: HostLogger,
}

impl PooledPack {
    fn register_request(
        &self,
        request_id: u64,
        context: *mut c_void,
        callback: Option<KapslRequestCancelledFn>,
        request: &mut InferenceRequest,
    ) -> CancellationRegistration {
        let token = CancellationToken::new();
        request.cancellation = Some(token.clone());
        self.cancellations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                request_id,
                CancellationProbe {
                    context: context as usize,
                    callback,
                    token,
                },
            );
        CancellationRegistration {
            request_id,
            active: Arc::clone(&self.cancellations),
        }
    }

    fn stop(&mut self) {
        self.backend
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .unload();
        self.monitor_stop.store(true, Ordering::Release);
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }
        self.logger
            .emit(KAPSL_LOG_INFO, "shut down pooled llama.cpp backend");
    }
}

unsafe fn pooled<'a>(handle: *mut c_void) -> Result<&'a PooledPack, (i32, String)> {
    if handle.is_null() {
        return Err((
            KAPSL_STATUS_INVALID_ARGUMENT,
            "null pack handle".to_string(),
        ));
    }
    // SAFETY: handles originate from `initialize` and live until `shutdown`.
    Ok(unsafe { &*(handle as *mut PooledPack) })
}

/// Invalid requests map to `INVALID_ARGUMENT` so the host answers them as
/// client errors rather than backend failures.
fn engine_status(error: EngineError) -> (i32, String) {
    let status = match &error {
        EngineError::InvalidInput { .. } => KAPSL_STATUS_INVALID_ARGUMENT,
        EngineError::Cancelled { .. } => KAPSL_STATUS_CANCELLED,
        _ => KAPSL_STATUS_BACKEND_ERROR,
    };
    (status, error.to_string())
}

fn write_json<T: serde::Serialize>(
    output: *mut KapslOwnedBuffer,
    value: &T,
    label: &str,
) -> Result<(), (i32, String)> {
    let bytes = serde_json::to_vec(value).map_err(json_error)?;
    // SAFETY: output storage is supplied by the ABI caller.
    unsafe { write_output(output, bytes) }
        .map_err(|status| (status, format!("null {label} output")))
}

/// The CPU pack never owns device memory: report every row on the host.
fn host_memory(mut report: MemoryReport) -> MemoryReport {
    for allocation in &mut report.allocations {
        allocation.domain = MemoryDomain::Host;
    }
    report
}

// ─── Tensor conversion ────────────────────────────────────────────────────────

fn dtype_from_abi(dtype: u32) -> Result<TensorDtype, (i32, String)> {
    Ok(match dtype {
        KAPSL_DTYPE_F32 => TensorDtype::Float32,
        KAPSL_DTYPE_F64 => TensorDtype::Float64,
        KAPSL_DTYPE_F16 => TensorDtype::Float16,
        KAPSL_DTYPE_I32 => TensorDtype::Int32,
        KAPSL_DTYPE_I64 => TensorDtype::Int64,
        KAPSL_DTYPE_U8 => TensorDtype::Uint8,
        KAPSL_DTYPE_UTF8 => TensorDtype::Utf8,
        other => {
            return Err((
                KAPSL_STATUS_UNSUPPORTED,
                format!("pooled llama.cpp pack does not accept tensor dtype {other}"),
            ))
        }
    })
}

fn dtype_to_abi(dtype: TensorDtype) -> u32 {
    match dtype {
        TensorDtype::Float32 => KAPSL_DTYPE_F32,
        TensorDtype::Float64 => KAPSL_DTYPE_F64,
        TensorDtype::Float16 => KAPSL_DTYPE_F16,
        TensorDtype::Int32 => KAPSL_DTYPE_I32,
        TensorDtype::Int64 => KAPSL_DTYPE_I64,
        TensorDtype::Uint8 => KAPSL_DTYPE_U8,
        TensorDtype::Utf8 => KAPSL_DTYPE_UTF8,
    }
}

/// Copy one borrowed host tensor into an owned packet.
unsafe fn packet_from_view(
    view: &KapslNamedTensorViewV1,
) -> Result<(String, BinaryTensorPacket), (i32, String)> {
    let invalid = |message: String| (KAPSL_STATUS_INVALID_ARGUMENT, message);
    if view.struct_size < std::mem::size_of::<KapslNamedTensorViewV1>() as u32
        || view.tensor.struct_size < std::mem::size_of::<KapslTensorViewV1>() as u32
    {
        return Err((
            KAPSL_STATUS_INCOMPATIBLE_ABI,
            "input tensor struct is smaller than ABI v1".to_string(),
        ));
    }
    // SAFETY: the name is borrowed for the duration of the ABI call.
    let name = unsafe { view.name.as_bytes() }
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .unwrap_or_default();
    let tensor = &view.tensor;
    if !matches!(
        tensor.memory_kind,
        KAPSL_MEMORY_HOST | KAPSL_MEMORY_HOST_PINNED
    ) {
        return Err(invalid(format!(
            "input `{name}` is not in host memory (kind {})",
            tensor.memory_kind
        )));
    }
    let rank = tensor.rank as usize;
    if rank > MAX_TENSOR_RANK || (rank > 0 && tensor.shape.is_null()) {
        return Err(invalid(format!("input `{name}` has invalid rank {rank}")));
    }
    let shape = if rank == 0 {
        Vec::new()
    } else {
        // SAFETY: rank is bounded and the shape array is borrowed for this call.
        unsafe { std::slice::from_raw_parts(tensor.shape, rank) }.to_vec()
    };
    if !tensor.strides.is_null() && tensor.flags & KAPSL_TENSOR_FLAG_CONTIGUOUS == 0 {
        return Err(invalid(format!("input `{name}` is not contiguous")));
    }
    let len = usize::try_from(tensor.byte_len)
        .map_err(|_| invalid(format!("input `{name}` is too large for this platform")))?;
    if len > 0 && tensor.data.is_null() {
        return Err(invalid(format!("input `{name}` has null storage")));
    }
    let data = if len == 0 {
        Vec::new()
    } else {
        // SAFETY: non-null storage of `len` bytes is borrowed for this call.
        unsafe { std::slice::from_raw_parts(tensor.data.cast::<u8>(), len) }.to_vec()
    };
    let packet = BinaryTensorPacket::new(shape, dtype_from_abi(tensor.dtype)?, data)
        .map_err(|error| invalid(format!("input `{name}`: {error}")))?;
    Ok((name, packet))
}

#[derive(serde::Deserialize, Default)]
struct WireMetadata {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    metadata: Option<RequestMetadata>,
}

/// Decode a wire request. The first input is the primary tensor; the rest
/// keep their names (`text.<n>`, `document.<n>`).
unsafe fn request_from_wire(
    request: *const KapslInferenceRequestV1,
) -> Result<(KapslInferenceRequestV1, InferenceRequest), (i32, String)> {
    if request.is_null() {
        return Err((KAPSL_STATUS_INVALID_ARGUMENT, "null request".to_string()));
    }
    // SAFETY: checked non-null; the ABI requires a full v1 request.
    let wire = unsafe { *request };
    if wire.struct_size < std::mem::size_of::<KapslInferenceRequestV1>() as u32
        || wire.wire_format != KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1
    {
        return Err((
            KAPSL_STATUS_INCOMPATIBLE_ABI,
            "unsupported request struct or wire format".to_string(),
        ));
    }
    let count = wire.input_count as usize;
    if count == 0 || wire.inputs.is_null() {
        return Err((
            KAPSL_STATUS_INVALID_ARGUMENT,
            "request has no input tensors".to_string(),
        ));
    }
    // SAFETY: the host passes `input_count` views borrowed for this call.
    let views = unsafe { std::slice::from_raw_parts(wire.inputs, count) };
    let mut packets = Vec::with_capacity(count);
    for view in views {
        // SAFETY: each view is borrowed for this call.
        packets.push(unsafe { packet_from_view(view) }?);
    }
    let mut packets = packets.into_iter();
    let (_, input) = packets.next().expect("count checked above");
    let mut decoded = InferenceRequest::new(input);
    decoded.additional_inputs = packets
        .map(|(name, tensor)| NamedTensor { name, tensor })
        .collect();
    // SAFETY: metadata bytes are borrowed for this call.
    if let Some(bytes) = unsafe { wire.metadata_json.as_bytes() }.filter(|b| !b.is_empty()) {
        let metadata: WireMetadata = serde_json::from_slice(bytes).map_err(|error| {
            (
                KAPSL_STATUS_INVALID_ARGUMENT,
                format!("decode request metadata JSON: {error}"),
            )
        })?;
        decoded.session_id = metadata.session_id;
        decoded.metadata = metadata.metadata;
    }
    Ok((wire, decoded))
}

/// One output tensor owned by the pack until `release_result`.
struct OwnedResult {
    view: KapslNamedTensorViewV1,
    name: Vec<u8>,
    shape: Vec<i64>,
    data: Vec<u8>,
}

fn owned_result(name: &str, packet: BinaryTensorPacket) -> KapslInferenceResultV1 {
    let owned = Box::into_raw(Box::new(OwnedResult {
        view: KapslNamedTensorViewV1 {
            struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
            reserved: 0,
            name: KapslSlice::empty(),
            tensor: KapslTensorViewV1::empty(),
        },
        name: name.as_bytes().to_vec(),
        shape: packet.shape,
        data: packet.data,
    }));
    // SAFETY: `owned` is a fresh, uniquely owned heap allocation. The view
    // points into it, and both are freed together in `release_result`.
    let owned_ref = unsafe { &mut *owned };
    owned_ref.view.name = KapslSlice::from_bytes(&owned_ref.name);
    owned_ref.view.tensor = KapslTensorViewV1 {
        struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
        dtype: dtype_to_abi(packet.dtype),
        memory_kind: KAPSL_MEMORY_HOST,
        flags: KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY,
        device_id: -1,
        rank: owned_ref.shape.len() as u32,
        shape: owned_ref.shape.as_ptr(),
        strides: std::ptr::null(),
        data: owned_ref.data.as_ptr().cast(),
        byte_len: owned_ref.data.len() as u64,
    };
    KapslInferenceResultV1 {
        struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
        output_count: 1,
        outputs: &owned_ref.view,
        metadata_json: KapslSlice::empty(),
        owner_context: owned.cast(),
    }
}

/// Run one wire request and return its output tensor and name.
unsafe fn run_request(
    pack: &PooledPack,
    request: *const KapslInferenceRequestV1,
) -> Result<(String, BinaryTensorPacket), (i32, String)> {
    // SAFETY: forwarded from the ABI caller.
    let (wire, mut decoded) = unsafe { request_from_wire(request) }?;
    let _registration = pack.register_request(
        wire.request_id,
        wire.cancellation_context,
        wire.is_cancelled,
        &mut decoded,
    );
    let backend = pack
        .backend
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let name = backend
        .model_info()
        .and_then(|info| info.output_names.into_iter().next())
        .unwrap_or_else(|| "output".to_string());
    let packet = backend.infer(&decoded).map_err(engine_status)?;
    Ok((name, packet))
}

// ─── ABI entry points ─────────────────────────────────────────────────────────

unsafe extern "C" fn describe(
    descriptor_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(descriptor_out) };
    unsafe {
        with_ffi_error(error_out, || {
            write_json(
                descriptor_out,
                &serde_json::json!({
                    "schema_version": KAPSL_BACKEND_DESCRIPTOR_SCHEMA_V1,
                    "backend": BACKEND,
                    "profiles": [PROFILE],
                    "formats": FORMATS,
                    "tasks": TASKS,
                    "backend_abi": KAPSL_BACKEND_ABI_VERSION,
                    "wire_format": KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
                    "execution_mode": "native",
                    "governed_device_memory": false,
                }),
                "descriptor",
            )
        })
    }
}

unsafe extern "C" fn initialize(
    config: *const KapslBackendConfigV1,
    handle_out: *mut *mut c_void,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    if !handle_out.is_null() {
        // SAFETY: checked non-null.
        unsafe { *handle_out = std::ptr::null_mut() };
    }
    unsafe {
        with_ffi_error(error_out, || {
            if config.is_null() || handle_out.is_null() {
                return Err((
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    "initialize requires config and handle output".to_string(),
                ));
            }
            let config = &*config;
            if config.struct_size < std::mem::size_of::<KapslBackendConfigV1>() as u32 {
                return Err((
                    KAPSL_STATUS_INCOMPATIBLE_ABI,
                    "backend config is smaller than ABI v1".to_string(),
                ));
            }
            let profile = config.profile.as_bytes().unwrap_or_default();
            if profile != PROFILE.as_bytes() {
                return Err((
                    KAPSL_STATUS_UNSUPPORTED,
                    format!(
                        "pooled llama.cpp table serves profile `{PROFILE}`, not `{}`",
                        String::from_utf8_lossy(profile)
                    ),
                ));
            }
            if config.require_governed_device_memory != 0 {
                return Err((
                    KAPSL_STATUS_UNSUPPORTED,
                    "the CPU llama.cpp pack owns no device memory to govern".to_string(),
                ));
            }
            let manifest: serde_json::Value = serde_json::from_slice(
                config.manifest_json.as_bytes().unwrap_or(b"{}"),
            )
            .map_err(|error| {
                (
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    format!("decode model manifest JSON: {error}"),
                )
            })?;
            let pooled_config = GgufPooledConfig::from_manifest_json(&manifest)
                .map_err(|message| (KAPSL_STATUS_INVALID_ARGUMENT, message))?;
            let logger = if config.host.is_null() {
                HostLogger {
                    user_data: 0,
                    callback: None,
                }
            } else {
                let host = &*config.host;
                if host.struct_size < std::mem::size_of::<KapslBackendHostV1>() as u32 {
                    return Err((
                        KAPSL_STATUS_INCOMPATIBLE_ABI,
                        "host callback table is smaller than ABI v1".to_string(),
                    ));
                }
                HostLogger {
                    user_data: host.user_data as usize,
                    callback: host.log,
                }
            };
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .thread_name("kapsl-llama-pooled")
                .build()
                .map_err(|error| {
                    (
                        KAPSL_STATUS_BACKEND_ERROR,
                        format!("create pooled llama.cpp runtime: {error}"),
                    )
                })?;
            let cancellations = ActiveCancellations::default();
            let monitor_stop = Arc::new(AtomicBool::new(false));
            let monitor = spawn_monitor(Arc::clone(&cancellations), Arc::clone(&monitor_stop))
                .map_err(|error| {
                    (
                        KAPSL_STATUS_BACKEND_ERROR,
                        format!("start cancellation monitor: {error}"),
                    )
                })?;
            logger.emit(
                KAPSL_LOG_INFO,
                &format!(
                    "initialized pooled llama.cpp backend for GGUF {}",
                    pooled_config.task.as_str()
                ),
            );
            let pack = PooledPack {
                runtime,
                backend: RwLock::new(GgufPooledBackend::new(
                    config.device_id as usize,
                    pooled_config,
                )),
                cancellations,
                monitor_stop,
                monitor: Some(monitor),
                logger,
            };
            *handle_out = Box::into_raw(Box::new(pack)).cast();
            Ok(())
        })
    }
}

unsafe extern "C" fn planned_memory(
    handle: *mut c_void,
    model_path: KapslSlice,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = pooled(handle)?;
            let path = path_from_slice(model_path)?;
            let report = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .planned_memory(&path)
                .map_err(engine_status)?;
            write_json(report_out, &host_memory(report), "memory report")
        })
    }
}

unsafe extern "C" fn load_model(
    handle: *mut c_void,
    model_path: KapslSlice,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            let pack = pooled(handle)?;
            let path = path_from_slice(model_path)?;
            pack.logger.emit(
                KAPSL_LOG_INFO,
                &format!("loading pooled GGUF model {}", path.display()),
            );
            let mut backend = pack
                .backend
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            pack.runtime
                .block_on(backend.load(&path))
                .map_err(engine_status)
        })
    }
}

unsafe extern "C" fn planned_request_memory(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = pooled(handle)?;
            let (_, decoded) = request_from_wire(request)?;
            let report = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .planned_request_memory(&decoded);
            write_json(report_out, &host_memory(report), "request memory report")
        })
    }
}

unsafe extern "C" fn infer(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    result_out: *mut KapslInferenceResultV1,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    if !result_out.is_null() {
        // SAFETY: caller-owned output storage.
        unsafe { *result_out = KapslInferenceResultV1::empty() };
    }
    unsafe {
        with_ffi_error(error_out, || {
            if result_out.is_null() {
                return Err((
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    "null inference result output".to_string(),
                ));
            }
            let pack = pooled(handle)?;
            let (name, packet) = run_request(pack, request)?;
            *result_out = owned_result(&name, packet);
            Ok(())
        })
    }
}

unsafe extern "C" fn infer_stream(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    user_data: *mut c_void,
    on_chunk: Option<KapslBackendStreamChunkFn>,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            let callback = on_chunk.ok_or_else(|| {
                (
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    "stream callback is required".to_string(),
                )
            })?;
            let pack = pooled(handle)?;
            let request_id = if request.is_null() {
                0
            } else {
                (*request).request_id
            };
            // A pooled task produces its whole output at once: one chunk.
            let (name, packet) = run_request(pack, request)?;
            let mut result = owned_result(&name, packet);
            let status = callback(user_data, request_id, &result);
            release_result(handle, &mut result);
            if status != KAPSL_STATUS_OK {
                return Err((
                    KAPSL_STATUS_CANCELLED,
                    "stream consumer cancelled request".to_string(),
                ));
            }
            Ok(())
        })
    }
}

unsafe extern "C" fn cancel(handle: *mut c_void, request_id: u64) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let pack = pooled(handle)?;
        let active = pack
            .cancellations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(probe) = active.get(&request_id) else {
            return Err((
                KAPSL_STATUS_INVALID_ARGUMENT,
                "unknown request id".to_string(),
            ));
        };
        probe.token.cancel();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => KAPSL_STATUS_OK,
        Ok(Err((status, _))) => status,
        Err(_) => KAPSL_STATUS_PANIC,
    }
}

unsafe extern "C" fn actual_memory(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = pooled(handle)?;
            let report = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .actual_memory();
            write_json(report_out, &host_memory(report), "memory report")
        })
    }
}

unsafe extern "C" fn metrics(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = pooled(handle)?;
            let metrics = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .metrics();
            write_json(report_out, &metrics, "metrics")
        })
    }
}

unsafe extern "C" fn model_info(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = pooled(handle)?;
            let info = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .model_info()
                .ok_or_else(|| {
                    (
                        KAPSL_STATUS_BACKEND_ERROR,
                        "pooled GGUF model is not loaded".to_string(),
                    )
                })?;
            write_json(report_out, &info, "model info")
        })
    }
}

unsafe extern "C" fn batching_policy(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            pooled(handle)?;
            // Inputs are batched within a request; requests are not coalesced,
            // so one caller's oversized input cannot fail another's request.
            write_json(
                report_out,
                &serde_json::json!({"mode": "none", "max_requests": 1}),
                "batching policy",
            )
        })
    }
}

unsafe extern "C" fn health_check(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            pooled(handle)?
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .health_check()
                .map_err(engine_status)
        })
    }
}

unsafe extern "C" fn unload(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            pooled(handle)?
                .backend
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .unload();
            Ok(())
        })
    }
}

unsafe extern "C" fn shutdown(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        // SAFETY: shutdown consumes the handle exactly once.
        let mut pack = Box::from_raw(handle as *mut PooledPack);
        pack.stop();
    }));
}

unsafe extern "C" fn release_result(_handle: *mut c_void, result: *mut KapslInferenceResultV1) {
    if result.is_null() {
        return;
    }
    // SAFETY: the host passes a result this table produced, at most once.
    let result = unsafe { &mut *result };
    if !result.owner_context.is_null() {
        // SAFETY: `owner_context` came from `Box::into_raw` in `owned_result`.
        unsafe { drop(Box::from_raw(result.owner_context.cast::<OwnedResult>())) };
    }
    *result = KapslInferenceResultV1::empty();
}

static API_V1: KapslBackendApiV1 = KapslBackendApiV1 {
    magic: KAPSL_BACKEND_ENTRYPOINT_MAGIC,
    abi_version: KAPSL_BACKEND_ABI_VERSION,
    struct_size: std::mem::size_of::<KapslBackendApiV1>() as u32,
    wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
    capabilities: CAPABILITIES,
    describe: Some(describe),
    initialize: Some(initialize),
    planned_memory: Some(planned_memory),
    load_model: Some(load_model),
    planned_request_memory: Some(planned_request_memory),
    infer: Some(infer),
    infer_batch: None,
    infer_stream: Some(infer_stream),
    cancel: Some(cancel),
    actual_memory: Some(actual_memory),
    metrics: Some(metrics),
    model_info: Some(model_info),
    kv_capabilities: None,
    kv_topology: None,
    batching_policy: Some(batching_policy),
    health_check: Some(health_check),
    unload: Some(unload),
    shutdown: Some(shutdown),
    release_result: Some(release_result),
    release_batch_result: None,
    free_buffer: Some(crate::ffi::free_buffer),
};

/// Return the immutable backend-neutral ABI v1 table for pooled GGUF tasks.
#[no_mangle]
pub extern "C" fn kapsl_backend_v1() -> *const KapslBackendApiV1 {
    &API_V1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_complete_and_consistent() {
        let api = unsafe { &*kapsl_backend_v1() };
        assert!(api.is_compatible());
        assert!(api.has_required_functions());
        assert!(api.capabilities_are_consistent());
        assert_eq!(
            api.capabilities & KAPSL_BACKEND_CAP_EXECUTION_MASK,
            KAPSL_BACKEND_CAP_CPU
        );
        assert_eq!(
            api.capabilities & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
            0
        );
        assert_eq!(api.capabilities & KAPSL_BACKEND_CAP_BATCHING, 0);
    }

    #[test]
    fn descriptor_matches_the_signed_contract() {
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        assert_eq!(
            unsafe { describe(&mut output, &mut error) },
            KAPSL_STATUS_OK
        );
        let bytes = unsafe { std::slice::from_raw_parts(output.ptr, output.len) }.to_vec();
        unsafe { crate::ffi::free_buffer(output) };
        let descriptor: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(descriptor["backend"], BACKEND);
        assert_eq!(descriptor["profiles"], serde_json::json!(["cpu"]));
        assert_eq!(descriptor["formats"], serde_json::json!(["gguf"]));
        assert_eq!(
            descriptor["tasks"],
            serde_json::json!(["classify", "embed", "rerank"])
        );
        assert_eq!(descriptor["execution_mode"], "native");
        assert_eq!(descriptor["governed_device_memory"], false);
    }

    fn config(profile: &[u8], manifest: &[u8], governed: u32) -> KapslBackendConfigV1 {
        KapslBackendConfigV1 {
            struct_size: std::mem::size_of::<KapslBackendConfigV1>() as u32,
            device_id: 0,
            model_id: 1,
            replica_id: 0,
            require_governed_device_memory: governed,
            reserved: 0,
            profile: KapslSlice::from_bytes(profile),
            manifest_json: KapslSlice::from_bytes(manifest),
            options_json: KapslSlice::from_bytes(b"{}"),
            host: std::ptr::null(),
        }
    }

    fn initialize_status(config: &KapslBackendConfigV1) -> (i32, *mut c_void) {
        let mut handle = std::ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe { initialize(config, &mut handle, &mut error) };
        if !error.ptr.is_null() {
            unsafe { crate::ffi::free_buffer(error) };
        }
        (status, handle)
    }

    #[test]
    fn initialize_fails_closed_on_wrong_contracts() {
        let embed = br#"{"task":"embed"}"#;
        let (status, handle) = initialize_status(&config(b"cuda12", embed, 0));
        assert_eq!(status, KAPSL_STATUS_UNSUPPORTED);
        assert!(handle.is_null());
        let (status, handle) = initialize_status(&config(b"cpu", embed, 1));
        assert_eq!(status, KAPSL_STATUS_UNSUPPORTED);
        assert!(handle.is_null());
        let (status, handle) = initialize_status(&config(b"cpu", br#"{"task":"generate"}"#, 0));
        assert_eq!(status, KAPSL_STATUS_INVALID_ARGUMENT);
        assert!(handle.is_null());
    }

    #[test]
    fn initialize_and_shutdown_an_unloaded_backend() {
        let (status, handle) = initialize_status(&config(b"cpu", br#"{"task":"rerank"}"#, 0));
        assert_eq!(status, KAPSL_STATUS_OK);
        assert!(!handle.is_null());
        let mut error = KapslOwnedBuffer::empty();
        assert_ne!(
            unsafe { health_check(handle, &mut error) },
            KAPSL_STATUS_OK,
            "an unloaded backend is unhealthy"
        );
        if !error.ptr.is_null() {
            unsafe { crate::ffi::free_buffer(error) };
        }
        unsafe { shutdown(handle) };
    }

    #[test]
    fn wire_requests_round_trip_named_text_inputs() {
        let query = b"what is a panda?";
        let document = b"a bear";
        let query_shape = [query.len() as i64];
        let document_shape = [document.len() as i64];
        let view = |name: &'static [u8], shape: &[i64; 1], data: &[u8]| KapslNamedTensorViewV1 {
            struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
            reserved: 0,
            name: KapslSlice::from_bytes(name),
            tensor: KapslTensorViewV1 {
                struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
                dtype: KAPSL_DTYPE_UTF8,
                memory_kind: KAPSL_MEMORY_HOST,
                flags: KAPSL_TENSOR_FLAG_CONTIGUOUS,
                device_id: -1,
                rank: 1,
                shape: shape.as_ptr(),
                strides: std::ptr::null(),
                data: data.as_ptr().cast(),
                byte_len: data.len() as u64,
            },
        };
        let views = [
            view(b"input", &query_shape, query),
            view(b"document.0", &document_shape, document),
        ];
        let metadata = br#"{"session_id":"s1","metadata":null}"#;
        let wire = KapslInferenceRequestV1 {
            struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
            wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            request_id: 7,
            inputs: views.as_ptr(),
            input_count: 2,
            reserved: 0,
            metadata_json: KapslSlice::from_bytes(metadata),
            cancellation_context: std::ptr::null_mut(),
            is_cancelled: None,
        };
        let (_, decoded) = unsafe { request_from_wire(&wire) }.unwrap();
        assert_eq!(decoded.input.data, query);
        assert_eq!(decoded.input.dtype, TensorDtype::Utf8);
        assert_eq!(decoded.additional_inputs.len(), 1);
        assert_eq!(decoded.additional_inputs[0].name, "document.0");
        assert_eq!(decoded.additional_inputs[0].tensor.data, document);
        assert_eq!(decoded.session_id.as_deref(), Some("s1"));
    }

    #[test]
    fn owned_results_expose_the_packet_until_released() {
        let packet = BinaryTensorPacket::new(
            vec![2],
            TensorDtype::Float32,
            [1.5f32, -2.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        )
        .unwrap();
        let mut result = owned_result("scores", packet);
        assert_eq!(result.output_count, 1);
        let view = unsafe { &*result.outputs };
        assert_eq!(unsafe { view.name.as_bytes() }, Some(&b"scores"[..]));
        assert_eq!(view.tensor.dtype, KAPSL_DTYPE_F32);
        assert_eq!(view.tensor.byte_len, 8);
        assert_eq!(unsafe { *view.tensor.shape }, 2);
        unsafe { release_result(std::ptr::null_mut(), &mut result) };
        assert!(result.owner_context.is_null());
        assert_eq!(result.output_count, 0);
    }
}
