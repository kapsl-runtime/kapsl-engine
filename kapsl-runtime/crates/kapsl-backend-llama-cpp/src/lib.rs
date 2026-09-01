//! Native llama.cpp pack behind Kapsl's stable C function-table ABI.
//!
//! The pack owns all Rust implementation details. Core sees only opaque
//! handles, versioned C structs, JSON wire messages, and explicit buffer
//! ownership. CUDA builds can expose either the runtime-owned shared-pool path
//! or the separately signed native-KV rollback profile.

use futures::StreamExt;
use kapsl_backend_abi::*;
use kapsl_engine_api::{
    CancellationToken, Engine as _, InferenceRequest, MemoryDomain, MemoryReport,
};
#[cfg(kapsl_llama_external_pool_sdk)]
use kapsl_llm::gguf_backend::{
    GgufExternalKvPool, GgufExternalKvPoolFactory, GgufExternalKvPoolGeometry,
};
use kapsl_llm::GgufBackend;
#[cfg(kapsl_llama_external_pool_sdk)]
use llama_cpp_sys_2::{llama_kapsl_kv_pool_desc, LLAMA_KAPSL_KV_DTYPE_F16};
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

#[derive(Clone, Copy)]
struct HostLogger {
    user_data: usize,
    callback: Option<KapslLogFn>,
}

unsafe impl Send for HostLogger {}
unsafe impl Sync for HostLogger {}

impl HostLogger {
    fn emit(self, level: u32, message: &str) {
        if let Some(callback) = self.callback {
            // SAFETY: the host promises that its callback context remains live
            // until shutdown returns. The message borrow lasts for this call.
            unsafe {
                callback(
                    self.user_data as *mut c_void,
                    level,
                    KapslSlice::from_bytes(message.as_bytes()),
                );
            }
        }
    }
}

#[cfg(kapsl_llama_external_pool_sdk)]
#[derive(Clone, Copy)]
struct SharedPoolHost {
    user_data: usize,
    create: KapslCreateSharedPoolFn,
    destroy: KapslDestroySharedPoolFn,
    bytes: KapslSharedPoolBytesFn,
}

#[cfg(kapsl_llama_external_pool_sdk)]
struct SharedPoolBridge {
    descriptor: KapslSharedPoolDescriptorV1,
    host_user_data: usize,
    destroy: KapslDestroySharedPoolFn,
}

#[cfg(kapsl_llama_external_pool_sdk)]
unsafe impl Send for SharedPoolBridge {}

#[cfg(kapsl_llama_external_pool_sdk)]
impl Drop for SharedPoolBridge {
    fn drop(&mut self) {
        // SAFETY: the core callback context remains live through pack shutdown,
        // and this is the one matching destroy for the successful create call.
        unsafe {
            (self.destroy)(
                self.host_user_data as *mut c_void,
                self.descriptor.pool_context,
            );
        }
    }
}

#[cfg(kapsl_llama_external_pool_sdk)]
unsafe fn shared_bridge<'a>(user_data: *mut c_void) -> Option<&'a SharedPoolBridge> {
    if user_data.is_null() {
        None
    } else {
        // SAFETY: the bridge Box is the external pool's drop guard and therefore
        // outlives the llama context that invokes these callbacks.
        Some(unsafe { &*(user_data as *const SharedPoolBridge) })
    }
}

#[cfg(kapsl_llama_external_pool_sdk)]
unsafe extern "C" fn bridge_reserve(
    user_data: *mut c_void,
    session_id: u64,
    tokens_needed: u32,
    block_table_device_out: *mut *mut u32,
    blocks_out: *mut u32,
) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| bridge.descriptor.reserve.map(|callback| (bridge, callback)))
        .is_some_and(|(bridge, callback)| unsafe {
            callback(
                bridge.descriptor.pool_context,
                session_id,
                tokens_needed,
                block_table_device_out,
                blocks_out,
            ) != 0
        })
}

#[cfg(kapsl_llama_external_pool_sdk)]
unsafe extern "C" fn bridge_reserve_sequence(
    user_data: *mut c_void,
    sequence_id: u64,
    tokens_needed: u32,
    block_table_device_out: *mut *mut u32,
    blocks_out: *mut u32,
) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| {
            bridge
                .descriptor
                .reserve_sequence
                .map(|callback| (bridge, callback))
        })
        .is_some_and(|(bridge, callback)| unsafe {
            callback(
                bridge.descriptor.pool_context,
                sequence_id,
                tokens_needed,
                block_table_device_out,
                blocks_out,
            ) != 0
        })
}

#[cfg(kapsl_llama_external_pool_sdk)]
unsafe extern "C" fn bridge_commit_sequences(
    user_data: *mut c_void,
    block_table_device_out: *mut *mut u32,
) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| {
            bridge
                .descriptor
                .commit_sequences
                .map(|callback| (bridge, callback))
        })
        .is_some_and(|(bridge, callback)| unsafe {
            callback(bridge.descriptor.pool_context, block_table_device_out) != 0
        })
}

#[cfg(kapsl_llama_external_pool_sdk)]
unsafe extern "C" fn bridge_release(user_data: *mut c_void, sequence_id: u64) {
    if let Some((bridge, callback)) = unsafe { shared_bridge(user_data) }
        .and_then(|bridge| bridge.descriptor.release.map(|callback| (bridge, callback)))
    {
        unsafe { callback(bridge.descriptor.pool_context, sequence_id) };
    }
}

#[cfg(kapsl_llama_external_pool_sdk)]
unsafe extern "C" fn bridge_touch(user_data: *mut c_void, sequence_id: u64) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| bridge.descriptor.touch.map(|callback| (bridge, callback)))
        .is_some_and(|(bridge, callback)| unsafe {
            callback(bridge.descriptor.pool_context, sequence_id) != 0
        })
}

#[cfg(kapsl_llama_external_pool_sdk)]
fn checked_u32(value: usize, field: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{field} exceeds llama.cpp ABI range"))
}

#[cfg(kapsl_llama_external_pool_sdk)]
fn external_pool_factory(host: SharedPoolHost) -> GgufExternalKvPoolFactory {
    Arc::new(move |geometry: GgufExternalKvPoolGeometry| {
        let wire_geometry = KapslSharedPoolGeometryV1 {
            struct_size: std::mem::size_of::<KapslSharedPoolGeometryV1>() as u32,
            device_id: checked_u32(geometry.device_id, "device id")?,
            requested_blocks: u64::try_from(geometry.requested_blocks)
                .map_err(|_| "requested blocks exceed Kapsl ABI range".to_string())?,
            block_size_tokens: checked_u32(geometry.block_size_tokens, "block size")?,
            num_layers: checked_u32(geometry.num_layers, "layer count")?,
            num_kv_heads: checked_u32(geometry.num_kv_heads, "KV head count")?,
            key_head_dim: checked_u32(geometry.key_head_dim, "key head dimension")?,
            value_head_dim: checked_u32(geometry.value_head_dim, "value head dimension")?,
            element_bytes: 2,
            max_sequences: checked_u32(geometry.max_sequences, "sequence count")?,
            max_blocks_per_sequence: checked_u32(
                geometry.max_blocks_per_sequence,
                "per-sequence block count",
            )?,
            model_fingerprint: geometry.model_fingerprint,
        };
        let mut descriptor = KapslSharedPoolDescriptorV1 {
            struct_size: 0,
            pool_context: std::ptr::null_mut(),
            device_base: std::ptr::null_mut(),
            addressable_blocks: 0,
            block_table_device: std::ptr::null_mut(),
            block_table_layer_stride: 0,
            block_table_sequence_stride: 0,
            sequence_slots: 0,
            reserve: None,
            reserve_sequence: None,
            commit_sequences: None,
            release: None,
            touch: None,
        };
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: all arguments are stack-owned for this synchronous host call.
        let status = unsafe {
            (host.create)(
                host.user_data as *mut c_void,
                &wire_geometry,
                &mut descriptor,
                &mut error,
            )
        };
        if !error.ptr.is_null() {
            return Err(
                "host create_shared_pool returned an invalid cross-allocator error buffer"
                    .to_string(),
            );
        }
        if status != KAPSL_STATUS_OK {
            return Err(format!(
                "host create_shared_pool failed with status {status}"
            ));
        }
        if descriptor.struct_size < std::mem::size_of::<KapslSharedPoolDescriptorV1>() as u32
            || descriptor.pool_context.is_null()
            || descriptor.device_base.is_null()
        {
            if !descriptor.pool_context.is_null() {
                unsafe { (host.destroy)(host.user_data as *mut c_void, descriptor.pool_context) };
            }
            return Err("host returned an incomplete shared-pool descriptor".to_string());
        }
        let num_blocks = match u32::try_from(descriptor.addressable_blocks) {
            Ok(blocks) => blocks,
            Err(_) => {
                // SAFETY: create succeeded, so its descriptor requires exactly
                // one matching destroy even though it cannot be represented by
                // llama.cpp's u32 block-count ABI.
                unsafe { (host.destroy)(host.user_data as *mut c_void, descriptor.pool_context) };
                return Err("shared pool has more than u32::MAX addressable blocks".to_string());
            }
        };
        let bridge = Box::new(SharedPoolBridge {
            descriptor,
            host_user_data: host.user_data,
            destroy: host.destroy,
        });
        let bridge_pointer = (&*bridge as *const SharedPoolBridge).cast_mut().cast();
        let raw = llama_kapsl_kv_pool_desc {
            user_data: bridge_pointer,
            device_id: wire_geometry.device_id,
            block_size: wire_geometry.block_size_tokens,
            num_blocks,
            num_kv_heads: wire_geometry.num_kv_heads,
            head_dim: wire_geometry.key_head_dim,
            dtype: LLAMA_KAPSL_KV_DTYPE_F16,
            device_base: descriptor.device_base,
            block_table_device: descriptor.block_table_device,
            block_table_layer_stride: descriptor.block_table_layer_stride,
            n_layers: wire_geometry.num_layers,
            max_blocks_per_seq: wire_geometry.max_blocks_per_sequence,
            block_table_seq_stride: descriptor.block_table_sequence_stride,
            n_seq_slots: descriptor.sequence_slots,
            model_fingerprint: wire_geometry.model_fingerprint,
            reserve: Some(bridge_reserve),
            reserve_seq: descriptor
                .reserve_sequence
                .map(|_| bridge_reserve_sequence as _),
            commit_seq: descriptor
                .commit_sequences
                .map(|_| bridge_commit_sequences as _),
            release: Some(bridge_release),
            touch: descriptor.touch.map(|_| bridge_touch as _),
            reserve_prefix: None,
            promote_prefix: None,
            needs_restore: None,
        };
        let bytes_host = host;
        let pool_context = descriptor.pool_context as usize;
        let usage_bytes: Arc<dyn Fn() -> usize + Send + Sync> = Arc::new(move || {
            // SAFETY: Kapsl core retains the host context through pack shutdown.
            let bytes = unsafe {
                (bytes_host.bytes)(
                    bytes_host.user_data as *mut c_void,
                    pool_context as *mut c_void,
                )
            };
            usize::try_from(bytes).unwrap_or(usize::MAX)
        });
        // SAFETY: bridge owns the raw callback context and its Drop invokes the
        // matching core destroy callback after llama.cpp releases the context.
        unsafe { GgufExternalKvPool::from_raw(geometry, raw, bridge, usage_bytes) }
    })
}

#[derive(Clone)]
struct CancellationProbe {
    context: usize,
    callback: Option<KapslRequestCancelledFn>,
    token: CancellationToken,
}

struct PackState {
    runtime: tokio::runtime::Runtime,
    backend: RwLock<GgufBackend>,
    cancellations: Arc<Mutex<HashMap<u64, CancellationProbe>>>,
    monitor_stop: Arc<AtomicBool>,
    monitor: Option<std::thread::JoinHandle<()>>,
    logger: HostLogger,
    profile: u32,
}

impl PackState {
    fn new(config: &KapslLlamaConfigV1) -> Result<Self, String> {
        let expected_profile = if cfg!(feature = "cuda12") {
            KAPSL_LLAMA_PROFILE_CUDA12
        } else {
            KAPSL_LLAMA_PROFILE_CPU
        };
        if config.profile != expected_profile {
            return Err(format!(
                "pack profile mismatch: library profile={} requested={}",
                expected_profile, config.profile
            ));
        }
        let host_callbacks = if config.host.is_null() {
            None
        } else {
            // SAFETY: initialize validates and consumes the borrowed host table
            // synchronously. Only its stable callback/context values are copied.
            let host = unsafe { &*config.host };
            if host.struct_size < std::mem::size_of::<KapslLlamaHostCallbacksV1>() as u32 {
                return Err("host callback table is smaller than ABI v1".to_string());
            }
            Some(*host)
        };
        let logger = if let Some(host) = host_callbacks {
            HostLogger {
                user_data: host.user_data as usize,
                callback: host.log,
            }
        } else {
            HostLogger {
                user_data: 0,
                callback: None,
            }
        };
        #[cfg(kapsl_llama_external_pool_sdk)]
        let backend = {
            if config.require_shared_pool == 0 {
                return Err(
                    "signed shared-pool pack requires Kapsl core to set require_shared_pool"
                        .to_string(),
                );
            }
            let host = host_callbacks.ok_or_else(|| {
                "runtime-owned shared-pool pack requires a host callback table".to_string()
            })?;
            let shared_host = SharedPoolHost {
                user_data: host.user_data as usize,
                create: host
                    .create_shared_pool
                    .ok_or_else(|| "host callback table lacks create_shared_pool".to_string())?,
                destroy: host
                    .destroy_shared_pool
                    .ok_or_else(|| "host callback table lacks destroy_shared_pool".to_string())?,
                bytes: host
                    .shared_pool_bytes
                    .ok_or_else(|| "host callback table lacks shared_pool_bytes".to_string())?,
            };
            GgufBackend::new_cuda_external_kv_pool(
                config.device_id as usize,
                external_pool_factory(shared_host),
            )
        };
        #[cfg(not(kapsl_llama_external_pool_sdk))]
        let backend = {
            if config.require_shared_pool != 0 {
                return Err(
                    "this pack exposes llama.cpp native KV only; a runtime-owned shared-pool pack is required"
                        .to_string(),
                );
            }
            GgufBackend::new_on_device(config.device_id as usize)
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("kapsl-llama-pack")
            .build()
            .map_err(|error| format!("create llama.cpp pack runtime: {error}"))?;
        let cancellations = Arc::new(Mutex::new(HashMap::<u64, CancellationProbe>::new()));
        let monitor_stop = Arc::new(AtomicBool::new(false));
        let monitor_cancellations = Arc::clone(&cancellations);
        let monitor_signal = Arc::clone(&monitor_stop);
        let monitor = std::thread::Builder::new()
            .name("kapsl-llama-cancel".to_string())
            .spawn(move || {
                while !monitor_signal.load(Ordering::Acquire) {
                    {
                        let active = monitor_cancellations
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        for (request_id, probe) in active.iter() {
                            let cancelled = probe.callback.is_some_and(|callback| {
                                // SAFETY: the request context remains alive until
                                // its synchronous ABI call unregisters this probe.
                                unsafe { callback(probe.context as *mut c_void, *request_id) != 0 }
                            });
                            if cancelled {
                                probe.token.cancel();
                            }
                        }
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
            .map_err(|error| format!("start cancellation monitor: {error}"))?;
        logger.emit(KAPSL_LOG_INFO, "initialized native llama.cpp backend pack");
        Ok(Self {
            runtime,
            backend: RwLock::new(backend),
            cancellations,
            monitor_stop,
            monitor: Some(monitor),
            logger,
            profile: config.profile,
        })
    }

    fn normalize_memory(&self, mut report: MemoryReport) -> MemoryReport {
        if self.profile == KAPSL_LLAMA_PROFILE_CPU {
            for allocation in &mut report.allocations {
                if matches!(allocation.domain, MemoryDomain::Cuda { .. }) {
                    allocation.domain = MemoryDomain::Host;
                }
            }
        }
        report
    }

    fn register_request(
        &self,
        wire: &KapslLlamaRequestV1,
        request: &mut InferenceRequest,
    ) -> CancellationRegistration {
        let token = CancellationToken::new();
        request.cancellation = Some(token.clone());
        self.cancellations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                wire.request_id,
                CancellationProbe {
                    context: wire.cancellation_context as usize,
                    callback: wire.is_cancelled,
                    token,
                },
            );
        CancellationRegistration {
            request_id: wire.request_id,
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
            .emit(KAPSL_LOG_INFO, "shut down native llama.cpp backend pack");
    }
}

struct CancellationRegistration {
    request_id: u64,
    active: Arc<Mutex<HashMap<u64, CancellationProbe>>>,
}

impl Drop for CancellationRegistration {
    fn drop(&mut self) {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.request_id);
    }
}

fn owned_buffer(bytes: Vec<u8>) -> KapslOwnedBuffer {
    let mut bytes = std::mem::ManuallyDrop::new(bytes);
    KapslOwnedBuffer {
        ptr: bytes.as_mut_ptr(),
        len: bytes.len(),
        capacity: bytes.capacity(),
    }
}

unsafe fn clear_output(output: *mut KapslOwnedBuffer) {
    if !output.is_null() {
        // SAFETY: caller supplied writable ABI output storage.
        unsafe { *output = KapslOwnedBuffer::empty() };
    }
}

unsafe fn write_output(output: *mut KapslOwnedBuffer, bytes: Vec<u8>) -> Result<(), i32> {
    if output.is_null() {
        return Err(KAPSL_STATUS_INVALID_ARGUMENT);
    }
    // SAFETY: checked non-null and required writable by the ABI.
    unsafe { *output = owned_buffer(bytes) };
    Ok(())
}

unsafe fn write_error(output: *mut KapslOwnedBuffer, message: impl AsRef<str>) {
    if !output.is_null() {
        // SAFETY: caller supplied writable ABI output storage.
        unsafe { *output = owned_buffer(message.as_ref().as_bytes().to_vec()) };
    }
}

unsafe fn with_ffi_error(
    error_out: *mut KapslOwnedBuffer,
    operation: impl FnOnce() -> Result<(), (i32, String)>,
) -> i32 {
    // SAFETY: forwarded from the ABI caller.
    unsafe { clear_output(error_out) };
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => KAPSL_STATUS_OK,
        Ok(Err((status, message))) => {
            // SAFETY: forwarded from the ABI caller.
            unsafe { write_error(error_out, message) };
            status
        }
        Err(_) => {
            // SAFETY: forwarded from the ABI caller.
            unsafe { write_error(error_out, "native llama.cpp pack panicked") };
            KAPSL_STATUS_PANIC
        }
    }
}

unsafe fn state<'a>(handle: *mut c_void) -> Result<&'a PackState, (i32, String)> {
    if handle.is_null() {
        return Err((
            KAPSL_STATUS_INVALID_ARGUMENT,
            "null pack handle".to_string(),
        ));
    }
    // SAFETY: handles originate from `initialize` and live until `shutdown`.
    Ok(unsafe { &*(handle as *mut PackState) })
}

unsafe fn request_from_wire(
    request: *const KapslLlamaRequestV1,
) -> Result<(KapslLlamaRequestV1, InferenceRequest), (i32, String)> {
    if request.is_null() {
        return Err((KAPSL_STATUS_INVALID_ARGUMENT, "null request".to_string()));
    }
    // SAFETY: checked non-null; the ABI requires a full v1 request.
    let wire = unsafe { *request };
    if wire.struct_size < std::mem::size_of::<KapslLlamaRequestV1>() as u32
        || wire.wire_format != KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1
    {
        return Err((
            KAPSL_STATUS_INCOMPATIBLE_ABI,
            "unsupported request struct or wire format".to_string(),
        ));
    }
    // SAFETY: request bytes are borrowed for the duration of this call.
    let bytes = unsafe { wire.request_json.as_bytes() }.ok_or_else(|| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            "request JSON has a null pointer".to_string(),
        )
    })?;
    let parsed = serde_json::from_slice(bytes).map_err(|error| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            format!("decode inference request JSON: {error}"),
        )
    })?;
    Ok((wire, parsed))
}

unsafe extern "C" fn initialize(
    config: *const KapslLlamaConfigV1,
    handle_out: *mut *mut c_void,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    // SAFETY: outputs are ABI caller-owned.
    unsafe { clear_output(error_out) };
    if !handle_out.is_null() {
        // SAFETY: checked non-null.
        unsafe { *handle_out = std::ptr::null_mut() };
    }
    // SAFETY: all raw pointer access is validated inside the closure.
    unsafe {
        with_ffi_error(error_out, || {
            if config.is_null() || handle_out.is_null() {
                return Err((
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    "initialize requires config and handle output".to_string(),
                ));
            }
            let config = &*config;
            if config.struct_size < std::mem::size_of::<KapslLlamaConfigV1>() as u32 {
                return Err((
                    KAPSL_STATUS_INCOMPATIBLE_ABI,
                    "llama.cpp config is smaller than ABI v1".to_string(),
                ));
            }
            let pack = PackState::new(config).map_err(|message| {
                let status = if config.require_shared_pool != 0 {
                    KAPSL_STATUS_UNSUPPORTED
                } else {
                    KAPSL_STATUS_BACKEND_ERROR
                };
                (status, message)
            })?;
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
    // SAFETY: outputs are ABI caller-owned.
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let path = path_from_slice(model_path)?;
            let report = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .planned_memory(&path)
                .map_err(engine_error)?;
            let report = pack.normalize_memory(report);
            let bytes = serde_json::to_vec(&report).map_err(json_error)?;
            write_output(report_out, bytes)
                .map_err(|status| (status, "null memory report output".to_string()))
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
            let pack = state(handle)?;
            let path = path_from_slice(model_path)?;
            pack.logger.emit(
                KAPSL_LOG_INFO,
                &format!("loading GGUF model {}", path.display()),
            );
            pack.runtime
                .block_on(
                    pack.backend
                        .write()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .load(&path),
                )
                .map_err(engine_error)
        })
    }
}

unsafe extern "C" fn planned_request_memory(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let (_, request) = request_from_wire(request)?;
            let report = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .planned_request_memory(&request);
            let bytes = serde_json::to_vec(&report).map_err(json_error)?;
            write_output(report_out, bytes)
                .map_err(|status| (status, "null request memory output".to_string()))
        })
    }
}

unsafe extern "C" fn infer(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    packet_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(packet_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let (wire, mut request) = request_from_wire(request)?;
            let _registration = pack.register_request(&wire, &mut request);
            let packet = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .infer(&request)
                .map_err(engine_error)?;
            let bytes = serde_json::to_vec(&packet).map_err(json_error)?;
            write_output(packet_out, bytes)
                .map_err(|status| (status, "null inference packet output".to_string()))
        })
    }
}

unsafe extern "C" fn infer_stream(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    user_data: *mut c_void,
    on_chunk: Option<KapslLlamaStreamChunkFn>,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let callback = on_chunk.ok_or_else(|| {
                (
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    "stream callback is required".to_string(),
                )
            })?;
            let (wire, mut request) = request_from_wire(request)?;
            let request_id = wire.request_id;
            let _registration = pack.register_request(&wire, &mut request);
            let mut stream = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .infer_stream(&request);
            pack.runtime.block_on(async {
                while let Some(item) = stream.next().await {
                    let packet = item.map_err(engine_error)?;
                    let bytes = serde_json::to_vec(&packet).map_err(json_error)?;
                    let status = callback(user_data, request_id, KapslSlice::from_bytes(&bytes));
                    if status != KAPSL_STATUS_OK {
                        if let Some(probe) = pack
                            .cancellations
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .get(&request_id)
                            .cloned()
                        {
                            probe.token.cancel();
                        }
                        return Err((
                            KAPSL_STATUS_CANCELLED,
                            "stream consumer cancelled request".to_string(),
                        ));
                    }
                }
                Ok(())
            })
        })
    }
}

unsafe extern "C" fn cancel(handle: *mut c_void, request_id: u64) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let pack = state(handle)?;
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
    json_report(handle, report_out, error_out, |pack, backend| {
        serde_json::to_vec(&pack.normalize_memory(backend.actual_memory()))
    })
}

unsafe extern "C" fn metrics(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, backend| {
        serde_json::to_vec(&backend.metrics())
    })
}

unsafe extern "C" fn model_info(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, backend| {
        serde_json::to_vec(&backend.model_info())
    })
}

unsafe extern "C" fn kv_capabilities(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    // kapsl-engine-api 0.1.3 predates the neutral KV capability methods used by
    // newer cores. Native-KV packs are conservatively unmanaged; a future
    // shared-pool-capable pack will return the versioned capability document.
    json_report(handle, report_out, error_out, |_pack, _backend| {
        serde_json::to_vec(&serde_json::Value::Null)
    })
}

unsafe extern "C" fn kv_topology(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, _backend| {
        serde_json::to_vec(&serde_json::Value::Null)
    })
}

#[derive(Serialize)]
struct PackBatchingPolicy {
    max_batch: usize,
    self_batches: bool,
}

unsafe extern "C" fn batching_policy(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, backend| {
        serde_json::to_vec(&PackBatchingPolicy {
            max_batch: backend.max_batch(),
            self_batches: backend.self_batches(),
        })
    })
}

unsafe fn json_report(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
    make: impl FnOnce(&PackState, &GgufBackend) -> Result<Vec<u8>, serde_json::Error>,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let backend = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let bytes = make(pack, &backend).map_err(json_error)?;
            write_output(report_out, bytes)
                .map_err(|status| (status, "null JSON report output".to_string()))
        })
    }
}

unsafe extern "C" fn health_check(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            state(handle)?
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .health_check()
                .map_err(engine_error)
        })
    }
}

unsafe extern "C" fn unload(handle: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        if let Ok(pack) = state(handle) {
            pack.backend
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .unload();
        }
    }));
}

unsafe extern "C" fn shutdown(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        // SAFETY: shutdown consumes the handle exactly once.
        let mut pack = Box::from_raw(handle as *mut PackState);
        pack.stop();
    }));
}

unsafe extern "C" fn free_buffer(buffer: KapslOwnedBuffer) {
    if buffer.ptr.is_null() {
        return;
    }
    // SAFETY: buffers originate from `owned_buffer` in this library and are
    // returned at most once through this function.
    unsafe {
        drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.capacity));
    }
}

unsafe fn path_from_slice(path: KapslSlice) -> Result<PathBuf, (i32, String)> {
    // SAFETY: path is borrowed for the current ABI call.
    let bytes = unsafe { path.as_bytes() }.ok_or_else(|| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            "model path has a null pointer".to_string(),
        )
    })?;
    let text = std::str::from_utf8(bytes).map_err(|error| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            format!("model path is not UTF-8: {error}"),
        )
    })?;
    Ok(Path::new(text).to_path_buf())
}

fn engine_error(error: kapsl_engine_api::EngineError) -> (i32, String) {
    let status = if matches!(error, kapsl_engine_api::EngineError::Cancelled { .. }) {
        KAPSL_STATUS_CANCELLED
    } else {
        KAPSL_STATUS_BACKEND_ERROR
    };
    (status, error.to_string())
}

fn json_error(error: serde_json::Error) -> (i32, String) {
    (
        KAPSL_STATUS_BACKEND_ERROR,
        format!("encode native-pack JSON: {error}"),
    )
}

#[cfg(kapsl_llama_external_pool_sdk)]
#[no_mangle]
pub static KAPSL_LLAMA_CPP_KV_MODE_V1: [u8; b"KAPSL_LLAMA_CPP_KV_MODE=shared_pool\0".len()] =
    *b"KAPSL_LLAMA_CPP_KV_MODE=shared_pool\0";

#[cfg(not(kapsl_llama_external_pool_sdk))]
#[no_mangle]
pub static KAPSL_LLAMA_CPP_KV_MODE_V1: [u8; b"KAPSL_LLAMA_CPP_KV_MODE=native\0".len()] =
    *b"KAPSL_LLAMA_CPP_KV_MODE=native\0";

#[cfg(kapsl_llama_external_pool_sdk)]
const CAPABILITIES: u64 = KAPSL_LLAMA_CAP_CUDA
    | KAPSL_LLAMA_CAP_SHARED_POOL
    | KAPSL_LLAMA_CAP_STREAMING
    | KAPSL_LLAMA_CAP_CANCELLATION
    | KAPSL_LLAMA_CAP_MEMORY_REPORTING;

#[cfg(not(kapsl_llama_external_pool_sdk))]
const CAPABILITIES: u64 = if cfg!(feature = "cuda12") {
    KAPSL_LLAMA_CAP_CUDA
        | KAPSL_LLAMA_CAP_NATIVE_KV
        | KAPSL_LLAMA_CAP_STREAMING
        | KAPSL_LLAMA_CAP_CANCELLATION
        | KAPSL_LLAMA_CAP_MEMORY_REPORTING
} else {
    KAPSL_LLAMA_CAP_CPU
        | KAPSL_LLAMA_CAP_NATIVE_KV
        | KAPSL_LLAMA_CAP_STREAMING
        | KAPSL_LLAMA_CAP_CANCELLATION
        | KAPSL_LLAMA_CAP_MEMORY_REPORTING
};

static API_V1: KapslLlamaCppApiV1 = KapslLlamaCppApiV1 {
    magic: KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC,
    abi_version: KAPSL_LLAMA_CPP_ABI_VERSION,
    struct_size: std::mem::size_of::<KapslLlamaCppApiV1>() as u32,
    wire_format: KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1,
    capabilities: CAPABILITIES,
    initialize: Some(initialize),
    planned_memory: Some(planned_memory),
    load_model: Some(load_model),
    planned_request_memory: Some(planned_request_memory),
    infer: Some(infer),
    infer_stream: Some(infer_stream),
    cancel: Some(cancel),
    actual_memory: Some(actual_memory),
    metrics: Some(metrics),
    model_info: Some(model_info),
    kv_capabilities: Some(kv_capabilities),
    kv_topology: Some(kv_topology),
    batching_policy: Some(batching_policy),
    health_check: Some(health_check),
    unload: Some(unload),
    shutdown: Some(shutdown),
    free_buffer: Some(free_buffer),
};

/// Return the immutable ABI v1 function table.
#[no_mangle]
pub extern "C" fn kapsl_llama_cpp_backend_v1() -> *const KapslLlamaCppApiV1 {
    &API_V1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(kapsl_llama_external_pool_sdk)]
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};

    #[cfg(kapsl_llama_external_pool_sdk)]
    struct AdapterHostState {
        creates: AtomicUsize,
        destroys: AtomicUsize,
        addressable_blocks: AtomicU64,
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    unsafe extern "C" fn adapter_reserve(
        _pool_context: *mut c_void,
        _session_id: u64,
        _tokens_needed: u32,
        _block_table_device_out: *mut *mut u32,
        _blocks_out: *mut u32,
    ) -> u32 {
        1
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    unsafe extern "C" fn adapter_commit(
        _pool_context: *mut c_void,
        _block_table_device_out: *mut *mut u32,
    ) -> u32 {
        1
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    unsafe extern "C" fn adapter_release(_pool_context: *mut c_void, _sequence_id: u64) {}

    #[cfg(kapsl_llama_external_pool_sdk)]
    unsafe extern "C" fn adapter_create(
        user_data: *mut c_void,
        geometry: *const KapslSharedPoolGeometryV1,
        descriptor_out: *mut KapslSharedPoolDescriptorV1,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        let state = unsafe { &*(user_data as *const AdapterHostState) };
        let geometry = unsafe { &*geometry };
        state.creates.fetch_add(1, AtomicOrdering::SeqCst);
        let configured_blocks = state.addressable_blocks.load(AtomicOrdering::SeqCst);
        let addressable_blocks = if configured_blocks == 0 {
            geometry.requested_blocks
        } else {
            configured_blocks
        };
        unsafe {
            descriptor_out.write(KapslSharedPoolDescriptorV1 {
                struct_size: std::mem::size_of::<KapslSharedPoolDescriptorV1>() as u32,
                pool_context: user_data,
                device_base: std::ptr::dangling_mut::<c_void>(),
                addressable_blocks,
                block_table_device: std::ptr::dangling_mut::<u32>(),
                block_table_layer_stride: geometry.max_blocks_per_sequence,
                block_table_sequence_stride: geometry
                    .num_layers
                    .saturating_mul(geometry.max_blocks_per_sequence),
                sequence_slots: geometry.max_sequences,
                reserve: Some(adapter_reserve),
                reserve_sequence: Some(adapter_reserve),
                commit_sequences: Some(adapter_commit),
                release: Some(adapter_release),
                touch: None,
            });
        }
        KAPSL_STATUS_OK
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    unsafe extern "C" fn adapter_destroy(user_data: *mut c_void, _pool_context: *mut c_void) {
        let state = unsafe { &*(user_data as *const AdapterHostState) };
        state.destroys.fetch_add(1, AtomicOrdering::SeqCst);
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    unsafe extern "C" fn adapter_bytes(_user_data: *mut c_void, _pool_context: *mut c_void) -> u64 {
        4096
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    fn adapter_geometry() -> GgufExternalKvPoolGeometry {
        GgufExternalKvPoolGeometry {
            device_id: 0,
            requested_blocks: 64,
            block_size_tokens: 16,
            num_layers: 8,
            num_kv_heads: 4,
            key_head_dim: 64,
            value_head_dim: 64,
            max_sequences: 2,
            max_blocks_per_sequence: 32,
            model_fingerprint: 0xC0DE,
        }
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    fn adapter_host(state: &AdapterHostState) -> SharedPoolHost {
        SharedPoolHost {
            user_data: (state as *const AdapterHostState) as usize,
            create: adapter_create,
            destroy: adapter_destroy,
            bytes: adapter_bytes,
        }
    }

    #[test]
    fn entrypoint_has_complete_native_pack_contract() {
        let api = unsafe { &*kapsl_llama_cpp_backend_v1() };
        assert_eq!(api.magic, KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC);
        assert_eq!(api.abi_version, KAPSL_LLAMA_CPP_ABI_VERSION);
        assert_eq!(api.struct_size as usize, std::mem::size_of_val(api));
        #[cfg(not(kapsl_llama_external_pool_sdk))]
        assert_ne!(api.capabilities & KAPSL_LLAMA_CAP_NATIVE_KV, 0);
        #[cfg(kapsl_llama_external_pool_sdk)]
        assert_ne!(api.capabilities & KAPSL_LLAMA_CAP_SHARED_POOL, 0);
        assert_ne!(api.capabilities & KAPSL_LLAMA_CAP_STREAMING, 0);
        #[cfg(not(kapsl_llama_external_pool_sdk))]
        assert_eq!(api.capabilities & KAPSL_LLAMA_CAP_SHARED_POOL, 0);
        assert!(api.initialize.is_some());
        assert!(api.load_model.is_some());
        assert!(api.infer.is_some());
        assert!(api.infer_stream.is_some());
        assert!(api.cancel.is_some());
        assert!(api.planned_memory.is_some());
        assert!(api.actual_memory.is_some());
        assert!(api.metrics.is_some());
        assert!(api.batching_policy.is_some());
        assert!(api.unload.is_some());
        assert!(api.shutdown.is_some());
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    #[test]
    fn external_pool_adapter_destroys_the_core_pool_with_its_guard() {
        let state = AdapterHostState {
            creates: AtomicUsize::new(0),
            destroys: AtomicUsize::new(0),
            addressable_blocks: AtomicU64::new(0),
        };
        let pool = external_pool_factory(adapter_host(&state))(adapter_geometry())
            .expect("valid core descriptor");
        assert_eq!(state.creates.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(state.destroys.load(AtomicOrdering::SeqCst), 0);
        drop(pool);
        assert_eq!(state.destroys.load(AtomicOrdering::SeqCst), 1);
    }

    #[cfg(kapsl_llama_external_pool_sdk)]
    #[test]
    fn external_pool_adapter_cleans_up_unrepresentable_block_counts() {
        let state = AdapterHostState {
            creates: AtomicUsize::new(0),
            destroys: AtomicUsize::new(0),
            addressable_blocks: AtomicU64::new(u64::from(u32::MAX) + 1),
        };
        let result = external_pool_factory(adapter_host(&state))(adapter_geometry());
        assert!(result.is_err());
        assert_eq!(state.creates.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(state.destroys.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    #[cfg(not(kapsl_llama_external_pool_sdk))]
    fn shared_pool_requirement_fails_closed() {
        let config = KapslLlamaConfigV1 {
            struct_size: std::mem::size_of::<KapslLlamaConfigV1>() as u32,
            profile: if cfg!(feature = "cuda12") {
                KAPSL_LLAMA_PROFILE_CUDA12
            } else {
                KAPSL_LLAMA_PROFILE_CPU
            },
            device_id: 0,
            model_id: 1,
            replica_id: 0,
            require_shared_pool: 1,
            host: std::ptr::null(),
        };
        let mut handle = std::ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe { initialize(&config, &mut handle, &mut error) };
        assert_eq!(status, KAPSL_STATUS_UNSUPPORTED);
        assert!(handle.is_null());
        assert!(!error.ptr.is_null());
        unsafe { free_buffer(error) };
    }
}
