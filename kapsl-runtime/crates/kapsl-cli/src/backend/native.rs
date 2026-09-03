//! Backend-neutral native-pack loading and engine bridging.
//!
//! The signed pack manager remains the trust and installation boundary. This
//! module owns one versioned loader for every in-process adapter and exposes the
//! runtime-owned device allocator through `KapslBackendHostV1`. ORT is the
//! first consumer, but no ORT-specific type crosses this boundary.

use super::{BackendExecutionMode, BackendPackManifest};
use crate::runtime::RuntimeResources;
use kapsl_backend_abi::*;
use kapsl_backends::OnnxRuntimeTuning;
use kapsl_core::Manifest;
use kapsl_engine_api::{
    BatchingMode, BatchingPolicy, BinaryTensorPacket, CancellationToken, Engine, EngineError,
    EngineMetrics, EngineModelInfo, EngineStream, InferenceRequest, KvBackendCapabilities,
    KvTopology, MemoryReport, TensorDtype,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "gpu-device-pool")]
use kapsl_hal::gpu_arena::{
    GpuAllocation, GpuDevicePool, PoolAllocationClass, PoolBackend, PoolOwner,
};
#[cfg(feature = "gpu-device-pool")]
use std::collections::HashMap;
#[cfg(feature = "gpu-device-pool")]
use std::panic::{catch_unwind, AssertUnwindSafe};

pub(crate) const GENERIC_NATIVE_PACKS_ENV: &str = "KAPSL_GENERIC_NATIVE_PACKS";
const MAX_JSON_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const MAX_RESULT_TENSORS: usize = 1024;
const MAX_TENSOR_RANK: usize = 32;
const NATIVE_STREAM_BUFFER_CHUNKS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct KapslBackendApiPrefixV1 {
    magic: u32,
    abi_version: u32,
    struct_size: u32,
    wire_format: u32,
}

struct ActiveNativePack {
    manifest: BackendPackManifest,
    api: KapslBackendApiV1,
    descriptor: serde_json::Value,
    root: PathBuf,
    entrypoint: PathBuf,
    #[allow(dead_code)]
    library: libloading::Library,
}

/// Immutable identity selected from the signed index before adapter loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeBackendPackIdentity {
    pub(crate) backend: String,
    pub(crate) profile: String,
    pub(crate) pack_version: String,
}

impl NativeBackendPackIdentity {
    pub(crate) fn from_manifest(manifest: &BackendPackManifest) -> Self {
        Self {
            backend: manifest.backend.clone(),
            profile: manifest.profile.clone(),
            pack_version: manifest.pack_version.clone(),
        }
    }
}

fn active_native_packs() -> &'static Mutex<Vec<Arc<ActiveNativePack>>> {
    static ACTIVE: OnceLock<Mutex<Vec<Arc<ActiveNativePack>>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Signed native packs are the bridge-release default. Setting this switch to
/// false is the explicit rollback that permits the embedded ORT path.
pub(crate) fn generic_native_backend_packs_enabled() -> Result<bool, String> {
    let value = match std::env::var(GENERIC_NATIVE_PACKS_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!(
                "{GENERIC_NATIVE_PACKS_ENV} must be valid UTF-8 and set to 0 or 1"
            ))
        }
    };
    resolve_generic_native_packs_switch(value.as_deref())
}

fn resolve_generic_native_packs_switch(value: Option<&str>) -> Result<bool, String> {
    value.map_or(Ok(true), parse_generic_native_packs_switch)
}

fn parse_generic_native_packs_switch(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "{GENERIC_NATIVE_PACKS_ENV} must be a boolean (0/1, false/true, no/yes, or off/on)"
        )),
    }
}

unsafe fn copy_native_backend_api(
    pointer: *const KapslBackendApiV1,
) -> Result<KapslBackendApiV1, String> {
    if pointer.is_null() {
        return Err("native backend ABI v1 entrypoint returned null".to_string());
    }
    // SAFETY: only the fixed prefix is read until the adapter proves that its
    // table is large enough for every v1 field this host will access.
    let header = unsafe { pointer.cast::<KapslBackendApiPrefixV1>().read() };
    if header.magic != KAPSL_BACKEND_ENTRYPOINT_MAGIC
        || header.abi_version != KAPSL_BACKEND_ABI_VERSION
        || header.wire_format != KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1
        || header.struct_size < std::mem::size_of::<KapslBackendApiV1>() as u32
    {
        return Err("native pack exposes an incompatible backend ABI v1 table".to_string());
    }
    // SAFETY: the adapter-declared struct size covers the complete v1 table.
    Ok(unsafe { pointer.read() })
}

pub(crate) fn activate_native_backend_pack(
    manifest: &BackendPackManifest,
    root: &Path,
) -> Result<(), String> {
    if manifest.execution_mode != BackendExecutionMode::Native {
        return Err(format!(
            "backend pack {}/{} is {}, not an in-process native pack",
            manifest.backend,
            manifest.profile,
            manifest.execution_mode.as_str()
        ));
    }
    let identity = (
        manifest.backend.as_str(),
        manifest.profile.as_str(),
        manifest.pack_version.as_str(),
    );
    let mut active = active_native_packs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if active.iter().any(|pack| {
        (
            pack.manifest.backend.as_str(),
            pack.manifest.profile.as_str(),
            pack.manifest.pack_version.as_str(),
        ) == identity
    }) {
        return Ok(());
    }

    let root = root
        .canonicalize()
        .map_err(|error| format!("resolve native pack {}: {error}", root.display()))?;
    let entrypoint = root
        .join(&manifest.entrypoint)
        .canonicalize()
        .map_err(|error| {
            format!(
                "resolve native pack entrypoint {}: {error}",
                root.join(&manifest.entrypoint).display()
            )
        })?;
    if !entrypoint.starts_with(&root) {
        return Err("native pack entrypoint escapes its installation root".to_string());
    }

    // SAFETY: the pack manager verified the signed archive and confined the
    // canonical path to its atomic installation root.
    let library = unsafe { libloading::Library::new(&entrypoint) }.map_err(|error| {
        format!(
            "load native backend entrypoint {}: {error}",
            entrypoint.display()
        )
    })?;
    // SAFETY: symbol name and return type are fixed by kapsl-backend-abi v1.
    let api = unsafe {
        let entrypoint: libloading::Symbol<'_, KapslBackendEntrypointV1> = library
            .get(KAPSL_BACKEND_ENTRYPOINT_SYMBOL)
            .map_err(|error| format!("resolve native backend ABI v1 entrypoint: {error}"))?;
        let pointer = entrypoint();
        copy_native_backend_api(pointer)?
    };
    validate_native_backend_api(manifest, &api)?;
    let descriptor = describe_backend(&api)?;
    validate_native_backend_descriptor(manifest, &api, &descriptor)?;
    log::info!(
        "Activated signed native backend pack {}/{} version {} from {}",
        manifest.backend,
        manifest.profile,
        manifest.pack_version,
        root.display()
    );
    active.push(Arc::new(ActiveNativePack {
        manifest: manifest.clone(),
        api,
        descriptor,
        root,
        entrypoint,
        library,
    }));
    Ok(())
}

fn validate_native_backend_api(
    manifest: &BackendPackManifest,
    api: &KapslBackendApiV1,
) -> Result<(), String> {
    if !api.is_compatible() {
        return Err("native pack exposes an incompatible backend ABI v1 table".to_string());
    }
    if !api.has_required_functions() {
        return Err("native pack ABI v1 function table is incomplete".to_string());
    }
    if !api.capabilities_are_consistent() {
        return Err("native pack ABI v1 capability table is contradictory".to_string());
    }
    let callable_capabilities = pack_capabilities_from_abi(api.capabilities);
    if manifest.capabilities != callable_capabilities {
        return Err(format!(
            "native {}/{} ABI capabilities do not match its signed pack capabilities",
            manifest.backend, manifest.profile
        ));
    }

    let execution = api.capabilities & KAPSL_BACKEND_CAP_EXECUTION_MASK;
    let expected = match manifest.accelerator_profile.as_str() {
        "cpu" => KAPSL_BACKEND_CAP_CPU,
        "cuda" => KAPSL_BACKEND_CAP_CUDA,
        "tensorrt" => KAPSL_BACKEND_CAP_CUDA | KAPSL_BACKEND_CAP_TENSORRT,
        profile => {
            return Err(format!(
                "native pack declares unsupported accelerator profile `{profile}`"
            ))
        }
    };
    if execution != expected {
        return Err(format!(
            "signed native pack profile `{}` requires execution capabilities 0x{expected:x}, but its ABI table declares 0x{execution:x}",
            manifest.accelerator_profile
        ));
    }
    if manifest.accelerator_profile != "cpu"
        && api.capabilities & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR == 0
    {
        return Err(format!(
            "native {}/{} pack does not require the governed device allocator",
            manifest.backend, manifest.profile
        ));
    }
    Ok(())
}

fn pack_capabilities_from_abi(capabilities: u64) -> super::BackendPackCapabilities {
    super::BackendPackCapabilities {
        batching: capabilities & KAPSL_BACKEND_CAP_BATCHING != 0,
        streaming: capabilities & KAPSL_BACKEND_CAP_STREAMING != 0,
        cancellation: capabilities & KAPSL_BACKEND_CAP_CANCELLATION != 0,
        memory_reporting: capabilities & KAPSL_BACKEND_CAP_MEMORY_REPORTING != 0,
        governed_device_allocator: capabilities & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR != 0,
        scoped_device_allocator: capabilities & KAPSL_BACKEND_CAP_SCOPED_DEVICE_ALLOCATOR != 0,
        kv_participation: capabilities & KAPSL_BACKEND_CAP_KV_PARTICIPANT != 0,
        concurrent_inference: capabilities & KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE != 0,
    }
}

fn validate_native_backend_descriptor(
    manifest: &BackendPackManifest,
    api: &KapslBackendApiV1,
    descriptor: &serde_json::Value,
) -> Result<(), String> {
    let expected_profiles = serde_json::json!([manifest.profile]);
    let checks = [
        (
            descriptor.get("schema_version")
                == Some(&serde_json::json!(KAPSL_BACKEND_DESCRIPTOR_SCHEMA_V1)),
            "schema version",
        ),
        (
            descriptor.get("backend") == Some(&serde_json::json!(manifest.backend)),
            "backend identity",
        ),
        (
            descriptor.get("profiles") == Some(&expected_profiles),
            "compiled profile",
        ),
        (
            descriptor.get("backend_abi") == Some(&serde_json::json!(KAPSL_BACKEND_ABI_VERSION)),
            "backend ABI",
        ),
        (
            descriptor.get("wire_format")
                == Some(&serde_json::json!(KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1)),
            "wire format",
        ),
        (
            descriptor.get("execution_mode") == Some(&serde_json::json!("native")),
            "execution mode",
        ),
    ];
    if let Some((_, label)) = checks.into_iter().find(|(matches, _)| !matches) {
        return Err(format!(
            "native descriptor {label} does not match signed {}/{} pack and ABI table",
            manifest.backend, manifest.profile
        ));
    }
    if let Some(governed) = descriptor.get("governed_device_memory") {
        let expected = api.capabilities & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR != 0;
        if governed != &serde_json::json!(expected) {
            return Err(format!(
                "native descriptor governed-device-memory declaration does not match signed {}/{} pack and ABI table",
                manifest.backend, manifest.profile
            ));
        }
    }
    for (field, signed) in [("formats", &manifest.formats), ("tasks", &manifest.tasks)] {
        let described = descriptor_contract_values(descriptor, field)?;
        let mut signed = signed
            .iter()
            .map(|value| value.trim().to_ascii_lowercase())
            .collect::<Vec<_>>();
        signed.sort();
        if described != signed {
            return Err(format!(
                "native descriptor {field} does not match signed {}/{} pack",
                manifest.backend, manifest.profile
            ));
        }
    }
    Ok(())
}

fn descriptor_contract_values(
    descriptor: &serde_json::Value,
    field: &str,
) -> Result<Vec<String>, String> {
    let values = descriptor
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("native backend descriptor `{field}` must be an array"))?;
    let mut result = values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(|value| value.trim().to_ascii_lowercase())
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    format!("native backend descriptor `{field}` contains a non-string value")
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    result.sort();
    result.dedup();
    if result.len() != values.len() {
        return Err(format!(
            "native backend descriptor `{field}` contains duplicate values"
        ));
    }
    Ok(result)
}

fn describe_backend(api: &KapslBackendApiV1) -> Result<serde_json::Value, String> {
    let mut output = KapslOwnedBuffer::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: function presence and table compatibility were validated above.
    let status =
        unsafe { api.describe.expect("validated describe function")(&mut output, &mut error) };
    if status != KAPSL_STATUS_OK {
        if !output.ptr.is_null() {
            let _ = take_owned_buffer(api, output, "discarded descriptor");
        }
        return Err(read_ffi_error(api, status, error));
    }
    if !error.ptr.is_null() {
        let _ = take_owned_buffer(api, error, "unexpected describe error");
    }
    let bytes = take_owned_buffer(api, output, "backend descriptor")?;
    let descriptor: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode native backend descriptor JSON: {error}"))?;
    if !descriptor.is_object() {
        return Err("native backend descriptor must be a JSON object".to_string());
    }
    Ok(descriptor)
}

fn active_native_pack(identity: &NativeBackendPackIdentity) -> Option<Arc<ActiveNativePack>> {
    active_native_packs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .rev()
        .find(|pack| {
            pack.manifest.backend == identity.backend
                && pack.manifest.profile == identity.profile
                && pack.manifest.pack_version == identity.pack_version
        })
        .cloned()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_native_backend_pack_engine(
    identity: &NativeBackendPackIdentity,
    manifest: &Manifest,
    resources: &RuntimeResources,
    device_id: usize,
    model_id: u32,
    replica_id: u32,
    tuning: Option<&OnnxRuntimeTuning>,
) -> Result<Box<dyn Engine>, String> {
    let pack = active_native_pack(identity).ok_or_else(|| {
        format!(
            "selected signed native backend pack {}/{} version {} is not active",
            identity.backend, identity.profile, identity.pack_version
        )
    })?;
    // Pass the signed canonical accelerator name into the adapter. Provider
    // aliases are accepted only at the engine policy boundary and cannot
    // weaken the pack's exact initialization contract.
    let canonical_provider = pack.manifest.accelerator_profile.clone();
    let instance = NativePackInstance::initialize(
        pack,
        manifest,
        &canonical_provider,
        resources,
        device_id,
        model_id,
        replica_id,
        tuning,
    )?;
    Ok(Box::new(NativePackedEngine { instance }))
}

struct NativePackInstance {
    pack: Arc<ActiveNativePack>,
    handle: *mut c_void,
    host: NativeBackendHost,
    cancellation_runtime: Option<tokio::runtime::Handle>,
    cancel_target: Arc<NativeCancelTarget>,
    next_request_id: AtomicU64,
    loaded: AtomicBool,
    call_lock: RwLock<()>,
}

// The adapter owns its handle. Concurrent inference tables take shared call
// guards; every control/lifecycle call and every non-concurrent inference takes
// the exclusive guard. Shutdown happens only after the engine lifecycle has
// drained outstanding calls.
unsafe impl Send for NativePackInstance {}
unsafe impl Sync for NativePackInstance {}

struct NativeCallGuard<'a> {
    _read: Option<RwLockReadGuard<'a, ()>>,
    _write: Option<RwLockWriteGuard<'a, ()>>,
}

#[derive(Clone, Copy)]
struct LiveNativeCancelTarget {
    handle: usize,
    function: KapslBackendCancelFn,
}

struct NativeCancelTarget {
    live: RwLock<Option<LiveNativeCancelTarget>>,
}

impl NativeCancelTarget {
    fn new(handle: *mut c_void, function: Option<KapslBackendCancelFn>) -> Self {
        Self {
            live: RwLock::new(function.map(|function| LiveNativeCancelTarget {
                handle: handle as usize,
                function,
            })),
        }
    }

    fn cancel(&self, request_id: u64) -> Option<i32> {
        let live = self
            .live
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let target = *live.as_ref()?;
        // SAFETY: the read guard prevents instance shutdown from invalidating
        // the adapter handle until this cancellation call returns.
        Some(unsafe { (target.function)(target.handle as *mut c_void, request_id) })
    }

    fn pause(&self) -> RwLockWriteGuard<'_, Option<LiveNativeCancelTarget>> {
        self.live
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn deactivate(&self) {
        *self.pause() = None;
    }
}

#[derive(Default)]
struct NativeCancellationWatches {
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl NativeCancellationWatches {
    fn spawn(
        runtime: Option<&tokio::runtime::Handle>,
        target: &Arc<NativeCancelTarget>,
        cancellations: impl IntoIterator<Item = (u64, CancellationToken)>,
    ) -> Self {
        let Some(runtime) = runtime else {
            return Self::default();
        };
        let tasks = cancellations
            .into_iter()
            .map(|(request_id, cancellation)| {
                let target = Arc::clone(target);
                runtime.spawn(async move {
                    cancellation.cancelled().await;
                    if let Some(status) = target.cancel(request_id) {
                        if status != KAPSL_STATUS_OK {
                            log::warn!(
                                "native backend cancellation for request {request_id} returned status {status}"
                            );
                        }
                    }
                })
            })
            .collect();
        Self { tasks }
    }
}

impl Drop for NativeCancellationWatches {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

struct NativeStreamCallbackContext {
    sender: async_channel::Sender<Result<BinaryTensorPacket, EngineError>>,
    callback_error: Option<EngineError>,
    consumer_closed: bool,
}

unsafe extern "C" fn native_stream_chunk(
    user_data: *mut c_void,
    _request_id: u64,
    result: *const KapslInferenceResultV1,
) -> i32 {
    if user_data.is_null() {
        return KAPSL_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: dispatch_stream retains this context until infer_stream returns.
    let context = unsafe { &mut *user_data.cast::<NativeStreamCallbackContext>() };
    if context.consumer_closed {
        return KAPSL_STATUS_CANCELLED;
    }
    if context.callback_error.is_some() {
        return KAPSL_STATUS_BACKEND_ERROR;
    }

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let packet = if result.is_null() {
            Err(EngineError::backend(
                "native backend stream callback received a null result",
            ))
        } else {
            // SAFETY: stream chunks are borrowed for this callback invocation.
            copy_single_result(unsafe { &*result })
        };
        match packet {
            Ok(packet) => match context.sender.send_blocking(Ok(packet)) {
                Ok(()) => KAPSL_STATUS_OK,
                Err(_) => {
                    context.consumer_closed = true;
                    KAPSL_STATUS_CANCELLED
                }
            },
            Err(error) => {
                context.callback_error = Some(error);
                KAPSL_STATUS_BACKEND_ERROR
            }
        }
    }))
    .unwrap_or_else(|_| {
        context.callback_error = Some(EngineError::backend(
            "native backend stream callback panicked",
        ));
        KAPSL_STATUS_PANIC
    })
}

struct NativeStreamDropGuard {
    target: Arc<NativeCancelTarget>,
    request_id: u64,
    completed: Arc<AtomicBool>,
}

impl Drop for NativeStreamDropGuard {
    fn drop(&mut self) {
        if !self.completed.load(Ordering::Acquire) {
            let _ = self.target.cancel(self.request_id);
        }
    }
}

struct NativeStreamReceiver {
    receiver: async_channel::Receiver<Result<BinaryTensorPacket, EngineError>>,
    _drop_guard: NativeStreamDropGuard,
}

fn native_bridge_runtime() -> Result<tokio::runtime::Handle, String> {
    static RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    match RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("kapsl-native-bridge")
            .build()
            .map_err(|error| format!("create native backend bridge runtime: {error}"))
    }) {
        Ok(runtime) => Ok(runtime.handle().clone()),
        Err(error) => Err(error.clone()),
    }
}

impl NativePackInstance {
    #[allow(clippy::too_many_arguments)]
    fn initialize(
        pack: Arc<ActiveNativePack>,
        manifest: &Manifest,
        provider: &str,
        resources: &RuntimeResources,
        device_id: usize,
        model_id: u32,
        replica_id: u32,
        tuning: Option<&OnnxRuntimeTuning>,
    ) -> Result<Arc<Self>, String> {
        let requires_governed_memory =
            pack.api.capabilities & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR != 0;
        let supports_cancellation = pack.api.capabilities & KAPSL_BACKEND_CAP_CANCELLATION != 0;
        let cancellation_runtime = supports_cancellation
            .then(native_bridge_runtime)
            .transpose()?;
        let host = NativeBackendHost::new(
            resources,
            &pack.manifest.backend,
            device_id,
            model_id,
            replica_id,
            requires_governed_memory,
        )?;
        let profile = pack.manifest.profile.as_bytes();
        let manifest_json = serde_json::to_vec(manifest)
            .map_err(|error| format!("encode model manifest for native backend: {error}"))?;
        let options_json = serde_json::to_vec(&serde_json::json!({
            "provider": provider,
            "accelerator_profile": pack.manifest.accelerator_profile,
            "pack_version": pack.manifest.pack_version,
            "descriptor": pack.descriptor,
            "pack_root": pack.root,
            "entrypoint": pack.entrypoint,
            "onnx_tuning": tuning.map(|tuning| serde_json::json!({
                "memory_pattern": tuning.memory_pattern,
                "disable_cpu_mem_arena": tuning.disable_cpu_mem_arena,
                "session_buckets": tuning.session_buckets,
                "bucket_dim_granularity": tuning.bucket_dim_granularity,
                "bucket_max_dims": tuning.bucket_max_dims,
                "peak_concurrency_hint": tuning.peak_concurrency_hint,
            })),
        }))
        .map_err(|error| format!("encode native backend options: {error}"))?;
        let config = KapslBackendConfigV1 {
            struct_size: std::mem::size_of::<KapslBackendConfigV1>() as u32,
            device_id: u32::try_from(device_id)
                .map_err(|_| format!("device id {device_id} exceeds backend ABI v1"))?,
            model_id,
            replica_id,
            require_governed_device_memory: u32::from(requires_governed_memory),
            reserved: 0,
            profile: KapslSlice::from_bytes(profile),
            manifest_json: KapslSlice::from_bytes(&manifest_json),
            options_json: KapslSlice::from_bytes(&options_json),
            host: host.table(),
        };
        let mut handle = std::ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: the table was validated and all borrowed config storage lives
        // through this synchronous call. The host table itself is boxed and is
        // retained until adapter shutdown.
        let status = unsafe {
            pack.api.initialize.expect("validated initialize function")(
                &config,
                &mut handle,
                &mut error,
            )
        };
        if status != KAPSL_STATUS_OK || handle.is_null() {
            let message = read_ffi_error(&pack.api, status, error);
            if !handle.is_null() {
                // SAFETY: an adapter that published a handle transferred it
                // to the host even though initialization failed. Shutdown is
                // the ABI's one terminal cleanup operation for that handle.
                unsafe {
                    pack.api.shutdown.expect("validated shutdown function")(handle);
                }
            }
            return Err(message);
        }
        if !error.ptr.is_null() {
            let _ = take_owned_buffer(&pack.api, error, "unexpected initialize error");
        }
        let cancel_function = if supports_cancellation {
            Some(
                pack.api
                    .cancel
                    .expect("cancellation capability validated cancel function"),
            )
        } else {
            None
        };
        Ok(Arc::new(Self {
            pack,
            handle,
            host,
            cancellation_runtime,
            cancel_target: Arc::new(NativeCancelTarget::new(handle, cancel_function)),
            next_request_id: AtomicU64::new(1),
            loaded: AtomicBool::new(false),
            call_lock: RwLock::new(()),
        }))
    }

    fn next_request_id(&self) -> Result<u64, EngineError> {
        self.next_request_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| EngineError::backend("native backend request ID space exhausted"))
    }

    fn watch_cancellations(
        &self,
        cancellations: impl IntoIterator<Item = (u64, CancellationToken)>,
    ) -> NativeCancellationWatches {
        NativeCancellationWatches::spawn(
            self.cancellation_runtime.as_ref(),
            &self.cancel_target,
            cancellations,
        )
    }

    fn inference_guard(&self) -> NativeCallGuard<'_> {
        if self.pack.api.capabilities & KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE != 0 {
            NativeCallGuard {
                _read: Some(
                    self.call_lock
                        .read()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                ),
                _write: None,
            }
        } else {
            self.exclusive_guard()
        }
    }

    fn exclusive_guard(&self) -> NativeCallGuard<'_> {
        NativeCallGuard {
            _read: None,
            _write: Some(
                self.call_lock
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
        }
    }

    fn call_path_report<T: DeserializeOwned>(
        &self,
        path: &Path,
        function: KapslBackendPathReportFn,
    ) -> Result<T, EngineError> {
        let path = path.to_str().ok_or_else(|| {
            EngineError::invalid_input(format!("model path is not UTF-8: {}", path.display()))
        })?;
        let _guard = self.exclusive_guard();
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: the handle is live and path bytes remain borrowed through the call.
        let status = unsafe {
            function(
                self.handle,
                KapslSlice::from_bytes(path.as_bytes()),
                &mut output,
                &mut error,
            )
        };
        self.decode_json(status, output, error)
    }

    fn call_json_report<T: DeserializeOwned>(
        &self,
        function: KapslBackendJsonReportFn,
    ) -> Result<T, EngineError> {
        let _guard = self.exclusive_guard();
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: the handle remains live for the synchronous call.
        let status = unsafe { function(self.handle, &mut output, &mut error) };
        self.decode_json(status, output, error)
    }

    fn call_request_report<T: DeserializeOwned>(
        &self,
        request: &InferenceRequest,
    ) -> Result<T, EngineError> {
        let request_id = self.next_request_id()?;
        let mut bridge = RequestBridge::new(request)?;
        let wire = bridge.wire(request_id);
        let _guard = self.exclusive_guard();
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: all request views and callback context remain live through
        // this synchronous ABI call.
        let status = unsafe {
            self.pack
                .api
                .planned_request_memory
                .expect("validated request-memory function")(
                self.handle,
                &wire,
                &mut output,
                &mut error,
            )
        };
        self.decode_json(status, output, error)
    }

    fn decode_json<T: DeserializeOwned>(
        &self,
        status: i32,
        output: KapslOwnedBuffer,
        error: KapslOwnedBuffer,
    ) -> Result<T, EngineError> {
        if status != KAPSL_STATUS_OK {
            if !output.ptr.is_null() {
                let _ = take_owned_buffer(&self.pack.api, output, "discarded JSON output");
            }
            return Err(status_engine_error(
                status,
                read_ffi_error(&self.pack.api, status, error),
            ));
        }
        if !error.ptr.is_null() {
            let _ = take_owned_buffer(&self.pack.api, error, "unexpected successful error");
        }
        let bytes = take_owned_buffer(&self.pack.api, output, "native backend JSON report")
            .map_err(EngineError::backend)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| EngineError::backend(format!("decode native backend JSON: {error}")))
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        if request
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(EngineError::cancelled(
                "native backend request was cancelled before dispatch",
            ));
        }
        let request_id = self.next_request_id()?;
        let mut bridge = RequestBridge::new(request)?;
        let wire = bridge.wire(request_id);
        let _guard = self.inference_guard();
        let _cancellation_watches = self.watch_cancellations(
            request
                .cancellation
                .clone()
                .map(|cancellation| (request_id, cancellation)),
        );
        let mut result = KapslInferenceResultV1::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: request and result storage follows the synchronous ABI lifetime.
        let status = unsafe {
            self.pack.api.infer.expect("validated infer function")(
                self.handle,
                &wire,
                &mut result,
                &mut error,
            )
        };
        if status != KAPSL_STATUS_OK {
            return Err(status_engine_error(
                status,
                read_ffi_error(&self.pack.api, status, error),
            ));
        }
        if !error.ptr.is_null() {
            let _ = take_owned_buffer(&self.pack.api, error, "unexpected infer error");
        }
        let _release = ResultReleaseGuard::single(&self.pack.api, self.handle, &mut result);
        if request
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(EngineError::cancelled(
                "native backend request was cancelled during execution",
            ));
        }
        copy_single_result(&result)
    }

    fn infer_stream(self: &Arc<Self>, request: InferenceRequest) -> EngineStream {
        if request
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Box::pin(futures::stream::once(async {
                Err(EngineError::cancelled(
                    "native backend stream was cancelled before dispatch",
                ))
            }));
        }
        let request_id = match self.next_request_id() {
            Ok(request_id) => request_id,
            Err(error) => return Box::pin(futures::stream::once(async move { Err(error) })),
        };
        let runtime = match native_bridge_runtime() {
            Ok(runtime) => runtime,
            Err(error) => {
                return Box::pin(futures::stream::once(async move {
                    Err(EngineError::backend(error))
                }))
            }
        };
        let (sender, receiver) = async_channel::bounded(NATIVE_STREAM_BUFFER_CHUNKS);
        let completed = Arc::new(AtomicBool::new(false));
        let worker_completed = Arc::clone(&completed);
        let instance = Arc::clone(self);
        runtime.spawn_blocking(move || {
            // Keep this sender alive until completion is published. Otherwise
            // the receiver can observe a closed channel in the small window
            // between dispatch_stream returning and the flag store, and its
            // drop guard would issue a late cancellation for completed work.
            instance.dispatch_stream(request_id, &request, sender.clone());
            worker_completed.store(true, Ordering::Release);
        });

        let state = NativeStreamReceiver {
            receiver,
            _drop_guard: NativeStreamDropGuard {
                target: Arc::clone(&self.cancel_target),
                request_id,
                completed,
            },
        };
        Box::pin(futures::stream::unfold(state, |state| async move {
            state.receiver.recv().await.ok().map(|item| (item, state))
        }))
    }

    fn dispatch_stream(
        &self,
        request_id: u64,
        request: &InferenceRequest,
        sender: async_channel::Sender<Result<BinaryTensorPacket, EngineError>>,
    ) {
        let terminal = (|| -> Result<(), EngineError> {
            let mut bridge = RequestBridge::new(request)?;
            let wire = bridge.wire(request_id);
            let _guard = self.inference_guard();
            let _cancellation_watches = self.watch_cancellations(
                request
                    .cancellation
                    .clone()
                    .map(|cancellation| (request_id, cancellation)),
            );
            let mut context = NativeStreamCallbackContext {
                sender: sender.clone(),
                callback_error: None,
                consumer_closed: false,
            };
            let mut error = KapslOwnedBuffer::empty();
            // SAFETY: the request bridge and callback context remain live until
            // the synchronous stream call returns. Every chunk is copied in the
            // callback before the adapter may reuse its borrowed storage.
            let status = unsafe {
                self.pack
                    .api
                    .infer_stream
                    .expect("streaming capability validated infer-stream function")(
                    self.handle,
                    &wire,
                    (&mut context as *mut NativeStreamCallbackContext).cast(),
                    Some(native_stream_chunk),
                    &mut error,
                )
            };
            let status_error = if status == KAPSL_STATUS_OK {
                if !error.ptr.is_null() {
                    let _ = take_owned_buffer(
                        &self.pack.api,
                        error,
                        "unexpected successful stream error",
                    );
                }
                None
            } else {
                Some(status_engine_error(
                    status,
                    read_ffi_error(&self.pack.api, status, error),
                ))
            };
            if context.consumer_closed {
                return Ok(());
            }
            if let Some(error) = context.callback_error {
                return Err(error);
            }
            if let Some(error) = status_error {
                return Err(error);
            }
            if request
                .cancellation
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err(EngineError::cancelled(
                    "native backend stream was cancelled during execution",
                ));
            }
            Ok(())
        })();
        if let Err(error) = terminal {
            let _ = sender.send_blocking(Err(error));
        }
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if requests.iter().any(|request| {
            request
                .cancellation
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
        }) {
            return Err(EngineError::cancelled(
                "native backend batch was cancelled before dispatch",
            ));
        }
        if self.pack.api.capabilities & KAPSL_BACKEND_CAP_BATCHING == 0 {
            return requests.iter().map(|request| self.infer(request)).collect();
        }

        let mut bridges = requests
            .iter()
            .map(RequestBridge::new)
            .collect::<Result<Vec<_>, _>>()?;
        let request_ids = (0..bridges.len())
            .map(|_| self.next_request_id())
            .collect::<Result<Vec<_>, _>>()?;
        let wires = bridges
            .iter_mut()
            .zip(&request_ids)
            .map(|(bridge, request_id)| bridge.wire(*request_id))
            .collect::<Vec<_>>();
        let batch = KapslInferenceBatchV1 {
            struct_size: std::mem::size_of::<KapslInferenceBatchV1>() as u32,
            request_count: u32::try_from(wires.len())
                .map_err(|_| EngineError::invalid_input("native batch exceeds ABI v1"))?,
            requests: wires.as_ptr(),
        };
        let _guard = self.inference_guard();
        let _cancellation_watches =
            self.watch_cancellations(requests.iter().zip(&request_ids).filter_map(
                |(request, request_id)| {
                    request
                        .cancellation
                        .clone()
                        .map(|cancellation| (*request_id, cancellation))
                },
            ));
        let mut result = KapslInferenceBatchResultV1::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: every request bridge and output slot remains live through the call.
        let status = unsafe {
            self.pack
                .api
                .infer_batch
                .expect("capability-validated batch function")(
                self.handle,
                &batch,
                &mut result,
                &mut error,
            )
        };
        if status != KAPSL_STATUS_OK {
            return Err(status_engine_error(
                status,
                read_ffi_error(&self.pack.api, status, error),
            ));
        }
        if !error.ptr.is_null() {
            let _ = take_owned_buffer(&self.pack.api, error, "unexpected batch error");
        }
        let _release = ResultReleaseGuard::batch(&self.pack.api, self.handle, &mut result);
        if requests.iter().any(|request| {
            request
                .cancellation
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
        }) {
            return Err(EngineError::cancelled(
                "native backend batch was cancelled during execution",
            ));
        }
        copy_batch_results(&result, requests.len())
    }

    fn unload(&self) -> Result<(), EngineError> {
        if !self.loaded.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        let _guard = self.exclusive_guard();
        let _cancel_pause = self.cancel_target.pause();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: lifecycle ownership guarantees no request is active here.
        let status = unsafe {
            self.pack.api.unload.expect("validated unload function")(self.handle, &mut error)
        };
        if status == KAPSL_STATUS_OK {
            if !error.ptr.is_null() {
                let _ = take_owned_buffer(&self.pack.api, error, "unexpected unload error");
            }
            Ok(())
        } else {
            Err(status_engine_error(
                status,
                read_ffi_error(&self.pack.api, status, error),
            ))
        }
    }
}

impl Drop for NativePackInstance {
    fn drop(&mut self) {
        // Prevent an already-woken cancellation task from reaching the handle
        // once terminal lifecycle teardown begins. Any call already in flight
        // completes before this write lock is acquired.
        self.cancel_target.deactivate();
        if self.loaded.load(Ordering::Acquire) {
            if let Err(error) = self.unload() {
                log::error!("native backend unload during shutdown failed: {error}");
            }
        }
        if !self.handle.is_null() {
            // SAFETY: the adapter handle remains valid until its one shutdown call.
            unsafe {
                self.pack.api.shutdown.expect("validated shutdown function")(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
        if self.host.live_allocations() != 0 {
            log::error!(
                "native backend returned from shutdown with {} governed allocations live",
                self.host.live_allocations()
            );
        }
    }
}

struct NativePackedEngine {
    instance: Arc<NativePackInstance>,
}

#[async_trait::async_trait]
impl Engine for NativePackedEngine {
    fn kv_capabilities(&self) -> KvBackendCapabilities {
        if self.instance.pack.api.capabilities & KAPSL_BACKEND_CAP_KV_PARTICIPANT == 0 {
            return KvBackendCapabilities::unmanaged();
        }
        self.instance
            .pack
            .api
            .kv_capabilities
            .and_then(|function| self.instance.call_json_report(function).ok())
            .unwrap_or_else(KvBackendCapabilities::unmanaged)
    }

    fn kv_topology(&self) -> Option<KvTopology> {
        if self.instance.pack.api.capabilities & KAPSL_BACKEND_CAP_KV_PARTICIPANT == 0 {
            return None;
        }
        self.instance
            .pack
            .api
            .kv_topology
            .and_then(|function| self.instance.call_json_report(function).ok())
    }

    fn planned_memory(&self, model_path: &Path) -> Result<MemoryReport, EngineError> {
        self.instance.call_path_report(
            model_path,
            self.instance
                .pack
                .api
                .planned_memory
                .expect("validated planned-memory function"),
        )
    }

    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        let path = model_path.to_str().ok_or_else(|| {
            EngineError::invalid_input(format!("model path is not UTF-8: {}", model_path.display()))
        })?;
        let _guard = self.instance.exclusive_guard();
        let _cancel_pause = self.instance.cancel_target.pause();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: path bytes remain valid through this synchronous call.
        let status = unsafe {
            self.instance
                .pack
                .api
                .load_model
                .expect("validated load function")(
                self.instance.handle,
                KapslSlice::from_bytes(path.as_bytes()),
                &mut error,
            )
        };
        if status == KAPSL_STATUS_OK {
            self.instance.loaded.store(true, Ordering::Release);
            if !error.ptr.is_null() {
                let _ = take_owned_buffer(&self.instance.pack.api, error, "unexpected load error");
            }
            Ok(())
        } else {
            Err(status_engine_error(
                status,
                read_ffi_error(&self.instance.pack.api, status, error),
            ))
        }
    }

    fn actual_memory(&self) -> MemoryReport {
        self.instance
            .call_json_report(
                self.instance
                    .pack
                    .api
                    .actual_memory
                    .expect("validated actual-memory function"),
            )
            .unwrap_or_else(|error| {
                log::error!("native backend actual-memory report failed: {error}");
                MemoryReport::default()
            })
    }

    fn planned_request_memory(&self, request: &InferenceRequest) -> MemoryReport {
        self.instance
            .call_request_report(request)
            .unwrap_or_else(|error| {
                log::error!("native backend request-memory report failed: {error}");
                MemoryReport::default()
            })
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.instance.infer(request)
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        self.instance.infer_batch(requests)
    }

    fn max_batch(&self) -> usize {
        self.pack_batching_policy().max_requests.max(1)
    }

    fn self_batches(&self) -> bool {
        matches!(
            self.pack_batching_policy().mode,
            BatchingMode::Continuous | BatchingMode::Delegated
        )
    }

    fn batching_policy(&self) -> BatchingPolicy {
        self.pack_batching_policy()
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        if self.instance.pack.api.capabilities & KAPSL_BACKEND_CAP_STREAMING != 0 {
            self.instance.infer_stream(request.clone())
        } else {
            let result = self.infer(request);
            Box::pin(futures::stream::once(async move { result }))
        }
    }

    fn unload(&mut self) {
        if let Err(error) = self.instance.unload() {
            log::error!("native backend unload failed: {error}");
        }
    }

    fn metrics(&self) -> EngineMetrics {
        self.instance
            .call_json_report(
                self.instance
                    .pack
                    .api
                    .metrics
                    .expect("validated metrics function"),
            )
            .unwrap_or_else(|error| {
                log::error!("native backend metrics failed: {error}");
                EngineMetrics::new()
            })
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        self.instance
            .pack
            .api
            .model_info
            .and_then(|function| self.instance.call_json_report(function).ok())
    }

    fn health_check(&self) -> Result<(), EngineError> {
        let _guard = self.instance.exclusive_guard();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: the validated health function borrows the live handle only.
        let status = unsafe {
            self.instance
                .pack
                .api
                .health_check
                .expect("validated health function")(self.instance.handle, &mut error)
        };
        if status == KAPSL_STATUS_OK {
            if !error.ptr.is_null() {
                let _ =
                    take_owned_buffer(&self.instance.pack.api, error, "unexpected health error");
            }
            Ok(())
        } else {
            Err(status_engine_error(
                status,
                read_ffi_error(&self.instance.pack.api, status, error),
            ))
        }
    }
}

impl NativePackedEngine {
    fn pack_batching_policy(&self) -> BatchingPolicy {
        let report = self.instance.call_json_report::<NativeBatchingPolicy>(
            self.instance
                .pack
                .api
                .batching_policy
                .expect("validated batching-policy function"),
        );
        let mut policy = report
            .map(NativeBatchingPolicy::into_policy)
            .unwrap_or_else(|error| {
                log::error!("native backend batching-policy report failed: {error}");
                BatchingPolicy::none()
            });
        if self.instance.pack.api.capabilities & KAPSL_BACKEND_CAP_BATCHING == 0 {
            policy = BatchingPolicy::none();
        }
        policy
    }
}

#[derive(Deserialize)]
struct NativeBatchingPolicy {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default, alias = "max_batch")]
    max_requests: Option<usize>,
    #[serde(default)]
    self_batches: bool,
    #[serde(default)]
    queue_delay_ms: Option<u64>,
    #[serde(default)]
    max_batched_tokens: Option<usize>,
    #[serde(default)]
    supports_priority: bool,
}

impl NativeBatchingPolicy {
    fn into_policy(self) -> BatchingPolicy {
        let max_requests = self.max_requests.unwrap_or(1).max(1);
        let mode = match self.mode.as_deref() {
            Some("request_coalescing") => BatchingMode::RequestCoalescing,
            Some("continuous") => BatchingMode::Continuous,
            Some("delegated") => BatchingMode::Delegated,
            Some("none") => BatchingMode::None,
            _ if self.self_batches => BatchingMode::Continuous,
            _ if max_requests > 1 => BatchingMode::RequestCoalescing,
            _ => BatchingMode::None,
        };
        BatchingPolicy {
            mode,
            max_requests,
            queue_delay_ms: self.queue_delay_ms,
            max_batched_tokens: self.max_batched_tokens,
            supports_priority: self.supports_priority,
        }
    }
}

struct RequestBridge<'a> {
    request: &'a InferenceRequest,
    views: Vec<KapslNamedTensorViewV1>,
    input_count: u32,
    metadata_json: Vec<u8>,
    cancellation: CancellationContext,
}

impl<'a> RequestBridge<'a> {
    fn new(request: &'a InferenceRequest) -> Result<Self, EngineError> {
        request.input.validate()?;
        for input in &request.additional_inputs {
            input.tensor.validate()?;
            if input.name.is_empty() {
                return Err(EngineError::invalid_input(
                    "native backend additional input name may not be empty",
                ));
            }
        }

        let mut views = Vec::with_capacity(request.additional_inputs.len() + 1);
        views.push(named_tensor_view(b"input", &request.input)?);
        for input in &request.additional_inputs {
            views.push(named_tensor_view(input.name.as_bytes(), &input.tensor)?);
        }
        let input_count = u32::try_from(views.len())
            .map_err(|_| EngineError::invalid_input("native input count exceeds backend ABI v1"))?;
        let metadata_json = serde_json::to_vec(&serde_json::json!({
            "session_id": request.session_id,
            "metadata": request.metadata,
        }))
        .map_err(|error| EngineError::invalid_input(format!("encode request metadata: {error}")))?;
        if metadata_json.len() > MAX_JSON_BUFFER_BYTES {
            return Err(EngineError::invalid_input(format!(
                "native request metadata exceeds {MAX_JSON_BUFFER_BYTES} bytes"
            )));
        }
        Ok(Self {
            request,
            views,
            input_count,
            metadata_json,
            cancellation: CancellationContext {
                token: request.cancellation.clone(),
            },
        })
    }

    fn wire(&mut self, request_id: u64) -> KapslInferenceRequestV1 {
        let _hold = self.request;
        KapslInferenceRequestV1 {
            struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
            wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            request_id,
            inputs: self.views.as_ptr(),
            input_count: self.input_count,
            reserved: 0,
            metadata_json: KapslSlice::from_bytes(&self.metadata_json),
            cancellation_context: (&mut self.cancellation as *mut CancellationContext).cast(),
            is_cancelled: Some(request_cancelled),
        }
    }
}

fn named_tensor_view(
    name: &[u8],
    packet: &BinaryTensorPacket,
) -> Result<KapslNamedTensorViewV1, EngineError> {
    let rank = u32::try_from(packet.shape.len())
        .map_err(|_| EngineError::invalid_input("tensor rank exceeds backend ABI v1"))?;
    let byte_len = u64::try_from(packet.data.len())
        .map_err(|_| EngineError::invalid_input("tensor byte length exceeds backend ABI v1"))?;
    Ok(KapslNamedTensorViewV1 {
        struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
        reserved: 0,
        name: KapslSlice::from_bytes(name),
        tensor: KapslTensorViewV1 {
            struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
            dtype: dtype_to_abi(packet.dtype),
            memory_kind: KAPSL_MEMORY_HOST,
            flags: KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY,
            device_id: -1,
            rank,
            shape: packet.shape.as_ptr(),
            strides: std::ptr::null(),
            data: packet.data.as_ptr().cast(),
            byte_len,
        },
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

fn dtype_from_abi(dtype: u32) -> Result<TensorDtype, EngineError> {
    match dtype {
        KAPSL_DTYPE_F32 => Ok(TensorDtype::Float32),
        KAPSL_DTYPE_F64 => Ok(TensorDtype::Float64),
        KAPSL_DTYPE_F16 => Ok(TensorDtype::Float16),
        KAPSL_DTYPE_I32 => Ok(TensorDtype::Int32),
        KAPSL_DTYPE_I64 => Ok(TensorDtype::Int64),
        KAPSL_DTYPE_U8 => Ok(TensorDtype::Uint8),
        KAPSL_DTYPE_UTF8 => Ok(TensorDtype::Utf8),
        other => Err(EngineError::backend(format!(
            "native backend returned unsupported tensor dtype {other}"
        ))),
    }
}

fn copy_single_result(result: &KapslInferenceResultV1) -> Result<BinaryTensorPacket, EngineError> {
    validate_struct_size::<KapslInferenceResultV1>(result.struct_size, "inference result")?;
    if result.output_count != 1 {
        return Err(EngineError::backend(format!(
            "native backend must return exactly one postprocessed output, returned {}",
            result.output_count
        )));
    }
    if result.outputs.is_null() {
        return Err(EngineError::backend(
            "native backend returned a null output array",
        ));
    }
    // SAFETY: the adapter owns one result entry until release_result below.
    let output = unsafe { &*result.outputs };
    copy_output_tensor(output)
}

fn copy_batch_results(
    result: &KapslInferenceBatchResultV1,
    expected: usize,
) -> Result<Vec<BinaryTensorPacket>, EngineError> {
    validate_struct_size::<KapslInferenceBatchResultV1>(
        result.struct_size,
        "batch inference result",
    )?;
    let count = usize::try_from(result.result_count)
        .map_err(|_| EngineError::backend("native batch result count exceeds this platform"))?;
    if count != expected || count > MAX_RESULT_TENSORS {
        return Err(EngineError::backend(format!(
            "native backend returned {count} batch results for {expected} requests"
        )));
    }
    if count > 0 && result.results.is_null() {
        return Err(EngineError::backend(
            "native backend returned a null batch result array",
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: count was bounded and the adapter retains the array until the
    // batch release guard is dropped.
    let results = unsafe { std::slice::from_raw_parts(result.results, count) };
    results.iter().map(copy_single_result).collect()
}

fn copy_output_tensor(output: &KapslNamedTensorViewV1) -> Result<BinaryTensorPacket, EngineError> {
    validate_struct_size::<KapslNamedTensorViewV1>(output.struct_size, "named tensor result")?;
    validate_struct_size::<KapslTensorViewV1>(output.tensor.struct_size, "tensor result")?;
    if output.reserved != 0 {
        return Err(EngineError::backend(
            "native backend returned non-zero reserved tensor metadata",
        ));
    }
    let supported_flags = KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY;
    if output.tensor.flags & !supported_flags != 0 {
        return Err(EngineError::backend(format!(
            "native backend returned unsupported tensor flags 0x{:x}",
            output.tensor.flags & !supported_flags
        )));
    }
    if !matches!(
        output.tensor.memory_kind,
        KAPSL_MEMORY_HOST | KAPSL_MEMORY_HOST_PINNED
    ) {
        return Err(EngineError::backend(format!(
            "native backend returned memory kind {}; engine tensor output requires host-visible storage",
            output.tensor.memory_kind
        )));
    }
    if output.tensor.flags & KAPSL_TENSOR_FLAG_CONTIGUOUS == 0 {
        return Err(EngineError::backend(
            "native backend returned a non-contiguous output tensor",
        ));
    }
    let rank = usize::try_from(output.tensor.rank)
        .map_err(|_| EngineError::backend("native output rank exceeds this platform"))?;
    if rank > MAX_TENSOR_RANK || (rank > 0 && output.tensor.shape.is_null()) {
        return Err(EngineError::backend(format!(
            "native backend returned invalid tensor rank {rank}"
        )));
    }
    let shape = if rank == 0 {
        Vec::new()
    } else {
        // SAFETY: the non-zero rank is bounded and shape storage belongs to
        // the live adapter result.
        unsafe { std::slice::from_raw_parts(output.tensor.shape, rank) }.to_vec()
    };
    let len = usize::try_from(output.tensor.byte_len)
        .map_err(|_| EngineError::backend("native output byte length exceeds this platform"))?;
    if len > 0 && output.tensor.data.is_null() {
        return Err(EngineError::backend(
            "native backend returned null storage for a non-empty tensor",
        ));
    }
    let data = if len == 0 {
        Vec::new()
    } else {
        // SAFETY: non-empty result storage is non-null and retained by the
        // release guard until after this copy.
        unsafe { std::slice::from_raw_parts(output.tensor.data.cast::<u8>(), len) }.to_vec()
    };
    BinaryTensorPacket::new(shape, dtype_from_abi(output.tensor.dtype)?, data)
}

fn validate_struct_size<T>(actual: u32, name: &str) -> Result<(), EngineError> {
    let expected = std::mem::size_of::<T>() as u32;
    if actual < expected {
        Err(EngineError::backend(format!(
            "native backend {name} struct is {actual} bytes, expected at least {expected}"
        )))
    } else {
        Ok(())
    }
}

struct ResultReleaseGuard<'a> {
    api: &'a KapslBackendApiV1,
    handle: *mut c_void,
    result: ResultPointer,
}

enum ResultPointer {
    Single(*mut KapslInferenceResultV1),
    Batch(*mut KapslInferenceBatchResultV1),
}

impl<'a> ResultReleaseGuard<'a> {
    fn single(
        api: &'a KapslBackendApiV1,
        handle: *mut c_void,
        result: *mut KapslInferenceResultV1,
    ) -> Self {
        Self {
            api,
            handle,
            result: ResultPointer::Single(result),
        }
    }

    fn batch(
        api: &'a KapslBackendApiV1,
        handle: *mut c_void,
        result: *mut KapslInferenceBatchResultV1,
    ) -> Self {
        Self {
            api,
            handle,
            result: ResultPointer::Batch(result),
        }
    }
}

impl Drop for ResultReleaseGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: each successful adapter result is released exactly once by
        // the matching function from the same validated table.
        unsafe {
            match self.result {
                ResultPointer::Single(result) => self
                    .api
                    .release_result
                    .expect("validated result release function")(
                    self.handle, result
                ),
                ResultPointer::Batch(result) => self
                    .api
                    .release_batch_result
                    .expect("batch capability validated release function")(
                    self.handle, result
                ),
            }
        }
    }
}

struct CancellationContext {
    token: Option<kapsl_engine_api::CancellationToken>,
}

unsafe extern "C" fn request_cancelled(user_data: *mut c_void, _request_id: u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            return 0;
        }
        // SAFETY: the request bridge retains this context until the synchronous
        // ABI call returns.
        let context = unsafe { &*(user_data as *const CancellationContext) };
        u32::from(
            context
                .token
                .as_ref()
                .is_some_and(|token| token.is_cancelled()),
        )
    }))
    .unwrap_or(1)
}

unsafe extern "C" fn host_log(_user_data: *mut c_void, level: u32, message: KapslSlice) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the backend promises a borrowed message for this callback.
        let text = unsafe { message.as_bytes() }
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .unwrap_or("<invalid native backend log message>");
        match level {
            KAPSL_LOG_ERROR => log::error!(target: "kapsl_backend", "{text}"),
            KAPSL_LOG_WARN => log::warn!(target: "kapsl_backend", "{text}"),
            KAPSL_LOG_INFO => log::info!(target: "kapsl_backend", "{text}"),
            KAPSL_LOG_DEBUG => log::debug!(target: "kapsl_backend", "{text}"),
            _ => log::trace!(target: "kapsl_backend", "{text}"),
        }
    }));
}

struct NativeBackendHost {
    table: Box<KapslBackendHostV1>,
    #[cfg(feature = "gpu-device-pool")]
    allocator: Option<Box<GovernedDeviceHost>>,
}

impl NativeBackendHost {
    #[allow(clippy::too_many_arguments)]
    fn new(
        resources: &RuntimeResources,
        backend: &str,
        device_id: usize,
        model_id: u32,
        replica_id: u32,
        require_governed: bool,
    ) -> Result<Self, String> {
        #[cfg(feature = "gpu-device-pool")]
        {
            let mut allocator = if require_governed {
                let pool = resources.device_pool(device_id).ok_or_else(|| {
                    format!(
                        "native {backend} pack requires governed device memory, but device {device_id} has no runtime-owned pool"
                    )
                })?;
                Some(Box::new(GovernedDeviceHost::new(
                    pool,
                    pool_backend(backend),
                    device_id,
                    model_id,
                    replica_id,
                )))
            } else {
                None
            };
            let user_data = allocator
                .as_mut()
                .map(|host| (&mut **host as *mut GovernedDeviceHost).cast())
                .unwrap_or(std::ptr::null_mut());
            let table = Box::new(KapslBackendHostV1 {
                struct_size: std::mem::size_of::<KapslBackendHostV1>() as u32,
                abi_version: KAPSL_BACKEND_ABI_VERSION,
                user_data,
                log: Some(host_log),
                allocate_device: allocator
                    .as_ref()
                    .map(|_| allocate_device as KapslDeviceAllocateFn),
                free_device: allocator.as_ref().map(|_| free_device as KapslDeviceFreeFn),
                synchronize_device: allocator
                    .as_ref()
                    .map(|_| synchronize_device as KapslDeviceSynchronizeFn),
            });
            Ok(Self { table, allocator })
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            let _ = (resources, backend, device_id, model_id, replica_id);
            if require_governed {
                return Err(
                    "native pack requires governed device memory, but this runtime was built without GPU pool authority"
                        .to_string(),
                );
            }
            Ok(Self {
                table: Box::new(KapslBackendHostV1 {
                    struct_size: std::mem::size_of::<KapslBackendHostV1>() as u32,
                    abi_version: KAPSL_BACKEND_ABI_VERSION,
                    user_data: std::ptr::null_mut(),
                    log: Some(host_log),
                    allocate_device: None,
                    free_device: None,
                    synchronize_device: None,
                }),
            })
        }
    }

    fn table(&self) -> *const KapslBackendHostV1 {
        self.table.as_ref()
    }

    fn live_allocations(&self) -> usize {
        #[cfg(feature = "gpu-device-pool")]
        {
            self.allocator
                .as_ref()
                .map(|allocator| allocator.live_allocations())
                .unwrap_or(0)
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            0
        }
    }
}

#[cfg(feature = "gpu-device-pool")]
fn pool_backend(backend: &str) -> PoolBackend {
    match backend.trim().to_ascii_lowercase().as_str() {
        "onnx" | "ort" | "onnxruntime" => PoolBackend::Onnx,
        "llama.cpp" | "llama_cpp" | "llama-cpp" => PoolBackend::Gguf,
        _ => PoolBackend::Native,
    }
}

#[cfg(feature = "gpu-device-pool")]
struct GovernedDeviceHost {
    pool: Arc<GpuDevicePool>,
    backend: PoolBackend,
    device_id: usize,
    model_id: u32,
    replica_id: u32,
    next_allocation_id: AtomicU64,
    allocations: Mutex<HashMap<u64, GpuAllocation>>,
}

#[cfg(feature = "gpu-device-pool")]
impl GovernedDeviceHost {
    fn new(
        pool: Arc<GpuDevicePool>,
        backend: PoolBackend,
        device_id: usize,
        model_id: u32,
        replica_id: u32,
    ) -> Self {
        Self {
            pool,
            backend,
            device_id,
            model_id,
            replica_id,
            next_allocation_id: AtomicU64::new(1),
            allocations: Mutex::new(HashMap::new()),
        }
    }

    fn allocate(
        &self,
        request: KapslDeviceAllocationRequestV1,
    ) -> Result<KapslDeviceAllocationV1, String> {
        if request.struct_size < std::mem::size_of::<KapslDeviceAllocationRequestV1>() as u32 {
            return Err("device allocation request struct is truncated".to_string());
        }
        if request.reserved != 0 || request.flags != 0 {
            return Err("device allocation request uses unsupported flags".to_string());
        }
        if request.memory_kind != KAPSL_MEMORY_CUDA {
            return Err(format!(
                "governed device allocator supports CUDA memory, not kind {}",
                request.memory_kind
            ));
        }
        if request.device_id as usize != self.device_id
            || request.model_id != self.model_id
            || request.replica_id != self.replica_id
        {
            return Err(
                "device allocation identity does not match its backend instance".to_string(),
            );
        }
        let bytes = usize::try_from(request.bytes)
            .map_err(|_| "device allocation byte count exceeds this platform".to_string())?;
        let alignment = usize::try_from(request.alignment)
            .map_err(|_| "device allocation alignment exceeds this platform".to_string())?;
        if bytes == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(
                "device allocation requires non-zero bytes and power-of-two alignment".to_string(),
            );
        }
        let class = match request.allocation_class {
            KAPSL_ALLOCATION_CLASS_WEIGHTS => PoolAllocationClass::PersistentWeights,
            KAPSL_ALLOCATION_CLASS_WORKSPACE => PoolAllocationClass::TransientWorkspace,
            KAPSL_ALLOCATION_CLASS_KV => PoolAllocationClass::KvCache,
            KAPSL_ALLOCATION_CLASS_REQUEST => PoolAllocationClass::RequestTransient,
            KAPSL_ALLOCATION_CLASS_OTHER => PoolAllocationClass::ExternallyOwned,
            other => return Err(format!("unknown device allocation class {other}")),
        };
        let owner = PoolOwner::new(self.backend, self.model_id, self.replica_id, class);
        let allocation = self
            .pool
            .alloc(owner, bytes, alignment)
            .map_err(|error| format!("runtime device-pool allocation failed: {error}"))?;
        let pointer = self.pool.allocation_ptr(&allocation);
        let granted_bytes = allocation.bytes() as u64;
        let allocation_id = match self.next_allocation_id.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |next| next.checked_add(1),
        ) {
            Ok(allocation_id) => allocation_id,
            Err(_) => {
                let _ = self.pool.free(allocation);
                return Err("native device allocation ID space exhausted".to_string());
            }
        };
        self.allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(allocation_id, allocation);
        Ok(KapslDeviceAllocationV1 {
            struct_size: std::mem::size_of::<KapslDeviceAllocationV1>() as u32,
            reserved: 0,
            allocation_id,
            device_ptr: pointer,
            granted_bytes,
        })
    }

    fn free(&self, returned: KapslDeviceAllocationV1) -> Result<(), String> {
        if returned.struct_size < std::mem::size_of::<KapslDeviceAllocationV1>() as u32
            || returned.reserved != 0
            || returned.allocation_id == 0
        {
            return Err("device free contains an invalid allocation identity".to_string());
        }
        let mut allocations = self
            .allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let stored = allocations
            .get(&returned.allocation_id)
            .ok_or_else(|| "device free references an unknown allocation ID".to_string())?;
        if self.pool.allocation_ptr(stored) != returned.device_ptr
            || stored.bytes() as u64 != returned.granted_bytes
        {
            return Err(
                "device free pointer or byte count does not match its allocation ID".to_string(),
            );
        }
        let allocation = allocations
            .remove(&returned.allocation_id)
            .expect("allocation checked above");
        drop(allocations);
        self.pool
            .free(allocation)
            .map_err(|error| format!("runtime device-pool free failed: {error}"))
    }

    fn synchronize(&self, device_id: u32) -> Result<(), String> {
        if device_id as usize != self.device_id {
            return Err(format!(
                "backend requested synchronization for device {device_id}, expected {}",
                self.device_id
            ));
        }
        self.pool
            .device()
            .bind_to_thread()
            .map_err(|error| format!("bind CUDA device before synchronize: {error}"))?;
        self.pool
            .device()
            .synchronize()
            .map_err(|error| format!("synchronize governed CUDA device: {error}"))
    }

    fn live_allocations(&self) -> usize {
        self.allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

#[cfg(feature = "gpu-device-pool")]
impl Drop for GovernedDeviceHost {
    fn drop(&mut self) {
        let allocations = self
            .allocations
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain()
            .map(|(_, allocation)| allocation)
            .collect::<Vec<_>>();
        if allocations.is_empty() {
            return;
        }
        log::error!(
            "native backend leaked {} governed allocations; reclaiming after shutdown",
            allocations.len()
        );
        if let Err(error) = self.pool.device().synchronize() {
            log::error!("synchronize before reclaiming leaked native allocations: {error}");
        }
        for allocation in allocations {
            if let Err(error) = self.pool.free(allocation) {
                log::error!("reclaim leaked native allocation: {error}");
            }
        }
    }
}

#[cfg(feature = "gpu-device-pool")]
unsafe extern "C" fn allocate_device(
    user_data: *mut c_void,
    request: *const KapslDeviceAllocationRequestV1,
    allocation_out: *mut KapslDeviceAllocationV1,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null() || request.is_null() || allocation_out.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: non-null callback pointers are borrowed for this invocation.
        let host = unsafe { &*(user_data as *const GovernedDeviceHost) };
        let request = unsafe { *request };
        match host.allocate(request) {
            Ok(allocation) => {
                // SAFETY: the caller provided a writable output slot.
                unsafe { *allocation_out = allocation };
                KAPSL_STATUS_OK
            }
            Err(error) => {
                log::error!("native backend governed allocation rejected: {error}");
                KAPSL_STATUS_BACKEND_ERROR
            }
        }
    }))
    .unwrap_or(KAPSL_STATUS_PANIC)
}

#[cfg(feature = "gpu-device-pool")]
unsafe extern "C" fn free_device(
    user_data: *mut c_void,
    allocation: *const KapslDeviceAllocationV1,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null() || allocation.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: non-null callback pointers are borrowed for this invocation.
        let host = unsafe { &*(user_data as *const GovernedDeviceHost) };
        let allocation = unsafe { *allocation };
        match host.free(allocation) {
            Ok(()) => KAPSL_STATUS_OK,
            Err(error) => {
                log::error!("native backend governed free rejected: {error}");
                KAPSL_STATUS_INVALID_ARGUMENT
            }
        }
    }))
    .unwrap_or(KAPSL_STATUS_PANIC)
}

#[cfg(feature = "gpu-device-pool")]
unsafe extern "C" fn synchronize_device(user_data: *mut c_void, device_id: u32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: user_data points to the retained governed host.
        let host = unsafe { &*(user_data as *const GovernedDeviceHost) };
        match host.synchronize(device_id) {
            Ok(()) => KAPSL_STATUS_OK,
            Err(error) => {
                log::error!("native backend device synchronize rejected: {error}");
                KAPSL_STATUS_BACKEND_ERROR
            }
        }
    }))
    .unwrap_or(KAPSL_STATUS_PANIC)
}

fn take_owned_buffer(
    api: &KapslBackendApiV1,
    buffer: KapslOwnedBuffer,
    label: &str,
) -> Result<Vec<u8>, String> {
    if buffer.ptr.is_null() {
        return if buffer.len == 0 && buffer.capacity == 0 {
            Ok(Vec::new())
        } else {
            Err(format!(
                "native backend returned an invalid null {label} buffer"
            ))
        };
    }
    let result = if buffer.len > buffer.capacity || buffer.len > MAX_JSON_BUFFER_BYTES {
        Err(format!(
            "native backend {label} buffer has invalid length {} and capacity {}",
            buffer.len, buffer.capacity
        ))
    } else {
        // SAFETY: the validated pack owns a readable buffer until free_buffer.
        Ok(unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec())
    };
    // SAFETY: the buffer is returned exactly once to the table that produced it.
    unsafe { api.free_buffer.expect("validated free-buffer function")(buffer) };
    result
}

fn read_ffi_error(api: &KapslBackendApiV1, status: i32, error: KapslOwnedBuffer) -> String {
    if error.ptr.is_null() {
        return format!("native backend returned status {status} without an error message");
    }
    match take_owned_buffer(api, error, "error") {
        Ok(bytes) => String::from_utf8(bytes).unwrap_or_else(|_| {
            format!("native backend returned non-UTF-8 error for status {status}")
        }),
        Err(message) => message,
    }
}

fn status_engine_error(status: i32, message: String) -> EngineError {
    match status {
        KAPSL_STATUS_INVALID_ARGUMENT | KAPSL_STATUS_INCOMPATIBLE_ABI => {
            EngineError::invalid_input(message)
        }
        KAPSL_STATUS_CANCELLED => EngineError::cancelled(message),
        _ => EngineError::backend(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CancelProbe {
        request_id: Mutex<Option<u64>>,
        changed: std::sync::Condvar,
    }

    unsafe extern "C" fn describe(
        output: *mut KapslOwnedBuffer,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        if !output.is_null() {
            let mut bytes = br#"{"backend":"test"}"#.to_vec();
            let buffer = KapslOwnedBuffer {
                ptr: bytes.as_mut_ptr(),
                len: bytes.len(),
                capacity: bytes.capacity(),
            };
            std::mem::forget(bytes);
            // SAFETY: output was checked above.
            unsafe { *output = buffer };
        }
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn initialize(
        _config: *const KapslBackendConfigV1,
        _handle: *mut *mut c_void,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn path_report(
        _handle: *mut c_void,
        _path: KapslSlice,
        _output: *mut KapslOwnedBuffer,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn load(
        _handle: *mut c_void,
        _path: KapslSlice,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn request_report(
        _handle: *mut c_void,
        _request: *const KapslInferenceRequestV1,
        _output: *mut KapslOwnedBuffer,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn infer(
        _handle: *mut c_void,
        _request: *const KapslInferenceRequestV1,
        _result: *mut KapslInferenceResultV1,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn infer_stream(
        _handle: *mut c_void,
        _request: *const KapslInferenceRequestV1,
        _user_data: *mut c_void,
        _on_chunk: Option<KapslBackendStreamChunkFn>,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn cancel(handle: *mut c_void, request_id: u64) -> i32 {
        if handle.is_null() {
            return KAPSL_STATUS_OK;
        }
        // SAFETY: cancellation tests retain this probe until the target is
        // deactivated and every watcher has been dropped.
        let probe = unsafe { &*handle.cast::<CancelProbe>() };
        *probe
            .request_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(request_id);
        probe.changed.notify_all();
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn json_report(
        _handle: *mut c_void,
        _output: *mut KapslOwnedBuffer,
        _error: *mut KapslOwnedBuffer,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn health(_handle: *mut c_void, _error: *mut KapslOwnedBuffer) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn unload(_handle: *mut c_void, _error: *mut KapslOwnedBuffer) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn shutdown(_handle: *mut c_void) {}
    unsafe extern "C" fn release(_handle: *mut c_void, _result: *mut KapslInferenceResultV1) {}

    unsafe extern "C" fn free_buffer(buffer: KapslOwnedBuffer) {
        if !buffer.ptr.is_null() {
            // SAFETY: tests create buffers from Vec with the recorded layout.
            unsafe {
                drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.capacity));
            }
        }
    }

    fn complete_api(capabilities: u64) -> KapslBackendApiV1 {
        KapslBackendApiV1 {
            magic: KAPSL_BACKEND_ENTRYPOINT_MAGIC,
            abi_version: KAPSL_BACKEND_ABI_VERSION,
            struct_size: std::mem::size_of::<KapslBackendApiV1>() as u32,
            wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            capabilities,
            describe: Some(describe),
            initialize: Some(initialize),
            planned_memory: Some(path_report),
            load_model: Some(load),
            planned_request_memory: Some(request_report),
            infer: Some(infer),
            infer_batch: None,
            infer_stream: if capabilities & KAPSL_BACKEND_CAP_STREAMING != 0 {
                Some(infer_stream)
            } else {
                None
            },
            cancel: if capabilities & KAPSL_BACKEND_CAP_CANCELLATION != 0 {
                Some(cancel)
            } else {
                None
            },
            actual_memory: Some(json_report),
            metrics: Some(json_report),
            model_info: Some(json_report),
            kv_capabilities: None,
            kv_topology: None,
            batching_policy: Some(json_report),
            health_check: Some(health),
            unload: Some(unload),
            shutdown: Some(shutdown),
            release_result: Some(release),
            release_batch_result: None,
            free_buffer: Some(free_buffer),
        }
    }

    fn test_manifest(accelerator_profile: &str, capabilities: u64) -> BackendPackManifest {
        let profile = match accelerator_profile {
            "cpu" => crate::backend::ONNX_CPU_PACK_PROFILE,
            "cuda" => crate::backend::ONNX_CUDA12_PACK_PROFILE,
            "tensorrt" => crate::backend::ONNX_TENSORRT10_PACK_PROFILE,
            other => other,
        };
        BackendPackManifest {
            schema_version: 1,
            backend: "onnx".to_string(),
            profile: profile.to_string(),
            pack_version: "1.0.0".to_string(),
            runtime_abi: 1,
            adapter_abi: Some(crate::backend::STANDARD_NATIVE_ADAPTER_ABI.to_string()),
            compatible_kapsl: "*".to_string(),
            platform: "linux-x86_64".to_string(),
            architecture: "x86_64".to_string(),
            accelerator_profile: accelerator_profile.to_string(),
            accelerator_requirements: super::super::BackendAcceleratorRequirements {
                kind: Some(accelerator_profile.to_string()),
                execution_providers: vec![accelerator_profile.to_string()],
                implicit_cpu_fallback: Some(false),
            },
            minimum_cuda: None,
            minimum_driver: None,
            execution_mode: BackendExecutionMode::Native,
            kv_mode: None,
            formats: vec!["onnx".to_string()],
            model_types: Vec::new(),
            tasks: vec!["forward".to_string()],
            capabilities: pack_capabilities_from_abi(capabilities),
            memory_behavior: super::super::BackendMemoryBehavior {
                allocation_scope: (capabilities & KAPSL_BACKEND_CAP_SCOPED_DEVICE_ALLOCATOR != 0)
                    .then(|| "kapsl-scoped-device-allocator-v1".to_string()),
                device_allocation: Some(if accelerator_profile == "cpu" {
                    "none".to_string()
                } else {
                    "host-governed-scoped".to_string()
                }),
                planned_reporting: true,
                live_reporting: true,
                request_reporting: true,
                synchronize_before_free: accelerator_profile != "cpu",
            },
            entrypoint: "lib/libbackend.so".to_string(),
            artifact: "artifact.tar.gz".to_string(),
            download_bytes: 1,
            installed_bytes: 1,
            sha256: "00".repeat(32),
            signature: "signature".to_string(),
            memory: Default::default(),
            installer: Default::default(),
            files: Default::default(),
            licenses: Vec::new(),
            priority: 0,
        }
    }

    #[test]
    fn signed_profile_and_execution_capabilities_must_match_exactly() {
        let cpu = complete_api(KAPSL_BACKEND_CAP_CPU | KAPSL_BACKEND_CAP_MEMORY_REPORTING);
        assert!(validate_native_backend_api(&test_manifest("cpu", cpu.capabilities), &cpu).is_ok());

        let mixed = complete_api(
            KAPSL_BACKEND_CAP_CPU | KAPSL_BACKEND_CAP_CUDA | KAPSL_BACKEND_CAP_MEMORY_REPORTING,
        );
        assert!(
            validate_native_backend_api(&test_manifest("cpu", mixed.capabilities), &mixed).is_err()
        );

        let unmanaged_cuda =
            complete_api(KAPSL_BACKEND_CAP_CUDA | KAPSL_BACKEND_CAP_MEMORY_REPORTING);
        assert!(validate_native_backend_api(
            &test_manifest("cuda", unmanaged_cuda.capabilities),
            &unmanaged_cuda
        )
        .is_err());

        let governed_cuda = complete_api(
            KAPSL_BACKEND_CAP_CUDA
                | KAPSL_BACKEND_CAP_MEMORY_REPORTING
                | KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
        );
        assert!(validate_native_backend_api(
            &test_manifest("cuda", governed_cuda.capabilities),
            &governed_cuda
        )
        .is_ok());

        let mut mismatched = test_manifest("cuda", governed_cuda.capabilities);
        mismatched.accelerator_profile = "tensorrt".to_string();
        assert!(validate_native_backend_api(&mismatched, &governed_cuda).is_err());
    }

    #[test]
    fn descriptor_must_match_the_signed_identity_and_capability_table() {
        let api = complete_api(
            KAPSL_BACKEND_CAP_CUDA
                | KAPSL_BACKEND_CAP_MEMORY_REPORTING
                | KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
        );
        let manifest = test_manifest("cuda", api.capabilities);
        let descriptor = serde_json::json!({
            "schema_version": KAPSL_BACKEND_DESCRIPTOR_SCHEMA_V1,
            "backend": "onnx",
            "profiles": ["cuda12"],
            "formats": ["onnx"],
            "tasks": ["forward"],
            "backend_abi": KAPSL_BACKEND_ABI_VERSION,
            "wire_format": KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            "execution_mode": "native",
            "governed_device_memory": true,
        });
        assert!(validate_native_backend_descriptor(&manifest, &api, &descriptor).is_ok());

        for (field, value) in [
            ("profiles", serde_json::json!(["cpu"])),
            ("governed_device_memory", serde_json::json!(false)),
            ("execution_mode", serde_json::json!("external")),
        ] {
            let mut invalid = descriptor.clone();
            invalid[field] = value;
            assert!(validate_native_backend_descriptor(&manifest, &api, &invalid).is_err());
        }
    }

    #[test]
    fn cancellation_capability_requires_and_accepts_the_cancel_hook() {
        let capabilities = KAPSL_BACKEND_CAP_CPU
            | KAPSL_BACKEND_CAP_MEMORY_REPORTING
            | KAPSL_BACKEND_CAP_CANCELLATION;
        let cancellable = complete_api(capabilities);
        assert!(validate_native_backend_api(
            &test_manifest("cpu", cancellable.capabilities),
            &cancellable
        )
        .is_ok());

        let mut contradictory = cancellable;
        contradictory.cancel = None;
        assert!(validate_native_backend_api(
            &test_manifest("cpu", contradictory.capabilities),
            &contradictory
        )
        .is_err());
    }

    #[test]
    fn streaming_capability_requires_and_accepts_the_stream_hook() {
        let capabilities = KAPSL_BACKEND_CAP_CPU
            | KAPSL_BACKEND_CAP_MEMORY_REPORTING
            | KAPSL_BACKEND_CAP_STREAMING;
        let streaming = complete_api(capabilities);
        assert!(validate_native_backend_api(
            &test_manifest("cpu", streaming.capabilities),
            &streaming
        )
        .is_ok());

        let mut contradictory = streaming;
        contradictory.infer_stream = None;
        assert!(validate_native_backend_api(
            &test_manifest("cpu", contradictory.capabilities),
            &contradictory
        )
        .is_err());
    }

    #[test]
    fn cancellation_token_invokes_native_hook_on_dedicated_runtime() {
        let runtime = native_bridge_runtime().unwrap();
        let probe = Box::new(CancelProbe::default());
        let handle = (&*probe as *const CancelProbe).cast_mut().cast::<c_void>();
        let target = Arc::new(NativeCancelTarget::new(handle, Some(cancel)));
        let cancellation = CancellationToken::new();
        let watches =
            NativeCancellationWatches::spawn(Some(&runtime), &target, [(91, cancellation.clone())]);

        cancellation.cancel();

        let observed = probe
            .request_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (observed, timeout) = probe
            .changed
            .wait_timeout_while(observed, std::time::Duration::from_secs(2), |value| {
                value.is_none()
            })
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(!timeout.timed_out());
        assert_eq!(*observed, Some(91));
        drop(observed);
        drop(watches);
        target.deactivate();
        assert_eq!(target.cancel(92), None);
    }

    #[test]
    fn dropping_an_unfinished_native_stream_invokes_the_cancel_hook() {
        let probe = Box::new(CancelProbe::default());
        let handle = (&*probe as *const CancelProbe).cast_mut().cast::<c_void>();
        let target = Arc::new(NativeCancelTarget::new(handle, Some(cancel)));
        let completed = Arc::new(AtomicBool::new(false));
        drop(NativeStreamDropGuard {
            target: Arc::clone(&target),
            request_id: 93,
            completed,
        });
        assert_eq!(
            *probe
                .request_id
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            Some(93)
        );
        target.deactivate();
    }

    #[test]
    fn dropping_a_completed_native_stream_does_not_cancel_it() {
        let probe = Box::new(CancelProbe::default());
        let handle = (&*probe as *const CancelProbe).cast_mut().cast::<c_void>();
        let target = Arc::new(NativeCancelTarget::new(handle, Some(cancel)));
        let completed = Arc::new(AtomicBool::new(true));
        drop(NativeStreamDropGuard {
            target: Arc::clone(&target),
            request_id: 94,
            completed,
        });
        assert_eq!(
            *probe
                .request_id
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            None
        );
        target.deactivate();
    }

    #[test]
    fn request_bridge_borrows_contiguous_tensor_views_without_json_payloads() {
        let input = BinaryTensorPacket::new(
            vec![1, 2],
            TensorDtype::Float32,
            vec![0; 2 * std::mem::size_of::<f32>()],
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        let mut request = InferenceRequest::new(input);
        request.cancellation = Some(cancellation.clone());
        let mut bridge = RequestBridge::new(&request).unwrap();
        let wire = bridge.wire(7);
        assert_eq!(wire.request_id, 7);
        assert_eq!(wire.input_count, 1);
        // SAFETY: the bridge and request remain live in this scope.
        let named = unsafe { &*wire.inputs };
        assert_eq!(named.tensor.data, request.input.data.as_ptr().cast());
        assert_eq!(named.tensor.byte_len, request.input.data.len() as u64);
        assert_eq!(named.tensor.memory_kind, KAPSL_MEMORY_HOST);
        assert_ne!(named.tensor.flags & KAPSL_TENSOR_FLAG_CONTIGUOUS, 0);
        // Only small request metadata is JSON; tensor bytes are direct views.
        assert!(bridge.metadata_json.len() < request.input.data.len() + 128);
        // The borrowed callback covers cancellation that races adapter dispatch
        // before the explicit cancel hook can find the request ID.
        assert_eq!(
            unsafe {
                wire.is_cancelled.expect("request cancellation callback")(
                    wire.cancellation_context,
                    wire.request_id,
                )
            },
            0
        );
        cancellation.cancel();
        assert_eq!(
            unsafe {
                wire.is_cancelled.expect("request cancellation callback")(
                    wire.cancellation_context,
                    wire.request_id,
                )
            },
            1
        );
    }

    #[test]
    fn output_bridge_rejects_device_or_noncontiguous_results() {
        let shape = [1_i64];
        let data = [0_u8; 4];
        let mut output = KapslNamedTensorViewV1 {
            struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
            reserved: 0,
            name: KapslSlice::from_bytes(b"output"),
            tensor: KapslTensorViewV1 {
                struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
                dtype: KAPSL_DTYPE_F32,
                memory_kind: KAPSL_MEMORY_CUDA,
                flags: KAPSL_TENSOR_FLAG_CONTIGUOUS,
                device_id: 0,
                rank: 1,
                shape: shape.as_ptr(),
                strides: std::ptr::null(),
                data: data.as_ptr().cast(),
                byte_len: data.len() as u64,
            },
        };
        assert!(copy_output_tensor(&output).is_err());
        output.tensor.memory_kind = KAPSL_MEMORY_HOST;
        output.tensor.flags = 0;
        assert!(copy_output_tensor(&output).is_err());
        output.tensor.flags = KAPSL_TENSOR_FLAG_CONTIGUOUS;
        assert!(copy_output_tensor(&output).is_ok());
        output.tensor.flags = KAPSL_TENSOR_FLAG_CONTIGUOUS | (1 << 31);
        assert!(copy_output_tensor(&output).is_err());
    }

    #[test]
    fn stream_callback_copies_borrowed_chunks_into_the_bounded_bridge() {
        let (sender, receiver) = async_channel::bounded(1);
        let mut context = NativeStreamCallbackContext {
            sender,
            callback_error: None,
            consumer_closed: false,
        };
        let shape = [1_i64, 5_i64];
        let mut data = b"hello".to_vec();
        let output = KapslNamedTensorViewV1 {
            struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
            reserved: 0,
            name: KapslSlice::from_bytes(b"token"),
            tensor: KapslTensorViewV1 {
                struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
                dtype: KAPSL_DTYPE_UTF8,
                memory_kind: KAPSL_MEMORY_HOST,
                flags: KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY,
                device_id: -1,
                rank: shape.len() as u32,
                shape: shape.as_ptr(),
                strides: std::ptr::null(),
                data: data.as_ptr().cast(),
                byte_len: data.len() as u64,
            },
        };
        let result = KapslInferenceResultV1 {
            struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
            output_count: 1,
            outputs: &output,
            metadata_json: KapslSlice::empty(),
            owner_context: std::ptr::null_mut(),
        };
        // SAFETY: every borrowed callback value remains live for this call.
        assert_eq!(
            unsafe {
                native_stream_chunk(
                    (&mut context as *mut NativeStreamCallbackContext).cast(),
                    9,
                    &result,
                )
            },
            KAPSL_STATUS_OK
        );
        data.fill(b'x');
        let packet = receiver.recv_blocking().expect("stream chunk").unwrap();
        assert_eq!(packet.dtype, TensorDtype::Utf8);
        assert_eq!(packet.shape, vec![1, 5]);
        assert_eq!(packet.data, b"hello");
    }

    #[test]
    fn stream_callback_applies_backpressure_when_the_bounded_bridge_is_full() {
        let (sender, receiver) = async_channel::bounded(1);
        sender
            .send_blocking(Err(EngineError::backend("occupied")))
            .unwrap();
        let producer = std::thread::spawn(move || {
            let mut context = NativeStreamCallbackContext {
                sender,
                callback_error: None,
                consumer_closed: false,
            };
            let shape = [1_i64];
            let data = *b"x";
            let output = KapslNamedTensorViewV1 {
                struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
                reserved: 0,
                name: KapslSlice::from_bytes(b"token"),
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
                    byte_len: 1,
                },
            };
            let result = KapslInferenceResultV1 {
                struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
                output_count: 1,
                outputs: &output,
                metadata_json: KapslSlice::empty(),
                owner_context: std::ptr::null_mut(),
            };
            // SAFETY: every borrowed callback value remains live for this call.
            let status = unsafe {
                native_stream_chunk(
                    (&mut context as *mut NativeStreamCallbackContext).cast(),
                    10,
                    &result,
                )
            };
            (
                status,
                context.consumer_closed,
                context.callback_error.is_none(),
            )
        });

        assert!(receiver.recv_blocking().expect("occupied slot").is_err());
        let (status, consumer_closed, callback_clean) = producer.join().unwrap();
        assert_eq!(status, KAPSL_STATUS_OK);
        assert!(!consumer_closed);
        assert!(callback_clean);
        let packet = receiver
            .recv_blocking()
            .expect("backpressured stream chunk")
            .unwrap();
        assert_eq!(packet.data, b"x");
    }

    #[test]
    fn stream_callback_cancels_when_the_bounded_bridge_is_closed() {
        let (sender, receiver) = async_channel::bounded(1);
        drop(receiver);
        let mut context = NativeStreamCallbackContext {
            sender,
            callback_error: None,
            consumer_closed: false,
        };
        let shape = [1_i64];
        let data = *b"x";
        let output = KapslNamedTensorViewV1 {
            struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
            reserved: 0,
            name: KapslSlice::from_bytes(b"token"),
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
                byte_len: 1,
            },
        };
        let result = KapslInferenceResultV1 {
            struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
            output_count: 1,
            outputs: &output,
            metadata_json: KapslSlice::empty(),
            owner_context: std::ptr::null_mut(),
        };
        // SAFETY: every borrowed callback value remains live for this call.
        assert_eq!(
            unsafe {
                native_stream_chunk(
                    (&mut context as *mut NativeStreamCallbackContext).cast(),
                    10,
                    &result,
                )
            },
            KAPSL_STATUS_CANCELLED
        );
        assert!(context.consumer_closed);
        assert!(context.callback_error.is_none());
    }

    #[test]
    fn descriptor_is_released_through_the_same_function_table() {
        let api = complete_api(KAPSL_BACKEND_CAP_CPU | KAPSL_BACKEND_CAP_MEMORY_REPORTING);
        let descriptor = describe_backend(&api).unwrap();
        assert_eq!(descriptor["backend"], "test");
    }

    #[test]
    fn migration_switch_is_strict_and_explicit() {
        assert!(resolve_generic_native_packs_switch(None).unwrap());
        assert!(parse_generic_native_packs_switch("1").unwrap());
        assert!(!parse_generic_native_packs_switch("off").unwrap());
        assert!(parse_generic_native_packs_switch("enabled").is_err());
    }

    #[test]
    fn truncated_api_table_is_rejected_before_full_table_read() {
        let prefix = KapslBackendApiPrefixV1 {
            magic: KAPSL_BACKEND_ENTRYPOINT_MAGIC,
            abi_version: KAPSL_BACKEND_ABI_VERSION,
            struct_size: std::mem::size_of::<KapslBackendApiPrefixV1>() as u32,
            wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
        };
        // SAFETY: the helper reads only `prefix` after observing its truncated
        // size and returns before attempting to read a complete function table.
        let result = unsafe {
            copy_native_backend_api(
                (&prefix as *const KapslBackendApiPrefixV1).cast::<KapslBackendApiV1>(),
            )
        };
        assert!(result.is_err());
    }
}
