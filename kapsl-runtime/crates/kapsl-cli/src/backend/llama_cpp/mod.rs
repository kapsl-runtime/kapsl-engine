//! Signed lazy activation and C-ABI loading for native llama.cpp packs.
//!
//! The certified eager CUDA shared-KV backend remains available as the
//! rollback path. Lazy CPU packs are enabled on Linux x86_64 by default; a
//! native-KV CUDA pack requires an explicit policy acknowledgement until a
//! pack advertises the shared-pool callback capability.

mod shared_pool;

use self::shared_pool::host_log_bridge;
#[cfg(feature = "gpu-device-pool")]
use self::shared_pool::LlamaCppSharedPoolHost;
use super::{
    guarded_host_memory_bytes, inspect_model_weight_bytes, MemoryAdmissionStatus,
    PreliminaryMemoryAdmission,
};
use super::{
    BackendAccelerator, BackendManager, BackendPackManifest, BackendTarget,
    LlamaCppBackendPackProfile,
};
use crate::runtime::RuntimeResources;
use crate::runtime::{MemoryDomain, MemorySnapshot};
use futures::channel::mpsc;
use kapsl_backend_abi::*;
use kapsl_core::{EngineKind, Manifest};
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo, EngineStream,
    InferenceRequest, MemoryReport,
};
use kapsl_hal::device::DeviceInfo;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const LAZY_BACKENDS_ENV: &str = "KAPSL_LAZY_BACKENDS";
const LAZY_LLAMA_PACKS_ENV: &str = "KAPSL_LAZY_LLAMA_CPP_PACKS";
const ALLOW_NATIVE_KV_ENV: &str = "KAPSL_LLAMA_CPP_ALLOW_NATIVE_KV";
const PRELIMINARY_MEMORY_GUARD_PERCENT: u64 = 10;
const MINIMUM_LLAMA_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;

static OFFLINE: AtomicBool = AtomicBool::new(false);

struct ActiveLlamaPack {
    identity: (String, String, String),
    profile: LlamaCppBackendPackProfile,
    kv_mode: LlamaCppPackKvMode,
    api: KapslLlamaCppApiV1,
    #[allow(dead_code)]
    library: libloading::Library,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LlamaCppPackKvMode {
    Native,
    SharedPool,
}

impl LlamaCppPackKvMode {
    fn from_manifest(pack: &BackendPackManifest) -> Result<Self, String> {
        match pack.kv_mode.as_deref() {
            Some("native") => Ok(Self::Native),
            Some("shared_pool") => Ok(Self::SharedPool),
            Some(mode) => Err(format!(
                "signed llama.cpp pack declares unsupported KV mode `{mode}`"
            )),
            None => Err("signed llama.cpp pack does not declare kv_mode".to_string()),
        }
    }
}

fn active_packs() -> &'static Mutex<Vec<Arc<ActiveLlamaPack>>> {
    static ACTIVE: OnceLock<Mutex<Vec<Arc<ActiveLlamaPack>>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(Vec::new()))
}

pub(crate) fn configure_llama_cpp_backend_packs(offline: bool) {
    if offline {
        OFFLINE.store(true, Ordering::Release);
    }
}

fn env_switch(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

pub(crate) fn llama_cpp_lazy_packs_supported_for_platform(platform: &str) -> bool {
    platform == "linux-x86_64"
}

fn eager_llama_backend_compiled() -> bool {
    cfg!(any(
        feature = "gguf-native",
        feature = "gguf-cuda",
        feature = "gguf-cuda-shared-kv"
    ))
}

pub(crate) fn lazy_llama_cpp_packs_enabled() -> bool {
    if !env_switch(LAZY_BACKENDS_ENV, true) {
        return false;
    }
    env_switch(
        LAZY_LLAMA_PACKS_ENV,
        llama_cpp_lazy_packs_supported_for_platform(&super::current_platform())
            && !eager_llama_backend_compiled(),
    )
}

pub(crate) fn llama_cpp_pack_profile_for_target(
    manifest: &Manifest,
    accelerator: BackendAccelerator,
) -> Option<LlamaCppBackendPackProfile> {
    if !EngineKind::resolve(manifest).is_gguf() {
        return None;
    }
    Some(match accelerator {
        BackendAccelerator::Cpu => LlamaCppBackendPackProfile::Cpu,
        BackendAccelerator::Cuda | BackendAccelerator::TensorRt => {
            LlamaCppBackendPackProfile::Cuda12
        }
    })
}

pub(crate) fn llama_cpp_pack_profile_for_manifest(
    manifest: &Manifest,
    device_info: &DeviceInfo,
) -> Option<LlamaCppBackendPackProfile> {
    // A portable core has no CUDA allocation authority and therefore selects
    // the CPU pack even if a driver happens to be visible. CUDA release builds
    // retain the eager shared-KV path unless an operator explicitly forces the
    // lazy pack during rollout.
    let accelerator = if cfg!(feature = "gpu-device-pool") && device_info.has_cuda {
        BackendAccelerator::Cuda
    } else {
        BackendAccelerator::Cpu
    };
    llama_cpp_pack_profile_for_target(manifest, accelerator)
}

fn target_for_profile(
    profile: LlamaCppBackendPackProfile,
    device_info: &DeviceInfo,
) -> BackendTarget {
    let mut target = BackendTarget::current(device_info);
    target.accelerator = profile.accelerator();
    if profile == LlamaCppBackendPackProfile::Cpu {
        target.cuda_version = None;
        target.driver_version = None;
    }
    target
}

pub(crate) fn ensure_llama_cpp_backend_pack(
    manifest: &Manifest,
    model_path: &Path,
    device_info: &DeviceInfo,
    memory_snapshot: Option<&MemorySnapshot>,
) -> Result<bool, String> {
    if !EngineKind::resolve(manifest).is_gguf() || !lazy_llama_cpp_packs_enabled() {
        return Ok(false);
    }
    let Some(profile) = llama_cpp_pack_profile_for_manifest(manifest, device_info) else {
        return Ok(false);
    };
    let manager = BackendManager::from_env(OFFLINE.load(Ordering::Acquire))
        .map_err(|error| error.to_string())?;
    let target = target_for_profile(profile, device_info);
    let plan = manager
        .plan_llama_cpp(profile, &target)
        .map_err(|error| error.to_string())?;
    let kv_mode = LlamaCppPackKvMode::from_manifest(&plan.manifest)?;
    if profile == LlamaCppBackendPackProfile::Cuda12
        && kv_mode == LlamaCppPackKvMode::Native
        && !env_switch(ALLOW_NATIVE_KV_ENV, false)
    {
        return Err(format!(
            "lazy llama.cpp CUDA pack uses backend-owned native KV and is disabled by memory policy; keep the certified eager shared-KV runtime or set {ALLOW_NATIVE_KV_ENV}=1 to select the explicit native-KV rollback"
        ));
    }
    let admission = preliminary_llama_cpp_memory_admission(
        profile,
        model_path,
        manifest,
        device_info,
        &plan.manifest,
        memory_snapshot,
    )?;
    if admission.status == MemoryAdmissionStatus::Rejected {
        return Err(format!(
            "preliminary memory admission rejected llama.cpp/{} before backend download: {}",
            profile.profile(),
            admission.reason
        ));
    }
    let root = manager
        .ensure_pack(&plan.manifest)
        .map_err(|error| error.to_string())?;
    activate_pack(profile, &plan.manifest, &root)?;
    Ok(true)
}

pub(crate) fn preliminary_llama_cpp_memory_admission(
    profile: LlamaCppBackendPackProfile,
    model_path: &Path,
    manifest: &Manifest,
    device_info: &DeviceInfo,
    pack: &BackendPackManifest,
    memory_snapshot: Option<&MemorySnapshot>,
) -> Result<PreliminaryMemoryAdmission, String> {
    let model_bytes = inspect_model_weight_bytes(model_path, manifest)?;
    let kv_capacity = estimated_native_kv_bytes(model_path)?;
    let fixed = match profile {
        LlamaCppBackendPackProfile::Cpu => pack.memory.host_bytes,
        LlamaCppBackendPackProfile::Cuda12 => pack.memory.accelerator_bytes,
    };
    let required = model_bytes.map(|model_bytes| {
        let weights = model_bytes.saturating_mul(5).saturating_div(4);
        let workspace = (model_bytes / 8)
            .max(MINIMUM_LLAMA_WORKSPACE_BYTES)
            .max(pack.memory.minimum_workspace_bytes)
            .saturating_add(
                model_bytes
                    .saturating_mul(pack.memory.workspace_weight_ppm)
                    .saturating_div(1_000_000),
            );
        weights
            .saturating_add(workspace)
            .saturating_add(kv_capacity.unwrap_or(0))
            .saturating_add(fixed)
    });
    let physical_available = match profile {
        LlamaCppBackendPackProfile::Cpu => {
            guarded_host_memory_bytes(device_info.total_memory, PRELIMINARY_MEMORY_GUARD_PERCENT)
        }
        LlamaCppBackendPackProfile::Cuda12 => device_info
            .devices
            .iter()
            .filter(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
            .map(|device| {
                device
                    .memory_mb
                    .saturating_mul(1024 * 1024)
                    .saturating_mul(100 - PRELIMINARY_MEMORY_GUARD_PERCENT)
                    .saturating_div(100)
            })
            .max()
            .unwrap_or(0),
    };
    let governed_available = memory_snapshot.and_then(|snapshot| match profile {
        LlamaCppBackendPackProfile::Cpu => snapshot
            .domain(&MemoryDomain::Host)
            .map(|domain| domain.available_bytes as u64),
        LlamaCppBackendPackProfile::Cuda12 => snapshot
            .domains
            .iter()
            .filter_map(|domain| match domain.domain {
                MemoryDomain::Cuda { .. } => Some(domain.available_bytes as u64),
                _ => None,
            })
            .max(),
    });
    let available = governed_available
        .map(|governed| governed.min(physical_available))
        .unwrap_or(physical_available);
    match required {
        Some(required) if required > available => Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Rejected,
            required_bytes: Some(required),
            available_bytes: Some(available),
            reason: format!(
                "estimated weights, workspace, and known KV require {required} bytes, but only {available} guarded bytes are available"
            ),
        }),
        Some(required) if kv_capacity.is_some() => Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Accepted,
            required_bytes: Some(required),
            available_bytes: Some(available),
            reason: format!(
                "estimated weights, workspace, and KV capacity require {required} bytes within a guarded {available} byte budget"
            ),
        }),
        Some(required) => Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Unknown,
            required_bytes: Some(required),
            available_bytes: Some(available),
            reason: "the packaged model exposes a conservative weight/workspace lower bound, but KV geometry is available only after GGUF extraction".to_string(),
        }),
        None => Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Unknown,
            required_bytes: None,
            available_bytes: Some(available),
            reason: "the package did not expose enough model size information for preliminary llama.cpp admission".to_string(),
        }),
    }
}

fn estimated_native_kv_bytes(model_path: &Path) -> Result<Option<u64>, String> {
    if model_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
    {
        return Ok(None);
    }
    let file = kapsl_loader::gguf_loader::GgufFile::open(model_path)
        .map_err(|error| format!("read GGUF metadata for preliminary admission: {error}"))?;
    let config = file
        .extract_config()
        .map_err(|error| format!("read GGUF config for preliminary admission: {error}"))?;
    let concurrency = std::env::var("KAPSL_GGUF_MAX_CONCURRENT")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .or_else(|| {
            std::env::var("KAPSL_GGUF_TARGET_CONCURRENCY")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
        })
        .unwrap_or(32);
    let context = std::env::var("KAPSL_GGUF_CTX_PER_SEQ")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2048)
        .min(config.max_position_embeddings.max(1) as u64);
    let bytes_per_cell = (config.num_hidden_layers as u64)
        .saturating_mul(config.num_kv_heads() as u64)
        .saturating_mul(config.head_dim() as u64)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<u16>() as u64);
    Ok(Some(
        concurrency
            .saturating_mul(context)
            .saturating_mul(bytes_per_cell),
    ))
}

fn activate_pack(
    profile: LlamaCppBackendPackProfile,
    pack: &BackendPackManifest,
    root: &Path,
) -> Result<(), String> {
    let identity = (
        pack.backend.clone(),
        pack.profile.clone(),
        pack.pack_version.clone(),
    );
    let mut active = active_packs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if active.iter().any(|item| item.identity == identity) {
        return Ok(());
    }
    let root = root
        .canonicalize()
        .map_err(|error| format!("resolve llama.cpp pack {}: {error}", root.display()))?;
    let entrypoint = root
        .join(&pack.entrypoint)
        .canonicalize()
        .map_err(|error| {
            format!(
                "resolve llama.cpp pack entrypoint {}: {error}",
                root.join(&pack.entrypoint).display()
            )
        })?;
    if !entrypoint.starts_with(&root) {
        return Err("llama.cpp pack entrypoint escapes its installation root".to_string());
    }
    // SAFETY: the artifact was signature/checksum verified and its canonical
    // path is confined to the atomic installation root.
    let library = unsafe { libloading::Library::new(&entrypoint) }.map_err(|error| {
        format!(
            "load llama.cpp pack entrypoint {}: {error}",
            entrypoint.display()
        )
    })?;
    type Entrypoint = unsafe extern "C" fn() -> *const KapslLlamaCppApiV1;
    // SAFETY: symbol name and return layout are the versioned pack contract.
    let api = unsafe {
        let entrypoint: libloading::Symbol<'_, Entrypoint> = library
            .get(KAPSL_LLAMA_CPP_ENTRYPOINT_SYMBOL)
            .map_err(|error| format!("resolve llama.cpp ABI v1 entrypoint: {error}"))?;
        let pointer = entrypoint();
        if pointer.is_null() {
            return Err("llama.cpp ABI v1 entrypoint returned null".to_string());
        }
        *pointer
    };
    validate_api(profile, &api)?;
    let kv_mode = LlamaCppPackKvMode::from_manifest(pack)?;
    if kv_mode == LlamaCppPackKvMode::SharedPool && profile != LlamaCppBackendPackProfile::Cuda12 {
        return Err("only a CUDA llama.cpp pack may declare shared_pool KV".to_string());
    }
    validate_kv_mode_capability(kv_mode, api.capabilities)?;
    log::info!(
        "Activated signed llama.cpp backend pack {}/{} from {}",
        pack.backend,
        pack.profile,
        root.display()
    );
    active.push(Arc::new(ActiveLlamaPack {
        identity,
        profile,
        kv_mode,
        api,
        library,
    }));
    Ok(())
}

fn validate_kv_mode_capability(
    kv_mode: LlamaCppPackKvMode,
    capabilities: u64,
) -> Result<(), String> {
    let selected = match kv_mode {
        LlamaCppPackKvMode::Native => KAPSL_LLAMA_CAP_NATIVE_KV,
        LlamaCppPackKvMode::SharedPool => KAPSL_LLAMA_CAP_SHARED_POOL,
    };
    let declared = capabilities & (KAPSL_LLAMA_CAP_NATIVE_KV | KAPSL_LLAMA_CAP_SHARED_POOL);
    if declared != selected {
        return Err(format!(
            "signed llama.cpp pack declares {:?} KV, but its function table advertises KV capabilities 0x{declared:x} instead of exactly 0x{selected:x}",
            kv_mode
        ));
    }
    Ok(())
}

fn validate_api(
    profile: LlamaCppBackendPackProfile,
    api: &KapslLlamaCppApiV1,
) -> Result<(), String> {
    if api.magic != KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC
        || api.abi_version != KAPSL_LLAMA_CPP_ABI_VERSION
        || api.struct_size < std::mem::size_of::<KapslLlamaCppApiV1>() as u32
        || api.wire_format != KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1
    {
        return Err("llama.cpp pack exposes an incompatible ABI v1 function table".to_string());
    }
    let profile_capability = match profile {
        LlamaCppBackendPackProfile::Cpu => KAPSL_LLAMA_CAP_CPU,
        LlamaCppBackendPackProfile::Cuda12 => KAPSL_LLAMA_CAP_CUDA,
    };
    let required_capabilities = profile_capability
        | KAPSL_LLAMA_CAP_STREAMING
        | KAPSL_LLAMA_CAP_CANCELLATION
        | KAPSL_LLAMA_CAP_MEMORY_REPORTING;
    if api.capabilities & required_capabilities != required_capabilities {
        return Err(format!(
            "llama.cpp pack is missing required capabilities 0x{:x}",
            required_capabilities & !api.capabilities
        ));
    }
    if api.capabilities & (KAPSL_LLAMA_CAP_NATIVE_KV | KAPSL_LLAMA_CAP_SHARED_POOL) == 0 {
        return Err("llama.cpp pack does not declare a governed KV mode".to_string());
    }
    if [
        api.initialize.is_some(),
        api.planned_memory.is_some(),
        api.load_model.is_some(),
        api.planned_request_memory.is_some(),
        api.infer.is_some(),
        api.infer_stream.is_some(),
        api.cancel.is_some(),
        api.actual_memory.is_some(),
        api.metrics.is_some(),
        api.model_info.is_some(),
        api.kv_capabilities.is_some(),
        api.kv_topology.is_some(),
        api.batching_policy.is_some(),
        api.health_check.is_some(),
        api.unload.is_some(),
        api.shutdown.is_some(),
        api.free_buffer.is_some(),
    ]
    .into_iter()
    .any(|present| !present)
    {
        return Err("llama.cpp pack ABI v1 function table is incomplete".to_string());
    }
    Ok(())
}

fn active_pack(profile: LlamaCppBackendPackProfile) -> Option<Arc<ActiveLlamaPack>> {
    active_packs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .rev()
        .find(|pack| pack.profile == profile)
        .cloned()
}

pub(crate) fn create_llama_cpp_pack_engine(
    manifest: &Manifest,
    device_info: &DeviceInfo,
    resources: &RuntimeResources,
    device_id: usize,
    model_id: u32,
    replica_id: u32,
) -> Result<Option<Box<dyn Engine>>, String> {
    if !EngineKind::resolve(manifest).is_gguf() || !lazy_llama_cpp_packs_enabled() {
        return Ok(None);
    }
    let profile = llama_cpp_pack_profile_for_manifest(manifest, device_info)
        .ok_or_else(|| "resolve llama.cpp pack profile".to_string())?;
    let pack = active_pack(profile).ok_or_else(|| {
        format!(
            "signed llama.cpp/{} pack was not activated before backend construction",
            profile.profile()
        )
    })?;
    let instance =
        PackInstance::initialize(pack, profile, resources, device_id, model_id, replica_id)?;
    Ok(Some(Box::new(PackedLlamaEngine { instance })))
}

struct PackInstance {
    pack: Arc<ActiveLlamaPack>,
    handle: *mut c_void,
    next_request_id: AtomicU64,
    // A pack may retain the host callback table until shutdown.
    _host: Box<KapslLlamaHostCallbacksV1>,
    // The callback table's user_data points here. It must outlive pack
    // shutdown, which is enforced by PackInstance::drop.
    #[cfg(feature = "gpu-device-pool")]
    _shared_pool_host: Option<Box<LlamaCppSharedPoolHost>>,
}

// The handle owns a pack-side `PackState`; its inference state is synchronized
// by that implementation, and Kapsl's Engine lifecycle guarantees shutdown
// only after outstanding calls have completed.
unsafe impl Send for PackInstance {}
unsafe impl Sync for PackInstance {}

impl PackInstance {
    fn initialize(
        pack: Arc<ActiveLlamaPack>,
        profile: LlamaCppBackendPackProfile,
        resources: &RuntimeResources,
        device_id: usize,
        model_id: u32,
        replica_id: u32,
    ) -> Result<Arc<Self>, String> {
        #[cfg(not(feature = "gpu-device-pool"))]
        let _ = resources;
        let requires_shared_pool = profile == LlamaCppBackendPackProfile::Cuda12
            && pack.kv_mode == LlamaCppPackKvMode::SharedPool;
        #[cfg(feature = "gpu-device-pool")]
        let mut shared_pool_host = if requires_shared_pool {
            let pool = resources.device_pool(device_id).ok_or_else(|| {
                format!(
                    "llama.cpp CUDA pack requires the runtime-owned pool for device {device_id}, but no pool was materialized"
                )
            })?;
            Some(Box::new(LlamaCppSharedPoolHost::new(
                pool, device_id, model_id, replica_id,
            )))
        } else {
            None
        };
        #[cfg(not(feature = "gpu-device-pool"))]
        if requires_shared_pool {
            return Err(
                "llama.cpp pack requires shared GPU allocation callbacks, but this core was built without GPU pool authority"
                    .to_string(),
            );
        }
        #[cfg(feature = "gpu-device-pool")]
        let host = Box::new(match shared_pool_host.as_mut() {
            Some(shared) => shared.callbacks(),
            None => KapslLlamaHostCallbacksV1 {
                struct_size: std::mem::size_of::<KapslLlamaHostCallbacksV1>() as u32,
                user_data: std::ptr::null_mut(),
                log: Some(host_log_bridge),
                create_shared_pool: None,
                destroy_shared_pool: None,
                shared_pool_bytes: None,
            },
        });
        #[cfg(not(feature = "gpu-device-pool"))]
        let host = Box::new(KapslLlamaHostCallbacksV1 {
            struct_size: std::mem::size_of::<KapslLlamaHostCallbacksV1>() as u32,
            user_data: std::ptr::null_mut(),
            log: Some(host_log_bridge),
            create_shared_pool: None,
            destroy_shared_pool: None,
            shared_pool_bytes: None,
        });
        let config = KapslLlamaConfigV1 {
            struct_size: std::mem::size_of::<KapslLlamaConfigV1>() as u32,
            profile: match profile {
                LlamaCppBackendPackProfile::Cpu => KAPSL_LLAMA_PROFILE_CPU,
                LlamaCppBackendPackProfile::Cuda12 => KAPSL_LLAMA_PROFILE_CUDA12,
            },
            device_id: u32::try_from(device_id)
                .map_err(|_| format!("llama.cpp device id {device_id} exceeds ABI v1"))?,
            model_id,
            replica_id,
            require_shared_pool: u32::from(requires_shared_pool),
            host: host.as_ref(),
        };
        let mut handle = std::ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe {
            pack.api.initialize.expect("validated initialize function")(
                &config,
                &mut handle,
                &mut error,
            )
        };
        if status != KAPSL_STATUS_OK || handle.is_null() {
            return Err(read_ffi_error(&pack, status, error));
        }
        if !error.ptr.is_null() {
            free_ffi_buffer(&pack, error);
        }
        Ok(Arc::new(Self {
            pack,
            handle,
            next_request_id: AtomicU64::new(1),
            _host: host,
            #[cfg(feature = "gpu-device-pool")]
            _shared_pool_host: shared_pool_host,
        }))
    }

    fn request_wire<'a>(
        &'a self,
        request_id: u64,
        json: &'a [u8],
        cancellation: &'a mut CancellationContext,
    ) -> KapslLlamaRequestV1 {
        KapslLlamaRequestV1 {
            struct_size: std::mem::size_of::<KapslLlamaRequestV1>() as u32,
            wire_format: KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1,
            request_id,
            request_json: KapslSlice::from_bytes(json),
            cancellation_context: (cancellation as *mut CancellationContext).cast(),
            is_cancelled: Some(request_cancelled),
        }
    }

    fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    fn call_path_report<T: serde::de::DeserializeOwned>(
        &self,
        path: &Path,
        function: KapslPathReportFn,
    ) -> Result<T, EngineError> {
        let path = path.to_str().ok_or_else(|| {
            EngineError::invalid_input(format!("model path is not UTF-8: {}", path.display()))
        })?;
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe {
            function(
                self.handle,
                KapslSlice::from_bytes(path.as_bytes()),
                &mut output,
                &mut error,
            )
        };
        self.decode_output(status, output, error)
    }

    fn call_json_report<T: serde::de::DeserializeOwned>(
        &self,
        function: KapslJsonReportFn,
    ) -> Result<T, EngineError> {
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe { function(self.handle, &mut output, &mut error) };
        self.decode_output(status, output, error)
    }

    fn decode_output<T: serde::de::DeserializeOwned>(
        &self,
        status: i32,
        output: KapslOwnedBuffer,
        error: KapslOwnedBuffer,
    ) -> Result<T, EngineError> {
        if status != KAPSL_STATUS_OK {
            if !output.ptr.is_null() {
                free_ffi_buffer(&self.pack, output);
            }
            return Err(status_engine_error(
                status,
                read_ffi_error(&self.pack, status, error),
            ));
        }
        if !error.ptr.is_null() {
            free_ffi_buffer(&self.pack, error);
        }
        let bytes = take_ffi_buffer(&self.pack, output)?;
        serde_json::from_slice(&bytes).map_err(|error| {
            EngineError::backend(format!("decode llama.cpp pack JSON response: {error}"))
        })
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let json = serde_json::to_vec(request).map_err(|error| {
            EngineError::invalid_input(format!("encode llama.cpp pack request: {error}"))
        })?;
        let request_id = self.next_request_id();
        let mut cancellation = CancellationContext::from_request(request);
        let wire = self.request_wire(request_id, &json, &mut cancellation);
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe {
            self.pack.api.infer.expect("validated infer function")(
                self.handle,
                &wire,
                &mut output,
                &mut error,
            )
        };
        self.decode_output(status, output, error)
    }

    fn planned_request_memory(
        &self,
        request: &InferenceRequest,
    ) -> Result<MemoryReport, EngineError> {
        let json = serde_json::to_vec(request).map_err(|error| {
            EngineError::invalid_input(format!("encode llama.cpp pack request: {error}"))
        })?;
        let request_id = self.next_request_id();
        let mut cancellation = CancellationContext::from_request(request);
        let wire = self.request_wire(request_id, &json, &mut cancellation);
        let mut output = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe {
            self.pack
                .api
                .planned_request_memory
                .expect("validated request memory function")(
                self.handle,
                &wire,
                &mut output,
                &mut error,
            )
        };
        self.decode_output(status, output, error)
    }
}

impl Drop for PackInstance {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                self.pack.api.shutdown.expect("validated shutdown function")(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

struct PackedLlamaEngine {
    instance: Arc<PackInstance>,
}

#[async_trait::async_trait]
impl Engine for PackedLlamaEngine {
    fn planned_memory(&self, model_path: &Path) -> Result<MemoryReport, EngineError> {
        self.instance.call_path_report(
            model_path,
            self.instance
                .pack
                .api
                .planned_memory
                .expect("validated planned memory function"),
        )
    }

    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        let path = model_path.to_str().ok_or_else(|| {
            EngineError::invalid_input(format!("model path is not UTF-8: {}", model_path.display()))
        })?;
        let mut error = KapslOwnedBuffer::empty();
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
            if !error.ptr.is_null() {
                free_ffi_buffer(&self.instance.pack, error);
            }
            Ok(())
        } else {
            Err(status_engine_error(
                status,
                read_ffi_error(&self.instance.pack, status, error),
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
                    .expect("validated actual memory function"),
            )
            .unwrap_or_else(|error| {
                log::error!("llama.cpp pack actual-memory report failed: {error}");
                MemoryReport::default()
            })
    }

    fn planned_request_memory(&self, request: &InferenceRequest) -> MemoryReport {
        self.instance
            .planned_request_memory(request)
            .unwrap_or_else(|error| {
                log::error!("llama.cpp pack request-memory report failed: {error}");
                MemoryReport::default()
            })
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.instance.infer(request)
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let json = match serde_json::to_vec(request) {
            Ok(json) => json,
            Err(error) => {
                return Box::pin(futures::stream::once(async move {
                    Err(EngineError::invalid_input(format!(
                        "encode llama.cpp pack request: {error}"
                    )))
                }))
            }
        };
        let request_id = self.instance.next_request_id();
        let cancellation = CancellationContext::from_request(request);
        let instance = Arc::clone(&self.instance);
        let (sender, receiver) = mpsc::unbounded();
        let spawn = std::thread::Builder::new()
            .name(format!("kapsl-llama-stream-{request_id}"))
            .spawn(move || {
                let mut context = StreamContext {
                    sender,
                    cancellation,
                };
                let wire = instance.request_wire(request_id, &json, &mut context.cancellation);
                let mut error = KapslOwnedBuffer::empty();
                let status = unsafe {
                    instance
                        .pack
                        .api
                        .infer_stream
                        .expect("validated streaming function")(
                        instance.handle,
                        &wire,
                        (&mut context as *mut StreamContext).cast(),
                        Some(stream_chunk),
                        &mut error,
                    )
                };
                if status != KAPSL_STATUS_OK
                    && status != KAPSL_STATUS_CANCELLED
                    && !context.sender.is_closed()
                {
                    let message = read_ffi_error(&instance.pack, status, error);
                    let _ = context
                        .sender
                        .unbounded_send(Err(status_engine_error(status, message)));
                } else if !error.ptr.is_null() {
                    free_ffi_buffer(&instance.pack, error);
                }
            });
        if let Err(error) = spawn {
            return Box::pin(futures::stream::once(async move {
                Err(EngineError::backend(format!(
                    "start llama.cpp streaming bridge: {error}"
                )))
            }));
        }
        Box::pin(receiver)
    }

    fn unload(&mut self) {
        unsafe {
            self.instance
                .pack
                .api
                .unload
                .expect("validated unload function")(self.instance.handle);
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
                log::error!("llama.cpp pack metrics failed: {error}");
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
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe {
            self.instance
                .pack
                .api
                .health_check
                .expect("validated health function")(self.instance.handle, &mut error)
        };
        if status == KAPSL_STATUS_OK {
            if !error.ptr.is_null() {
                free_ffi_buffer(&self.instance.pack, error);
            }
            Ok(())
        } else {
            Err(status_engine_error(
                status,
                read_ffi_error(&self.instance.pack, status, error),
            ))
        }
    }

    fn max_batch(&self) -> usize {
        self.pack_batching_policy()
            .map(|policy| policy.max_batch.max(1))
            .unwrap_or(1)
    }

    fn self_batches(&self) -> bool {
        self.pack_batching_policy()
            .map(|policy| policy.self_batches)
            .unwrap_or(false)
    }
}

impl PackedLlamaEngine {
    fn pack_batching_policy(&self) -> Result<PackBatchingPolicy, EngineError> {
        self.instance.call_json_report(
            self.instance
                .pack
                .api
                .batching_policy
                .expect("validated batching policy function"),
        )
    }
}

#[derive(serde::Deserialize)]
struct PackBatchingPolicy {
    max_batch: usize,
    self_batches: bool,
}

struct CancellationContext {
    token: Option<kapsl_engine_api::CancellationToken>,
}

impl CancellationContext {
    fn from_request(request: &InferenceRequest) -> Self {
        Self {
            token: request.cancellation.clone(),
        }
    }

    fn is_cancelled(&self) -> bool {
        self.token
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
    }
}

struct StreamContext {
    sender: mpsc::UnboundedSender<Result<BinaryTensorPacket, EngineError>>,
    cancellation: CancellationContext,
}

unsafe extern "C" fn request_cancelled(user_data: *mut c_void, _request_id: u64) -> u32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null() {
            return 0;
        }
        // SAFETY: request wire calls retain this context until the pack unregisters
        // its cancellation probe and returns from the synchronous ABI call.
        let context = unsafe { &*(user_data as *const CancellationContext) };
        u32::from(context.is_cancelled())
    }))
    .unwrap_or(1)
}

unsafe extern "C" fn stream_chunk(
    user_data: *mut c_void,
    _request_id: u64,
    packet_json: KapslSlice,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if user_data.is_null() {
            return KAPSL_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the streaming bridge owns this context until infer_stream returns.
        let context = unsafe { &mut *(user_data as *mut StreamContext) };
        if context.cancellation.is_cancelled() || context.sender.is_closed() {
            return KAPSL_STATUS_CANCELLED;
        }
        let Some(bytes) = (unsafe { packet_json.as_bytes() }) else {
            let _ = context.sender.unbounded_send(Err(EngineError::backend(
                "llama.cpp pack returned a null stream packet",
            )));
            return KAPSL_STATUS_BACKEND_ERROR;
        };
        match serde_json::from_slice(bytes) {
            Ok(packet) => match context.sender.unbounded_send(Ok(packet)) {
                Ok(()) => KAPSL_STATUS_OK,
                Err(_) => KAPSL_STATUS_CANCELLED,
            },
            Err(error) => {
                let _ = context
                    .sender
                    .unbounded_send(Err(EngineError::backend(format!(
                        "decode llama.cpp stream packet: {error}"
                    ))));
                KAPSL_STATUS_BACKEND_ERROR
            }
        }
    }))
    .unwrap_or(KAPSL_STATUS_PANIC)
}

fn take_ffi_buffer(
    pack: &ActiveLlamaPack,
    buffer: KapslOwnedBuffer,
) -> Result<Vec<u8>, EngineError> {
    if buffer.len == 0 {
        if !buffer.ptr.is_null() {
            free_ffi_buffer(pack, buffer);
        }
        return Ok(Vec::new());
    }
    if buffer.ptr.is_null() || buffer.capacity < buffer.len {
        return Err(EngineError::backend(
            "llama.cpp pack returned an invalid owned buffer",
        ));
    }
    // SAFETY: validated ABI buffer remains live until free_buffer below.
    let bytes = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec();
    free_ffi_buffer(pack, buffer);
    Ok(bytes)
}

fn read_ffi_error(pack: &ActiveLlamaPack, status: i32, error: KapslOwnedBuffer) -> String {
    if error.ptr.is_null() || error.len == 0 || error.capacity < error.len {
        if !error.ptr.is_null() && error.capacity >= error.len {
            free_ffi_buffer(pack, error);
        }
        return format!("llama.cpp pack failed with status {status}");
    }
    // SAFETY: validated ABI buffer remains live until free_buffer below.
    let message = unsafe { std::slice::from_raw_parts(error.ptr, error.len) };
    let message = String::from_utf8_lossy(message).into_owned();
    free_ffi_buffer(pack, error);
    message
}

fn free_ffi_buffer(pack: &ActiveLlamaPack, buffer: KapslOwnedBuffer) {
    unsafe {
        pack.api
            .free_buffer
            .expect("validated free-buffer function")(buffer);
    }
}

fn status_engine_error(status: i32, message: String) -> EngineError {
    if status == KAPSL_STATUS_CANCELLED {
        EngineError::cancelled(message)
    } else {
        EngineError::backend(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(framework: &str) -> Manifest {
        Manifest {
            project_name: "test-model".to_string(),
            framework: framework.to_string(),
            version: "1.0.0".to_string(),
            created_at: String::new(),
            model_file: format!("model.{framework}"),
            format: Some(framework.to_string()),
            model_type: Some("causal-lm".to_string()),
            task: Some("generate".to_string()),
            metadata: None,
            hardware_requirements: kapsl_core::HardwareRequirements::default(),
            cron_jobs: Vec::new(),
        }
    }

    fn cpu_backend_pack() -> BackendPackManifest {
        BackendPackManifest {
            schema_version: 1,
            backend: "llama_cpp".to_string(),
            profile: crate::backend::LLAMA_CPP_CPU_PACK_PROFILE.to_string(),
            pack_version: "test".to_string(),
            runtime_abi: 1,
            adapter_abi: None,
            compatible_kapsl: "=0.2.3".to_string(),
            platform: crate::backend::current_platform(),
            architecture: std::env::consts::ARCH.to_string(),
            accelerator_profile: "cpu".to_string(),
            minimum_cuda: None,
            minimum_driver: None,
            execution_mode: crate::backend::BackendExecutionMode::Native,
            kv_mode: Some("native".to_string()),
            entrypoint: "libkapsl_backend_llama_cpp.so".to_string(),
            artifact: "https://downloads.kapsl.net/fixture.tar.gz".to_string(),
            download_bytes: 1,
            installed_bytes: 1,
            sha256: "0".repeat(64),
            signature: "fixture".to_string(),
            memory: crate::backend::BackendMemoryManifest::default(),
            installer: crate::backend::BackendInstaller::Extract,
            files: std::collections::BTreeMap::new(),
            licenses: Vec::new(),
            priority: 0,
        }
    }

    #[test]
    fn portable_build_selects_cpu_pack_for_gguf() {
        let manifest = manifest("gguf");
        assert_eq!(
            llama_cpp_pack_profile_for_target(&manifest, BackendAccelerator::Cpu),
            Some(LlamaCppBackendPackProfile::Cpu)
        );
        assert_eq!(
            llama_cpp_pack_profile_for_target(&manifest, BackendAccelerator::Cuda),
            Some(LlamaCppBackendPackProfile::Cuda12)
        );
    }

    #[test]
    fn non_gguf_models_do_not_select_llama_pack() {
        let manifest = manifest("onnx");
        assert_eq!(
            llama_cpp_pack_profile_for_target(&manifest, BackendAccelerator::Cpu),
            None
        );
    }

    #[test]
    fn preliminary_cpu_admission_treats_host_capacity_as_kib() {
        let model = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        std::fs::write(model.path(), vec![0_u8; 1024]).unwrap();
        let info = DeviceInfo {
            cpu_cores: 4,
            // DeviceInfo reports host capacity in KiB: this is one GiB.
            total_memory: 1024 * 1024,
            os_type: "linux".to_string(),
            os_release: "test".to_string(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: Vec::new(),
        };

        let admission = preliminary_llama_cpp_memory_admission(
            LlamaCppBackendPackProfile::Cpu,
            model.path(),
            &manifest("safetensors"),
            &info,
            &cpu_backend_pack(),
            None,
        )
        .unwrap();
        assert_ne!(admission.status, MemoryAdmissionStatus::Rejected);
        assert_eq!(admission.available_bytes, Some(1024 * 1024 * 1024 * 9 / 10));
    }

    #[test]
    fn api_validation_rejects_missing_shared_contract_functions() {
        let api = KapslLlamaCppApiV1 {
            magic: KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC,
            abi_version: KAPSL_LLAMA_CPP_ABI_VERSION,
            struct_size: std::mem::size_of::<KapslLlamaCppApiV1>() as u32,
            wire_format: KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1,
            capabilities: KAPSL_LLAMA_CAP_CPU,
            initialize: None,
            planned_memory: None,
            load_model: None,
            planned_request_memory: None,
            infer: None,
            infer_stream: None,
            cancel: None,
            actual_memory: None,
            metrics: None,
            model_info: None,
            kv_capabilities: None,
            kv_topology: None,
            batching_policy: None,
            health_check: None,
            unload: None,
            shutdown: None,
            free_buffer: None,
        };
        assert!(validate_api(LlamaCppBackendPackProfile::Cpu, &api)
            .unwrap_err()
            .contains("missing required capabilities"));
    }

    #[test]
    fn signed_kv_mode_requires_one_exact_pack_capability() {
        assert!(validate_kv_mode_capability(
            LlamaCppPackKvMode::SharedPool,
            KAPSL_LLAMA_CAP_SHARED_POOL
        )
        .is_ok());
        assert!(validate_kv_mode_capability(
            LlamaCppPackKvMode::Native,
            KAPSL_LLAMA_CAP_NATIVE_KV | KAPSL_LLAMA_CAP_SHARED_POOL
        )
        .unwrap_err()
        .contains("instead of exactly"));
        assert!(validate_kv_mode_capability(
            LlamaCppPackKvMode::SharedPool,
            KAPSL_LLAMA_CAP_NATIVE_KV
        )
        .is_err());
    }
}
