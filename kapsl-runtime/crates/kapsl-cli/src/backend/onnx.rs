//! Lazy, signed ONNX Runtime accelerator-pack activation.
//!
//! ONNX inference remains in-process. The pack manager verifies and installs
//! the selected provider family, then this module opens its ORT sidecars by
//! pack-local absolute path. It deliberately never mutates `LD_LIBRARY_PATH`.

use crate::backend::{
    activate_native_backend_pack, generic_native_backend_packs_enabled, BackendAccelerator,
    BackendManager, BackendPackManifest, BackendTarget, OnnxBackendPackProfile,
    ONNX_CPU_PACK_PROFILE, ONNX_CUDA12_PACK_PROFILE, ONNX_TENSORRT10_PACK_PROFILE,
};
use crate::runtime::{provider_policy, select_mesh_devices, MemoryDomain, MemorySnapshot};
use kapsl_core::{EngineKind, Manifest};
use kapsl_hal::device::DeviceInfo;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

const LAZY_BACKENDS_ENV: &str = "KAPSL_LAZY_BACKENDS";
const LAZY_ONNX_PACKS_ENV: &str = "KAPSL_LAZY_ONNX_PACKS";
const PROVIDER_PATH_ENV: &str = "KAPSL_PROVIDER_PATH";
const PRELIMINARY_MEMORY_GUARD_PERCENT: u64 = 10;
const MINIMUM_ONNX_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
const ONNX_PACK_ENTRYPOINT_SYMBOL: &[u8] = b"kapsl_onnx_backend_pack_v1\0";
const ONNX_PACK_ENTRYPOINT_MAGIC: u32 = 0x4b4f_4e58;

static OFFLINE: AtomicBool = AtomicBool::new(false);

struct ActiveOnnxPack {
    identity: (String, String, String),
    #[allow(dead_code)]
    libraries: Vec<libloading::Library>,
}

#[repr(C)]
struct OnnxPackEntrypointV1 {
    magic: u32,
    struct_size: u32,
    runtime_abi: u32,
    profile: u32,
}

fn active_packs() -> &'static Mutex<Vec<ActiveOnnxPack>> {
    static ACTIVE: OnceLock<Mutex<Vec<ActiveOnnxPack>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Configure the process-wide lazy manager before any model lifecycle work.
/// Once an invocation is offline, later lifecycle operations cannot weaken it.
pub(crate) fn configure_onnx_backend_packs(offline: bool) -> Result<(), String> {
    if offline {
        OFFLINE.store(true, Ordering::Release);
    }
    if !lazy_onnx_packs_enabled() {
        return Ok(());
    }
    let cache_root = crate::backend::backend_cache_root()
        .ok_or_else(|| "resolve the Kapsl backend cache directory".to_string())?;
    let cache_root = if cache_root.is_absolute() {
        cache_root
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve the current directory: {error}"))?
            .join(cache_root)
    };
    let onnx_root = cache_root
        .join(crate::backend::runtime_release_version())
        .join("onnx");
    // Register all stable profile roots before model loading can become
    // parallel. The directories may not exist yet; the signed manager creates
    // them atomically before provider discovery is queried.
    for profile in [
        ONNX_CPU_PACK_PROFILE,
        ONNX_CUDA12_PACK_PROFILE,
        ONNX_TENSORRT10_PACK_PROFILE,
    ] {
        add_provider_search_root(&onnx_root.join(profile))?;
    }
    Ok(())
}

pub(crate) fn backend_packs_are_offline() -> bool {
    OFFLINE.load(Ordering::Acquire)
}

pub(crate) fn onnx_pack_profile_for_manifest(
    manifest: &Manifest,
    device_info: &DeviceInfo,
) -> Result<Option<OnnxBackendPackProfile>, String> {
    if !EngineKind::resolve(manifest).uses_onnx_session() {
        return Ok(None);
    }
    let selection = select_mesh_devices(&manifest.hardware_requirements, device_info)?;
    profile_for_selected_provider(&selection.logical_provider, manifest).map(Some)
}

/// Resolve a cross-target bundle without pretending the preparation host owns
/// the target accelerator. TensorRT is considered only when the package names
/// it explicitly as a preferred or fallback provider.
pub(crate) fn onnx_pack_profile_for_target(
    manifest: &Manifest,
    target: BackendAccelerator,
) -> Result<Option<OnnxBackendPackProfile>, String> {
    if !EngineKind::resolve(manifest).uses_onnx_session() {
        return Ok(None);
    }

    let requirements = &manifest.hardware_requirements;
    let mut providers = Vec::new();
    if let Some(preferred) = requirements.preferred_provider.as_deref() {
        push_unique_provider(&mut providers, preferred);
    }
    for fallback in &requirements.fallback_providers {
        push_unique_provider(&mut providers, fallback);
    }
    if providers.is_empty() && provider_policy() != "manifest" {
        match target {
            BackendAccelerator::Cpu => providers.push("cpu".to_string()),
            BackendAccelerator::Cuda | BackendAccelerator::TensorRt => {
                providers.push("cuda".to_string());
                providers.push("cpu".to_string());
            }
        }
    }

    for provider in providers {
        let available = match provider.as_str() {
            "cpu" => true,
            "cuda" => target != BackendAccelerator::Cpu,
            // TensorRT is a policy choice on CUDA-capable hardware. The
            // TensorRT pack supplies its user-space runtime, so both `cuda`
            // and `tensorrt` bundle targets can satisfy an explicit request.
            "tensorrt" => target != BackendAccelerator::Cpu,
            _ => false,
        };
        if available {
            return profile_for_selected_provider(&provider, manifest).map(Some);
        }
    }
    Err(format!(
        "ONNX package `{}` declares no provider compatible with the {} target",
        manifest.project_name,
        target.as_str()
    ))
}

fn push_unique_provider(providers: &mut Vec<String>, provider: &str) {
    let provider = provider.trim().to_ascii_lowercase();
    if !provider.is_empty() && !providers.iter().any(|item| item == &provider) {
        providers.push(provider);
    }
}

fn profile_for_selected_provider(
    provider: &str,
    manifest: &Manifest,
) -> Result<OnnxBackendPackProfile, String> {
    match provider.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(OnnxBackendPackProfile::Cpu),
        "cuda" => Ok(OnnxBackendPackProfile::Cuda12),
        "tensorrt" => {
            let explicitly_declared = manifest
                .hardware_requirements
                .preferred_provider
                .iter()
                .chain(manifest.hardware_requirements.fallback_providers.iter())
                .any(|candidate| candidate.trim().eq_ignore_ascii_case("tensorrt"));
            if !explicitly_declared {
                return Err(
                    "TensorRT may only be selected when the .aimod explicitly declares it as a preferred or fallback provider"
                        .to_string(),
                );
            }
            Ok(OnnxBackendPackProfile::TensorRt10)
        }
        other => Err(format!(
            "ONNX provider `{other}` is not supplied by the CPU/CUDA 12/TensorRT 10 lazy pack family"
        )),
    }
}

fn target_for_profile(profile: OnnxBackendPackProfile, device_info: &DeviceInfo) -> BackendTarget {
    let mut target = BackendTarget::current(device_info);
    target.accelerator = profile.accelerator();
    if profile == OnnxBackendPackProfile::Cpu {
        target.cuda_version = None;
        target.driver_version = None;
    }
    target
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

pub(crate) fn onnx_lazy_packs_supported_for_platform(platform: &str) -> bool {
    platform == "linux-x86_64"
}

pub(crate) fn lazy_onnx_packs_enabled() -> bool {
    if !env_switch(LAZY_BACKENDS_ENV, true) {
        return false;
    }
    env_switch(
        LAZY_ONNX_PACKS_ENV,
        onnx_lazy_packs_supported_for_platform(&crate::backend::current_platform()),
    )
}

/// Ensure and activate the exact ONNX pack selected for a model. This is safe
/// to call for every lifecycle load; cache locks and the active-pack registry
/// make the operation idempotent.
pub(crate) fn ensure_onnx_backend_pack(
    manifest: &Manifest,
    model_path: &Path,
    device_info: &DeviceInfo,
    memory_snapshot: Option<&MemorySnapshot>,
) -> Result<Option<PathBuf>, String> {
    if !EngineKind::resolve(manifest).uses_onnx_session() {
        return Ok(None);
    }
    if !lazy_onnx_packs_enabled() {
        log::debug!("Lazy ONNX packs are disabled; using the runtime's eager provider deployment");
        return Ok(None);
    }
    let Some(profile) = onnx_pack_profile_for_manifest(manifest, device_info)? else {
        return Ok(None);
    };
    let selected_cuda_devices = select_mesh_devices(&manifest.hardware_requirements, device_info)?
        .devices
        .into_iter()
        .filter(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
        .map(|device| device.id)
        .collect::<HashSet<_>>();
    let offline = OFFLINE.load(Ordering::Acquire);
    let manager = BackendManager::from_env(offline).map_err(|error| error.to_string())?;
    let target = target_for_profile(profile, device_info);
    let pack_plan = manager
        .plan_onnx(profile, &target)
        .map_err(|error| error.to_string())?;
    preliminary_memory_admission(
        profile,
        model_path,
        device_info,
        &pack_plan.manifest,
        memory_snapshot,
        &selected_cuda_devices,
    )?;
    let installed = manager
        .ensure_pack(&pack_plan.manifest)
        .map_err(|error| error.to_string())?;
    if generic_native_backend_packs_enabled()? {
        activate_native_backend_pack(&pack_plan.manifest, &installed)?;
    } else {
        activate_onnx_pack(&pack_plan.manifest, &installed)?;
    }
    Ok(Some(installed))
}

fn preliminary_memory_admission(
    profile: OnnxBackendPackProfile,
    model_path: &Path,
    device_info: &DeviceInfo,
    pack: &BackendPackManifest,
    memory_snapshot: Option<&MemorySnapshot>,
    selected_cuda_devices: &HashSet<usize>,
) -> Result<(), String> {
    let model_bytes = std::fs::metadata(model_path)
        .map_err(|error| format!("stat ONNX model {}: {error}", model_path.display()))?
        .len();
    let decoded_weights = model_bytes.saturating_mul(5).saturating_div(4);
    let estimated_workspace = (model_bytes / 4)
        .max(MINIMUM_ONNX_WORKSPACE_BYTES)
        .max(pack.memory.minimum_workspace_bytes)
        .saturating_add(
            model_bytes
                .saturating_mul(pack.memory.workspace_weight_ppm)
                .saturating_div(1_000_000),
        );
    let fixed = match profile {
        OnnxBackendPackProfile::Cpu => pack.memory.host_bytes,
        OnnxBackendPackProfile::Cuda12 | OnnxBackendPackProfile::TensorRt10 => {
            pack.memory.accelerator_bytes
        }
    };
    let required = decoded_weights
        .saturating_add(estimated_workspace)
        .saturating_add(fixed);
    let physical_available = match profile {
        OnnxBackendPackProfile::Cpu => device_info
            .total_memory
            .saturating_mul(100 - PRELIMINARY_MEMORY_GUARD_PERCENT)
            .saturating_div(100),
        OnnxBackendPackProfile::Cuda12 | OnnxBackendPackProfile::TensorRt10 => device_info
            .devices
            .iter()
            .filter(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
            .filter(|device| {
                selected_cuda_devices.is_empty() || selected_cuda_devices.contains(&device.id)
            })
            .map(|device| {
                device
                    .memory_mb
                    .saturating_mul(1024 * 1024)
                    .saturating_mul(100 - PRELIMINARY_MEMORY_GUARD_PERCENT)
                    .saturating_div(100)
            })
            .min()
            .unwrap_or(0),
    };
    let governed_available = memory_snapshot.and_then(|snapshot| match profile {
        OnnxBackendPackProfile::Cpu => snapshot
            .domain(&MemoryDomain::Host)
            .map(|domain| domain.available_bytes as u64),
        OnnxBackendPackProfile::Cuda12 | OnnxBackendPackProfile::TensorRt10 => snapshot
            .domains
            .iter()
            .filter(|domain| match domain.domain {
                MemoryDomain::Cuda { device_id } => {
                    selected_cuda_devices.is_empty() || selected_cuda_devices.contains(&device_id)
                }
                _ => false,
            })
            .map(|domain| domain.available_bytes as u64)
            .min(),
    });
    let available = governed_available
        .map(|governed| governed.min(physical_available))
        .unwrap_or(physical_available);
    if required > available {
        return Err(format!(
            "preliminary memory admission rejected ONNX {}/{} before backend download: required={} available={} bytes",
            profile.provider(),
            profile.profile(),
            required,
            available
        ));
    }
    Ok(())
}

fn activate_onnx_pack(pack: &BackendPackManifest, root: &Path) -> Result<(), String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("resolve ONNX pack {}: {error}", root.display()))?;
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

    let library_paths = onnx_library_load_order(pack, &root)?;
    let entrypoint_path = root
        .join(&pack.entrypoint)
        .canonicalize()
        .map_err(|error| {
            format!(
                "resolve ONNX pack entrypoint {}: {error}",
                root.join(&pack.entrypoint).display()
            )
        })?;
    library_paths
        .iter()
        .any(|path| path == &entrypoint_path)
        .then_some(())
        .ok_or_else(|| {
            format!(
                "ONNX pack entrypoint {} was not loaded",
                root.join(&pack.entrypoint).display()
            )
        })?;
    // Validate the pack ABI using its dependency-free descriptor before any
    // ORT or accelerator provider object can run a load-time initializer.
    let entrypoint_library = load_pack_library(&entrypoint_path)?;
    validate_pack_entrypoint(pack, &entrypoint_library)?;
    let mut libraries = Vec::with_capacity(library_paths.len());
    libraries.push(entrypoint_library);
    for path in &library_paths {
        if path != &entrypoint_path {
            libraries.push(load_pack_library(path)?);
        }
    }
    validate_ort_provider_available(pack)?;
    log::info!(
        "Activated signed ONNX backend pack {}/{} from {} ({} pack-local libraries)",
        pack.backend,
        pack.profile,
        root.display(),
        libraries.len()
    );
    active.push(ActiveOnnxPack {
        identity,
        libraries,
    });
    Ok(())
}

fn validate_ort_provider_available(pack: &BackendPackManifest) -> Result<(), String> {
    use ort::execution_providers::ExecutionProvider as _;
    let available = match pack.profile.as_str() {
        crate::backend::ONNX_CPU_PACK_PROFILE => true,
        crate::backend::ONNX_CUDA12_PACK_PROFILE => {
            ort::execution_providers::CUDAExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
        }
        crate::backend::ONNX_TENSORRT10_PACK_PROFILE => {
            ort::execution_providers::CUDAExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
                && ort::execution_providers::TensorRTExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
        }
        other => return Err(format!("unknown ONNX pack profile `{other}`")),
    };
    if !available {
        return Err(format!(
            "signed ONNX pack {}/{} loaded, but ONNX Runtime did not expose its requested execution provider; refusing CPU fallback",
            pack.backend, pack.profile
        ));
    }
    Ok(())
}

fn validate_pack_entrypoint(
    pack: &BackendPackManifest,
    library: &libloading::Library,
) -> Result<(), String> {
    type Entrypoint = unsafe extern "C" fn() -> *const OnnxPackEntrypointV1;
    let function =
        unsafe { library.get::<Entrypoint>(ONNX_PACK_ENTRYPOINT_SYMBOL) }.map_err(|error| {
            format!(
                "ONNX pack {}/{} entrypoint does not export kapsl_onnx_backend_pack_v1: {error}",
                pack.backend, pack.profile
            )
        })?;
    let descriptor = unsafe { function() };
    let descriptor = unsafe { descriptor.as_ref() }
        .ok_or_else(|| "ONNX pack entrypoint returned a null descriptor".to_string())?;
    let expected_profile = match pack.profile.as_str() {
        crate::backend::ONNX_CPU_PACK_PROFILE => 1,
        crate::backend::ONNX_CUDA12_PACK_PROFILE => 2,
        crate::backend::ONNX_TENSORRT10_PACK_PROFILE => 3,
        other => return Err(format!("unknown ONNX pack profile `{other}`")),
    };
    if descriptor.magic != ONNX_PACK_ENTRYPOINT_MAGIC
        || descriptor.struct_size < std::mem::size_of::<OnnxPackEntrypointV1>() as u32
        || descriptor.runtime_abi != crate::backend::BACKEND_RUNTIME_ABI
        || descriptor.profile != expected_profile
    {
        return Err(format!(
            "ONNX pack native descriptor is incompatible: magic={:#x}, size={}, ABI={}, profile={} (expected ABI={}, profile={})",
            descriptor.magic,
            descriptor.struct_size,
            descriptor.runtime_abi,
            descriptor.profile,
            crate::backend::BACKEND_RUNTIME_ABI,
            expected_profile
        ));
    }
    Ok(())
}

fn onnx_library_load_order(
    pack: &BackendPackManifest,
    root: &Path,
) -> Result<Vec<PathBuf>, String> {
    let mut candidates = pack
        .files
        .keys()
        .filter(|relative| is_onnx_provider_library(relative))
        .cloned()
        .collect::<Vec<_>>();
    if !candidates.iter().any(|path| path == &pack.entrypoint) {
        candidates.push(pack.entrypoint.clone());
    }
    candidates.sort_by_key(|path| {
        let lower = path.to_ascii_lowercase();
        if lower.contains("providers_shared") {
            (0, lower)
        } else if lower.contains("providers_cuda") {
            (1, lower)
        } else if lower.contains("providers_tensorrt") {
            (2, lower)
        } else {
            (3, lower)
        }
    });
    candidates.dedup();
    if candidates.is_empty() {
        return Err(format!(
            "ONNX pack {}/{} declares no loadable native entrypoint",
            pack.backend, pack.profile
        ));
    }

    candidates
        .into_iter()
        .map(|relative| {
            let path = root.join(&relative);
            let canonical = path.canonicalize().map_err(|error| {
                format!("resolve ONNX pack library {}: {error}", path.display())
            })?;
            if !canonical.starts_with(root) || !canonical.is_file() {
                return Err(format!(
                    "ONNX pack library escapes its verified root: {}",
                    canonical.display()
                ));
            }
            Ok(canonical)
        })
        .collect()
}

fn is_onnx_provider_library(path: &str) -> bool {
    let name = Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    name.contains("onnxruntime_providers_")
        && (name.ends_with(".dll")
            || name.contains(".so")
            || name.ends_with(".dylib")
            || name.contains(".dylib."))
}

fn load_pack_library(path: &Path) -> Result<libloading::Library, String> {
    if !path.is_absolute() {
        return Err(format!(
            "refusing to load a non-absolute ONNX pack library path: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        // RTLD_GLOBAL allows ORT's later provider registration lookup to reuse
        // the exact already-verified object without a bare-name search.
        let library = unsafe {
            libloading::os::unix::Library::open(Some(path), libc::RTLD_NOW | libc::RTLD_GLOBAL)
        }
        .map_err(|error| format!("load ONNX pack library {}: {error}", path.display()))?;
        Ok(library.into())
    }
    #[cfg(windows)]
    {
        // The top-level DLL is always named by an absolute path. Packager
        // dependency closure keeps its provider DLL dependencies beside it.
        unsafe { libloading::Library::new(path) }
            .map_err(|error| format!("load ONNX pack library {}: {error}", path.display()))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Err("native ONNX pack loading is unsupported on this platform".to_string())
    }
}

fn add_provider_search_root(root: &Path) -> Result<(), String> {
    let mut roots = std::env::var_os(PROVIDER_PATH_ENV)
        .map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut seen = roots.iter().cloned().collect::<HashSet<_>>();
    if !seen.insert(root.to_path_buf()) {
        return Ok(());
    }
    roots.push(root.to_path_buf());
    let joined = std::env::join_paths(roots)
        .map_err(|error| format!("build {PROVIDER_PATH_ENV}: {error}"))?;
    // SAFETY: this function is called once on the startup path before model
    // lifecycle work begins. No loader search environment such as
    // LD_LIBRARY_PATH is modified.
    unsafe { std::env::set_var(PROVIDER_PATH_ENV, joined) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_core::HardwareRequirements;
    use kapsl_hal::device::{Device, DeviceBackend};

    fn onnx_manifest(provider: Option<&str>, fallbacks: &[&str]) -> Manifest {
        Manifest {
            project_name: "onnx-pack-test".to_string(),
            framework: "onnx".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2026-08-25T00:00:00Z".to_string(),
            format: Some("onnx".to_string()),
            model_type: Some("embedding".to_string()),
            task: Some("embed".to_string()),
            model_file: "model.onnx".to_string(),
            metadata: None,
            hardware_requirements: HardwareRequirements {
                preferred_provider: provider.map(str::to_string),
                fallback_providers: fallbacks.iter().map(|item| item.to_string()).collect(),
                ..HardwareRequirements::default()
            },
            cron_jobs: Vec::new(),
        }
    }

    #[test]
    fn target_resolution_never_auto_selects_tensorrt() {
        let automatic = onnx_manifest(None, &[]);
        assert_eq!(
            onnx_pack_profile_for_target(&automatic, BackendAccelerator::TensorRt).unwrap(),
            Some(OnnxBackendPackProfile::Cuda12)
        );

        let explicit = onnx_manifest(Some("tensorrt"), &["cuda"]);
        assert_eq!(
            onnx_pack_profile_for_target(&explicit, BackendAccelerator::Cuda).unwrap(),
            Some(OnnxBackendPackProfile::TensorRt10)
        );
        assert_eq!(
            onnx_pack_profile_for_target(&explicit, BackendAccelerator::TensorRt).unwrap(),
            Some(OnnxBackendPackProfile::TensorRt10)
        );
    }

    #[test]
    fn declared_fallback_controls_cross_target_pack() {
        let manifest = onnx_manifest(Some("cuda"), &["cpu"]);
        assert_eq!(
            onnx_pack_profile_for_target(&manifest, BackendAccelerator::Cpu).unwrap(),
            Some(OnnxBackendPackProfile::Cpu)
        );
        assert_eq!(
            onnx_pack_profile_for_target(&manifest, BackendAccelerator::Cuda).unwrap(),
            Some(OnnxBackendPackProfile::Cuda12)
        );

        let gpu_only = onnx_manifest(Some("cuda"), &[]);
        assert!(
            onnx_pack_profile_for_target(&gpu_only, BackendAccelerator::Cpu)
                .unwrap_err()
                .contains("no provider compatible")
        );
    }

    #[test]
    fn provider_library_filter_excludes_cuda_runtime_dependencies() {
        assert!(is_onnx_provider_library("libonnxruntime_providers_cuda.so"));
        assert!(is_onnx_provider_library(
            "libonnxruntime_providers_shared.so"
        ));
        assert!(!is_onnx_provider_library("libcudnn.so.9"));
        assert!(!is_onnx_provider_library("licenses/LICENSE"));
    }

    #[test]
    fn preliminary_gpu_rejection_happens_before_pack_installation() {
        let model = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(model.path(), vec![0_u8; 1024]).unwrap();
        let pack = BackendPackManifest {
            schema_version: 1,
            backend: "onnx".to_string(),
            profile: crate::backend::ONNX_CUDA12_PACK_PROFILE.to_string(),
            pack_version: "test".to_string(),
            runtime_abi: 1,
            compatible_kapsl: "=0.2.3".to_string(),
            platform: crate::backend::current_platform(),
            architecture: std::env::consts::ARCH.to_string(),
            accelerator_profile: "cuda".to_string(),
            minimum_cuda: None,
            minimum_driver: None,
            execution_mode: crate::backend::BackendExecutionMode::Native,
            kv_mode: None,
            entrypoint: "libkapsl_backend_onnx.so".to_string(),
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
        };
        let info = DeviceInfo {
            cpu_cores: 4,
            total_memory: 8 * 1024 * 1024 * 1024,
            os_type: "linux".to_string(),
            os_release: "test".to_string(),
            has_cuda: true,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![Device {
                id: 0,
                name: "tiny GPU".to_string(),
                backend: DeviceBackend::Cuda,
                memory_mb: 64,
                compute_units: 1,
                pci_bus_id: None,
                partition_id: None,
                driver_version: Some("580.1".to_string()),
                cuda_version: Some("13.0".to_string()),
                compute_capability: Some("8.0".to_string()),
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
            }],
        };

        let error = preliminary_memory_admission(
            OnnxBackendPackProfile::Cuda12,
            model.path(),
            &info,
            &pack,
            None,
            &HashSet::from([0]),
        )
        .unwrap_err();
        assert!(error.contains("before backend download"));
    }
}
