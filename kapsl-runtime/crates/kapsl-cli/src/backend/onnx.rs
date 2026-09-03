//! Signed ONNX backend-pack selection and bridge-release rollback routing.
//!
//! ONNX inference remains in-process. The pack manager resolves, verifies, and
//! installs one compatible native adapter, then carries that immutable signed
//! identity to replica construction. Embedded ORT is reachable only through
//! the explicit rollback switch; selection or activation errors fail closed.

use crate::backend::{
    activate_native_backend_pack, generic_native_backend_packs_enabled, guarded_host_memory_bytes,
    manifest_backend_pack_pin, BackendAccelerator, BackendExecutionMode, BackendManager,
    BackendPackCapabilities, BackendPackManifest, BackendPackRequirements, BackendTarget,
    NativeBackendPackIdentity, OnnxBackendPackProfile, GENERIC_NATIVE_PACKS_ENV,
    ONNX_CPU_PACK_PROFILE, ONNX_CUDA12_PACK_PROFILE, ONNX_TENSORRT10_PACK_PROFILE,
    STANDARD_NATIVE_ADAPTER_ABI,
};
use crate::runtime::{provider_policy, select_mesh_devices, MemoryDomain, MemorySnapshot};
use kapsl_core::{EngineKind, Manifest};
use kapsl_hal::device::DeviceInfo;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

const LAZY_BACKENDS_ENV: &str = "KAPSL_LAZY_BACKENDS";
const LAZY_ONNX_PACKS_ENV: &str = "KAPSL_LAZY_ONNX_PACKS";
const PROVIDER_PATH_ENV: &str = "KAPSL_PROVIDER_PATH";
const PRELIMINARY_MEMORY_GUARD_PERCENT: u64 = 10;
const MINIMUM_ONNX_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
const SCOPED_DEVICE_ALLOCATOR_V1: &str = "kapsl-scoped-device-allocator-v1";

static OFFLINE: AtomicBool = AtomicBool::new(false);

/// The route is resolved once per model load and carried to backend creation;
/// backend construction may not infer a different route from process state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OnnxBackendRoute {
    SignedPack {
        identity: NativeBackendPackIdentity,
        reason: String,
    },
    EmbeddedRollback {
        reason: String,
    },
}

/// Configure the process-wide lazy manager before any model lifecycle work.
/// Once an invocation is offline, later lifecycle operations cannot weaken it.
pub(crate) fn configure_onnx_backend_packs(offline: bool) -> Result<(), String> {
    if offline {
        OFFLINE.store(true, Ordering::Release);
    }
    if !generic_native_backend_packs_enabled()? {
        return Ok(());
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

pub(crate) fn onnx_backend_pack_requirements(
    manifest: &Manifest,
    profile: OnnxBackendPackProfile,
) -> Result<BackendPackRequirements, String> {
    let uses_accelerator = profile != OnnxBackendPackProfile::Cpu;
    let mut requirements = BackendPackRequirements::for_model(manifest);
    requirements.backend_pin = manifest_backend_pack_pin(manifest)?;
    requirements.execution_provider = Some(profile.provider().to_string());
    requirements.execution_mode = Some(BackendExecutionMode::Native);
    requirements.adapter_abi = Some(STANDARD_NATIVE_ADAPTER_ABI.to_string());
    requirements.capabilities = BackendPackCapabilities {
        batching: true,
        streaming: kapsl_core::engine_kind::effective_task(manifest) == "generate",
        cancellation: true,
        memory_reporting: true,
        governed_device_allocator: uses_accelerator,
        scoped_device_allocator: uses_accelerator,
        kv_participation: false,
        concurrent_inference: false,
    };
    requirements.allocation_scope =
        uses_accelerator.then(|| SCOPED_DEVICE_ALLOCATOR_V1.to_string());
    requirements.synchronize_before_free = uses_accelerator;
    Ok(requirements)
}

pub(crate) fn embedded_onnx_rollback_reason(manifest: &Manifest) -> Result<String, String> {
    if let Some(backend) = manifest_backend_pack_pin(manifest)? {
        return Err(format!(
            "model `{}` explicitly pins signed backend `{backend}`, which conflicts with the embedded ORT rollback selected by {GENERIC_NATIVE_PACKS_ENV}=0",
            manifest.project_name
        ));
    }
    Ok(format!(
        "explicit rollback selected by {GENERIC_NATIVE_PACKS_ENV}=0"
    ))
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
) -> Result<Option<OnnxBackendRoute>, String> {
    if !EngineKind::resolve(manifest).uses_onnx_session() {
        return Ok(None);
    }
    if !generic_native_backend_packs_enabled()? {
        let reason = embedded_onnx_rollback_reason(manifest)?;
        log::warn!(
            "Using embedded ORT rollback for model `{}`: {}",
            manifest.project_name,
            reason
        );
        return Ok(Some(OnnxBackendRoute::EmbeddedRollback { reason }));
    }
    if !lazy_onnx_packs_enabled() {
        return Err(format!(
            "model `{}` requires a signed native backend pack, but ONNX pack installation is disabled or unsupported on {}; set {GENERIC_NATIVE_PACKS_ENV}=0 only for an explicit embedded ORT rollback",
            manifest.project_name,
            crate::backend::current_platform()
        ));
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
    let requirements = onnx_backend_pack_requirements(manifest, profile)?;
    let pack_plan = manager
        .plan_compatible_backend(&requirements, &target)
        .map_err(|error| error.to_string())?;
    let standard_adapter = pack_plan.manifest.adapter_abi.as_deref()
        == Some(crate::backend::STANDARD_NATIVE_ADAPTER_ABI);
    if !standard_adapter {
        return Err(format!(
            "capability resolver selected {}/{}, but it does not implement {STANDARD_NATIVE_ADAPTER_ABI}",
            pack_plan.manifest.backend, pack_plan.manifest.profile
        ));
    }
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
    activate_native_backend_pack(&pack_plan.manifest, &installed)?;
    let identity = NativeBackendPackIdentity::from_manifest(&pack_plan.manifest);
    log::info!(
        "Selected signed native backend route {}/{} version {} for model `{}`: {}",
        identity.backend,
        identity.profile,
        identity.pack_version,
        manifest.project_name,
        pack_plan.selection_reason
    );
    Ok(Some(OnnxBackendRoute::SignedPack {
        identity,
        reason: pack_plan.selection_reason,
    }))
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
        OnnxBackendPackProfile::Cpu => {
            guarded_host_memory_bytes(device_info.total_memory, PRELIMINARY_MEMORY_GUARD_PERCENT)
        }
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

    fn backend_pack(profile: &str, accelerator_profile: &str) -> BackendPackManifest {
        BackendPackManifest {
            schema_version: 1,
            backend: "onnx".to_string(),
            profile: profile.to_string(),
            pack_version: "test".to_string(),
            runtime_abi: 1,
            adapter_abi: None,
            compatible_kapsl: "=0.2.3".to_string(),
            platform: crate::backend::current_platform(),
            architecture: std::env::consts::ARCH.to_string(),
            accelerator_profile: accelerator_profile.to_string(),
            accelerator_requirements: Default::default(),
            minimum_cuda: None,
            minimum_driver: None,
            execution_mode: crate::backend::BackendExecutionMode::Native,
            kv_mode: None,
            formats: Vec::new(),
            model_types: Vec::new(),
            tasks: Vec::new(),
            capabilities: Default::default(),
            memory_behavior: Default::default(),
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
    fn cpu_pack_requirements_preserve_model_axes_without_accelerator_scope() {
        let mut manifest = onnx_manifest(Some("cpu"), &[]);
        manifest.metadata = Some(
            serde_yaml::from_str("serving:\n  backend_pack: Acme.ORT\n")
                .expect("backend-pack pin metadata"),
        );

        let requirements = onnx_backend_pack_requirements(&manifest, OnnxBackendPackProfile::Cpu)
            .expect("CPU pack requirements");

        assert_eq!(requirements.backend_pin.as_deref(), Some("acme.ort"));
        assert_eq!(requirements.format.as_deref(), Some("onnx"));
        assert_eq!(requirements.model_type.as_deref(), Some("embedding"));
        assert_eq!(requirements.task.as_deref(), Some("embed"));
        assert_eq!(requirements.execution_provider.as_deref(), Some("cpu"));
        assert_eq!(
            requirements.execution_mode,
            Some(BackendExecutionMode::Native)
        );
        assert_eq!(
            requirements.adapter_abi.as_deref(),
            Some(STANDARD_NATIVE_ADAPTER_ABI)
        );
        assert!(requirements.capabilities.batching);
        assert!(requirements.capabilities.cancellation);
        assert!(requirements.capabilities.memory_reporting);
        assert!(!requirements.capabilities.streaming);
        assert!(!requirements.capabilities.governed_device_allocator);
        assert!(!requirements.capabilities.scoped_device_allocator);
        assert_eq!(requirements.allocation_scope, None);
        assert!(!requirements.synchronize_before_free);
    }

    #[test]
    fn accelerator_generation_requires_streaming_and_scoped_governed_allocation() {
        let mut manifest = onnx_manifest(Some("cuda"), &[]);
        manifest.model_type = Some("causal-lm".to_string());
        manifest.task = Some("generate".to_string());

        let requirements =
            onnx_backend_pack_requirements(&manifest, OnnxBackendPackProfile::Cuda12)
                .expect("CUDA generation pack requirements");

        assert!(requirements.capabilities.streaming);
        assert_eq!(requirements.execution_provider.as_deref(), Some("cuda"));
        assert!(requirements.capabilities.governed_device_allocator);
        assert!(requirements.capabilities.scoped_device_allocator);
        assert_eq!(
            requirements.allocation_scope.as_deref(),
            Some(SCOPED_DEVICE_ALLOCATOR_V1)
        );
        assert!(requirements.synchronize_before_free);
    }

    #[test]
    fn explicit_signed_backend_pin_cannot_cross_to_embedded_rollback() {
        let mut manifest = onnx_manifest(Some("cpu"), &[]);
        manifest.metadata = Some(
            serde_yaml::from_str("serving:\n  backend_pack: vendor-ort\n")
                .expect("backend-pack pin metadata"),
        );

        let error = embedded_onnx_rollback_reason(&manifest).unwrap_err();
        assert!(error.contains("explicitly pins signed backend `vendor-ort`"));
        assert!(error.contains("conflicts with the embedded ORT rollback"));
    }

    #[test]
    fn preliminary_cpu_admission_treats_host_capacity_as_kib() {
        let model = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(model.path(), vec![0_u8; 1024]).unwrap();
        let pack = backend_pack(crate::backend::ONNX_CPU_PACK_PROFILE, "cpu");
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

        preliminary_memory_admission(
            OnnxBackendPackProfile::Cpu,
            model.path(),
            &info,
            &pack,
            None,
            &HashSet::new(),
        )
        .unwrap();
    }

    #[test]
    fn preliminary_gpu_rejection_happens_before_pack_installation() {
        let model = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(model.path(), vec![0_u8; 1024]).unwrap();
        let pack = backend_pack(crate::backend::ONNX_CUDA12_PACK_PROFILE, "cuda");
        let info = DeviceInfo {
            cpu_cores: 4,
            total_memory: 8 * 1024 * 1024,
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
