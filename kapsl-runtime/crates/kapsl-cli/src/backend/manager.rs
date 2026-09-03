//! Signed backend-pack resolution and installation.
//!
//! A model package may select a backend policy, but it never supplies an
//! executable or URL.  Those deployment details come only from the signed
//! runtime-specific backend index handled here.

use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64_URL_SAFE};
use base64::Engine as _;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use flate2::read::GzDecoder;
use fs2::FileExt;
use kapsl_hal::device::{DeviceBackend, DeviceInfo};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{SystemTime, UNIX_EPOCH};
use tar::Archive;

pub(crate) const BACKEND_INDEX_SCHEMA_VERSION: u32 = 1;
pub(crate) const BACKEND_PACK_SCHEMA_VERSION: u32 = 1;
pub(crate) const BACKEND_RUNTIME_ABI: u32 = 1;
pub(crate) const MANAGED_VLLM_PACK_PROFILE: &str = "cu130-flash-attn";
pub(crate) const ONNX_CPU_PACK_PROFILE: &str = "cpu";
pub(crate) const ONNX_CUDA12_PACK_PROFILE: &str = "cuda12";
pub(crate) const ONNX_TENSORRT10_PACK_PROFILE: &str = "tensorrt10";
pub(crate) const STANDARD_NATIVE_ADAPTER_ABI: &str = "kapsl-backend-v1";
pub(crate) const LLAMA_CPP_CPU_PACK_PROFILE: &str = "cpu";
pub(crate) const LLAMA_CPP_CUDA12_PACK_PROFILE: &str = "cuda12";

const BACKEND_CACHE_ENV: &str = "KAPSL_BACKEND_CACHE_DIR";
const BACKEND_INDEX_URL_ENV: &str = "KAPSL_BACKEND_INDEX_URL";
const BACKEND_INDEX_PATH_ENV: &str = "KAPSL_BACKEND_INDEX_PATH";
const BACKEND_PUBLIC_KEYS_ENV: &str = "KAPSL_BACKEND_PUBLIC_KEYS";
const OFFLINE_ENV: &str = "KAPSL_OFFLINE";
const DEFAULT_DOWNLOAD_BASE_URL: &str = "https://downloads.kapsl.net";
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const INSTALL_SPACE_OVERHEAD_BYTES: u64 = 256 * 1024 * 1024;
const INSTALLED_SIZE_TOLERANCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 250_000;
const INSTALL_RECORD_NAME: &str = ".kapsl-backend.json";
const PAYLOAD_MANIFEST_NAME: &str = "backend-pack.json";

#[derive(Debug)]
pub(crate) struct BackendManagerError(String);

impl BackendManagerError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BackendManagerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BackendManagerError {}

impl From<std::io::Error> for BackendManagerError {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

pub(crate) type ManagerResult<T> = Result<T, BackendManagerError>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BackendExecutionMode {
    Native,
    External,
}

impl BackendExecutionMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::External => "external",
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BackendInstaller {
    /// The archive payload is already the installed pack.
    #[default]
    Extract,
    /// Run a trusted pack-local bootstrap with PAYLOAD_ROOT and INSTALL_ROOT.
    Bootstrap { path: String },
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BackendMemoryManifest {
    /// Fixed host bytes needed before model-specific allocations.
    #[serde(default)]
    pub(crate) host_bytes: u64,
    /// Fixed accelerator bytes needed before model-specific allocations.
    #[serde(default)]
    pub(crate) accelerator_bytes: u64,
    /// Workspace bytes added per byte of model weights, in millionths.
    #[serde(default)]
    pub(crate) workspace_weight_ppm: u64,
    /// Minimum workspace even for small models.
    #[serde(default)]
    pub(crate) minimum_workspace_bytes: u64,
}

/// Capabilities covered by the signed pack index.
///
/// The adapter's runtime function table must make the same claims before the
/// host will initialize it. Keeping these values in the signed index lets the
/// resolver reject an incompatible multi-gigabyte pack before downloading it.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BackendPackCapabilities {
    #[serde(default)]
    pub(crate) batching: bool,
    #[serde(default)]
    pub(crate) streaming: bool,
    #[serde(default)]
    pub(crate) cancellation: bool,
    #[serde(default)]
    pub(crate) memory_reporting: bool,
    #[serde(default)]
    pub(crate) governed_device_allocator: bool,
    #[serde(default)]
    pub(crate) scoped_device_allocator: bool,
    #[serde(default)]
    pub(crate) kv_participation: bool,
    #[serde(default)]
    pub(crate) concurrent_inference: bool,
}

impl BackendPackCapabilities {
    fn missing_required(&self, required: &Self) -> Vec<&'static str> {
        [
            (required.batching, self.batching, "batching"),
            (required.streaming, self.streaming, "streaming"),
            (required.cancellation, self.cancellation, "cancellation"),
            (
                required.memory_reporting,
                self.memory_reporting,
                "memory_reporting",
            ),
            (
                required.governed_device_allocator,
                self.governed_device_allocator,
                "governed_device_allocator",
            ),
            (
                required.scoped_device_allocator,
                self.scoped_device_allocator,
                "scoped_device_allocator",
            ),
            (
                required.kv_participation,
                self.kv_participation,
                "kv_participation",
            ),
            (
                required.concurrent_inference,
                self.concurrent_inference,
                "concurrent_inference",
            ),
        ]
        .into_iter()
        .filter_map(|(needed, available, label)| (needed && !available).then_some(label))
        .collect()
    }
}

/// Accelerator policy asserted by the pack producer.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BackendAcceleratorRequirements {
    #[serde(default)]
    pub(crate) kind: Option<String>,
    #[serde(default)]
    pub(crate) execution_providers: Vec<String>,
    /// `None` means an older pack made no signed claim and is therefore not
    /// eligible for the standard native adapter route.
    #[serde(default)]
    pub(crate) implicit_cpu_fallback: Option<bool>,
}

/// Memory semantics covered by the signed pack index.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BackendMemoryBehavior {
    #[serde(default)]
    pub(crate) allocation_scope: Option<String>,
    #[serde(default)]
    pub(crate) device_allocation: Option<String>,
    #[serde(default)]
    pub(crate) planned_reporting: bool,
    #[serde(default)]
    pub(crate) live_reporting: bool,
    #[serde(default)]
    pub(crate) request_reporting: bool,
    #[serde(default)]
    pub(crate) synchronize_before_free: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BackendLicenseNotice {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) path: Option<String>,
    #[serde(default)]
    pub(crate) url: Option<String>,
}

/// One signed entry in `backend-index.json`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BackendPackManifest {
    pub(crate) schema_version: u32,
    pub(crate) backend: String,
    pub(crate) profile: String,
    pub(crate) pack_version: String,
    pub(crate) runtime_abi: u32,
    /// Standard in-process adapter contract. Legacy provider-only packs omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) adapter_abi: Option<String>,
    /// A semver requirement evaluated against the Kapsl runtime version.
    pub(crate) compatible_kapsl: String,
    /// Canonical OS/architecture pair, for example `linux-x86_64`.
    pub(crate) platform: String,
    pub(crate) architecture: String,
    /// `cpu`, `cuda`, or `tensorrt`.
    pub(crate) accelerator_profile: String,
    #[serde(default)]
    pub(crate) accelerator_requirements: BackendAcceleratorRequirements,
    #[serde(default)]
    pub(crate) minimum_cuda: Option<String>,
    #[serde(default)]
    pub(crate) minimum_driver: Option<String>,
    pub(crate) execution_mode: BackendExecutionMode,
    /// KV allocation authority for llama.cpp packs. Omitted by non-llama
    /// backends and legacy native-KV records.
    #[serde(default)]
    pub(crate) kv_mode: Option<String>,
    /// Model contracts this signed pack can execute. Empty lists are accepted
    /// only for legacy packs that do not use the standard native adapter ABI.
    #[serde(default)]
    pub(crate) formats: Vec<String>,
    #[serde(default)]
    pub(crate) model_types: Vec<String>,
    #[serde(default)]
    pub(crate) tasks: Vec<String>,
    #[serde(default)]
    pub(crate) capabilities: BackendPackCapabilities,
    #[serde(default)]
    pub(crate) memory_behavior: BackendMemoryBehavior,
    /// Pack-local native library or external executable.
    pub(crate) entrypoint: String,
    pub(crate) artifact: String,
    pub(crate) download_bytes: u64,
    pub(crate) installed_bytes: u64,
    pub(crate) sha256: String,
    /// Ed25519 signature over the domain-separated artifact digest.
    pub(crate) signature: String,
    #[serde(default)]
    pub(crate) memory: BackendMemoryManifest,
    #[serde(default)]
    pub(crate) installer: BackendInstaller,
    /// Optional hashes for security-sensitive installed files.
    #[serde(default)]
    pub(crate) files: BTreeMap<String, String>,
    #[serde(default)]
    pub(crate) licenses: Vec<BackendLicenseNotice>,
    #[serde(default)]
    pub(crate) priority: i32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BackendIndex {
    pub(crate) schema_version: u32,
    pub(crate) runtime_version: String,
    pub(crate) generated_at: String,
    pub(crate) packs: Vec<BackendPackManifest>,
}

/// Model and runtime requirements used to query a signed backend index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackendPackRequirements {
    pub(crate) backend_pin: Option<String>,
    pub(crate) preferred_profile: Option<String>,
    pub(crate) format: Option<String>,
    pub(crate) model_type: Option<String>,
    pub(crate) task: Option<String>,
    pub(crate) execution_provider: Option<String>,
    pub(crate) execution_mode: Option<BackendExecutionMode>,
    pub(crate) adapter_abi: Option<String>,
    pub(crate) capabilities: BackendPackCapabilities,
    pub(crate) allocation_scope: Option<String>,
    pub(crate) synchronize_before_free: bool,
}

impl BackendPackRequirements {
    pub(crate) fn for_model(manifest: &kapsl_core::Manifest) -> Self {
        Self {
            format: Some(kapsl_core::engine_kind::effective_format(manifest)),
            model_type: Some(kapsl_core::engine_kind::effective_model_type(manifest)),
            task: Some(kapsl_core::engine_kind::effective_task(manifest)),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BackendPackSelection {
    pub(crate) manifest: BackendPackManifest,
    pub(crate) reason: String,
}

/// Small manifest physically carried by a pack. Artifact hashes/signatures are
/// deliberately absent because embedding an archive's own digest is circular.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct BackendPayloadManifest {
    schema_version: u32,
    backend: String,
    profile: String,
    pack_version: String,
    runtime_abi: u32,
    #[serde(default)]
    adapter_abi: Option<String>,
    platform: String,
    execution_mode: BackendExecutionMode,
    #[serde(default)]
    kv_mode: Option<String>,
    entrypoint: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct InstalledPackRecord {
    manifest: BackendPackManifest,
    installed_at_unix_seconds: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendAccelerator {
    Cpu,
    Cuda,
    TensorRt,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum OnnxBackendPackProfile {
    Cpu,
    Cuda12,
    TensorRt10,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum LlamaCppBackendPackProfile {
    Cpu,
    Cuda12,
}

impl LlamaCppBackendPackProfile {
    pub(crate) fn profile(self) -> &'static str {
        match self {
            Self::Cpu => LLAMA_CPP_CPU_PACK_PROFILE,
            Self::Cuda12 => LLAMA_CPP_CUDA12_PACK_PROFILE,
        }
    }

    pub(crate) fn accelerator(self) -> BackendAccelerator {
        match self {
            Self::Cpu => BackendAccelerator::Cpu,
            Self::Cuda12 => BackendAccelerator::Cuda,
        }
    }
}

impl OnnxBackendPackProfile {
    pub(crate) fn profile(self) -> &'static str {
        match self {
            Self::Cpu => ONNX_CPU_PACK_PROFILE,
            Self::Cuda12 => ONNX_CUDA12_PACK_PROFILE,
            Self::TensorRt10 => ONNX_TENSORRT10_PACK_PROFILE,
        }
    }

    pub(crate) fn provider(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda12 => "cuda",
            Self::TensorRt10 => "tensorrt",
        }
    }

    pub(crate) fn accelerator(self) -> BackendAccelerator {
        match self {
            Self::Cpu => BackendAccelerator::Cpu,
            Self::Cuda12 => BackendAccelerator::Cuda,
            Self::TensorRt10 => BackendAccelerator::TensorRt,
        }
    }
}

impl BackendAccelerator {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BackendTarget {
    pub(crate) platform: String,
    pub(crate) architecture: String,
    pub(crate) accelerator: BackendAccelerator,
    pub(crate) cuda_version: Option<String>,
    pub(crate) driver_version: Option<String>,
}

impl BackendTarget {
    pub(crate) fn current(device_info: &DeviceInfo) -> Self {
        let cuda = device_info
            .devices
            .iter()
            .find(|device| device.backend == DeviceBackend::Cuda);
        Self {
            platform: current_platform(),
            architecture: std::env::consts::ARCH.to_string(),
            accelerator: if device_info.has_cuda {
                BackendAccelerator::Cuda
            } else {
                BackendAccelerator::Cpu
            },
            cuda_version: cuda.and_then(|device| device.cuda_version.clone()),
            driver_version: cuda.and_then(|device| device.driver_version.clone()),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BackendPackPlan {
    pub(crate) selected_backend: String,
    pub(crate) profile: String,
    pub(crate) selection_reason: String,
    pub(crate) installed: bool,
    pub(crate) download_required: bool,
    pub(crate) download_bytes: u64,
    pub(crate) execution_mode: String,
    #[serde(skip_serializing)]
    pub(crate) manifest: BackendPackManifest,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BackendCacheEntry {
    pub(crate) runtime_version: String,
    pub(crate) backend: String,
    pub(crate) profile: String,
    pub(crate) pack_version: String,
    pub(crate) execution_mode: String,
    pub(crate) installed_bytes: u64,
    pub(crate) path: PathBuf,
    pub(crate) valid: bool,
}

#[derive(Clone)]
struct BackendManagerConfig {
    cache_root: PathBuf,
    runtime_version: String,
    index_url: String,
    index_path: Option<PathBuf>,
    offline: bool,
    trusted_keys: Vec<VerifyingKey>,
    allow_file_artifacts: bool,
}

#[derive(Clone)]
pub(crate) struct BackendManager {
    config: BackendManagerConfig,
}

struct MaintenanceReadGuard {
    _process: RwLockReadGuard<'static, ()>,
    _file: File,
}

struct MaintenanceWriteGuard {
    _process: RwLockWriteGuard<'static, ()>,
    _file: File,
}

impl BackendManager {
    pub(crate) fn from_env(force_offline: bool) -> ManagerResult<Self> {
        let runtime_version = runtime_release_version();
        let cache_root = backend_cache_root().ok_or_else(|| {
            BackendManagerError::new(
                "could not determine the local data directory for the backend cache; set KAPSL_BACKEND_CACHE_DIR",
            )
        })?;
        let index_url = std::env::var(BACKEND_INDEX_URL_ENV)
            .unwrap_or_else(|_| default_backend_index_url(&runtime_version));
        let index_path = std::env::var_os(BACKEND_INDEX_PATH_ENV).map(PathBuf::from);
        let offline = force_offline || env_flag(OFFLINE_ENV);
        let trusted_keys = trusted_public_keys()?;
        Ok(Self {
            config: BackendManagerConfig {
                cache_root,
                runtime_version,
                index_url,
                index_path,
                offline,
                trusted_keys,
                allow_file_artifacts: false,
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        cache_root: PathBuf,
        runtime_version: &str,
        index_path: PathBuf,
        trusted_key: VerifyingKey,
        offline: bool,
    ) -> Self {
        Self {
            config: BackendManagerConfig {
                cache_root,
                runtime_version: runtime_version.to_string(),
                index_url: String::new(),
                index_path: Some(index_path),
                offline,
                trusted_keys: vec![trusted_key],
                allow_file_artifacts: true,
            },
        }
    }

    pub(crate) fn runtime_version(&self) -> &str {
        &self.config.runtime_version
    }

    pub(crate) fn installed_path(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        if pack.pack_version.trim().is_empty()
            || pack.platform.trim().is_empty()
            || pack.architecture.trim().is_empty()
        {
            return Err(BackendManagerError::new(
                "backend pack version, platform, and architecture must be non-empty",
            ));
        }
        Ok(self
            .runtime_cache_root()
            .join(&pack.backend)
            .join(&pack.profile))
    }

    /// Resolve one model contract against the verified signed index without
    /// downloading or loading adapter code.
    pub(crate) fn plan_compatible_backend(
        &self,
        requirements: &BackendPackRequirements,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackPlan> {
        let index = self.load_index()?;
        let selection = self.select_compatible_pack(&index, requirements, target)?;
        let pack = selection.manifest;
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: selection.reason,
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn plan_vllm(&self, target: &BackendTarget) -> ManagerResult<BackendPackPlan> {
        let index = self.load_index()?;
        let pack = self.select_pack(&index, "vllm", Some(MANAGED_VLLM_PACK_PROFILE), target)?;
        if pack.execution_mode != BackendExecutionMode::External {
            return Err(BackendManagerError::new(
                "the managed vLLM backend pack must use external execution mode",
            ));
        }
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: format!(
                "explicit vLLM profile `{}` matched the signed backend index",
                pack.profile
            ),
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn plan_onnx(
        &self,
        profile: OnnxBackendPackProfile,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackPlan> {
        if target.accelerator != profile.accelerator() {
            return Err(BackendManagerError::new(format!(
                "ONNX profile `{}` requires a {} target, not {}",
                profile.profile(),
                profile.accelerator().as_str(),
                target.accelerator.as_str()
            )));
        }
        let index = self.load_index()?;
        let pack = self.select_pack(&index, "onnx", Some(profile.profile()), target)?;
        if pack.execution_mode != BackendExecutionMode::Native {
            return Err(BackendManagerError::new(format!(
                "the ONNX backend pack {}/{} must use native execution mode",
                pack.backend, pack.profile
            )));
        }
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: format!(
                "explicit ONNX profile `{}` matched the signed backend index",
                pack.profile
            ),
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn plan_llama_cpp(
        &self,
        profile: LlamaCppBackendPackProfile,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackPlan> {
        if target.accelerator != profile.accelerator() {
            return Err(BackendManagerError::new(format!(
                "llama.cpp profile `{}` requires a {} target, not {}",
                profile.profile(),
                profile.accelerator().as_str(),
                target.accelerator.as_str()
            )));
        }
        let index = self.load_index()?;
        let pack = self.select_pack(&index, "llama-cpp", Some(profile.profile()), target)?;
        if pack.execution_mode != BackendExecutionMode::Native {
            return Err(BackendManagerError::new(format!(
                "the llama.cpp backend pack {}/{} must use native execution mode",
                pack.backend, pack.profile
            )));
        }
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: format!(
                "explicit llama.cpp profile `{}` matched the signed backend index",
                pack.profile
            ),
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn ensure_vllm(&self, target: &BackendTarget) -> ManagerResult<PathBuf> {
        let plan = self.plan_vllm(target)?;
        self.ensure_pack(&plan.manifest)
    }

    pub(crate) fn ensure_pack(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        let _maintenance = self.acquire_maintenance_read()?;
        self.validate_pack_manifest(pack, None)?;
        let final_path = self.installed_path(pack)?;
        let lock_path = self.install_lock_path(pack)?;
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!(
                "lock backend installation {}: {error}",
                lock_path.display()
            ))
        })?;

        self.cleanup_stale_install_stages(pack)?;
        if self.installed_pack_is_valid(pack)? {
            let _ = fs2::FileExt::unlock(&lock);
            return Ok(final_path);
        }
        if self.config.offline {
            let _ = fs2::FileExt::unlock(&lock);
            return Err(BackendManagerError::new(format!(
                "backend `{}/{}` is not installed and Kapsl is offline; prepare a .kapsl-bundle on a connected machine or run `kapsl backend ensure MODEL` while online",
                pack.backend, pack.profile
            )));
        }

        let staging_parent = self.pack_staging_root(pack)?;
        fs::create_dir_all(&staging_parent)?;
        ensure_available_space(
            &staging_parent,
            pack.download_bytes
                .saturating_mul(2)
                .saturating_add(pack.installed_bytes)
                .saturating_add(INSTALL_SPACE_OVERHEAD_BYTES),
            "download and install backend pack",
        )?;
        let stage = tempfile::Builder::new()
            .prefix("install-")
            .tempdir_in(&staging_parent)
            .map_err(|error| {
                BackendManagerError::new(format!(
                    "create backend staging directory under {}: {error}",
                    staging_parent.display()
                ))
            })?;
        let archive_path = stage.path().join("backend-pack.tar.gz");
        eprintln!(
            "Downloading Kapsl backend {}/{} ({} bytes)...",
            pack.backend, pack.profile, pack.download_bytes
        );
        self.download_artifact(pack, &archive_path)?;
        self.install_verified_archive_locked(pack, &archive_path, &final_path, stage.path())?;
        let _ = fs2::FileExt::unlock(&lock);
        Ok(final_path)
    }

    /// Install an archive that was carried by a verified offline bundle. The
    /// caller supplies the signed index entry; this method re-verifies the
    /// artifact before it can enter the shared cache.
    pub(crate) fn install_pack_from_archive(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
    ) -> ManagerResult<PathBuf> {
        let _maintenance = self.acquire_maintenance_read()?;
        self.validate_pack_manifest(pack, None)?;
        self.verify_artifact_file(pack, archive_path)?;
        let final_path = self.installed_path(pack)?;
        let lock_path = self.install_lock_path(pack)?;
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!(
                "lock backend installation {}: {error}",
                lock_path.display()
            ))
        })?;
        self.cleanup_stale_install_stages(pack)?;
        if !self.installed_pack_is_valid(pack)? {
            let staging_parent = self.pack_staging_root(pack)?;
            fs::create_dir_all(&staging_parent)?;
            let stage = tempfile::Builder::new()
                .prefix("install-")
                .tempdir_in(&staging_parent)
                .map_err(|error| BackendManagerError::new(error.to_string()))?;
            self.install_verified_archive_locked(pack, archive_path, &final_path, stage.path())?;
        }
        let _ = fs2::FileExt::unlock(&lock);
        Ok(final_path)
    }

    pub(crate) fn fetch_pack_archive(
        &self,
        pack: &BackendPackManifest,
        destination: &Path,
    ) -> ManagerResult<()> {
        if self.config.offline {
            return Err(BackendManagerError::new(format!(
                "cannot download backend `{}/{}` while offline",
                pack.backend, pack.profile
            )));
        }
        self.download_artifact(pack, destination)
    }

    /// Verify a caller-supplied archive against its signed index entry without
    /// installing it. Offline bundle creation uses this for release artifacts
    /// already present on the preparation host.
    pub(crate) fn verify_pack_archive(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
    ) -> ManagerResult<()> {
        self.validate_pack_manifest(pack, None)?;
        self.verify_artifact_file(pack, archive_path)
    }

    pub(crate) fn load_index(&self) -> ManagerResult<BackendIndex> {
        let index_dir = self.runtime_cache_root().join(".index");
        fs::create_dir_all(&index_dir)?;
        let lock_path = index_dir.join("index.lock");
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!("lock backend index cache: {error}"))
        })?;

        let cached_index = index_dir.join("backend-index.json");
        let cached_signature = index_dir.join("backend-index.json.sig");
        if cached_index.is_file() && cached_signature.is_file() {
            match self.read_and_verify_index(&cached_index, &cached_signature) {
                Ok(index) => {
                    let _ = fs2::FileExt::unlock(&lock);
                    return Ok(index);
                }
                Err(error) if self.config.offline => {
                    let _ = fs2::FileExt::unlock(&lock);
                    return Err(BackendManagerError::new(format!(
                        "cached backend index is invalid while offline: {error}"
                    )));
                }
                Err(error) => {
                    eprintln!("Ignoring invalid cached backend index: {error}");
                    quarantine_file(&cached_index)?;
                    quarantine_file(&cached_signature)?;
                }
            }
        }

        if self.config.offline {
            let _ = fs2::FileExt::unlock(&lock);
            return Err(BackendManagerError::new(
                "no verified backend index is cached and Kapsl is offline",
            ));
        }

        let (index_bytes, signature_bytes) = self.read_index_source()?;
        let signature_text = std::str::from_utf8(&signature_bytes)
            .map_err(|error| {
                BackendManagerError::new(format!(
                    "backend index signature is not UTF-8 text: {error}"
                ))
            })?
            .trim();
        self.verify_index_signature(&index_bytes, signature_text)?;
        let index = self.decode_and_validate_index(&index_bytes)?;
        atomic_write(&cached_index, &index_bytes)?;
        atomic_write(&cached_signature, signature_text.as_bytes())?;
        let _ = fs2::FileExt::unlock(&lock);
        Ok(index)
    }

    pub(crate) fn verify_index_bytes(
        &self,
        index_bytes: &[u8],
        signature: &str,
    ) -> ManagerResult<BackendIndex> {
        self.verify_index_signature(index_bytes, signature.trim())?;
        self.decode_and_validate_index(index_bytes)
    }

    /// Anchor an index carried by a verified offline bundle into the ordinary
    /// runtime cache. Subsequent lazy-pack validation therefore uses the same
    /// signed trust path without requiring network access. A different valid
    /// index for the same immutable runtime release is rejected.
    pub(crate) fn cache_verified_index_bytes(
        &self,
        index_bytes: &[u8],
        signature: &str,
    ) -> ManagerResult<BackendIndex> {
        let index = self.verify_index_bytes(index_bytes, signature)?;
        let index_dir = self.runtime_cache_root().join(".index");
        fs::create_dir_all(&index_dir)?;
        let lock_path = index_dir.join("index.lock");
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!("lock backend index cache: {error}"))
        })?;

        let cached_index = index_dir.join("backend-index.json");
        let cached_signature = index_dir.join("backend-index.json.sig");
        if cached_index.is_file() || cached_signature.is_file() {
            if !cached_index.is_file() || !cached_signature.is_file() {
                let _ = fs2::FileExt::unlock(&lock);
                return Err(BackendManagerError::new(
                    "backend index cache is incomplete; refusing to replace it from an offline bundle",
                ));
            }
            let existing_bytes = read_file_bounded(&cached_index, MAX_INDEX_BYTES)?;
            let existing_signature = read_file_bounded(&cached_signature, 64 * 1024)?;
            let existing_signature = std::str::from_utf8(&existing_signature)
                .map_err(|error| BackendManagerError::new(error.to_string()))?
                .trim();
            self.verify_index_bytes(&existing_bytes, existing_signature)?;
            if existing_bytes != index_bytes || existing_signature != signature.trim() {
                let _ = fs2::FileExt::unlock(&lock);
                return Err(BackendManagerError::new(
                    "offline bundle carries a different signed backend index for this immutable runtime release",
                ));
            }
            let _ = fs2::FileExt::unlock(&lock);
            return Ok(index);
        }

        atomic_write(&cached_index, index_bytes)?;
        atomic_write(&cached_signature, signature.trim().as_bytes())?;
        let _ = fs2::FileExt::unlock(&lock);
        Ok(index)
    }

    pub(crate) fn verified_index_material(&self) -> ManagerResult<(BackendIndex, Vec<u8>, String)> {
        let _ = self.load_index()?;
        let index_path = self.runtime_cache_root().join(".index/backend-index.json");
        let signature_path = self
            .runtime_cache_root()
            .join(".index/backend-index.json.sig");
        let bytes = read_file_bounded(&index_path, MAX_INDEX_BYTES)?;
        let signature_bytes = read_file_bounded(&signature_path, 64 * 1024)?;
        let signature = std::str::from_utf8(&signature_bytes)
            .map_err(|error| BackendManagerError::new(error.to_string()))?
            .trim()
            .to_string();
        let index = self.verify_index_bytes(&bytes, &signature)?;
        Ok((index, bytes, signature))
    }

    pub(crate) fn select_compatible_pack(
        &self,
        index: &BackendIndex,
        requirements: &BackendPackRequirements,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackSelection> {
        let backend_pin = requirements
            .backend_pin
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let preferred_profile = requirements
            .preferred_profile
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let mut rejected = Vec::new();
        let mut candidates = Vec::new();

        for pack in &index.packs {
            if backend_pin.is_some_and(|backend| !pack.backend.eq_ignore_ascii_case(backend)) {
                continue;
            }
            if preferred_profile.is_some_and(|profile| pack.profile != profile) {
                continue;
            }

            let identity = format!("{}/{}", pack.backend, pack.profile);
            if let Err(error) = self.validate_pack_manifest(pack, Some(target)) {
                rejected.push(format!("{identity}: {error}"));
                continue;
            }
            if requirements
                .execution_mode
                .is_some_and(|mode| pack.execution_mode != mode)
            {
                rejected.push(format!(
                    "{identity}: execution mode {} does not match required {}",
                    pack.execution_mode.as_str(),
                    requirements
                        .execution_mode
                        .expect("checked execution mode")
                        .as_str()
                ));
                continue;
            }
            if requirements.adapter_abi.is_some()
                && pack.adapter_abi.as_deref() != requirements.adapter_abi.as_deref()
            {
                rejected.push(format!(
                    "{identity}: adapter ABI {:?} does not match required {:?}",
                    pack.adapter_abi, requirements.adapter_abi
                ));
                continue;
            }
            if let Some(format) = requirements.format.as_deref() {
                if !signed_contract_contains(&pack.formats, format) {
                    rejected.push(format!(
                        "{identity}: format `{format}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(model_type) = requirements.model_type.as_deref() {
                if !pack.model_types.is_empty()
                    && !signed_contract_contains(&pack.model_types, model_type)
                {
                    rejected.push(format!(
                        "{identity}: model type `{model_type}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(task) = requirements.task.as_deref() {
                if !signed_contract_contains(&pack.tasks, task) {
                    rejected.push(format!(
                        "{identity}: task `{task}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(provider) = requirements.execution_provider.as_deref() {
                if !signed_contract_contains(
                    &pack.accelerator_requirements.execution_providers,
                    provider,
                ) {
                    rejected.push(format!(
                        "{identity}: execution provider `{provider}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            let missing = pack
                .capabilities
                .missing_required(&requirements.capabilities);
            if !missing.is_empty() {
                rejected.push(format!(
                    "{identity}: missing required capabilities {}",
                    missing.join(", ")
                ));
                continue;
            }
            if let Some(scope) = requirements.allocation_scope.as_deref() {
                if pack.memory_behavior.allocation_scope.as_deref() != Some(scope) {
                    rejected.push(format!(
                        "{identity}: allocation scope {:?} does not match required `{scope}`",
                        pack.memory_behavior.allocation_scope
                    ));
                    continue;
                }
            }
            if requirements.synchronize_before_free && !pack.memory_behavior.synchronize_before_free
            {
                rejected.push(format!(
                    "{identity}: signed memory behavior does not synchronize the device before freeing governed allocations"
                ));
                continue;
            }
            candidates.push(pack.clone());
        }

        candidates.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| compare_pack_versions(&right.pack_version, &left.pack_version))
                .then_with(|| left.backend.cmp(&right.backend))
                .then_with(|| left.profile.cmp(&right.profile))
        });

        let Some(selected) = candidates.first() else {
            let selection = backend_pin
                .map(|backend| format!("explicit backend `{backend}`"))
                .unwrap_or_else(|| "automatic backend selection".to_string());
            return Err(BackendManagerError::new(format!(
                "{selection} found no compatible signed pack for format `{}`, model type `{}`, task `{}`, provider `{}` on {} ({}){}",
                requirements.format.as_deref().unwrap_or("any"),
                requirements.model_type.as_deref().unwrap_or("any"),
                requirements.task.as_deref().unwrap_or("any"),
                requirements.execution_provider.as_deref().unwrap_or("any"),
                target.platform,
                target.accelerator.as_str(),
                if rejected.is_empty() {
                    String::new()
                } else {
                    format!("; rejected candidates: {}", rejected.join("; "))
                }
            )));
        };

        let tied = candidates
            .iter()
            .take_while(|candidate| {
                candidate.priority == selected.priority
                    && compare_pack_versions(&candidate.pack_version, &selected.pack_version)
                        == std::cmp::Ordering::Equal
            })
            .map(|candidate| format!("{}/{}", candidate.backend, candidate.profile))
            .collect::<Vec<_>>();
        if tied.len() > 1 {
            return Err(BackendManagerError::new(format!(
                "backend selection is ambiguous between equally ranked signed packs: {}; use an explicit backend pin or assign distinct signed priorities",
                tied.join(", ")
            )));
        }

        let selection_kind = backend_pin
            .map(|backend| format!("explicit backend pin `{backend}`"))
            .unwrap_or_else(|| "capability-based automatic selection".to_string());
        Ok(BackendPackSelection {
            manifest: selected.clone(),
            reason: format!(
                "{selection_kind} matched signed {}/{} version {} for format `{}`, model type `{}`, task `{}`, provider `{}`, and {} execution",
                selected.backend,
                selected.profile,
                selected.pack_version,
                requirements.format.as_deref().unwrap_or("any"),
                requirements.model_type.as_deref().unwrap_or("any"),
                requirements.task.as_deref().unwrap_or("any"),
                requirements.execution_provider.as_deref().unwrap_or("any"),
                target.accelerator.as_str()
            ),
        })
    }

    pub(crate) fn validate_pack_for_target(
        &self,
        pack: &BackendPackManifest,
        target: &BackendTarget,
    ) -> ManagerResult<()> {
        self.validate_pack_manifest(pack, Some(target))
    }

    pub(crate) fn select_pack(
        &self,
        index: &BackendIndex,
        backend: &str,
        preferred_profile: Option<&str>,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackManifest> {
        let mut rejected = Vec::new();
        let mut candidates = Vec::new();
        for pack in index.packs.iter().filter(|pack| pack.backend == backend) {
            if preferred_profile.is_some_and(|profile| pack.profile != profile) {
                continue;
            }
            match self.validate_pack_manifest(pack, Some(target)) {
                Ok(()) => candidates.push(pack.clone()),
                Err(error) => rejected.push(format!("{}: {error}", pack.profile)),
            }
        }
        candidates.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| compare_pack_versions(&right.pack_version, &left.pack_version))
        });
        candidates.into_iter().next().ok_or_else(|| {
            BackendManagerError::new(format!(
                "no compatible signed backend pack for `{backend}` profile `{}` on {} ({}, CUDA {:?}, driver {:?}){}",
                preferred_profile.unwrap_or("auto"),
                target.platform,
                target.accelerator.as_str(),
                target.cuda_version,
                target.driver_version,
                if rejected.is_empty() {
                    String::new()
                } else {
                    format!("; rejected candidates: {}", rejected.join("; "))
                }
            ))
        })
    }

    pub(crate) fn list(&self) -> ManagerResult<Vec<BackendCacheEntry>> {
        let mut result = Vec::new();
        if !self.config.cache_root.is_dir() {
            return Ok(result);
        }
        // A local install record is not a trust root. Only label a current
        // entry valid when it exactly matches a pack from the cached,
        // signature-verified runtime index.
        let signed_current_packs = self
            .load_index()
            .map(|index| index.packs)
            .unwrap_or_default();
        for runtime_entry in read_dirs(&self.config.cache_root)? {
            let runtime_path = runtime_entry.path();
            if !runtime_path.is_dir() || is_internal_name(&runtime_entry.file_name()) {
                continue;
            }
            let runtime_version = runtime_entry.file_name().to_string_lossy().into_owned();
            for backend_entry in read_dirs(&runtime_path)? {
                let backend_path = backend_entry.path();
                if !backend_path.is_dir() || is_internal_name(&backend_entry.file_name()) {
                    continue;
                }
                for profile_entry in read_dirs(&backend_path)? {
                    let profile_path = profile_entry.path();
                    if !profile_path.is_dir() || is_internal_name(&profile_entry.file_name()) {
                        continue;
                    }
                    let record = read_json_bounded::<InstalledPackRecord>(
                        &profile_path.join(INSTALL_RECORD_NAME),
                        MAX_MANIFEST_BYTES,
                    );
                    let (backend, profile, pack_version, execution_mode, valid) = match record {
                        Ok(record) => {
                            let valid = runtime_version == self.config.runtime_version
                                && signed_current_packs
                                    .iter()
                                    .any(|pack| pack == &record.manifest)
                                && self
                                    .installed_pack_is_valid(&record.manifest)
                                    .unwrap_or(false);
                            (
                                record.manifest.backend,
                                record.manifest.profile,
                                record.manifest.pack_version,
                                record.manifest.execution_mode.as_str().to_string(),
                                valid,
                            )
                        }
                        Err(_) => (
                            backend_entry.file_name().to_string_lossy().into_owned(),
                            profile_entry.file_name().to_string_lossy().into_owned(),
                            "unknown".to_string(),
                            "unknown".to_string(),
                            false,
                        ),
                    };
                    result.push(BackendCacheEntry {
                        runtime_version: runtime_version.clone(),
                        backend,
                        profile,
                        pack_version,
                        execution_mode,
                        installed_bytes: directory_size(&profile_path)?,
                        path: profile_path,
                        valid,
                    });
                }
            }
        }
        result.sort_by(|left, right| {
            left.runtime_version
                .cmp(&right.runtime_version)
                .then_with(|| left.backend.cmp(&right.backend))
                .then_with(|| left.profile.cmp(&right.profile))
        });
        Ok(result)
    }

    /// Remove abandoned staging/quarantine data. With `old_versions`, also
    /// remove cache trees belonging to other runtime versions.
    pub(crate) fn prune(&self, old_versions: bool) -> ManagerResult<(u64, usize)> {
        let _maintenance = self.acquire_maintenance_write()?;
        let mut bytes = 0_u64;
        let mut entries = 0_usize;
        if !self.config.cache_root.is_dir() {
            return Ok((0, 0));
        }
        for runtime_entry in read_dirs(&self.config.cache_root)? {
            let runtime_path = runtime_entry.path();
            if !runtime_path.is_dir() {
                continue;
            }
            let runtime_name = runtime_entry.file_name().to_string_lossy().into_owned();
            if old_versions
                && !is_internal_name(&runtime_entry.file_name())
                && runtime_name != self.config.runtime_version
            {
                bytes = bytes.saturating_add(directory_size(&runtime_path)?);
                fs::remove_dir_all(&runtime_path).map_err(|error| {
                    BackendManagerError::new(format!(
                        "remove old backend cache {}: {error}",
                        runtime_path.display()
                    ))
                })?;
                entries += 1;
                continue;
            }
            for name in [".staging", ".quarantine"] {
                let path = runtime_path.join(name);
                if path.is_dir() {
                    bytes = bytes.saturating_add(directory_size(&path)?);
                    fs::remove_dir_all(&path).map_err(|error| {
                        BackendManagerError::new(format!(
                            "remove stale backend data {}: {error}",
                            path.display()
                        ))
                    })?;
                    entries += 1;
                }
            }
        }
        Ok((bytes, entries))
    }

    fn runtime_cache_root(&self) -> PathBuf {
        self.config.cache_root.join(&self.config.runtime_version)
    }

    fn acquire_maintenance_read(&self) -> ManagerResult<MaintenanceReadGuard> {
        fs::create_dir_all(&self.config.cache_root)?;
        let process = backend_maintenance_lock()
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let file = lock_file(&self.config.cache_root.join(".maintenance.lock"))?;
        fs2::FileExt::lock_shared(&file).map_err(|error| {
            BackendManagerError::new(format!("lock backend cache for installation: {error}"))
        })?;
        Ok(MaintenanceReadGuard {
            _process: process,
            _file: file,
        })
    }

    fn acquire_maintenance_write(&self) -> ManagerResult<MaintenanceWriteGuard> {
        fs::create_dir_all(&self.config.cache_root)?;
        let process = backend_maintenance_lock()
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let file = lock_file(&self.config.cache_root.join(".maintenance.lock"))?;
        file.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!("lock backend cache for pruning: {error}"))
        })?;
        Ok(MaintenanceWriteGuard {
            _process: process,
            _file: file,
        })
    }

    fn install_lock_path(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        Ok(self
            .runtime_cache_root()
            .join(".locks")
            .join(&pack.backend)
            .join(format!("{}.lock", pack.profile)))
    }

    fn pack_staging_root(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        Ok(self
            .runtime_cache_root()
            .join(".staging")
            .join(&pack.backend)
            .join(&pack.profile))
    }

    fn read_index_source(&self) -> ManagerResult<(Vec<u8>, Vec<u8>)> {
        if let Some(index_path) = self.config.index_path.as_ref() {
            let bytes = read_file_bounded(index_path, MAX_INDEX_BYTES)?;
            let signature_path = PathBuf::from(format!("{}.sig", index_path.display()));
            let signature = read_file_bounded(&signature_path, 64 * 1024)?;
            return Ok((bytes, signature));
        }
        if !self.config.index_url.starts_with("https://") {
            return Err(BackendManagerError::new(format!(
                "backend index URL must use HTTPS: {}",
                self.config.index_url
            )));
        }
        let bytes = download_bytes(&self.config.index_url, MAX_INDEX_BYTES)?;
        let signature = download_bytes(&format!("{}.sig", self.config.index_url), 64 * 1024)?;
        Ok((bytes, signature))
    }

    fn read_and_verify_index(
        &self,
        index_path: &Path,
        signature_path: &Path,
    ) -> ManagerResult<BackendIndex> {
        let bytes = read_file_bounded(index_path, MAX_INDEX_BYTES)?;
        let signature_bytes = read_file_bounded(signature_path, 64 * 1024)?;
        let signature = std::str::from_utf8(&signature_bytes)
            .map_err(|error| BackendManagerError::new(error.to_string()))?
            .trim();
        self.verify_index_bytes(&bytes, signature)
    }

    fn decode_and_validate_index(&self, bytes: &[u8]) -> ManagerResult<BackendIndex> {
        let index: BackendIndex = serde_json::from_slice(bytes).map_err(|error| {
            BackendManagerError::new(format!("invalid backend-index.json: {error}"))
        })?;
        if index.schema_version != BACKEND_INDEX_SCHEMA_VERSION {
            return Err(BackendManagerError::new(format!(
                "backend index schema {} is unsupported; expected {}",
                index.schema_version, BACKEND_INDEX_SCHEMA_VERSION
            )));
        }
        if index.runtime_version != self.config.runtime_version {
            return Err(BackendManagerError::new(format!(
                "backend index targets Kapsl {}, but this runtime is {}",
                index.runtime_version, self.config.runtime_version
            )));
        }
        if index.packs.is_empty() {
            return Err(BackendManagerError::new("backend index contains no packs"));
        }
        let mut identities = HashSet::new();
        for pack in &index.packs {
            self.validate_pack_manifest(pack, None)?;
            if !identities.insert((
                pack.backend.clone(),
                pack.profile.clone(),
                pack.pack_version.clone(),
                pack.platform.clone(),
                normalize_architecture(&pack.architecture).to_string(),
                pack.accelerator_profile.clone(),
            )) {
                return Err(BackendManagerError::new(format!(
                    "backend index contains duplicate pack identity {}/{} version {} for {}-{} ({})",
                    pack.backend,
                    pack.profile,
                    pack.pack_version,
                    pack.platform,
                    pack.architecture,
                    pack.accelerator_profile
                )));
            }
        }
        Ok(index)
    }

    fn verify_index_signature(&self, bytes: &[u8], encoded: &str) -> ManagerResult<()> {
        let mut message = b"kapsl-backend-index-v1\0".to_vec();
        message.extend_from_slice(bytes);
        self.verify_signature(&message, encoded, "backend index")
    }

    fn verify_artifact_signature(&self, digest: &str, encoded: &str) -> ManagerResult<()> {
        let message = format!("kapsl-backend-artifact-v1\0sha256:{digest}");
        self.verify_signature(message.as_bytes(), encoded, "backend artifact")
    }

    fn verify_signature(&self, message: &[u8], encoded: &str, label: &str) -> ManagerResult<()> {
        if self.config.trusted_keys.is_empty() {
            return Err(BackendManagerError::new(format!(
                "cannot verify {label}: this Kapsl build has no trusted backend signing key; official builds must embed KAPSL_BACKEND_PUBLIC_KEYS"
            )));
        }
        let signature = decode_signature(encoded)?;
        if self
            .config
            .trusted_keys
            .iter()
            .any(|key| key.verify(message, &signature).is_ok())
        {
            return Ok(());
        }
        Err(BackendManagerError::new(format!(
            "{label} signature verification failed"
        )))
    }

    fn validate_pack_manifest(
        &self,
        pack: &BackendPackManifest,
        target: Option<&BackendTarget>,
    ) -> ManagerResult<()> {
        if pack.schema_version != BACKEND_PACK_SCHEMA_VERSION {
            return Err(BackendManagerError::new(format!(
                "pack schema {} is unsupported",
                pack.schema_version
            )));
        }
        if pack.runtime_abi != BACKEND_RUNTIME_ABI {
            return Err(BackendManagerError::new(format!(
                "backend ABI {} does not match runtime ABI {}",
                pack.runtime_abi, BACKEND_RUNTIME_ABI
            )));
        }
        if let Some(adapter_abi) = pack.adapter_abi.as_deref() {
            if pack.execution_mode != BackendExecutionMode::Native {
                return Err(BackendManagerError::new(
                    "only native backend packs may declare adapter_abi",
                ));
            }
            if adapter_abi != STANDARD_NATIVE_ADAPTER_ABI {
                return Err(BackendManagerError::new(format!(
                    "unsupported native adapter ABI `{adapter_abi}`"
                )));
            }
            validate_signed_contract_values("formats", &pack.formats, false)?;
            validate_signed_contract_values("model_types", &pack.model_types, true)?;
            validate_signed_contract_values("tasks", &pack.tasks, false)?;
            if !pack.capabilities.memory_reporting {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must declare memory_reporting",
                    pack.backend, pack.profile
                )));
            }
            if pack.capabilities.scoped_device_allocator
                && !pack.capabilities.governed_device_allocator
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} declares a scoped allocator without a governed device allocator",
                    pack.backend, pack.profile
                )));
            }
            if pack.accelerator_profile != "cpu" && !pack.capabilities.governed_device_allocator {
                return Err(BackendManagerError::new(format!(
                    "standard native accelerator pack {}/{} must use the governed device allocator",
                    pack.backend, pack.profile
                )));
            }
            if pack.capabilities.scoped_device_allocator
                && pack
                    .memory_behavior
                    .allocation_scope
                    .as_deref()
                    .is_none_or(|scope| scope.trim().is_empty())
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must name its scoped allocation contract",
                    pack.backend, pack.profile
                )));
            }
            if pack
                .memory_behavior
                .device_allocation
                .as_deref()
                .is_none_or(|behavior| behavior.trim().is_empty())
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must declare its device allocation behavior",
                    pack.backend, pack.profile
                )));
            }
            if pack.memory_behavior.synchronize_before_free
                && !pack.capabilities.governed_device_allocator
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} cannot request host synchronization without a governed device allocator",
                    pack.backend, pack.profile
                )));
            }
            if !pack.memory_behavior.planned_reporting
                || !pack.memory_behavior.live_reporting
                || !pack.memory_behavior.request_reporting
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must declare planned, live, and request memory reporting",
                    pack.backend, pack.profile
                )));
            }
            if pack.accelerator_requirements.kind.as_deref()
                != Some(pack.accelerator_profile.as_str())
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} accelerator requirement does not match profile `{}`",
                    pack.backend, pack.profile, pack.accelerator_profile
                )));
            }
            validate_signed_contract_values(
                "execution providers",
                &pack.accelerator_requirements.execution_providers,
                false,
            )?;
            if pack.accelerator_requirements.implicit_cpu_fallback != Some(false) {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must explicitly disable implicit CPU fallback",
                    pack.backend, pack.profile
                )));
            }
        }
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        if !matches!(
            pack.accelerator_profile.as_str(),
            "cpu" | "cuda" | "tensorrt"
        ) {
            return Err(BackendManagerError::new(format!(
                "unsupported accelerator profile `{}`",
                pack.accelerator_profile
            )));
        }
        match (pack.backend.as_str(), pack.kv_mode.as_deref()) {
            ("llama-cpp", Some("native" | "shared_pool")) => {}
            ("llama-cpp", None) => {
                return Err(BackendManagerError::new(
                    "llama.cpp backend packs must sign an explicit kv_mode",
                ));
            }
            ("llama-cpp", Some(kv_mode)) => {
                return Err(BackendManagerError::new(format!(
                    "unsupported backend KV mode `{kv_mode}` for llama-cpp"
                )));
            }
            (backend, Some(kv_mode)) => {
                return Err(BackendManagerError::new(format!(
                    "backend {backend} cannot declare KV mode `{kv_mode}`"
                )));
            }
            (_, None) => {}
        }
        validate_relative_pack_path("entrypoint", &pack.entrypoint)?;
        if let BackendInstaller::Bootstrap { path } = &pack.installer {
            validate_relative_pack_path("bootstrap path", path)?;
        }
        validate_sha256(&pack.sha256)?;
        let _ = decode_signature(&pack.signature)?;
        if !(pack.artifact.starts_with("https://")
            || self.config.allow_file_artifacts && pack.artifact.starts_with("file://"))
        {
            return Err(BackendManagerError::new(format!(
                "backend artifact URL must use HTTPS: {}",
                pack.artifact
            )));
        }
        if pack.download_bytes == 0 || pack.installed_bytes == 0 {
            return Err(BackendManagerError::new(
                "backend pack sizes must both be greater than zero",
            ));
        }
        let runtime = Version::parse(&self.config.runtime_version).map_err(|error| {
            BackendManagerError::new(format!(
                "runtime version `{}` is not valid semver: {error}",
                self.config.runtime_version
            ))
        })?;
        let compatibility = VersionReq::parse(&pack.compatible_kapsl).map_err(|error| {
            BackendManagerError::new(format!(
                "invalid compatible_kapsl requirement `{}`: {error}",
                pack.compatible_kapsl
            ))
        })?;
        if !compatibility.matches(&runtime) {
            return Err(BackendManagerError::new(format!(
                "pack requires Kapsl `{}`, runtime is {}",
                pack.compatible_kapsl, runtime
            )));
        }
        for (path, digest) in &pack.files {
            validate_relative_pack_path("file checksum path", path)?;
            validate_sha256(digest)?;
        }
        if !pack.files.contains_key(&pack.entrypoint) {
            return Err(BackendManagerError::new(format!(
                "backend entrypoint `{}` requires a signed installed-file checksum",
                pack.entrypoint
            )));
        }
        for notice in &pack.licenses {
            if let Some(path) = notice.path.as_deref() {
                validate_relative_pack_path("license path", path)?;
            }
        }
        for (label, minimum) in [
            ("minimum_cuda", pack.minimum_cuda.as_deref()),
            ("minimum_driver", pack.minimum_driver.as_deref()),
        ] {
            if minimum.is_some_and(|value| numeric_version_components(value).is_empty()) {
                return Err(BackendManagerError::new(format!(
                    "backend {label} must contain a numeric version"
                )));
            }
        }
        if let Some(target) = target {
            if pack.platform != target.platform {
                return Err(BackendManagerError::new(format!(
                    "platform {} does not match {}",
                    pack.platform, target.platform
                )));
            }
            if normalize_architecture(&pack.architecture)
                != normalize_architecture(&target.architecture)
            {
                return Err(BackendManagerError::new(format!(
                    "architecture {} does not match {}",
                    pack.architecture, target.architecture
                )));
            }
            if !accelerator_matches(&pack.accelerator_profile, target.accelerator) {
                return Err(BackendManagerError::new(format!(
                    "accelerator profile {} does not match {}",
                    pack.accelerator_profile,
                    target.accelerator.as_str()
                )));
            }
            if let Some(minimum) = pack.minimum_cuda.as_deref() {
                let actual = target.cuda_version.as_deref().ok_or_else(|| {
                    BackendManagerError::new(format!("CUDA {minimum} or newer is required"))
                })?;
                if !version_at_least(actual, minimum) {
                    return Err(BackendManagerError::new(format!(
                        "CUDA {minimum} or newer is required; detected {actual}"
                    )));
                }
            }
            if let Some(minimum) = pack.minimum_driver.as_deref() {
                let actual = target.driver_version.as_deref().ok_or_else(|| {
                    BackendManagerError::new(format!("driver {minimum} or newer is required"))
                })?;
                if !version_at_least(actual, minimum) {
                    return Err(BackendManagerError::new(format!(
                        "driver {minimum} or newer is required; detected {actual}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn installed_pack_is_valid(&self, expected: &BackendPackManifest) -> ManagerResult<bool> {
        let root = self.installed_path(expected)?;
        if !root.is_dir() {
            return Ok(false);
        }
        let record_path = root.join(INSTALL_RECORD_NAME);
        let record =
            match read_json_bounded::<InstalledPackRecord>(&record_path, MAX_MANIFEST_BYTES) {
                Ok(record) => record,
                Err(_) => return Ok(false),
            };
        if record.manifest != *expected {
            return Ok(false);
        }
        if validate_tree_symlinks(&root, &root).is_err() {
            return Ok(false);
        }
        if !entrypoint_is_usable(&root, expected)? {
            return Ok(false);
        }
        for (relative, expected_digest) in &expected.files {
            let path = root.join(relative);
            if !path.is_file() || sha256_file(&path)? != expected_digest.to_ascii_lowercase() {
                return Ok(false);
            }
        }
        for notice in &expected.licenses {
            if notice
                .path
                .as_deref()
                .is_some_and(|relative| !root.join(relative).is_file())
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn download_artifact(
        &self,
        pack: &BackendPackManifest,
        destination: &Path,
    ) -> ManagerResult<()> {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(path) = pack.artifact.strip_prefix("file://") {
            if !self.config.allow_file_artifacts {
                return Err(BackendManagerError::new(
                    "file:// backend artifacts are disabled",
                ));
            }
            fs::copy(Path::new(path), destination).map_err(|error| {
                BackendManagerError::new(format!("copy backend artifact: {error}"))
            })?;
        } else {
            if !pack.artifact.starts_with("https://") {
                return Err(BackendManagerError::new(format!(
                    "backend artifact URL must use HTTPS: {}",
                    pack.artifact
                )));
            }
            let agent = crate::features::http_client::http_agent_for_transfer();
            let mut response = agent.get(&pack.artifact).call().map_err(|error| {
                BackendManagerError::new(format!(
                    "download backend artifact {}: {}",
                    pack.artifact,
                    crate::features::http_client::format_remote_http_error(error)
                ))
            })?;
            let file = File::create(destination)?;
            let mut writer = BufWriter::new(file);
            let mut reader = response
                .body_mut()
                .as_reader()
                .take(pack.download_bytes + 1);
            let copied = std::io::copy(&mut reader, &mut writer)?;
            writer.flush()?;
            if copied != pack.download_bytes {
                return Err(BackendManagerError::new(format!(
                    "backend artifact size mismatch: expected {} bytes, received {}",
                    pack.download_bytes, copied
                )));
            }
        }
        self.verify_artifact_file(pack, destination)
    }

    fn verify_artifact_file(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
    ) -> ManagerResult<()> {
        let size = fs::metadata(archive_path)?.len();
        if size != pack.download_bytes {
            return Err(BackendManagerError::new(format!(
                "backend artifact size mismatch: expected {} bytes, got {}",
                pack.download_bytes, size
            )));
        }
        let actual = sha256_file(archive_path)?;
        if actual != pack.sha256.to_ascii_lowercase() {
            return Err(BackendManagerError::new(format!(
                "backend artifact checksum mismatch: expected {}, got {}",
                pack.sha256, actual
            )));
        }
        self.verify_artifact_signature(&actual, &pack.signature)
    }

    fn install_verified_archive_locked(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
        final_path: &Path,
        stage_root: &Path,
    ) -> ManagerResult<()> {
        // Always re-check at the installation boundary. Bundle extraction or a
        // local filesystem race cannot turn prior verification into trust.
        self.verify_artifact_file(pack, archive_path)?;
        ensure_available_space(
            stage_root,
            pack.download_bytes
                .saturating_add(pack.installed_bytes)
                .saturating_add(INSTALL_SPACE_OVERHEAD_BYTES),
            "extract and install backend pack",
        )?;
        let extract_root = stage_root.join("extracted");
        fs::create_dir_all(&extract_root)?;
        safe_extract_tar_gz(
            archive_path,
            &extract_root,
            pack.installed_bytes.saturating_add(pack.download_bytes),
        )?;
        let payload_root = find_payload_root(&extract_root)?;
        let payload: BackendPayloadManifest = read_json_bounded(
            &payload_root.join(PAYLOAD_MANIFEST_NAME),
            MAX_MANIFEST_BYTES,
        )?;
        validate_payload_manifest(pack, &payload)?;

        let install_root = stage_root.join("installed");
        match &pack.installer {
            BackendInstaller::Extract => {
                fs::rename(&payload_root, &install_root).map_err(|error| {
                    BackendManagerError::new(format!(
                        "stage extracted backend payload {}: {error}",
                        payload_root.display()
                    ))
                })?;
            }
            BackendInstaller::Bootstrap { path } => {
                let script = payload_root.join(path);
                if !script.is_file() {
                    return Err(BackendManagerError::new(format!(
                        "backend bootstrap is missing: {}",
                        script.display()
                    )));
                }
                let status = Command::new("bash")
                    .arg(&script)
                    .arg(&payload_root)
                    .arg(&install_root)
                    .stdin(Stdio::null())
                    .status()
                    .map_err(|error| {
                        BackendManagerError::new(format!(
                            "start trusted backend bootstrap {}: {error}",
                            script.display()
                        ))
                    })?;
                if !status.success() {
                    return Err(BackendManagerError::new(format!(
                        "trusted backend bootstrap failed with {}",
                        status
                    )));
                }
                fs::copy(
                    payload_root.join(PAYLOAD_MANIFEST_NAME),
                    install_root.join(PAYLOAD_MANIFEST_NAME),
                )?;
            }
        }

        validate_tree_symlinks(&install_root, &install_root)?;
        let installed_size = directory_size(&install_root)?;
        let installed_size_limit = pack
            .installed_bytes
            .saturating_add((pack.installed_bytes / 20).max(INSTALLED_SIZE_TOLERANCE_BYTES));
        if installed_size > installed_size_limit {
            return Err(BackendManagerError::new(format!(
                "installed backend expanded to {installed_size} bytes, beyond its signed {installed_size_limit} byte allowance"
            )));
        }

        if !entrypoint_is_usable(&install_root, pack)? {
            return Err(BackendManagerError::new(format!(
                "installed backend entrypoint is missing or unusable: {}",
                install_root.join(&pack.entrypoint).display()
            )));
        }
        for (relative, expected) in &pack.files {
            let path = install_root.join(relative);
            if !path.is_file() {
                return Err(BackendManagerError::new(format!(
                    "installed backend is missing checksummed file {}",
                    relative
                )));
            }
            let actual = sha256_file(&path)?;
            if actual != expected.to_ascii_lowercase() {
                return Err(BackendManagerError::new(format!(
                    "installed backend file {} failed checksum verification",
                    relative
                )));
            }
        }
        for notice in &pack.licenses {
            if let Some(relative) = notice.path.as_deref() {
                let path = install_root.join(relative);
                if !path.is_file() {
                    return Err(BackendManagerError::new(format!(
                        "installed backend is missing license notice {}",
                        relative
                    )));
                }
            }
        }
        let record = InstalledPackRecord {
            manifest: pack.clone(),
            installed_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let record_bytes = serde_json::to_vec_pretty(&record).map_err(|error| {
            BackendManagerError::new(format!("encode installed backend record: {error}"))
        })?;
        atomic_write(&install_root.join(INSTALL_RECORD_NAME), &record_bytes)?;
        sync_tree_security(&install_root)?;

        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let quarantine = if final_path.exists() {
            let quarantine_root = self
                .runtime_cache_root()
                .join(".quarantine")
                .join(&pack.backend);
            fs::create_dir_all(&quarantine_root)?;
            let path = quarantine_root.join(format!("{}-{}", pack.profile, unique_nonce()));
            fs::rename(final_path, &path).map_err(|error| {
                BackendManagerError::new(format!(
                    "quarantine incomplete backend {}: {error}",
                    final_path.display()
                ))
            })?;
            Some(path)
        } else {
            None
        };
        if let Err(error) = fs::rename(&install_root, final_path) {
            if let Some(previous) = quarantine.as_ref() {
                let _ = fs::rename(previous, final_path);
            }
            return Err(BackendManagerError::new(format!(
                "atomically activate backend {}: {error}",
                final_path.display()
            )));
        }
        sync_parent(final_path)?;
        if let Some(previous) = quarantine {
            fs::remove_dir_all(previous)?;
        }
        eprintln!(
            "Installed Kapsl backend {}/{} at {}",
            pack.backend,
            pack.profile,
            final_path.display()
        );
        Ok(())
    }

    fn cleanup_stale_install_stages(&self, pack: &BackendPackManifest) -> ManagerResult<()> {
        let root = self.pack_staging_root(pack)?;
        if !root.is_dir() {
            return Ok(());
        }
        for entry in read_dirs(&root)? {
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn execute_backend_command(
    args: crate::app::BackendCommandArgs,
) -> Result<(), crate::DynError> {
    match args.command {
        crate::app::BackendSubcommand::Ensure(args) => {
            let device_info = DeviceInfo::probe();
            let target = BackendTarget::current(&device_info);
            let model_paths = crate::backend::expand_run_bundles(&args.model, &device_info)?;
            let manager = BackendManager::from_env(args.offline)?;
            let mut ensured = HashSet::<(String, String)>::new();
            for model in model_paths {
                let absolute = model.canonicalize().map_err(|error| {
                    BackendManagerError::new(format!(
                        "invalid model path {}: {error}",
                        model.display()
                    ))
                })?;
                let manifest = crate::backend::inspect_serving_manifest(&absolute)?;
                crate::backend::validate_model_contract(&manifest)?;
                let decision =
                    crate::backend::resolve_serving_backend(&manifest, device_info.has_cuda)?;
                crate::backend::validate_runtime_serving_backend(&manifest, decision)?;
                let memory = crate::backend::preliminary_memory_admission(
                    &absolute,
                    &manifest,
                    decision,
                    &device_info,
                )?;
                if memory.status == crate::backend::MemoryAdmissionStatus::Rejected {
                    return Err(format!(
                        "preliminary memory admission rejected `{}` before backend download: {}",
                        manifest.project_name, memory.reason
                    )
                    .into());
                }
                let uses_onnx = kapsl_core::EngineKind::resolve(&manifest).uses_onnx_session();
                let signed_onnx_route = crate::backend::generic_native_backend_packs_enabled()?;
                let onnx_profile = if uses_onnx && signed_onnx_route {
                    if !crate::backend::lazy_onnx_packs_enabled() {
                        return Err(format!(
                            "model `{}` requires a signed native backend pack, but ONNX pack installation is disabled or unsupported on {}; set {}=0 only for an explicit embedded ORT rollback",
                            manifest.project_name,
                            crate::backend::current_platform(),
                            crate::backend::GENERIC_NATIVE_PACKS_ENV
                        )
                        .into());
                    }
                    crate::backend::onnx_pack_profile_for_manifest(&manifest, &device_info)?
                } else {
                    None
                };
                let llama_profile = if crate::backend::lazy_llama_cpp_packs_enabled() {
                    crate::backend::llama_cpp_pack_profile_for_manifest(&manifest, &device_info)
                } else {
                    None
                };
                if decision.selected == crate::backend::ResolvedServingBackend::Vllm
                    && ensured.insert(("vllm".to_string(), MANAGED_VLLM_PACK_PROFILE.to_string()))
                {
                    let installed = manager.ensure_vllm(&target)?;
                    println!(
                        "Ensured backend vllm/{} at {}",
                        MANAGED_VLLM_PACK_PROFILE,
                        installed.display()
                    );
                } else if let Some(profile) = onnx_profile {
                    let mut onnx_target = target.clone();
                    onnx_target.accelerator = profile.accelerator();
                    if profile == OnnxBackendPackProfile::Cpu {
                        onnx_target.cuda_version = None;
                        onnx_target.driver_version = None;
                    }
                    let requirements =
                        crate::backend::onnx_backend_pack_requirements(&manifest, profile)?;
                    let plan = manager.plan_compatible_backend(&requirements, &onnx_target)?;
                    let identity = (plan.selected_backend.clone(), plan.profile.clone());
                    if ensured.insert(identity) {
                        let installed = manager.ensure_pack(&plan.manifest)?;
                        println!(
                            "Ensured backend {}/{} at {} ({})",
                            plan.selected_backend,
                            plan.profile,
                            installed.display(),
                            plan.selection_reason
                        );
                    }
                } else if let Some(profile) = llama_profile {
                    let identity = ("llama-cpp".to_string(), profile.profile().to_string());
                    if ensured.insert(identity) {
                        let mut llama_target = target.clone();
                        llama_target.accelerator = profile.accelerator();
                        if profile == LlamaCppBackendPackProfile::Cpu {
                            llama_target.cuda_version = None;
                            llama_target.driver_version = None;
                        }
                        let plan = manager.plan_llama_cpp(profile, &llama_target)?;
                        let admission = crate::backend::preliminary_llama_cpp_memory_admission(
                            profile,
                            &absolute,
                            &manifest,
                            &device_info,
                            &plan.manifest,
                            None,
                        )?;
                        if admission.status == crate::backend::MemoryAdmissionStatus::Rejected {
                            return Err(format!(
                                "preliminary memory admission rejected `{}` before backend download: {}",
                                manifest.project_name, admission.reason
                            )
                            .into());
                        }
                        let installed = manager.ensure_pack(&plan.manifest)?;
                        println!(
                            "Ensured backend llama-cpp/{} at {}",
                            profile.profile(),
                            installed.display()
                        );
                    }
                } else if uses_onnx && !signed_onnx_route {
                    let reason = crate::backend::embedded_onnx_rollback_reason(&manifest)?;
                    println!(
                        "{} uses the embedded ORT rollback ({})",
                        manifest.project_name, reason
                    );
                } else if decision.selected != crate::backend::ResolvedServingBackend::Vllm {
                    println!(
                        "{} uses the in-process {} backend; no lazy pack is required by this runtime build",
                        manifest.project_name,
                        decision.selected.as_str()
                    );
                }
            }
            Ok(())
        }
        crate::app::BackendSubcommand::List(args) => {
            let manager = BackendManager::from_env(true)?;
            let entries = manager.list()?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else if entries.is_empty() {
                println!("No backend packs are installed.");
            } else {
                println!("RUNTIME\tBACKEND\tPROFILE\tVERSION\tMODE\tSIZE\tSTATUS\tPATH");
                for entry in entries {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        entry.runtime_version,
                        entry.backend,
                        entry.profile,
                        entry.pack_version,
                        entry.execution_mode,
                        entry.installed_bytes,
                        if entry.valid { "valid" } else { "invalid" },
                        entry.path.display()
                    );
                }
            }
            Ok(())
        }
        crate::app::BackendSubcommand::Prune(args) => {
            let manager = BackendManager::from_env(true)?;
            let (bytes, entries) = manager.prune(args.old_versions)?;
            println!("Pruned {entries} backend cache item(s), reclaiming {bytes} bytes.");
            Ok(())
        }
    }
}

pub(crate) fn runtime_release_version() -> String {
    // The release workflow injects KAPSL_VERSION at compile time. Never accept
    // a process-time override here: cache namespaces, signed indexes, and ABI
    // compatibility must describe the binary that is actually executing.
    option_env!("KAPSL_VERSION")
        .unwrap_or(env!("CARGO_PKG_VERSION"))
        .to_string()
}

pub(crate) fn backend_cache_root() -> Option<PathBuf> {
    std::env::var_os(BACKEND_CACHE_ENV)
        .map(PathBuf::from)
        .or_else(|| dirs::data_local_dir().map(|path| path.join("kapsl/backends")))
}

pub(crate) fn current_platform() -> String {
    format!(
        "{}-{}",
        std::env::consts::OS,
        normalize_architecture(std::env::consts::ARCH)
    )
}

fn default_backend_index_url(runtime_version: &str) -> String {
    let channel = if runtime_version.contains("-beta.") {
        "runtime/beta"
    } else {
        "runtime"
    };
    let base =
        std::env::var("KAPSL_BASE_URL").unwrap_or_else(|_| DEFAULT_DOWNLOAD_BASE_URL.to_string());
    format!(
        "{}/{channel}/v{runtime_version}/backend-index.json",
        base.trim_end_matches('/')
    )
}

fn trusted_public_keys() -> ManagerResult<Vec<VerifyingKey>> {
    let mut encoded = Vec::new();
    if let Some(keys) = option_env!("KAPSL_BACKEND_PUBLIC_KEYS") {
        encoded.push(keys.to_string());
    }
    if let Ok(keys) = std::env::var(BACKEND_PUBLIC_KEYS_ENV) {
        encoded.push(keys);
    }
    let mut result = Vec::new();
    for value in encoded {
        for candidate in value
            .split(|character: char| {
                character == ',' || character == ';' || character.is_whitespace()
            })
            .filter(|candidate| !candidate.is_empty())
        {
            let candidate = candidate.strip_prefix("ed25519:").unwrap_or(candidate);
            let bytes = decode_base64(candidate).map_err(|error| {
                BackendManagerError::new(format!("invalid trusted backend public key: {error}"))
            })?;
            let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
                BackendManagerError::new("trusted Ed25519 public keys must contain 32 bytes")
            })?;
            let key = VerifyingKey::from_bytes(&bytes).map_err(|error| {
                BackendManagerError::new(format!("invalid Ed25519 public key: {error}"))
            })?;
            if !result.contains(&key) {
                result.push(key);
            }
        }
    }
    Ok(result)
}

fn decode_signature(encoded: &str) -> ManagerResult<Signature> {
    let encoded = encoded.strip_prefix("ed25519:").unwrap_or(encoded);
    let bytes = decode_base64(encoded).map_err(|error| {
        BackendManagerError::new(format!("invalid Ed25519 signature encoding: {error}"))
    })?;
    Signature::from_slice(&bytes).map_err(|error| {
        BackendManagerError::new(format!("invalid Ed25519 signature length: {error}"))
    })
}

fn decode_base64(value: &str) -> Result<Vec<u8>, base64::DecodeError> {
    BASE64
        .decode(value)
        .or_else(|_| BASE64_URL_SAFE.decode(value))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn signed_contract_contains(values: &[String], required: &str) -> bool {
    let required = required.trim();
    !required.is_empty()
        && values
            .iter()
            .any(|value| value.trim().eq_ignore_ascii_case(required))
}

fn validate_signed_contract_values(
    label: &str,
    values: &[String],
    allow_empty: bool,
) -> ManagerResult<()> {
    if values.is_empty() && !allow_empty {
        return Err(BackendManagerError::new(format!(
            "standard native backend pack must declare non-empty {label}"
        )));
    }
    let mut normalized = HashSet::new();
    for value in values {
        let value = value.trim().to_ascii_lowercase();
        if value.is_empty() {
            return Err(BackendManagerError::new(format!(
                "standard native backend pack contains an empty {label} value"
            )));
        }
        if !normalized.insert(value.clone()) {
            return Err(BackendManagerError::new(format!(
                "standard native backend pack repeats {label} value `{value}`"
            )));
        }
    }
    Ok(())
}

fn accelerator_matches(profile: &str, target: BackendAccelerator) -> bool {
    match profile.trim().to_ascii_lowercase().as_str() {
        "cpu" => target == BackendAccelerator::Cpu,
        "cuda" => matches!(
            target,
            BackendAccelerator::Cuda | BackendAccelerator::TensorRt
        ),
        "tensorrt" => target == BackendAccelerator::TensorRt,
        _ => false,
    }
}

fn normalize_architecture(value: &str) -> &str {
    match value {
        "amd64" => "x86_64",
        "arm64" => "aarch64",
        other => other,
    }
}

fn compare_pack_versions(left: &str, right: &str) -> std::cmp::Ordering {
    match (semver::Version::parse(left), semver::Version::parse(right)) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        (Ok(_), Err(_)) => std::cmp::Ordering::Greater,
        (Err(_), Ok(_)) => std::cmp::Ordering::Less,
        (Err(_), Err(_)) => {
            let left_numeric = numeric_version_components(left);
            let right_numeric = numeric_version_components(right);
            left_numeric
                .cmp(&right_numeric)
                .then_with(|| left.cmp(right))
        }
    }
}

fn version_at_least(actual: &str, minimum: &str) -> bool {
    let mut actual = numeric_version_components(actual);
    let mut minimum = numeric_version_components(minimum);
    if actual.is_empty() || minimum.is_empty() {
        return false;
    }
    let length = actual.len().max(minimum.len());
    actual.resize(length, 0);
    minimum.resize(length, 0);
    actual >= minimum
}

fn numeric_version_components(value: &str) -> Vec<u64> {
    value
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .take(4)
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect()
}

fn validate_cache_component(label: &str, value: &str) -> ManagerResult<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'+'))
    {
        return Err(BackendManagerError::new(format!(
            "invalid backend {label} `{value}`"
        )));
    }
    Ok(())
}

fn validate_relative_pack_path(label: &str, value: &str) -> ManagerResult<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BackendManagerError::new(format!(
            "invalid {label} `{value}`: expected a normalized relative path"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> ManagerResult<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(BackendManagerError::new(format!(
            "invalid SHA-256 digest `{value}`"
        )));
    }
    Ok(())
}

fn lock_file(path: &Path) -> ManagerResult<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| {
            BackendManagerError::new(format!("open lock file {}: {error}", path.display()))
        })
}

fn ensure_available_space(path: &Path, required: u64, operation: &str) -> ManagerResult<()> {
    let available = fs2::available_space(path).map_err(|error| {
        BackendManagerError::new(format!(
            "inspect available space at {}: {error}",
            path.display()
        ))
    })?;
    if required > available {
        return Err(BackendManagerError::new(format!(
            "insufficient disk space to {operation}: required {required} bytes, available {available} at {}",
            path.display()
        )));
    }
    Ok(())
}

fn process_path_lock(path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut locks = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn backend_maintenance_lock() -> &'static RwLock<()> {
    static LOCK: OnceLock<RwLock<()>> = OnceLock::new();
    LOCK.get_or_init(|| RwLock::new(()))
}

fn read_file_bounded(path: &Path, limit: u64) -> ManagerResult<Vec<u8>> {
    let file = File::open(path)
        .map_err(|error| BackendManagerError::new(format!("open {}: {error}", path.display())))?;
    if file.metadata()?.len() > limit {
        return Err(BackendManagerError::new(format!(
            "{} exceeds the {} byte limit",
            path.display(),
            limit
        )));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(BackendManagerError::new(format!(
            "{} exceeds the {} byte limit",
            path.display(),
            limit
        )));
    }
    Ok(bytes)
}

fn read_json_bounded<T: for<'de> Deserialize<'de>>(path: &Path, limit: u64) -> ManagerResult<T> {
    let bytes = read_file_bounded(path, limit)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        BackendManagerError::new(format!("invalid JSON in {}: {error}", path.display()))
    })
}

fn download_bytes(url: &str, limit: u64) -> ManagerResult<Vec<u8>> {
    let agent = crate::features::http_client::http_agent_for_transfer();
    let mut response = agent.get(url).call().map_err(|error| {
        BackendManagerError::new(format!(
            "download {url}: {}",
            crate::features::http_client::format_remote_http_error(error)
        ))
    })?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(BackendManagerError::new(format!(
            "download from {url} exceeds the {limit} byte limit"
        )));
    }
    Ok(bytes)
}

pub(crate) fn sha256_file(path: &Path) -> ManagerResult<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> ManagerResult<()> {
    let parent = path.parent().ok_or_else(|| {
        BackendManagerError::new(format!("{} has no parent directory", path.display()))
    })?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name().and_then(OsStr::to_str).unwrap_or("file"),
        unique_nonce()
    ));
    {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    if path.exists() {
        quarantine_file(path)?;
    }
    fs::rename(&temporary, path)?;
    sync_parent(path)
}

fn quarantine_file(path: &Path) -> ManagerResult<()> {
    if !path.exists() {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or_else(|| BackendManagerError::new(format!("{} has no parent", path.display())))?;
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or("file");
    let destination = parent.join(format!(".{name}.invalid-{}", unique_nonce()));
    fs::rename(path, destination)?;
    Ok(())
}

fn unique_nonce() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

pub(crate) fn safe_extract_tar_gz(
    archive: &Path,
    destination: &Path,
    max_bytes: u64,
) -> ManagerResult<()> {
    let file = File::open(archive)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let mut total = 0_u64;
    for (position, entry) in archive
        .entries()
        .map_err(|error| BackendManagerError::new(format!("read backend archive: {error}")))?
        .enumerate()
    {
        if position >= MAX_ARCHIVE_ENTRIES {
            return Err(BackendManagerError::new(format!(
                "backend archive contains more than {MAX_ARCHIVE_ENTRIES} entries"
            )));
        }
        let mut entry = entry.map_err(|error| {
            BackendManagerError::new(format!("read backend archive entry: {error}"))
        })?;
        let relative = entry
            .path()
            .map_err(|error| BackendManagerError::new(error.to_string()))?
            .into_owned();
        if relative.is_absolute()
            || relative.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(BackendManagerError::new(format!(
                "backend archive contains unsafe path {}",
                relative.display()
            )));
        }
        let entry_type = entry.header().entry_type();
        if !(entry_type.is_file() || entry_type.is_dir() || entry_type.is_symlink()) {
            return Err(BackendManagerError::new(format!(
                "backend archive contains unsupported entry type for {}",
                relative.display()
            )));
        }
        total = total.saturating_add(entry.size());
        if total > max_bytes {
            return Err(BackendManagerError::new(format!(
                "backend archive expands beyond its signed {} byte limit",
                max_bytes
            )));
        }
        ensure_no_symlink_ancestor(destination, &relative)?;
        if !entry.unpack_in(destination).map_err(|error| {
            BackendManagerError::new(format!(
                "extract backend archive entry {}: {error}",
                relative.display()
            ))
        })? {
            return Err(BackendManagerError::new(format!(
                "backend archive entry escaped the staging directory: {}",
                relative.display()
            )));
        }
    }
    validate_tree_symlinks(destination, destination)?;
    Ok(())
}

fn ensure_no_symlink_ancestor(root: &Path, relative: &Path) -> ManagerResult<()> {
    let mut current = root.to_path_buf();
    let mut components = relative.components().peekable();
    while let Some(component) = components.next() {
        let Component::Normal(component) = component else {
            continue;
        };
        if components.peek().is_none() {
            break;
        }
        current.push(component);
        if fs::symlink_metadata(&current)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false)
        {
            return Err(BackendManagerError::new(format!(
                "backend archive writes through symlink {}",
                current.display()
            )));
        }
    }
    Ok(())
}

fn validate_tree_symlinks(root: &Path, directory: &Path) -> ManagerResult<()> {
    for entry in read_dirs(directory)? {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            let target = fs::read_link(&path)?;
            if target.is_absolute() || !lexical_target_stays_inside(root, &path, &target) {
                return Err(BackendManagerError::new(format!(
                    "backend archive contains escaping symlink {} -> {}",
                    path.display(),
                    target.display()
                )));
            }
        } else if metadata.is_dir() {
            validate_tree_symlinks(root, &path)?;
        }
    }
    Ok(())
}

fn lexical_target_stays_inside(root: &Path, link: &Path, target: &Path) -> bool {
    let Ok(relative_parent) = link.parent().unwrap_or(root).strip_prefix(root) else {
        return false;
    };
    let mut depth = 0_i64;
    for component in relative_parent.components().chain(target.components()) {
        match component {
            Component::CurDir => {}
            Component::Normal(_) => depth += 1,
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            Component::RootDir | Component::Prefix(_) => return false,
        }
    }
    true
}

fn find_payload_root(root: &Path) -> ManagerResult<PathBuf> {
    fn visit(root: &Path, depth: usize, found: &mut Vec<PathBuf>) -> ManagerResult<()> {
        if depth > 4 {
            return Ok(());
        }
        if root.join(PAYLOAD_MANIFEST_NAME).is_file() {
            found.push(root.to_path_buf());
            return Ok(());
        }
        for entry in read_dirs(root)? {
            let path = entry.path();
            if path.is_dir() && !fs::symlink_metadata(&path)?.file_type().is_symlink() {
                visit(&path, depth + 1, found)?;
            }
        }
        Ok(())
    }

    let mut found = Vec::new();
    visit(root, 0, &mut found)?;
    match found.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(BackendManagerError::new(format!(
            "backend archive does not contain {PAYLOAD_MANIFEST_NAME}"
        ))),
        _ => Err(BackendManagerError::new(format!(
            "backend archive contains multiple {PAYLOAD_MANIFEST_NAME} files"
        ))),
    }
}

fn validate_payload_manifest(
    signed: &BackendPackManifest,
    payload: &BackendPayloadManifest,
) -> ManagerResult<()> {
    let matches = payload.schema_version == signed.schema_version
        && payload.backend == signed.backend
        && payload.profile == signed.profile
        && payload.pack_version == signed.pack_version
        && payload.runtime_abi == signed.runtime_abi
        && payload.adapter_abi == signed.adapter_abi
        && payload.platform == signed.platform
        && payload.execution_mode == signed.execution_mode
        && payload.kv_mode == signed.kv_mode
        && payload.entrypoint == signed.entrypoint;
    if matches {
        Ok(())
    } else {
        Err(BackendManagerError::new(format!(
            "backend payload manifest does not match the signed index entry: payload={payload:?}"
        )))
    }
}

fn entrypoint_is_usable(root: &Path, pack: &BackendPackManifest) -> ManagerResult<bool> {
    let path = root.join(&pack.entrypoint);
    if !path.is_file() {
        return Ok(false);
    }
    #[cfg(unix)]
    if pack.execution_mode == BackendExecutionMode::External {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(path)?.permissions().mode() & 0o111 == 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sync_tree_security(root: &Path) -> ManagerResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fn visit(path: &Path) -> std::io::Result<()> {
            let metadata = fs::symlink_metadata(path)?;
            if metadata.file_type().is_symlink() {
                return Ok(());
            }
            let mut permissions = metadata.permissions();
            // Installed code is user-owned: strip setuid/setgid and prevent a
            // group or another local account from rewriting a verified pack.
            let mode = permissions.mode() & !0o6022;
            permissions.set_mode(mode);
            fs::set_permissions(path, permissions)?;
            if !metadata.is_dir() {
                return Ok(());
            }
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let child = entry.path();
                visit(&child)?;
            }
            Ok(())
        }
        visit(root)?;
    }
    Ok(())
}

fn sync_parent(path: &Path) -> ManagerResult<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn read_dirs(path: &Path) -> ManagerResult<Vec<fs::DirEntry>> {
    let mut entries = fs::read_dir(path)
        .map_err(|error| {
            BackendManagerError::new(format!("read directory {}: {error}", path.display()))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BackendManagerError::new(error.to_string()))?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

fn directory_size(path: &Path) -> ManagerResult<u64> {
    let mut total = 0_u64;
    for entry in read_dirs(path)? {
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            total = total.saturating_add(directory_size(&entry.path())?);
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn is_internal_name(name: &OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::sync::{Arc, Barrier};
    use tar::Builder;

    fn sign_index(key: &SigningKey, bytes: &[u8]) -> String {
        let mut message = b"kapsl-backend-index-v1\0".to_vec();
        message.extend_from_slice(bytes);
        format!("ed25519:{}", BASE64.encode(key.sign(&message).to_bytes()))
    }

    fn sign_artifact(key: &SigningKey, digest: &str) -> String {
        let message = format!("kapsl-backend-artifact-v1\0sha256:{digest}");
        format!(
            "ed25519:{}",
            BASE64.encode(key.sign(message.as_bytes()).to_bytes())
        )
    }

    fn build_pack(path: &Path) {
        let file = File::create(path).unwrap();
        let gzip = GzEncoder::new(file, Compression::fast());
        let mut builder = Builder::new(gzip);
        let payload = BackendPayloadManifest {
            schema_version: 1,
            backend: "vllm".to_string(),
            profile: MANAGED_VLLM_PACK_PROFILE.to_string(),
            pack_version: "0.26.1".to_string(),
            runtime_abi: 1,
            adapter_abi: None,
            platform: current_platform(),
            execution_mode: BackendExecutionMode::External,
            kv_mode: None,
            entrypoint: "bin/python".to_string(),
        };
        let manifest = serde_json::to_vec_pretty(&payload).unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_size(manifest.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, PAYLOAD_MANIFEST_NAME, manifest.as_slice())
            .unwrap();
        let executable = b"#!/bin/sh\nexit 0\n";
        let mut header = tar::Header::new_gnu();
        header.set_size(executable.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, "bin/python", executable.as_slice())
            .unwrap();
        builder.finish().unwrap();
    }

    fn fixture() -> (
        tempfile::TempDir,
        BackendManager,
        BackendPackManifest,
        SigningKey,
    ) {
        let root = tempfile::tempdir().unwrap();
        let artifact = root.path().join("pack.tar.gz");
        build_pack(&artifact);
        let signing = SigningKey::from_bytes(&[7_u8; 32]);
        let digest = sha256_file(&artifact).unwrap();
        let executable_digest = format!("{:x}", Sha256::digest(b"#!/bin/sh\nexit 0\n"));
        let pack = BackendPackManifest {
            schema_version: 1,
            backend: "vllm".to_string(),
            profile: MANAGED_VLLM_PACK_PROFILE.to_string(),
            pack_version: "0.26.1".to_string(),
            runtime_abi: 1,
            adapter_abi: None,
            compatible_kapsl: ">=0.2.3, <0.3.0".to_string(),
            platform: current_platform(),
            architecture: std::env::consts::ARCH.to_string(),
            accelerator_profile: "cuda".to_string(),
            accelerator_requirements: Default::default(),
            minimum_cuda: Some("12.0".to_string()),
            minimum_driver: None,
            execution_mode: BackendExecutionMode::External,
            kv_mode: None,
            formats: Vec::new(),
            model_types: Vec::new(),
            tasks: Vec::new(),
            capabilities: Default::default(),
            memory_behavior: Default::default(),
            entrypoint: "bin/python".to_string(),
            artifact: format!("file://{}", artifact.display()),
            download_bytes: fs::metadata(&artifact).unwrap().len(),
            installed_bytes: 4096,
            sha256: digest.clone(),
            signature: sign_artifact(&signing, &digest),
            memory: BackendMemoryManifest::default(),
            installer: BackendInstaller::Extract,
            files: BTreeMap::from([("bin/python".to_string(), executable_digest)]),
            licenses: Vec::new(),
            priority: 1,
        };
        let index = BackendIndex {
            schema_version: 1,
            runtime_version: "0.2.3".to_string(),
            generated_at: "2026-08-25T00:00:00Z".to_string(),
            packs: vec![pack.clone()],
        };
        let index_path = root.path().join("backend-index.json");
        let bytes = serde_json::to_vec_pretty(&index).unwrap();
        fs::write(&index_path, &bytes).unwrap();
        fs::write(
            format!("{}.sig", index_path.display()),
            sign_index(&signing, &bytes),
        )
        .unwrap();
        let manager = BackendManager::for_test(
            root.path().join("cache"),
            "0.2.3",
            index_path,
            signing.verifying_key(),
            false,
        );
        (root, manager, pack, signing)
    }

    fn cuda_target() -> BackendTarget {
        BackendTarget {
            platform: current_platform(),
            architecture: std::env::consts::ARCH.to_string(),
            accelerator: BackendAccelerator::Cuda,
            cuda_version: Some("13.0".to_string()),
            driver_version: Some("580.1".to_string()),
        }
    }

    fn cpu_target() -> BackendTarget {
        BackendTarget {
            platform: current_platform(),
            architecture: std::env::consts::ARCH.to_string(),
            accelerator: BackendAccelerator::Cpu,
            cuda_version: None,
            driver_version: None,
        }
    }

    fn tensorrt_target() -> BackendTarget {
        BackendTarget {
            accelerator: BackendAccelerator::TensorRt,
            ..cuda_target()
        }
    }

    fn standard_native_pack(
        mut pack: BackendPackManifest,
        backend: &str,
        profile: &str,
        accelerator: &str,
    ) -> BackendPackManifest {
        let uses_device = accelerator != "cpu";
        pack.backend = backend.to_string();
        pack.profile = profile.to_string();
        pack.adapter_abi = Some(STANDARD_NATIVE_ADAPTER_ABI.to_string());
        pack.accelerator_profile = accelerator.to_string();
        pack.accelerator_requirements = BackendAcceleratorRequirements {
            kind: Some(accelerator.to_string()),
            execution_providers: vec![accelerator.to_string()],
            implicit_cpu_fallback: Some(false),
        };
        pack.minimum_cuda = uses_device.then(|| "12.0".to_string());
        pack.execution_mode = BackendExecutionMode::Native;
        pack.kv_mode = None;
        pack.formats = vec!["onnx".to_string()];
        pack.model_types = Vec::new();
        pack.tasks = vec!["forward".to_string(), "generate".to_string()];
        pack.capabilities = BackendPackCapabilities {
            batching: true,
            streaming: true,
            cancellation: true,
            memory_reporting: true,
            governed_device_allocator: uses_device,
            scoped_device_allocator: uses_device,
            kv_participation: false,
            concurrent_inference: true,
        };
        pack.memory_behavior = BackendMemoryBehavior {
            allocation_scope: uses_device.then(|| "kapsl-scoped-device-allocator-v1".to_string()),
            device_allocation: Some(if uses_device {
                "host-governed-scoped".to_string()
            } else {
                "none".to_string()
            }),
            planned_reporting: true,
            live_reporting: true,
            request_reporting: true,
            synchronize_before_free: uses_device,
        };
        pack
    }

    #[test]
    fn signed_pack_installs_atomically_and_is_reused() {
        let (_root, manager, pack, _) = fixture();
        let selected = manager.plan_vllm(&cuda_target()).unwrap();
        assert!(!selected.installed);
        let installed = manager.ensure_pack(&pack).unwrap();
        assert!(installed.join("bin/python").is_file());
        assert!(manager.installed_pack_is_valid(&pack).unwrap());
        assert_eq!(manager.ensure_pack(&pack).unwrap(), installed);
        let listed = manager.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].valid);
    }

    #[test]
    fn onnx_plan_requires_an_exact_native_profile() {
        let (_root, manager, pack, signing) = fixture();
        let pack = standard_native_pack(pack, "onnx", ONNX_CPU_PACK_PROFILE, "cpu");
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-08-25T00:00:00Z".to_string(),
            packs: vec![pack],
        };
        let bytes = serde_json::to_vec_pretty(&index).unwrap();
        let index_path = manager.config.index_path.as_ref().unwrap();
        fs::write(index_path, &bytes).unwrap();
        fs::write(
            format!("{}.sig", index_path.display()),
            sign_index(&signing, &bytes),
        )
        .unwrap();

        let plan = manager
            .plan_onnx(OnnxBackendPackProfile::Cpu, &cpu_target())
            .unwrap();
        assert_eq!(plan.selected_backend, "onnx");
        assert_eq!(plan.profile, ONNX_CPU_PACK_PROFILE);
        assert_eq!(plan.execution_mode, "native");
        assert_eq!(
            plan.manifest.adapter_abi.as_deref(),
            Some(STANDARD_NATIVE_ADAPTER_ABI)
        );
        assert!(manager
            .plan_onnx(OnnxBackendPackProfile::Cuda12, &cpu_target())
            .unwrap_err()
            .to_string()
            .contains("requires a cuda target"));
    }

    #[test]
    fn capability_resolver_filters_model_contract_and_records_reason() {
        let (_root, manager, pack, _) = fixture();
        let mut compatible = standard_native_pack(pack.clone(), "fake-a", "cpu", "cpu");
        compatible.priority = 20;
        let mut wrong_task = standard_native_pack(pack, "fake-b", "cpu", "cpu");
        wrong_task.priority = 100;
        wrong_task.tasks = vec!["classify".to_string()];
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-09-03T00:00:00Z".to_string(),
            packs: vec![wrong_task, compatible],
        };
        let requirements = BackendPackRequirements {
            format: Some("onnx".to_string()),
            model_type: Some("causal-lm".to_string()),
            task: Some("generate".to_string()),
            execution_mode: Some(BackendExecutionMode::Native),
            capabilities: BackendPackCapabilities {
                batching: true,
                streaming: true,
                cancellation: true,
                memory_reporting: true,
                ..BackendPackCapabilities::default()
            },
            ..BackendPackRequirements::default()
        };

        let selected = manager
            .select_compatible_pack(&index, &requirements, &cpu_target())
            .unwrap();
        assert_eq!(selected.manifest.backend, "fake-a");
        assert!(selected
            .reason
            .contains("capability-based automatic selection"));
        assert!(selected.reason.contains("causal-lm"));
        assert!(selected.reason.contains("generate"));
    }

    #[test]
    fn capability_resolver_requires_the_selected_execution_provider() {
        let (_root, manager, pack, _) = fixture();
        let mut cuda = standard_native_pack(pack.clone(), "fake-cuda", "cuda12", "cuda");
        cuda.priority = 100;
        let tensorrt = standard_native_pack(pack, "fake-tensorrt", "tensorrt10", "tensorrt");
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-09-03T00:00:00Z".to_string(),
            packs: vec![cuda, tensorrt],
        };
        let requirements = BackendPackRequirements {
            format: Some("onnx".to_string()),
            task: Some("forward".to_string()),
            execution_provider: Some("tensorrt".to_string()),
            ..BackendPackRequirements::default()
        };

        let selected = manager
            .select_compatible_pack(&index, &requirements, &tensorrt_target())
            .unwrap();
        assert_eq!(selected.manifest.backend, "fake-tensorrt");
        assert!(selected.reason.contains("provider `tensorrt`"));
    }

    #[test]
    fn capability_resolver_orders_semantic_versions_numerically() {
        let (_root, manager, pack, _) = fixture();
        let mut older = standard_native_pack(pack.clone(), "fake", "cpu", "cpu");
        older.pack_version = "0.9.0".to_string();
        let mut newer = standard_native_pack(pack, "fake", "cpu", "cpu");
        newer.pack_version = "0.10.0".to_string();
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-09-03T00:00:00Z".to_string(),
            packs: vec![older, newer],
        };
        let requirements = BackendPackRequirements {
            backend_pin: Some("fake".to_string()),
            format: Some("onnx".to_string()),
            task: Some("forward".to_string()),
            ..BackendPackRequirements::default()
        };

        let selected = manager
            .select_compatible_pack(&index, &requirements, &cpu_target())
            .unwrap();
        assert_eq!(selected.manifest.pack_version, "0.10.0");
    }

    #[test]
    fn explicit_backend_pin_fails_without_substitution() {
        let (_root, manager, pack, _) = fixture();
        let available = standard_native_pack(pack, "available", "cpu", "cpu");
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-09-03T00:00:00Z".to_string(),
            packs: vec![available],
        };
        let requirements = BackendPackRequirements {
            backend_pin: Some("required".to_string()),
            format: Some("onnx".to_string()),
            task: Some("forward".to_string()),
            ..BackendPackRequirements::default()
        };

        let error = manager
            .select_compatible_pack(&index, &requirements, &cpu_target())
            .unwrap_err()
            .to_string();
        assert!(error.contains("explicit backend `required`"));
        assert!(error.contains("no compatible signed pack"));
        assert!(!error.contains("selected available"));
    }

    #[test]
    fn equally_ranked_automatic_candidates_are_ambiguous() {
        let (_root, manager, pack, _) = fixture();
        let first = standard_native_pack(pack.clone(), "fake-a", "cpu", "cpu");
        let second = standard_native_pack(pack, "fake-b", "cpu", "cpu");
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-09-03T00:00:00Z".to_string(),
            packs: vec![second, first],
        };
        let mut requirements = BackendPackRequirements {
            format: Some("onnx".to_string()),
            task: Some("forward".to_string()),
            ..BackendPackRequirements::default()
        };

        let error = manager
            .select_compatible_pack(&index, &requirements, &cpu_target())
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous"));
        assert!(error.contains("fake-a/cpu"));
        assert!(error.contains("fake-b/cpu"));

        requirements.backend_pin = Some("fake-b".to_string());
        let selected = manager
            .select_compatible_pack(&index, &requirements, &cpu_target())
            .unwrap();
        assert_eq!(selected.manifest.backend, "fake-b");
    }

    #[test]
    fn scoped_allocator_requirement_rejects_unscoped_accelerator_pack() {
        let (_root, manager, pack, _) = fixture();
        let mut unscoped = standard_native_pack(pack, "fake-gpu", "cuda12", "cuda");
        unscoped.capabilities.scoped_device_allocator = false;
        unscoped.memory_behavior.allocation_scope = None;
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-09-03T00:00:00Z".to_string(),
            packs: vec![unscoped],
        };
        let requirements = BackendPackRequirements {
            backend_pin: Some("fake-gpu".to_string()),
            format: Some("onnx".to_string()),
            task: Some("generate".to_string()),
            execution_mode: Some(BackendExecutionMode::Native),
            capabilities: BackendPackCapabilities {
                memory_reporting: true,
                governed_device_allocator: true,
                scoped_device_allocator: true,
                ..BackendPackCapabilities::default()
            },
            allocation_scope: Some("kapsl-scoped-device-allocator-v1".to_string()),
            ..BackendPackRequirements::default()
        };

        let error = manager
            .select_compatible_pack(&index, &requirements, &cuda_target())
            .unwrap_err()
            .to_string();
        assert!(error.contains("scoped_device_allocator"));
    }

    #[test]
    fn governed_allocator_requirement_rejects_missing_device_synchronization() {
        let (_root, manager, pack, _) = fixture();
        let mut unsynchronized = standard_native_pack(pack, "fake-gpu", "cuda12", "cuda");
        unsynchronized.memory_behavior.synchronize_before_free = false;
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-09-03T00:00:00Z".to_string(),
            packs: vec![unsynchronized],
        };
        let requirements = BackendPackRequirements {
            backend_pin: Some("fake-gpu".to_string()),
            format: Some("onnx".to_string()),
            task: Some("generate".to_string()),
            execution_provider: Some("cuda".to_string()),
            capabilities: BackendPackCapabilities {
                governed_device_allocator: true,
                scoped_device_allocator: true,
                ..BackendPackCapabilities::default()
            },
            allocation_scope: Some("kapsl-scoped-device-allocator-v1".to_string()),
            synchronize_before_free: true,
            ..BackendPackRequirements::default()
        };

        let error = manager
            .select_compatible_pack(&index, &requirements, &cuda_target())
            .unwrap_err()
            .to_string();
        assert!(error.contains("synchronize the device"));
    }

    #[test]
    fn adapter_abi_is_explicit_and_fail_closed() {
        let (_root, manager, mut pack, _) = fixture();
        pack.execution_mode = BackendExecutionMode::Native;
        pack.adapter_abi = Some("vendor-private-v1".to_string());
        let error = manager
            .validate_pack_manifest(&pack, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsupported native adapter ABI"));

        pack.adapter_abi = Some(STANDARD_NATIVE_ADAPTER_ABI.to_string());
        pack.execution_mode = BackendExecutionMode::External;
        let error = manager
            .validate_pack_manifest(&pack, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("only native backend packs"));
    }

    #[test]
    fn payload_must_repeat_the_signed_adapter_abi() {
        let (_root, _manager, mut signed, _) = fixture();
        signed.execution_mode = BackendExecutionMode::Native;
        signed.adapter_abi = Some(STANDARD_NATIVE_ADAPTER_ABI.to_string());
        let mut payload = BackendPayloadManifest {
            schema_version: signed.schema_version,
            backend: signed.backend.clone(),
            profile: signed.profile.clone(),
            pack_version: signed.pack_version.clone(),
            runtime_abi: signed.runtime_abi,
            adapter_abi: None,
            platform: signed.platform.clone(),
            execution_mode: signed.execution_mode,
            kv_mode: signed.kv_mode.clone(),
            entrypoint: signed.entrypoint.clone(),
        };
        assert!(validate_payload_manifest(&signed, &payload).is_err());
        payload.adapter_abi = signed.adapter_abi.clone();
        assert!(validate_payload_manifest(&signed, &payload).is_ok());
    }

    #[test]
    fn llama_cpp_plan_requires_an_exact_native_profile() {
        let (_root, manager, mut pack, signing) = fixture();
        pack.backend = "llama-cpp".to_string();
        pack.profile = LLAMA_CPP_CPU_PACK_PROFILE.to_string();
        pack.accelerator_profile = "cpu".to_string();
        pack.minimum_cuda = None;
        pack.execution_mode = BackendExecutionMode::Native;
        pack.kv_mode = Some("native".to_string());
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-08-25T00:00:00Z".to_string(),
            packs: vec![pack],
        };
        let bytes = serde_json::to_vec_pretty(&index).unwrap();
        let index_path = manager.config.index_path.as_ref().unwrap();
        fs::write(index_path, &bytes).unwrap();
        fs::write(
            format!("{}.sig", index_path.display()),
            sign_index(&signing, &bytes),
        )
        .unwrap();

        let plan = manager
            .plan_llama_cpp(LlamaCppBackendPackProfile::Cpu, &cpu_target())
            .unwrap();
        assert_eq!(plan.selected_backend, "llama-cpp");
        assert_eq!(plan.profile, LLAMA_CPP_CPU_PACK_PROFILE);
        assert_eq!(plan.execution_mode, "native");
        assert!(manager
            .plan_llama_cpp(LlamaCppBackendPackProfile::Cuda12, &cpu_target())
            .unwrap_err()
            .to_string()
            .contains("requires a cuda target"));
    }

    #[test]
    fn llama_cpp_pack_requires_a_signed_kv_mode() {
        let (_root, manager, mut pack, _) = fixture();
        pack.backend = "llama-cpp".to_string();
        pack.profile = LLAMA_CPP_CPU_PACK_PROFILE.to_string();
        pack.accelerator_profile = "cpu".to_string();
        pack.minimum_cuda = None;
        pack.execution_mode = BackendExecutionMode::Native;

        let error = manager
            .validate_pack_manifest(&pack, Some(&cpu_target()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("must sign an explicit kv_mode"));
    }

    #[test]
    fn offline_index_anchor_rejects_a_conflicting_signed_release_index() {
        let (_root, manager, pack, signing) = fixture();
        let original = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: manager.runtime_version().to_string(),
            generated_at: "2026-08-25T00:00:00Z".to_string(),
            packs: vec![pack],
        };
        let original_bytes = serde_json::to_vec_pretty(&original).unwrap();
        let original_signature = sign_index(&signing, &original_bytes);
        manager
            .cache_verified_index_bytes(&original_bytes, &original_signature)
            .unwrap();

        let mut conflicting = original;
        conflicting.generated_at = "2026-08-25T00:00:01Z".to_string();
        let conflicting_bytes = serde_json::to_vec_pretty(&conflicting).unwrap();
        let conflicting_signature = sign_index(&signing, &conflicting_bytes);
        let error = manager
            .cache_verified_index_bytes(&conflicting_bytes, &conflicting_signature)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different signed backend index"));
    }

    #[test]
    fn corrupted_archive_is_rejected_without_partial_install() {
        let (root, manager, pack, _) = fixture();
        let artifact = PathBuf::from(pack.artifact.strip_prefix("file://").unwrap());
        fs::write(&artifact, b"corrupt").unwrap();
        let error = manager.ensure_pack(&pack).unwrap_err().to_string();
        assert!(error.contains("size mismatch") || error.contains("checksum mismatch"));
        assert!(!manager.installed_path(&pack).unwrap().exists());
        assert!(root.path().join("cache").exists());
    }

    #[test]
    fn invalid_index_signature_fails_closed() {
        let (_root, manager, _pack, _) = fixture();
        let signature_path = format!(
            "{}.sig",
            manager.config.index_path.as_ref().unwrap().display()
        );
        fs::write(
            signature_path,
            format!("ed25519:{}", BASE64.encode([0_u8; 64])),
        )
        .unwrap();
        assert!(manager
            .load_index()
            .unwrap_err()
            .to_string()
            .contains("signature verification failed"));
    }

    #[test]
    fn abi_and_platform_mismatches_are_rejected() {
        let (_root, manager, mut pack, _) = fixture();
        pack.runtime_abi = 2;
        assert!(manager
            .validate_pack_manifest(&pack, Some(&cuda_target()))
            .unwrap_err()
            .to_string()
            .contains("ABI"));
        pack.runtime_abi = 1;
        pack.platform = "somewhere-else".to_string();
        assert!(manager
            .validate_pack_manifest(&pack, Some(&cuda_target()))
            .unwrap_err()
            .to_string()
            .contains("platform"));
    }

    #[test]
    fn offline_missing_pack_has_actionable_error() {
        let (root, _manager, _pack, signing) = fixture();
        let index_path = root.path().join("backend-index.json");
        let manager = BackendManager::for_test(
            root.path().join("offline-cache"),
            "0.2.3",
            index_path,
            signing.verifying_key(),
            true,
        );
        assert!(manager
            .plan_vllm(&cuda_target())
            .unwrap_err()
            .to_string()
            .contains("offline"));
    }

    #[test]
    fn concurrent_ensure_produces_one_complete_install() {
        let (_root, manager, pack, _) = fixture();
        let manager = Arc::new(manager);
        let pack = Arc::new(pack);
        let barrier = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let manager = manager.clone();
                let pack = pack.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    manager.ensure_pack(&pack)
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let paths = handles
            .into_iter()
            .map(|handle| handle.join().unwrap().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(paths[0], paths[1]);
        assert!(manager.installed_pack_is_valid(&pack).unwrap());
    }

    #[test]
    fn interrupted_stage_is_removed_before_retry() {
        let (_root, manager, pack, _signing) = fixture();
        let stale = manager
            .pack_staging_root(&pack)
            .unwrap()
            .join("install-interrupted");
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("partial-download"), b"partial").unwrap();

        let installed = manager.ensure_pack(&pack).unwrap();
        assert!(installed.join("bin/python").is_file());
        assert!(!stale.exists());
    }

    #[test]
    fn corrupted_cached_file_is_replaced_from_signed_pack() {
        let (_root, manager, pack, _signing) = fixture();
        let installed = manager.ensure_pack(&pack).unwrap();
        fs::write(installed.join("bin/python"), b"corrupt").unwrap();

        let repaired = manager.ensure_pack(&pack).unwrap();
        assert_eq!(
            fs::read(repaired.join("bin/python")).unwrap(),
            b"#!/bin/sh\nexit 0\n"
        );
    }

    #[test]
    fn escaping_symlink_is_rejected() {
        assert!(!lexical_target_stays_inside(
            Path::new("/cache"),
            Path::new("/cache/bin/python"),
            Path::new("../../outside")
        ));
        assert!(lexical_target_stays_inside(
            Path::new("/cache"),
            Path::new("/cache/bin/python"),
            Path::new("../python3.12")
        ));
    }

    #[test]
    fn numeric_versions_compare_without_lexical_errors() {
        assert!(version_at_least("13.0", "12.6"));
        assert!(version_at_least("580.10.2", "580.9"));
        assert!(!version_at_least("12.1", "12.10"));
    }
}
