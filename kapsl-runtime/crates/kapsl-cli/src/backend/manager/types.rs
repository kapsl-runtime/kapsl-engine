//! Signed backend index schema and the target/profile types used to query it.

use super::*;

pub(crate) const ONNX_CPU_PACK_PROFILE: &str = "cpu";
pub(crate) const ONNX_CUDA12_PACK_PROFILE: &str = "cuda12";
pub(crate) const ONNX_TENSORRT10_PACK_PROFILE: &str = "tensorrt10";
pub(crate) const LLAMA_CPP_CPU_PACK_PROFILE: &str = "cpu";
pub(crate) const LLAMA_CPP_CUDA12_PACK_PROFILE: &str = "cuda12";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BackendExecutionMode {
    /// The entrypoint is a shared library loaded into the runtime process.
    /// Used by ONNX and llama.cpp packs; the only mode that may declare an
    /// `adapter_abi`.
    Native,
    /// The entrypoint is an executable run as a separate process. Used by the
    /// managed vLLM pack.
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
    pub(super) fn missing_required(&self, required: &Self) -> Vec<&'static str> {
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
    /// Manifest format version; must equal `BACKEND_PACK_SCHEMA_VERSION`.
    pub(crate) schema_version: u32,
    /// Backend family, for example `onnx`, `llama-cpp`, or `vllm`. Also names
    /// the pack's cache directory.
    pub(crate) backend: String,
    /// Build variant within the backend, for example `cpu` or `cuda12`. Also
    /// names the cache subdirectory under `backend`.
    pub(crate) profile: String,
    /// Version of this pack build; the newer version wins a priority tie.
    pub(crate) pack_version: String,
    /// Host/pack interface version; must equal `BACKEND_RUNTIME_ABI`.
    pub(crate) runtime_abi: u32,
    /// Standard in-process adapter contract. Legacy provider-only packs omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) adapter_abi: Option<String>,
    /// A semver requirement evaluated against the Kapsl runtime version.
    pub(crate) compatible_kapsl: String,
    /// Canonical OS/architecture pair, for example `linux-x86_64`.
    pub(crate) platform: String,
    /// CPU architecture, for example `x86_64`. The `amd64` and `arm64`
    /// aliases are normalized before matching.
    pub(crate) architecture: String,
    /// `cpu`, `cuda`, or `tensorrt`. A `cuda` pack also serves TensorRT
    /// targets.
    pub(crate) accelerator_profile: String,
    /// Execution providers and CPU-fallback policy the pack producer signs.
    #[serde(default)]
    pub(crate) accelerator_requirements: BackendAcceleratorRequirements,
    /// Lowest CUDA version the target must report. `None` skips the check.
    #[serde(default)]
    pub(crate) minimum_cuda: Option<String>,
    /// Lowest GPU driver version the target must report. `None` skips the
    /// check.
    #[serde(default)]
    pub(crate) minimum_driver: Option<String>,
    /// Whether the entrypoint is loaded in-process or run as its own process.
    pub(crate) execution_mode: BackendExecutionMode,
    /// KV allocation authority for llama.cpp packs (`native` or
    /// `shared_pool`). Omitted by non-llama backends and legacy native-KV
    /// records.
    #[serde(default)]
    pub(crate) kv_mode: Option<String>,
    /// Model formats this pack can execute. Must be non-empty for standard
    /// native adapter packs; only legacy packs leave it empty.
    #[serde(default)]
    pub(crate) formats: Vec<String>,
    /// Model types this pack can execute. Empty means any model type.
    #[serde(default)]
    pub(crate) model_types: Vec<String>,
    /// Tasks this pack can serve. Must be non-empty for standard native
    /// adapter packs.
    #[serde(default)]
    pub(crate) tasks: Vec<String>,
    /// Runtime features the pack claims; the adapter must confirm them when
    /// it is loaded.
    #[serde(default)]
    pub(crate) capabilities: BackendPackCapabilities,
    /// How the pack allocates, reports, and frees memory.
    #[serde(default)]
    pub(crate) memory_behavior: BackendMemoryBehavior,
    /// Pack-local native library or external executable. Must have a
    /// checksum in `files`.
    pub(crate) entrypoint: String,
    /// Archive download URL. Must be `https://`; `file://` is accepted only
    /// when local artifacts are allowed.
    pub(crate) artifact: String,
    /// Exact archive size; a download of any other length is rejected.
    pub(crate) download_bytes: u64,
    /// Expected size after extraction. Used for the free-space check, and an
    /// install more than 5% (at least 64 MiB) larger is rejected.
    pub(crate) installed_bytes: u64,
    /// Hex-encoded SHA-256 of the archive.
    pub(crate) sha256: String,
    /// Ed25519 signature over the domain-separated artifact digest.
    pub(crate) signature: String,
    /// Fixed and per-model memory the pack needs, used for memory admission.
    #[serde(default)]
    pub(crate) memory: BackendMemoryManifest,
    /// How the extracted archive becomes an installed pack.
    #[serde(default)]
    pub(crate) installer: BackendInstaller,
    /// SHA-256 digests of security-sensitive installed files, keyed by
    /// pack-relative path. Must include the entrypoint.
    #[serde(default)]
    pub(crate) files: BTreeMap<String, String>,
    /// Optional complete BLAKE3 digests authenticated by the signed index.
    /// SHA-256 remains required for installation and older engine versions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) files_blake3: Option<BTreeMap<String, String>>,
    /// Third-party license notices; any notice `path` must exist in the
    /// installed pack.
    #[serde(default)]
    pub(crate) licenses: Vec<BackendLicenseNotice>,
    /// Rank among compatible packs; the highest priority is selected.
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
pub(super) struct BackendPayloadManifest {
    pub(super) schema_version: u32,
    pub(super) backend: String,
    pub(super) profile: String,
    pub(super) pack_version: String,
    pub(super) runtime_abi: u32,
    #[serde(default)]
    pub(super) adapter_abi: Option<String>,
    pub(super) platform: String,
    pub(super) execution_mode: BackendExecutionMode,
    #[serde(default)]
    pub(super) kv_mode: Option<String>,
    pub(super) entrypoint: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct InstalledPackRecord {
    pub(super) manifest: BackendPackManifest,
    pub(super) installed_at_unix_seconds: u64,
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
