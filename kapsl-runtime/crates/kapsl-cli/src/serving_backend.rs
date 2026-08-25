use super::*;
use clap::ValueEnum;

const SERVING_METADATA_KEY: &str = "serving";
const BACKEND_METADATA_KEY: &str = "backend";
const BACKEND_PLAN_MANIFEST_MAX_BYTES: u64 = 1024 * 1024;

#[derive(clap::Args, Debug)]
pub(crate) struct BackendPlanCommandArgs {
    /// Model package or raw model path to inspect.
    #[arg(value_name = "MODEL")]
    pub(crate) model: PathBuf,

    /// Override detected CUDA availability when planning for another host.
    #[arg(long, value_name = "BOOL")]
    pub(crate) cuda: Option<bool>,

    /// Do not fetch a backend index; use only a previously verified cached copy.
    #[arg(long)]
    pub(crate) offline: bool,
}

/// Deployment-time serving policy embedded in `metadata.serving.backend`.
///
/// This is deliberately separate from `EngineKind`: `EngineKind` describes the
/// model/file contract, while this policy chooses the process that serves a
/// compatible model.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ServingBackendPolicy {
    Auto,
    #[value(name = "llama_cpp", alias = "llama-cpp", alias = "llama.cpp")]
    #[serde(alias = "llama-cpp", alias = "llama.cpp")]
    LlamaCpp,
    Vllm,
}

impl ServingBackendPolicy {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::LlamaCpp => "llama_cpp",
            Self::Vllm => "vllm",
        }
    }

    fn parse_manifest_value(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "llama_cpp" | "llama-cpp" | "llama.cpp" => Ok(Self::LlamaCpp),
            "vllm" => Ok(Self::Vllm),
            other => Err(format!(
                "unknown metadata.serving.backend `{other}`; expected `auto`, `llama_cpp`, or `vllm`"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResolvedServingBackend {
    /// Preserve the backend factory behavior used by packages created before
    /// `metadata.serving.backend` existed, and for formats outside this policy.
    Builtin,
    LlamaCpp,
    Vllm,
}

impl ResolvedServingBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::LlamaCpp => "llama_cpp",
            Self::Vllm => "vllm",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ServingBackendDecision {
    pub(crate) requested: Option<ServingBackendPolicy>,
    pub(crate) selected: ResolvedServingBackend,
    pub(crate) reason: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MemoryAdmissionStatus {
    Accepted,
    Rejected,
    Unknown,
}

#[derive(Clone, Debug)]
pub(crate) struct PreliminaryMemoryAdmission {
    pub(crate) status: MemoryAdmissionStatus,
    pub(crate) required_bytes: Option<u64>,
    pub(crate) available_bytes: Option<u64>,
    pub(crate) reason: String,
}

/// Validate the model format at the CLI boundary before either packaging or
/// runtime dispatch. Released SDK versions historically treated unknown legacy
/// frameworks as ONNX, so this guard is intentionally duplicated here until
/// every consumer is on the fail-closed SDK resolver.
pub(crate) fn validate_model_contract(manifest: &Manifest) -> Result<(), String> {
    EngineKind::validate(manifest)?;

    if manifest.format.is_none()
        && !matches!(
            manifest.framework.trim().to_ascii_lowercase().as_str(),
            "onnx" | "llm" | "gguf" | "native" | "safetensors"
        )
    {
        return Err(format!(
            "unsupported framework `{}`: PyTorch, TensorFlow, and unknown frameworks are not \
             ONNX models and will not be routed to ONNX Runtime; export the model to ONNX, \
             GGUF, or a supported SafeTensors deployment, or declare the converted artifact's \
             explicit format",
            manifest.framework.trim()
        ));
    }

    let extension = Path::new(&manifest.model_file)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "onnx" | "gguf" | "safetensors" => Ok(()),
        "pt" | "pth" => Err(format!(
            "model_file `{}` is a PyTorch weight file, but no PyTorch serving backend is enabled. \
             Refusing to pass it to ONNX Runtime; export it to ONNX, GGUF, or a supported \
             SafeTensors deployment first",
            manifest.model_file
        )),
        "pb" => Err(format!(
            "model_file `{}` is a TensorFlow protobuf, but no TensorFlow serving backend exists. \
             Refusing to pass it to ONNX Runtime; export it to ONNX first",
            manifest.model_file
        )),
        "" => Err(format!(
            "model_file `{}` has no extension; expected `.onnx`, `.gguf`, or `.safetensors`",
            manifest.model_file
        )),
        other => Err(format!(
            "model_file `{}` has unsupported extension `.{other}`; expected `.onnx`, `.gguf`, or `.safetensors`",
            manifest.model_file
        )),
    }
}

/// Read the optional policy without treating its absence as `auto`.
///
/// Absence is a distinct legacy state so adding this feature cannot redirect an
/// existing SafeTensors package from its current in-process backend to vLLM.
pub(crate) fn manifest_serving_backend(
    manifest: &Manifest,
) -> Result<Option<ServingBackendPolicy>, String> {
    let Some(metadata) = manifest.metadata.as_ref() else {
        return Ok(None);
    };
    let Some(serving) = metadata.get(SERVING_METADATA_KEY) else {
        return Ok(None);
    };
    let serving = serving.as_mapping().ok_or_else(|| {
        "metadata.serving must be an object containing an optional `backend` string".to_string()
    })?;
    let key = serde_yaml::Value::String(BACKEND_METADATA_KEY.to_string());
    let Some(value) = serving.get(&key) else {
        return Ok(None);
    };
    let value = value.as_str().ok_or_else(|| {
        "metadata.serving.backend must be a string: `auto`, `llama_cpp`, or `vllm`".to_string()
    })?;
    ServingBackendPolicy::parse_manifest_value(value).map(Some)
}

fn is_vllm_model_contract(manifest: &Manifest) -> bool {
    manifest
        .format
        .as_deref()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("safetensors"))
        && manifest
            .model_type
            .as_deref()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("causal-lm"))
        && manifest
            .task
            .as_deref()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("generate"))
}

/// Validate model-policy compatibility without making a hardware decision.
pub(crate) fn validate_serving_backend_declaration(manifest: &Manifest) -> Result<(), String> {
    match manifest_serving_backend(manifest)? {
        None | Some(ServingBackendPolicy::Auto) => Ok(()),
        Some(ServingBackendPolicy::LlamaCpp) if EngineKind::resolve(manifest).is_gguf() => Ok(()),
        Some(ServingBackendPolicy::LlamaCpp) => Err(format!(
            "metadata.serving.backend `llama_cpp` requires a GGUF package; model `{}` resolves to {}",
            manifest.project_name,
            EngineKind::resolve(manifest).label()
        )),
        Some(ServingBackendPolicy::Vllm) if is_vllm_model_contract(manifest) => Ok(()),
        Some(ServingBackendPolicy::Vllm) => Err(format!(
            "metadata.serving.backend `vllm` requires explicit format=`safetensors`, model_type=`causal-lm`, and task=`generate` axes for model `{}`",
            manifest.project_name
        )),
    }
}

/// Resolve a package policy for the hardware visible to the deployment.
pub(crate) fn resolve_serving_backend(
    manifest: &Manifest,
    has_cuda: bool,
) -> Result<ServingBackendDecision, String> {
    validate_serving_backend_declaration(manifest)?;
    let requested = manifest_serving_backend(manifest)?;
    let kind = EngineKind::resolve(manifest);

    let (selected, reason) = match requested {
        None => (
            ResolvedServingBackend::Builtin,
            "legacy package: preserve the existing backend factory",
        ),
        Some(ServingBackendPolicy::LlamaCpp) => (
            ResolvedServingBackend::LlamaCpp,
            "package explicitly requires llama.cpp",
        ),
        Some(ServingBackendPolicy::Vllm) if has_cuda => (
            ResolvedServingBackend::Vllm,
            "package explicitly requires vLLM on CUDA",
        ),
        Some(ServingBackendPolicy::Vllm) => {
            return Err(format!(
                "model `{}` requires serving backend `vllm`, but no CUDA device is available",
                manifest.project_name
            ));
        }
        Some(ServingBackendPolicy::Auto) if kind.is_gguf() => (
            ResolvedServingBackend::LlamaCpp,
            "auto policy: GGUF is served by llama.cpp",
        ),
        Some(ServingBackendPolicy::Auto) if is_vllm_model_contract(manifest) && has_cuda => (
            ResolvedServingBackend::Vllm,
            "auto policy: CUDA SafeTensors causal-LM generation is served by vLLM",
        ),
        Some(ServingBackendPolicy::Auto) => (
            ResolvedServingBackend::Builtin,
            "auto policy: no external LLM backend matches; use the built-in backend factory",
        ),
    };

    Ok(ServingBackendDecision {
        requested,
        selected,
        reason,
    })
}

/// Enforce the execution boundary of the current runtime build.
pub(crate) fn validate_runtime_serving_backend(
    manifest: &Manifest,
    decision: ServingBackendDecision,
) -> Result<(), String> {
    if decision.selected == ResolvedServingBackend::Vllm
        && !cfg!(all(feature = "gpu-device-pool", target_os = "linux"))
    {
        return Err(format!(
            "selected managed serving backend `vllm` ({}), but this Kapsl binary is not a Linux gpu-device-pool build. Use the Kapsl CUDA distribution; refusing to fall back to a different backend.",
            decision.reason
        ));
    }
    if decision.selected == ResolvedServingBackend::LlamaCpp && cfg!(feature = "gguf-native") {
        return Err(
            "selected `llama_cpp`, but this binary was compiled with `gguf-native`, which replaces llama.cpp compute. Use the stable default/`cuda` runtime profile or remove the explicit policy after reviewing the backend change."
                .to_string(),
        );
    }
    if decision.selected == ResolvedServingBackend::Builtin
        && EngineKind::resolve(manifest) == EngineKind::Native
        && !cfg!(feature = "native")
    {
        return Err(
            "SafeTensors selected the built-in native backend, but this binary was compiled \
             without the `native` feature. Refusing to pass SafeTensors weights to ONNX Runtime; \
             use a native-enabled binary or select the certified managed vLLM deployment"
                .to_string(),
        );
    }
    Ok(())
}

/// Merge a CLI policy override into an arbitrary package metadata object.
pub(crate) fn apply_serving_backend_override(
    metadata: Option<serde_json::Value>,
    backend: Option<ServingBackendPolicy>,
) -> Result<Option<serde_json::Value>, String> {
    let Some(backend) = backend else {
        return Ok(metadata);
    };

    let mut metadata = metadata.unwrap_or_else(|| serde_json::json!({}));
    let metadata_object = metadata.as_object_mut().ok_or_else(|| {
        "package metadata must be a JSON object when --serving-backend is used".to_string()
    })?;
    let serving = metadata_object
        .entry(SERVING_METADATA_KEY.to_string())
        .or_insert_with(|| serde_json::json!({}));
    let serving_object = serving.as_object_mut().ok_or_else(|| {
        "metadata.serving must be a JSON object when --serving-backend is used".to_string()
    })?;
    serving_object.insert(
        BACKEND_METADATA_KEY.to_string(),
        serde_json::Value::String(backend.as_str().to_string()),
    );
    Ok(Some(metadata))
}

fn decode_bounded_manifest<R: Read>(
    reader: R,
    declared_size: Option<u64>,
) -> Result<Manifest, String> {
    if declared_size.is_some_and(|size| size > BACKEND_PLAN_MANIFEST_MAX_BYTES) {
        return Err(format!(
            "metadata.json exceeds the {} byte backend-planning limit",
            BACKEND_PLAN_MANIFEST_MAX_BYTES
        ));
    }
    let mut bytes = Vec::new();
    reader
        .take(BACKEND_PLAN_MANIFEST_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Failed to read metadata.json: {error}"))?;
    if bytes.len() as u64 > BACKEND_PLAN_MANIFEST_MAX_BYTES {
        return Err(format!(
            "metadata.json exceeds the {} byte backend-planning limit",
            BACKEND_PLAN_MANIFEST_MAX_BYTES
        ));
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("Invalid metadata.json: {error}"))
}

/// Read only the manifest needed for a policy decision.
///
/// In particular, this does not invoke `PackageLoader::load`, populate the
/// model cache, or honor discard-after-load settings.
pub(crate) fn inspect_serving_manifest(path: &Path) -> Result<Manifest, String> {
    if path.is_dir() {
        let metadata_path = path.join("metadata.json");
        if metadata_path.exists() {
            let file = File::open(&metadata_path)
                .map_err(|error| format!("Failed to open {}: {error}", metadata_path.display()))?;
            let size = file.metadata().ok().map(|metadata| metadata.len());
            return decode_bounded_manifest(file, size);
        }
        return PackageLoader::from_directory(path)
            .map(|loader| loader.manifest)
            .map_err(|error| format!("Failed to inspect model directory: {error}"));
    }

    if looks_like_model_file_path(path) {
        return PackageLoader::from_raw_file(path)
            .map(|loader| loader.manifest)
            .map_err(|error| format!("Failed to inspect raw model: {error}"));
    }

    let file = File::open(path)
        .map_err(|error| format!("Failed to open package {}: {error}", path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|error| format!("Failed to read package {}: {error}", path.display()))?;
    for entry in entries {
        let mut entry = entry.map_err(|error| {
            format!(
                "Failed to read package entry in {}: {error}",
                path.display()
            )
        })?;
        let entry_path = entry.path().map_err(|error| {
            format!(
                "Failed to read package entry path in {}: {error}",
                path.display()
            )
        })?;
        if entry_path.as_ref() == Path::new("metadata.json") {
            let size = entry.size();
            return decode_bounded_manifest(&mut entry, Some(size));
        }
    }
    Err(format!(
        "Package {} does not contain metadata.json",
        path.display()
    ))
}

/// A host-only gate used before a multi-gigabyte backend download. The normal
/// memory authority still performs the final transactional admission directly
/// before model initialization; this estimate exists only to reject obvious
/// over-capacity vLLM deployments early.
pub(crate) fn preliminary_memory_admission(
    model_path: &Path,
    manifest: &Manifest,
    decision: ServingBackendDecision,
    device_info: &DeviceInfo,
) -> Result<PreliminaryMemoryAdmission, String> {
    if decision.selected != ResolvedServingBackend::Vllm {
        return Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Accepted,
            required_bytes: None,
            available_bytes: None,
            reason:
                "the selected in-process backend performs authoritative admission at model load"
                    .to_string(),
        });
    }

    let weight_bytes = inspect_model_weight_bytes(model_path, manifest)?;
    let declared_bytes = manifest
        .hardware_requirements
        .gpu_memory_reservation_mb
        .or(manifest.hardware_requirements.gpu_memory_limit_mb)
        .or(manifest.hardware_requirements.min_vram_mb)
        .map(|mib| mib.saturating_mul(1024 * 1024));
    let estimated_bytes = weight_bytes.map(|weights| {
        let workspace = (weights / 8).max(256 * 1024 * 1024);
        weights.saturating_add(workspace)
    });
    let required = match (declared_bytes, estimated_bytes) {
        (Some(declared), Some(estimated)) => Some(declared.max(estimated)),
        (Some(declared), None) => Some(declared),
        (None, estimated) => estimated,
    };
    let allowed_ids = &manifest.hardware_requirements.gpu_device_ids;
    let requested_id = manifest.hardware_requirements.device_id;
    let available = device_info
        .devices
        .iter()
        .filter(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
        .filter(|device| {
            (allowed_ids.is_empty() || allowed_ids.contains(&(device.id as i32)))
                && requested_id.is_none_or(|id| id == device.id as i32)
        })
        // Keep a conservative ten-percent driver/runtime guard. The runtime's
        // device authority may apply a stricter configured ceiling later.
        .map(|device| {
            device
                .memory_mb
                .saturating_mul(1024 * 1024)
                .saturating_mul(9)
                / 10
        })
        .max();

    match (required, available) {
        (Some(required), Some(available)) if required > available => {
            Ok(PreliminaryMemoryAdmission {
                status: MemoryAdmissionStatus::Rejected,
                required_bytes: Some(required),
                available_bytes: Some(available),
                reason: format!(
                    "estimated model weights and workspace require {required} bytes, but the largest eligible GPU has {available} guarded bytes"
                ),
            })
        }
        (Some(required), Some(available)) => Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Accepted,
            required_bytes: Some(required),
            available_bytes: Some(available),
            reason: format!(
                "estimated model weights and workspace require {required} bytes within a guarded {available} byte GPU budget"
            ),
        }),
        (_, None) => Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Rejected,
            required_bytes: required,
            available_bytes: None,
            reason: "no eligible CUDA GPU is available".to_string(),
        }),
        (None, Some(available)) => Ok(PreliminaryMemoryAdmission {
            status: MemoryAdmissionStatus::Unknown,
            required_bytes: None,
            available_bytes: Some(available),
            reason: "the package did not expose enough size information for a host-only estimate"
                .to_string(),
        }),
    }
}

pub(crate) fn inspect_model_weight_bytes(
    path: &Path,
    manifest: &Manifest,
) -> Result<Option<u64>, String> {
    if path.is_dir() {
        return sum_safetensors_in_directory(path).map(Some);
    }
    if looks_like_model_file_path(path) {
        return fs::metadata(path)
            .map(|metadata| Some(metadata.len()))
            .map_err(|error| format!("Failed to inspect model size {}: {error}", path.display()));
    }

    let file = File::open(path)
        .map_err(|error| format!("Failed to open package {}: {error}", path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let mut bytes = 0_u64;
    let mut found = false;
    let entries = archive
        .entries()
        .map_err(|error| format!("Failed to inspect package {}: {error}", path.display()))?;
    for (index, entry) in entries.enumerate() {
        if index >= 100_000 {
            return Err("Package contains too many entries for backend planning".to_string());
        }
        let entry = entry.map_err(|error| format!("Failed to inspect package entry: {error}"))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let entry_path = entry
            .path()
            .map_err(|error| format!("Failed to inspect package entry path: {error}"))?;
        let is_safetensors = entry_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("safetensors"));
        let is_declared_model = entry_path.as_ref() == Path::new(&manifest.model_file);
        if is_safetensors || is_declared_model {
            found = true;
            bytes = bytes.saturating_add(entry.size());
        }
    }
    Ok(found.then_some(bytes))
}

fn sum_safetensors_in_directory(path: &Path) -> Result<u64, String> {
    fn visit(path: &Path, depth: usize, total: &mut u64) -> Result<(), String> {
        if depth > 8 {
            return Err("Model directory nesting exceeds the planning limit".to_string());
        }
        for entry in fs::read_dir(path)
            .map_err(|error| format!("Failed to inspect {}: {error}", path.display()))?
        {
            let entry = entry.map_err(|error| format!("Failed to inspect model entry: {error}"))?;
            let metadata = entry.metadata().map_err(|error| {
                format!("Failed to inspect {}: {error}", entry.path().display())
            })?;
            if metadata.is_dir() {
                visit(&entry.path(), depth + 1, total)?;
            } else if metadata.is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case("safetensors"))
            {
                *total = total.saturating_add(metadata.len());
            }
        }
        Ok(())
    }
    let mut total = 0;
    visit(path, 0, &mut total)?;
    Ok(total)
}

pub(crate) fn execute_backend_plan_command(args: BackendPlanCommandArgs) -> Result<(), DynError> {
    let absolute_path = args.model.canonicalize().map_err(|error| {
        format!(
            "Invalid model path {}: {error}",
            args.model.to_string_lossy()
        )
    })?;
    let manifest = inspect_serving_manifest(&absolute_path)?;
    validate_model_contract(&manifest)?;
    let (has_cuda, cuda_source) = match args.cuda {
        Some(value) => (value, "override"),
        None => (DeviceInfo::probe().has_cuda, "detected"),
    };
    let decision = resolve_serving_backend(&manifest, has_cuda)?;
    let mut planned_device_info = DeviceInfo::probe();
    planned_device_info.has_cuda = has_cuda;
    let mut memory =
        preliminary_memory_admission(&absolute_path, &manifest, decision, &planned_device_info)?;
    let synthetic_cuda_target = args.cuda == Some(true)
        && !planned_device_info
            .devices
            .iter()
            .any(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"));
    if synthetic_cuda_target && memory.status == MemoryAdmissionStatus::Rejected {
        memory.status = MemoryAdmissionStatus::Unknown;
        memory.reason = "CUDA was enabled by --cuda, but this host exposes no GPU capacity; final memory admission must run on the target host".to_string();
    }
    let onnx_profile = if crate::onnx_backend_pack::lazy_onnx_packs_enabled() {
        match args.cuda {
            Some(true) => crate::onnx_backend_pack::onnx_pack_profile_for_target(
                &manifest,
                BackendAccelerator::Cuda,
            )?,
            Some(false) => crate::onnx_backend_pack::onnx_pack_profile_for_target(
                &manifest,
                BackendAccelerator::Cpu,
            )?,
            None => crate::onnx_backend_pack::onnx_pack_profile_for_manifest(
                &manifest,
                &planned_device_info,
            )?,
        }
    } else {
        None
    };
    let llama_profile = if crate::llama_cpp_backend_pack::lazy_llama_cpp_packs_enabled() {
        match args.cuda {
            Some(true) => crate::llama_cpp_backend_pack::llama_cpp_pack_profile_for_target(
                &manifest,
                BackendAccelerator::Cuda,
            ),
            Some(false) => crate::llama_cpp_backend_pack::llama_cpp_pack_profile_for_target(
                &manifest,
                BackendAccelerator::Cpu,
            ),
            None => crate::llama_cpp_backend_pack::llama_cpp_pack_profile_for_manifest(
                &manifest,
                &planned_device_info,
            ),
        }
    } else {
        None
    };
    let (selected_backend, installed, download_required, download_bytes, execution_mode, profile) =
        if decision.selected == ResolvedServingBackend::Vllm
            && memory.status != MemoryAdmissionStatus::Rejected
        {
            let mut target = BackendTarget::current(&planned_device_info);
            if synthetic_cuda_target {
                // `--cuda true` is a contract-planning override, so allow the
                // signed CUDA profile to resolve without pretending that this
                // host measured a particular target driver/runtime version.
                target.cuda_version = Some("999.0".to_string());
                target.driver_version = Some("9999.0".to_string());
            }
            let manager = BackendManager::from_env(args.offline)?;
            let plan = manager.plan_vllm(&target)?;
            (
                "vllm".to_string(),
                plan.installed,
                plan.download_required,
                plan.download_bytes,
                plan.execution_mode,
                Some(plan.profile),
            )
        } else if decision.selected == ResolvedServingBackend::Vllm {
            (
                "vllm".to_string(),
                false,
                false,
                0,
                "external".to_string(),
                None,
            )
        } else if let Some(onnx_profile) = onnx_profile {
            let mut target = BackendTarget::current(&planned_device_info);
            target.accelerator = onnx_profile.accelerator();
            if onnx_profile == OnnxBackendPackProfile::Cpu {
                target.cuda_version = None;
                target.driver_version = None;
            } else if synthetic_cuda_target {
                target.cuda_version = Some("999.0".to_string());
                target.driver_version = Some("9999.0".to_string());
            }
            let manager = BackendManager::from_env(args.offline)?;
            let plan = manager.plan_onnx(onnx_profile, &target)?;
            (
                "onnx".to_string(),
                plan.installed,
                plan.download_required,
                plan.download_bytes,
                plan.execution_mode,
                Some(plan.profile),
            )
        } else if let Some(llama_profile) = llama_profile {
            let mut target = BackendTarget::current(&planned_device_info);
            target.accelerator = llama_profile.accelerator();
            if llama_profile == LlamaCppBackendPackProfile::Cpu {
                target.cuda_version = None;
                target.driver_version = None;
            } else if synthetic_cuda_target {
                target.cuda_version = Some("999.0".to_string());
                target.driver_version = Some("9999.0".to_string());
            }
            let manager = BackendManager::from_env(args.offline)?;
            let plan = manager.plan_llama_cpp(llama_profile, &target)?;
            memory = crate::llama_cpp_backend_pack::preliminary_llama_cpp_memory_admission(
                llama_profile,
                &absolute_path,
                &manifest,
                &planned_device_info,
                &plan.manifest,
                None,
            )?;
            if synthetic_cuda_target && memory.status == MemoryAdmissionStatus::Rejected {
                memory.status = MemoryAdmissionStatus::Unknown;
                memory.reason = "CUDA was enabled by --cuda, but this host exposes no GPU capacity; final memory admission must run on the target host".to_string();
            }
            let admitted = memory.status != MemoryAdmissionStatus::Rejected;
            (
                "llama_cpp".to_string(),
                plan.installed,
                admitted && plan.download_required,
                if admitted { plan.download_bytes } else { 0 },
                plan.execution_mode,
                Some(plan.profile),
            )
        } else {
            (
                decision.selected.as_str().to_string(),
                true,
                false,
                0,
                "native".to_string(),
                None,
            )
        };
    let policy = decision
        .requested
        .map(ServingBackendPolicy::as_str)
        .unwrap_or("legacy");
    let output = serde_json::json!({
        "model": manifest.project_name,
        "policy": policy,
        "selected_backend": selected_backend,
        "profile": profile,
        "installed": installed,
        "download_required": download_required,
        "download_bytes": download_bytes,
        "memory_admission": memory.status,
        "memory_required_bytes": memory.required_bytes,
        "memory_available_bytes": memory.available_bytes,
        "memory_reason": memory.reason,
        "execution_mode": execution_mode,
        "external_process": decision.selected == ResolvedServingBackend::Vllm,
        "cuda": has_cuda,
        "cuda_source": cuda_source,
        "reason": decision.reason,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("Failed to encode backend plan: {error}"))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend};

    fn manifest(format: &str, model_type: &str, task: &str) -> Manifest {
        Manifest {
            project_name: "test-model".to_string(),
            framework: if format == "gguf" {
                "gguf".to_string()
            } else {
                "safetensors".to_string()
            },
            version: "1.0.0".to_string(),
            created_at: String::new(),
            model_file: format!("model.{format}"),
            format: Some(format.to_string()),
            model_type: Some(model_type.to_string()),
            task: Some(task.to_string()),
            metadata: None,
            hardware_requirements: kapsl_core::HardwareRequirements::default(),
            cron_jobs: Vec::new(),
        }
    }

    fn set_policy(manifest: &mut Manifest, backend: &str) {
        manifest.metadata = Some(
            serde_yaml::from_str(&format!("serving:\n  backend: {backend}\n"))
                .expect("policy metadata"),
        );
    }

    #[test]
    fn legacy_package_preserves_builtin_selection() {
        let decision =
            resolve_serving_backend(&manifest("safetensors", "causal-lm", "generate"), true)
                .expect("legacy decision");
        assert_eq!(decision.requested, None);
        assert_eq!(decision.selected, ResolvedServingBackend::Builtin);
    }

    #[test]
    fn auto_selects_llama_cpp_for_gguf() {
        let mut manifest = manifest("gguf", "causal-lm", "generate");
        set_policy(&mut manifest, "auto");
        let decision = resolve_serving_backend(&manifest, true).expect("GGUF decision");
        assert_eq!(decision.selected, ResolvedServingBackend::LlamaCpp);
    }

    #[test]
    fn auto_selects_vllm_only_for_cuda_safetensors_generation() {
        let mut manifest = manifest("safetensors", "causal-lm", "generate");
        set_policy(&mut manifest, "auto");
        assert_eq!(
            resolve_serving_backend(&manifest, true)
                .expect("CUDA decision")
                .selected,
            ResolvedServingBackend::Vllm
        );
        assert_eq!(
            resolve_serving_backend(&manifest, false)
                .expect("CPU decision")
                .selected,
            ResolvedServingBackend::Builtin
        );
    }

    #[test]
    fn explicit_backends_reject_incompatible_models() {
        let mut safetensors = manifest("safetensors", "causal-lm", "generate");
        set_policy(&mut safetensors, "llama_cpp");
        assert!(validate_serving_backend_declaration(&safetensors)
            .unwrap_err()
            .contains("requires a GGUF package"));

        let mut gguf = manifest("gguf", "causal-lm", "generate");
        set_policy(&mut gguf, "vllm");
        assert!(validate_serving_backend_declaration(&gguf)
            .unwrap_err()
            .contains("requires explicit format=`safetensors`"));
    }

    #[test]
    fn malformed_policy_fails_instead_of_falling_back() {
        let mut manifest = manifest("gguf", "causal-lm", "generate");
        set_policy(&mut manifest, "something_else");
        assert!(manifest_serving_backend(&manifest)
            .unwrap_err()
            .contains("unknown metadata.serving.backend"));
    }

    #[test]
    fn unsupported_framework_and_weight_extensions_fail_before_onnx() {
        let mut pytorch = manifest("safetensors", "causal-lm", "generate");
        pytorch.framework = "pytorch".to_string();
        pytorch.format = None;
        pytorch.model_type = None;
        pytorch.task = None;
        pytorch.model_file = "model.pt".to_string();
        let error = validate_model_contract(&pytorch).unwrap_err();
        assert!(
            error.contains("unsupported framework"),
            "unexpected: {error}"
        );
        assert!(error.contains("will not be routed to ONNX Runtime"));

        let mut mislabeled = manifest("safetensors", "causal-lm", "generate");
        mislabeled.framework = "onnx".to_string();
        mislabeled.format = None;
        mislabeled.model_type = None;
        mislabeled.task = None;
        mislabeled.model_file = "model.pth".to_string();
        let error = validate_model_contract(&mislabeled).unwrap_err();
        assert!(error.contains("PyTorch weight file"), "unexpected: {error}");
        assert!(error.contains("Refusing to pass it to ONNX Runtime"));
    }

    #[test]
    fn explicit_onnx_format_accepts_a_converted_pytorch_source_label() {
        let mut converted = manifest("safetensors", "opaque", "forward");
        converted.framework = "pytorch".to_string();
        converted.format = Some("onnx".to_string());
        converted.model_file = "converted.onnx".to_string();
        assert!(validate_model_contract(&converted).is_ok());
    }

    #[test]
    fn runtime_requires_the_managed_vllm_build_contract() {
        let decision = ServingBackendDecision {
            requested: Some(ServingBackendPolicy::Vllm),
            selected: ResolvedServingBackend::Vllm,
            reason: "test selection",
        };
        let manifest = manifest("safetensors", "causal-lm", "generate");
        let result = validate_runtime_serving_backend(&manifest, decision);
        if cfg!(all(feature = "gpu-device-pool", target_os = "linux")) {
            assert!(result.is_ok());
        } else {
            let error = result.unwrap_err();
            assert!(error.contains("not a Linux gpu-device-pool build"));
            assert!(error.contains("refusing to fall back"));
        }
    }

    #[cfg(not(feature = "native"))]
    #[test]
    fn in_process_loader_rejects_safetensors_without_native_feature() {
        let manifest = manifest("safetensors", "opaque", "forward");
        let decision = ServingBackendDecision {
            requested: None,
            selected: ResolvedServingBackend::Builtin,
            reason: "legacy package",
        };
        let error = validate_runtime_serving_backend(&manifest, decision).unwrap_err();
        assert!(error.contains("without the `native` feature"));
        assert!(error.contains("Refusing to pass SafeTensors weights to ONNX Runtime"));
    }

    #[test]
    fn cli_override_preserves_other_serving_metadata() {
        let metadata = serde_json::json!({
            "owner": "test",
            "serving": {"max_context": 4096, "backend": "auto"}
        });
        let merged =
            apply_serving_backend_override(Some(metadata), Some(ServingBackendPolicy::LlamaCpp))
                .expect("merge")
                .expect("metadata");
        assert_eq!(merged["owner"], "test");
        assert_eq!(merged["serving"]["max_context"], 4096);
        assert_eq!(merged["serving"]["backend"], "llama_cpp");
    }

    #[test]
    fn preliminary_vllm_admission_rejects_an_undersized_gpu() {
        let root = tempfile::tempdir().unwrap();
        let model_path = root.path().join("model.safetensors");
        fs::write(&model_path, b"small fixture").unwrap();
        let mut manifest = manifest("safetensors", "causal-lm", "generate");
        set_policy(&mut manifest, "vllm");
        let decision = resolve_serving_backend(&manifest, true).unwrap();
        let device_info = DeviceInfo {
            cpu_cores: 1,
            total_memory: 1024 * 1024 * 1024,
            os_type: "linux".to_string(),
            os_release: "test".to_string(),
            has_cuda: true,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![Device {
                id: 0,
                name: "fixture GPU".to_string(),
                backend: DeviceBackend::Cuda,
                memory_mb: 128,
                compute_units: 1,
                pci_bus_id: None,
                partition_id: None,
                driver_version: Some("999.0".to_string()),
                cuda_version: Some("999.0".to_string()),
                compute_capability: Some("9.0".to_string()),
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
            }],
        };

        let admission =
            preliminary_memory_admission(&model_path, &manifest, decision, &device_info).unwrap();
        assert_eq!(admission.status, MemoryAdmissionStatus::Rejected);
        assert!(admission.required_bytes.unwrap() > admission.available_bytes.unwrap());
    }
}
