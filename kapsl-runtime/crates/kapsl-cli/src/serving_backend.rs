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
    let policy = decision
        .requested
        .map(ServingBackendPolicy::as_str)
        .unwrap_or("legacy");
    let output = serde_json::json!({
        "model": manifest.project_name,
        "policy": policy,
        "selected_backend": decision.selected.as_str(),
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
}
