//! Runtime configuration resolved independently of the CLI surface.

use super::*;

mod auto_sizing;
mod environment;
mod load_policy;
mod system;

pub(crate) use auto_sizing::*;
pub(crate) use environment::*;
pub(crate) use load_policy::*;
pub(crate) use system::*;

pub(crate) fn parse_env_bool(key: &str) -> Option<bool> {
    let value = optional_env_var(key)?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

pub(crate) fn yaml_lookup<'a>(
    value: &'a serde_yaml::Value,
    path: &[&str],
) -> Option<&'a serde_yaml::Value> {
    let mut current = value;
    for key in path {
        let mapping = match current {
            serde_yaml::Value::Mapping(mapping) => mapping,
            _ => return None,
        };
        current = mapping.get(*key)?;
    }
    Some(current)
}

pub(crate) fn yaml_bool(value: &serde_yaml::Value) -> Option<bool> {
    match value {
        serde_yaml::Value::Bool(val) => Some(*val),
        serde_yaml::Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn yaml_u32(value: &serde_yaml::Value) -> Option<u32> {
    match value {
        serde_yaml::Value::Number(number) => number.as_u64().and_then(|v| u32::try_from(v).ok()),
        serde_yaml::Value::String(text) => text.trim().parse::<u32>().ok(),
        _ => None,
    }
}

pub(crate) fn rag_storage_root() -> PathBuf {
    optional_env_var(RAG_STORAGE_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("rag-data"))
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeStateLayout {
    pub(crate) rag_root: PathBuf,
    pub(crate) extensions_root: PathBuf,
    pub(crate) extensions_config_root: PathBuf,
    pub(crate) auth_store_path: PathBuf,
}

pub(crate) fn resolve_runtime_state_layout(state_dir: Option<&Path>) -> RuntimeStateLayout {
    if let Some(state_dir) = state_dir {
        RuntimeStateLayout {
            rag_root: state_dir.join("rag-data"),
            extensions_root: state_dir.join("extensions"),
            extensions_config_root: state_dir.join("extensions-config"),
            auth_store_path: state_dir.join(DEFAULT_AUTH_STORE_FILENAME),
        }
    } else {
        let extensions_root = PathBuf::from(
            optional_env_var(EXTENSIONS_ROOT_ENV).unwrap_or_else(|| "extensions".to_string()),
        );
        let extensions_config_root = PathBuf::from(
            optional_env_var(EXT_CONFIG_ROOT_ENV)
                .unwrap_or_else(|| "extensions-config".to_string()),
        );
        RuntimeStateLayout {
            rag_root: rag_storage_root(),
            extensions_root,
            extensions_config_root,
            auth_store_path: resolve_auth_store_path(),
        }
    }
}

pub(crate) fn parse_expected_dtype(value: Option<&String>) -> Option<TensorDtype> {
    let raw = value?;
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("unknown") {
        return None;
    }
    trimmed.parse::<TensorDtype>().ok()
}

pub(crate) fn validate_tensor_against_model_spec(
    label: &str,
    tensor: &BinaryTensorPacket,
    expected_shape: &[i64],
    expected_dtype: Option<TensorDtype>,
) -> Result<(), String> {
    if let Some(expected_dtype) = expected_dtype {
        if tensor.dtype != expected_dtype {
            return Err(format!(
                "{} dtype mismatch: expected `{}`, got `{}`",
                label, expected_dtype, tensor.dtype
            ));
        }
    }

    if expected_shape.is_empty() {
        return Ok(());
    }

    if tensor.shape.len() != expected_shape.len() {
        return Err(format!(
            "{} rank mismatch: expected {} dims {:?}, got {} dims {:?}",
            label,
            expected_shape.len(),
            expected_shape,
            tensor.shape.len(),
            tensor.shape
        ));
    }

    for (index, (actual, expected)) in tensor.shape.iter().zip(expected_shape.iter()).enumerate() {
        // <= 0 is treated as dynamic/unknown in model metadata.
        if *expected <= 0 {
            continue;
        }
        if actual != expected {
            return Err(format!(
                "{} shape mismatch at dim {}: expected {}, got {} (expected shape {:?}, got {:?})",
                label, index, expected, actual, expected_shape, tensor.shape
            ));
        }
    }

    Ok(())
}

pub(crate) fn validate_inference_request_against_model_info(
    request: &InferenceRequest,
    model_info: &EngineModelInfo,
) -> Result<(), String> {
    if model_info.input_names.is_empty() {
        return Ok(());
    }

    let mut input_index: HashMap<&str, usize> = HashMap::new();
    for (index, name) in model_info.input_names.iter().enumerate() {
        input_index.insert(name.as_str(), index);
    }

    let primary_name = model_info.input_names[0].as_str();
    let primary_shape = model_info.input_shapes.first().cloned().unwrap_or_default();
    let primary_dtype = parse_expected_dtype(model_info.input_dtypes.first());
    validate_tensor_against_model_spec(
        &format!("primary input `{}`", primary_name),
        &request.input,
        &primary_shape,
        primary_dtype,
    )?;

    for additional in &request.additional_inputs {
        let Some(&index) = input_index.get(additional.name.as_str()) else {
            return Err(format!(
                "unknown additional input `{}`. Model inputs: {}",
                additional.name,
                model_info.input_names.join(", ")
            ));
        };
        let expected_shape = model_info
            .input_shapes
            .get(index)
            .cloned()
            .unwrap_or_default();
        let expected_dtype = parse_expected_dtype(model_info.input_dtypes.get(index));
        validate_tensor_against_model_spec(
            &format!("additional input `{}`", additional.name),
            &additional.tensor,
            &expected_shape,
            expected_dtype,
        )?;
    }

    Ok(())
}

pub(crate) fn manifest_llm_flag(manifest: &Manifest, key: &str) -> Option<bool> {
    let meta = manifest.metadata.as_ref()?;
    let value = yaml_lookup(meta, &["llm", key])?;
    yaml_bool(value)
}

pub(crate) fn manifest_llm_pipeline_stages(manifest: &Manifest) -> Option<Vec<String>> {
    let meta = manifest.metadata.as_ref()?;
    let value = yaml_lookup(meta, &["llm", "pipeline", "stages"])?;
    match value {
        serde_yaml::Value::Sequence(items) => {
            let stages: Vec<String> = items
                .iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect();
            if stages.is_empty() {
                None
            } else {
                Some(stages)
            }
        }
        _ => None,
    }
}

pub(crate) fn resolve_isolate_process(manifest: &Manifest) -> bool {
    if let Some(env) = parse_env_bool(LLM_ISOLATE_PROCESS_ENV) {
        return env;
    }
    if manifest_llm_flag(manifest, "isolate_process_strict").unwrap_or(false) {
        return true;
    }
    if manifest_llm_flag(manifest, "isolate_process").unwrap_or(false) {
        log::info!(
            "metadata.llm.isolate_process=true is advisory; running in-process by default. Set {}=1 or metadata.llm.isolate_process_strict=true to force worker isolation.",
            LLM_ISOLATE_PROCESS_ENV
        );
    }
    false
}

/// Whether process isolation is *required* (fail-closed). When true, a model
/// that requested isolation but whose worker cannot start fails to load instead
/// of silently falling back to in-process — which would drop the isolation
/// guarantee and rejoin the shared KV pool. Defaults to false for backward
/// compatibility (silent fallback preserved, but logged prominently).
pub(crate) fn resolve_isolate_process_strict(manifest: &Manifest) -> bool {
    if let Some(env) = parse_env_bool(LLM_ISOLATE_PROCESS_STRICT_ENV) {
        return env;
    }
    manifest_llm_flag(manifest, "isolate_process_strict").unwrap_or(false)
}

pub(crate) fn resolve_scheduler_tuning_for_framework(
    manifest: &Manifest,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
) -> (usize, u64) {
    if !EngineKind::resolve(manifest).is_onnx_generate() {
        return (scheduler_max_micro_batch, scheduler_queue_delay_ms);
    }

    if env_flag(LLM_ALLOW_SCHEDULER_MICROBATCH_ENV) {
        return (scheduler_max_micro_batch, scheduler_queue_delay_ms);
    }

    let mut resolved_micro_batch = scheduler_max_micro_batch.max(1);
    let mut resolved_queue_delay_ms = scheduler_queue_delay_ms;

    if resolved_micro_batch > 1 {
        log::info!(
            "Framework=llm: overriding scheduler_max_micro_batch {} -> 1 to avoid outer micro-batch serialization. Set {}=1 to keep configured value.",
            resolved_micro_batch,
            LLM_ALLOW_SCHEDULER_MICROBATCH_ENV
        );
        resolved_micro_batch = 1;
    }
    if resolved_queue_delay_ms > 0 {
        log::info!(
            "Framework=llm: overriding scheduler_queue_delay_ms {} -> 0 to reduce queueing delay. Set {}=1 to keep configured value.",
            resolved_queue_delay_ms,
            LLM_ALLOW_SCHEDULER_MICROBATCH_ENV
        );
        resolved_queue_delay_ms = 0;
    }

    (resolved_micro_batch, resolved_queue_delay_ms)
}

pub(crate) fn maybe_export_gguf_prefill_chunk_hint(
    model_file_path: Option<&Path>,
    batch_size: usize,
) {
    if std::env::var_os(GGUF_PREFILL_CHUNK_SIZE_ENV).is_some() {
        log::info!(
            "Framework=gguf: using explicit {} override.",
            GGUF_PREFILL_CHUNK_SIZE_ENV
        );
        return;
    }

    let Some(model_file_path) = model_file_path else {
        return;
    };
    let model_size_mb = largest_model_size_mb(&[model_file_path.to_path_buf()]);
    let available_ram_mb = available_ram_mb();
    let Some(chunk_size) =
        auto_tuned_gguf_prefill_chunk_size(model_size_mb, available_ram_mb, batch_size)
    else {
        return;
    };

    std::env::set_var(GGUF_PREFILL_CHUNK_SIZE_ENV, chunk_size.to_string());
    log::info!(
        "Framework=gguf: setting {}={} from model_size={}MB and available_ram={}MB.",
        GGUF_PREFILL_CHUNK_SIZE_ENV,
        chunk_size,
        model_size_mb,
        available_ram_mb
    );
}

/// Set once the runtime is serving, after which the GGUF auto-sizing hints must
/// not touch the process environment.
static ENV_AUTO_SIZING_SEALED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Close the window in which GGUF auto-sizing may write to the environment.
///
/// The GGUF knobs are published to `kapsl-llm` as environment variables rather
/// than passed as config, so exporting them means calling `std::env::set_var`.
/// That is not thread-safe: a write racing another thread's `std::env::var` is
/// undefined behaviour (hence `unsafe` in edition 2024). Startup is the only
/// point where the runtime controls which threads exist, so once the model-load
/// phase is done we seal the environment and later loads inherit whatever
/// startup resolved.
pub(crate) fn seal_env_auto_sizing() {
    ENV_AUTO_SIZING_SEALED.store(true, std::sync::atomic::Ordering::SeqCst);
}

fn env_auto_sizing_sealed() -> bool {
    ENV_AUTO_SIZING_SEALED.load(std::sync::atomic::Ordering::SeqCst)
}

/// Reopen the window. Tests share a process, so a test that seals would
/// otherwise poison every test that runs after it.
#[cfg(test)]
pub(crate) fn unseal_env_auto_sizing() {
    ENV_AUTO_SIZING_SEALED.store(false, std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn export_gguf_auto_sizing_hint(
    manifest: &Manifest,
    batch_size: usize,
    model_file_path: Option<&Path>,
) {
    if !EngineKind::resolve(manifest).is_gguf() {
        return;
    }
    if env_auto_sizing_sealed() {
        log::debug!(
            "Framework=gguf: runtime already serving, keeping the startup {} / {} values.",
            GGUF_PREFILL_CHUNK_SIZE_ENV,
            GGUF_TARGET_CONCURRENCY_ENV
        );
        return;
    }
    maybe_export_gguf_prefill_chunk_hint(model_file_path, batch_size);

    if std::env::var_os(GGUF_MAX_CONCURRENT_ENV).is_some()
        || std::env::var_os(GGUF_TARGET_CONCURRENCY_ENV).is_some()
    {
        log::info!(
            "Framework=gguf: using explicit {} / {} concurrency override.",
            GGUF_MAX_CONCURRENT_ENV,
            GGUF_TARGET_CONCURRENCY_ENV
        );
        return;
    }

    let target = batch_size.max(1);
    std::env::set_var(GGUF_TARGET_CONCURRENCY_ENV, target.to_string());
    log::info!(
        "Framework=gguf: setting {}={} from runtime batch_size.",
        GGUF_TARGET_CONCURRENCY_ENV,
        target
    );
}

pub(crate) fn parse_priority_weight_override(raw: &str, model_id: u32) -> Option<u32> {
    for entry in raw.split(',') {
        let Some((selector, value)) = entry.split_once('=') else {
            continue;
        };
        let selector = selector.trim();
        let selector_matches = selector == "*" || selector.parse::<u32>().ok() == Some(model_id);
        if !selector_matches {
            continue;
        }
        if let Ok(weight) = value.trim().parse::<u32>() {
            return Some(weight.max(1));
        }
    }
    None
}

pub(crate) fn manifest_priority_weight(manifest: &Manifest) -> Option<u32> {
    let meta = manifest.metadata.as_ref()?;
    for path in [
        &["scheduling", "priority_weight"][..],
        &["runtime", "scheduling", "priority_weight"][..],
        &["llm", "priority_weight"][..],
        &["priority_weight"][..],
    ] {
        if let Some(value) = yaml_lookup(meta, path).and_then(yaml_u32) {
            return Some(value.max(1));
        }
    }
    None
}

pub(crate) fn resolve_model_priority_weight(manifest: &Manifest, model_id: u32) -> u32 {
    if let Some(raw) = optional_env_var(MODEL_PRIORITY_WEIGHTS_ENV) {
        if let Some(weight) = parse_priority_weight_override(&raw, model_id) {
            return weight;
        }
    }

    manifest_priority_weight(manifest).unwrap_or(1).max(1)
}

pub(crate) fn parse_queue_overflow_policy_literal(
    value: &str,
) -> Option<kapsl_scheduler::QueueOverflowPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "block" | "blocking" => Some(kapsl_scheduler::QueueOverflowPolicy::Block),
        "drop_newest" | "drop-newest" | "latest_only" | "latest-only" | "latest" => {
            Some(kapsl_scheduler::QueueOverflowPolicy::DropNewest)
        }
        "drop_oldest" | "drop-oldest" => Some(kapsl_scheduler::QueueOverflowPolicy::DropOldest),
        _ => None,
    }
}

pub(crate) fn manifest_queue_overflow_policy(
    manifest: &Manifest,
) -> Option<kapsl_scheduler::QueueOverflowPolicy> {
    let meta = manifest.metadata.as_ref()?;
    for path in [
        &["runtime", "server", "queue_overflow_policy"][..],
        &["runtime", "queue_overflow_policy"][..],
        &["scheduler", "queue_overflow_policy"][..],
        &["queue_overflow_policy"][..],
    ] {
        if let Some(value) = yaml_lookup(meta, path) {
            if let Some(raw) = value.as_str() {
                if let Some(policy) = parse_queue_overflow_policy_literal(raw) {
                    return Some(policy);
                }
            }
        }
    }
    None
}

pub(crate) fn resolve_queue_overflow_policy(
    manifest: &Manifest,
) -> kapsl_scheduler::QueueOverflowPolicy {
    if let Some(value) = optional_env_var(SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV) {
        if let Some(policy) = parse_queue_overflow_policy_literal(&value) {
            return policy;
        }
        log::warn!(
            "Invalid {} value '{}'; expected block|drop_newest|drop_oldest. Falling back to manifest/default.",
            SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV,
            value
        );
    }

    manifest_queue_overflow_policy(manifest).unwrap_or(kapsl_scheduler::QueueOverflowPolicy::Block)
}

pub(crate) fn log_queue_policy_caveat(policy: kapsl_scheduler::QueueOverflowPolicy) {
    if matches!(policy, kapsl_scheduler::QueueOverflowPolicy::DropOldest) {
        log::warn!(
            "Scheduler queue policy 'drop_oldest' evicts the oldest queued request when capacity is reached"
        );
    }
}

pub(crate) struct EffectiveTopologyChoice {
    pub(crate) mesh_topology: kapsl_hal::device_mesh::MeshTopology,
    pub(crate) worker_topology: &'static str,
    pub(crate) worker_tp_degree: usize,
    pub(crate) use_pipeline_backend: bool,
}

pub(crate) fn resolve_effective_topology_choice(
    manifest: &Manifest,
    requested_topology: &str,
    requested_tp_degree: usize,
    pipeline_stages: Option<&[String]>,
) -> EffectiveTopologyChoice {
    use kapsl_hal::device_mesh::MeshTopology;

    let requested = requested_topology.trim().to_ascii_lowercase();
    let requested_degree = requested_tp_degree.max(1);
    let pipeline_stage_count = pipeline_stages.map(|stages| stages.len()).unwrap_or(0);
    let pipeline_ready =
        EngineKind::resolve(manifest).is_onnx_generate() && pipeline_stage_count > 0;

    match requested.as_str() {
        "pipeline" | "pipeline-parallel" => {
            if pipeline_ready {
                if requested_degree != pipeline_stage_count {
                    log::warn!(
                        "Ignoring --tp-degree={} for pipeline mode; using metadata stage count={}",
                        requested_degree,
                        pipeline_stage_count
                    );
                }
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::PipelineParallel {
                        stages: pipeline_stage_count,
                    },
                    worker_topology: "pipeline-parallel",
                    worker_tp_degree: pipeline_stage_count,
                    use_pipeline_backend: true,
                }
            } else {
                log::warn!(
                    "Pipeline topology requested but no usable LLM pipeline metadata found; falling back to data-parallel"
                );
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::DataParallel,
                    worker_topology: "data-parallel",
                    worker_tp_degree: 1,
                    use_pipeline_backend: false,
                }
            }
        }
        "mixed" => {
            if pipeline_ready {
                log::warn!(
                    "Mixed topology is not fully implemented; using pipeline-parallel execution based on metadata stages"
                );
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::PipelineParallel {
                        stages: pipeline_stage_count,
                    },
                    worker_topology: "pipeline-parallel",
                    worker_tp_degree: pipeline_stage_count,
                    use_pipeline_backend: true,
                }
            } else {
                log::warn!(
                    "Mixed topology requested but no usable LLM pipeline metadata found; falling back to data-parallel"
                );
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::DataParallel,
                    worker_topology: "data-parallel",
                    worker_tp_degree: 1,
                    use_pipeline_backend: false,
                }
            }
        }
        "tensor-parallel" => {
            log::warn!(
                "Tensor-parallel topology is not fully implemented in backend execution; falling back to data-parallel"
            );
            EffectiveTopologyChoice {
                mesh_topology: MeshTopology::DataParallel,
                worker_topology: "data-parallel",
                worker_tp_degree: 1,
                use_pipeline_backend: false,
            }
        }
        _ => EffectiveTopologyChoice {
            mesh_topology: MeshTopology::DataParallel,
            worker_topology: "data-parallel",
            worker_tp_degree: 1,
            use_pipeline_backend: false,
        },
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MeshDeviceSelection {
    pub(crate) devices: Vec<kapsl_hal::device::Device>,
    /// Execution provider requested by policy/manifest. This is deliberately
    /// separate from the physical device backend: TensorRT executes on CUDA
    /// devices and CoreML executes on Metal devices.
    pub(crate) logical_provider: String,
}

/// An explicitly requested CUDA device that this process cannot use.
///
/// Typed so the load path can reject the request at admission with a clear
/// message instead of failing later inside a backend, and so it is never
/// mistaken for a memory denial (which the priority arbiter retries by
/// evicting lower-priority models; no eviction can make a hidden device appear).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceUnavailable {
    pub(crate) device_id: usize,
    pub(crate) detail: String,
}

impl std::fmt::Display for DeviceUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the model requests CUDA device {}, which this process cannot use: {}",
            self.device_id, self.detail
        )
    }
}

impl std::error::Error for DeviceUnavailable {}

fn is_cuda_device(device: &kapsl_hal::device::Device) -> bool {
    device.backend.to_string().eq_ignore_ascii_case("cuda")
}

/// Why a CUDA inventory device cannot be used by this process, or `None` when
/// it can (or when no CUDA mapping is installed, which is the legacy identity).
fn cuda_device_problem(
    map: Option<&kapsl_llm::cuda_ordinal::CudaOrdinalMap>,
    device: &kapsl_hal::device::Device,
) -> Option<kapsl_llm::cuda_ordinal::CudaOrdinalError> {
    if !is_cuda_device(device) {
        return None;
    }
    map?.ordinal_or_error(device.id).err()
}

/// The inventory restricted to devices this process can use: CUDA devices that
/// are hidden from the process (for example by `CUDA_VISIBLE_DEVICES`) or that
/// could not be matched to a CUDA ordinal are not managed by the memory layer
/// and must never be a placement target.
fn usable_device_info(
    device_info: &DeviceInfo,
    map: Option<&kapsl_llm::cuda_ordinal::CudaOrdinalMap>,
) -> DeviceInfo {
    let mut usable = device_info.clone();
    usable
        .devices
        .retain(|device| cuda_device_problem(map, device).is_none());
    if usable.has_cuda && !usable.devices.iter().any(is_cuda_device) {
        usable.has_cuda = false;
    }
    usable
}

fn candidate_providers(
    requirements: &kapsl_core::HardwareRequirements,
    device_info: &DeviceInfo,
) -> Vec<String> {
    let mut providers: Vec<String> = Vec::new();
    if let Some(pref) = &requirements.preferred_provider {
        providers.push(pref.clone());
    }
    providers.extend(requirements.fallback_providers.clone());
    let provider_policy = provider_policy();
    let cpu_only_or_empty = providers.is_empty()
        || providers
            .iter()
            .all(|provider| matches!(provider.trim().to_ascii_lowercase().as_str(), "" | "cpu"));
    if provider_policy != "manifest" && cpu_only_or_empty {
        let mut push_if_missing = |provider: &str| {
            if providers
                .iter()
                .all(|candidate| !candidate.eq_ignore_ascii_case(provider))
            {
                providers.push(provider.to_string());
            }
        };
        if device_info.has_cuda {
            // Do not auto-upgrade cpu/missing-provider manifests to TensorRT:
            // its EP rejects some quantized ONNX graphs (non-zero zero-points in
            // Q/DQ). Only use TensorRT when a manifest explicitly requests it.
            push_if_missing("cuda");
        }
        if device_info.has_metal {
            push_if_missing("coreml");
        }
        if device_info.has_rocm {
            push_if_missing("rocm");
        }
        if device_info.has_directml {
            push_if_missing("directml");
        }
        push_if_missing("cpu");
    }
    providers
}

/// Deny a request that names a CUDA device this process cannot use. The
/// request is checked whatever the topology: a pool or data-parallel strategy
/// does not make an explicit `device_id` optional.
pub(crate) fn check_requested_device_usable(
    requirements: &kapsl_core::HardwareRequirements,
    device_info: &DeviceInfo,
    map: Option<&kapsl_llm::cuda_ordinal::CudaOrdinalMap>,
) -> Result<(), DeviceUnavailable> {
    let Some(requested) = requirements
        .device_id
        .and_then(|id| usize::try_from(id).ok())
    else {
        return Ok(());
    };
    let providers = candidate_providers(requirements, device_info);
    let wants_cuda = providers.iter().any(|provider| {
        matches!(
            provider.trim().to_ascii_lowercase().as_str(),
            "cuda" | "tensorrt"
        )
    }) || (providers.is_empty() && device_info.has_cuda);
    if !wants_cuda {
        return Ok(());
    }
    let Some(device) = device_info
        .devices
        .iter()
        .find(|device| is_cuda_device(device) && device.id == requested)
    else {
        return Ok(());
    };
    let Some(problem) = cuda_device_problem(map, device) else {
        return Ok(());
    };
    let usable: Vec<usize> = device_info
        .devices
        .iter()
        .filter(|device| is_cuda_device(device) && cuda_device_problem(map, device).is_none())
        .map(|device| device.id)
        .collect();
    Err(DeviceUnavailable {
        device_id: requested,
        detail: format!(
            "{problem}; CUDA devices this process can use: {usable:?} \
             (set CUDA_VISIBLE_DEVICES to include it, or request one of those devices)"
        ),
    })
}

pub(crate) fn select_mesh_devices(
    requirements: &kapsl_core::HardwareRequirements,
    device_info: &DeviceInfo,
) -> Result<MeshDeviceSelection, String> {
    select_mesh_devices_with(
        requirements,
        device_info,
        kapsl_llm::cuda_ordinal::installed().as_deref(),
    )
}

/// [`select_mesh_devices`] with the process's CUDA visibility passed in. Every
/// placement path goes through here, so hidden or unmatched CUDA devices are
/// removed once, before any provider branch or fallback can pick them.
pub(crate) fn select_mesh_devices_with(
    requirements: &kapsl_core::HardwareRequirements,
    device_info: &DeviceInfo,
    map: Option<&kapsl_llm::cuda_ordinal::CudaOrdinalMap>,
) -> Result<MeshDeviceSelection, String> {
    check_requested_device_usable(requirements, device_info, map)
        .map_err(|error| error.to_string())?;
    let usable = usable_device_info(device_info, map);
    let device_info = &usable;
    let strategy = requirements
        .strategy
        .as_deref()
        .unwrap_or("")
        .to_ascii_lowercase();
    let allow_multi = matches!(
        strategy.as_str(),
        "pool"
            | "round-robin"
            | "data-parallel"
            | "pipeline"
            | "pipeline-parallel"
            | "tensor-parallel"
            | "auto"
    );
    let mut preferred_device_id = requirements.device_id.map(|id| id as usize);
    if allow_multi {
        preferred_device_id = None;
    }
    let providers = candidate_providers(requirements, device_info);
    let mut selected: Vec<kapsl_hal::device::Device> = Vec::new();
    let mut selected_provider: Option<String> = None;
    if !providers.is_empty() {
        for provider in &providers {
            let provider_lower = provider.trim().to_ascii_lowercase();
            let backend_key = match provider_lower.as_str() {
                "tensorrt" => "cuda".to_string(),
                "coreml" => "metal".to_string(),
                other => other.to_string(),
            };
            let mut matches: Vec<kapsl_hal::device::Device> = device_info
                .devices
                .iter()
                .filter(|d| d.backend.to_string().to_lowercase() == backend_key)
                .cloned()
                .collect();
            if backend_key != "cpu" {
                if let Some(min_vram) = requirements.min_vram_mb {
                    matches.retain(|d| d.memory_mb >= min_vram);
                }
                if backend_key == "cuda" {
                    if let Some(min_ver) = &requirements.min_cuda_version {
                        matches.retain(|d| {
                            d.cuda_version
                                .as_ref()
                                .map(|ver| ver >= min_ver)
                                .unwrap_or(false)
                        });
                    }
                }
            }
            if let Some(dev_id) = preferred_device_id {
                if backend_key != "cpu" {
                    matches.retain(|d| d.id == dev_id);
                }
            }
            if !matches.is_empty() {
                selected = matches;
                selected_provider = Some(provider_lower);
                break;
            }
        }
    } else {
        let best_provider = device_info.get_best_provider().to_ascii_lowercase();
        selected_provider = Some(best_provider.clone());
        selected = device_info
            .devices
            .iter()
            .filter(|d| d.backend.to_string().to_ascii_lowercase() == best_provider)
            .cloned()
            .collect();
        if selected.is_empty() {
            selected = device_info.devices.clone();
        }
        if let Some(dev_id) = preferred_device_id {
            if best_provider != "cpu" {
                selected.retain(|d| d.id == dev_id);
            }
        }
    }

    if selected.is_empty() {
        if let Some(dev_id) = preferred_device_id {
            if providers.is_empty() {
                return Err(format!(
                    "Device ID {} not found in available devices",
                    dev_id
                ));
            }
            return Err(format!(
                "Device ID {} not found for providers {:?}",
                dev_id, providers
            ));
        }
        selected = device_info.devices.clone();
    }

    let logical_provider = selected_provider
        .or_else(|| {
            providers
                .iter()
                .find(|provider| !provider.trim().is_empty())
                .map(|provider| provider.trim().to_ascii_lowercase())
        })
        .or_else(|| {
            selected
                .first()
                .map(|device| device.backend.to_string().to_ascii_lowercase())
        })
        .unwrap_or_else(|| "cpu".to_string());

    Ok(MeshDeviceSelection {
        devices: selected,
        logical_provider,
    })
}

#[cfg(test)]
#[path = "visibility_tests.rs"]
mod visibility_tests;
