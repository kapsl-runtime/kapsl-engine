use super::*;

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

pub(crate) fn resolve_runtime_state_layout(args: &Args) -> RuntimeStateLayout {
    if let Some(state_dir) = args.state_dir.as_ref() {
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

pub(crate) fn export_gguf_auto_sizing_hint(
    manifest: &Manifest,
    batch_size: usize,
    model_file_path: Option<&Path>,
) {
    if !EngineKind::resolve(manifest).is_gguf() {
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
    if let Some(value) = optional_env_var_alias(
        SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV,
        LEGACY_SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV,
    ) {
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

pub(crate) fn select_mesh_devices(
    requirements: &kapsl_core::HardwareRequirements,
    device_info: &DeviceInfo,
) -> Result<Vec<kapsl_hal::device::Device>, String> {
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
            push_if_missing("tensorrt");
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

    let mut selected: Vec<kapsl_hal::device::Device> = Vec::new();
    if !providers.is_empty() {
        for provider in &providers {
            let provider_lower = provider.to_lowercase();
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
                break;
            }
        }
    } else {
        let best_provider = device_info.get_best_provider().to_ascii_lowercase();
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

    Ok(selected)
}
