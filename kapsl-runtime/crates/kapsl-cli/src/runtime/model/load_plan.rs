use super::*;
use kapsl_backends::OnnxRuntimeTuning;

/// Immutable runtime defaults injected into model-load planning.
///
/// CLI parsing is intentionally kept outside the planner. The composition root
/// translates CLI values into this domain configuration once, and every startup,
/// lifecycle, worker, and autoscaling path shares the same defaults.
#[derive(Clone, Debug)]
pub(crate) struct ModelLoadDefaults {
    pub(crate) batch_size: usize,
    pub(crate) scheduler_queue_size: usize,
    pub(crate) scheduler_max_micro_batch: usize,
    pub(crate) scheduler_queue_delay_ms: u64,
    pub(crate) placement: ModelLoadPlacement,
}

/// Requested topology for one model load.
///
/// Most loads use [`ModelLoadDefaults::placement`], while API-triggered loads can
/// inject an explicit request-scoped placement without rebuilding the planner.
#[derive(Clone, Debug)]
pub(crate) struct ModelLoadPlacement {
    pub(crate) topology: String,
    pub(crate) tp_degree: usize,
}

impl ModelLoadPlacement {
    pub(crate) fn new(topology: impl Into<String>, tp_degree: usize) -> Self {
        Self {
            topology: topology.into(),
            tp_degree,
        }
    }
}

/// Backend-specific tuning resolved after a package's engine kind is known.
///
/// Keeping this on [`ModelLoadPlan`] prevents ONNX settings from appearing in
/// backend-neutral load APIs. Additional backend families can add variants
/// without changing every caller.
#[derive(Clone, Debug, Default)]
pub(crate) enum BackendLoadTuning {
    #[default]
    None,
    Onnx(OnnxRuntimeTuning),
}

impl BackendLoadTuning {
    pub(crate) fn onnx(&self) -> Option<&OnnxRuntimeTuning> {
        match self {
            Self::Onnx(tuning) => Some(tuning),
            Self::None => None,
        }
    }
}

/// Injected source of backend-specific model tuning.
pub(crate) trait BackendTuningProvider: Send + Sync {
    fn resolve(&self, model_id: u32, engine_kind: EngineKind) -> BackendLoadTuning;
}

/// Plans model loads from immutable defaults and an injected tuning provider.
#[derive(Clone)]
pub(crate) struct ModelLoadPlanner {
    device_info: Arc<DeviceInfo>,
    defaults: ModelLoadDefaults,
    tuning_provider: Arc<dyn BackendTuningProvider>,
}

impl ModelLoadPlanner {
    pub(crate) fn new(
        device_info: Arc<DeviceInfo>,
        defaults: ModelLoadDefaults,
        tuning_provider: Arc<dyn BackendTuningProvider>,
    ) -> Self {
        Self {
            device_info,
            defaults,
            tuning_provider,
        }
    }

    /// Returns the immutable hardware snapshot shared by every load plan.
    pub(crate) fn device_info(&self) -> &DeviceInfo {
        self.device_info.as_ref()
    }

    /// Topology used by loads that do not request their own.
    pub(crate) fn default_placement(&self) -> &ModelLoadPlacement {
        &self.defaults.placement
    }

    /// Creates an immutable plan, applying an optional request-scoped placement.
    pub(crate) fn prepare(
        &self,
        base_model_id: u32,
        runtime_model_id: u32,
        replica_id: u32,
        model_path: &Path,
        placement: Option<&ModelLoadPlacement>,
    ) -> Result<ModelLoadPlan, DynError> {
        build_model_load_plan(
            base_model_id,
            runtime_model_id,
            replica_id,
            model_path,
            self.device_info(),
            &self.defaults,
            placement.unwrap_or(&self.defaults.placement),
            self.tuning_provider.as_ref(),
        )
    }
}

/// Immutable inputs derived once before any backend allocation starts.
/// Primary loads and scale-up replicas consume the same package, policy, and
/// topology decisions so those paths cannot silently drift apart.
pub(crate) struct ModelLoadPlan {
    pub(super) base_model_id: u32,
    pub(super) runtime_model_id: u32,
    pub(super) replica_id: u32,
    pub(super) absolute_path: PathBuf,
    pub(super) loader: PackageLoader,
    pub(super) model_file_path: PathBuf,
    pub(super) serving_backend: ServingBackendDecision,
    pub(super) batch_size: usize,
    pub(super) scheduler_queue_size: usize,
    pub(super) scheduler_max_micro_batch: usize,
    pub(super) scheduler_queue_delay_ms: u64,
    pub(super) queue_overflow_policy: kapsl_scheduler::QueueOverflowPolicy,
    pub(super) priority_weight: u32,
    pub(super) pipeline_stages: Option<Vec<String>>,
    pub(super) mesh_topology: kapsl_hal::device_mesh::MeshTopology,
    pub(super) worker_topology: &'static str,
    pub(super) worker_tp_degree: usize,
    pub(super) use_pipeline_backend: bool,
    pub(super) backend_tuning: BackendLoadTuning,
    #[cfg(feature = "gpu-device-pool")]
    pub(super) onnx_peak_concurrency: usize,
    pub(super) isolate_process: bool,
    pub(super) isolate_strict: bool,
}

impl ModelLoadPlan {
    pub(crate) fn base_model_id(&self) -> u32 {
        self.base_model_id
    }

    pub(crate) fn uses_managed_vllm(&self) -> bool {
        self.serving_backend.selected == ResolvedServingBackend::Vllm
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_model_load_plan(
    base_model_id: u32,
    runtime_model_id: u32,
    replica_id: u32,
    model_path: &Path,
    device_info: &DeviceInfo,
    defaults: &ModelLoadDefaults,
    placement: &ModelLoadPlacement,
    tuning_provider: &dyn BackendTuningProvider,
) -> Result<ModelLoadPlan, Box<dyn std::error::Error + Send + Sync>> {
    let absolute_path = model_path.canonicalize().map_err(|error| {
        format!(
            "Invalid model path {:?}: {} (CWD: {:?})",
            model_path,
            error,
            std::env::current_dir().unwrap_or_default()
        )
    })?;
    let policy_manifest = inspect_serving_manifest(&absolute_path).map_err(|error| {
        format!(
            "Failed to inspect serving backend policy for model {} replica {}: {}",
            base_model_id, replica_id, error
        )
    })?;
    validate_model_contract(&policy_manifest).map_err(|error| {
        format!(
            "Model contract rejected model {} replica {}: {}",
            base_model_id, replica_id, error
        )
    })?;
    let preflight_serving_backend = resolve_serving_backend(&policy_manifest, device_info.has_cuda)
        .map_err(|error| {
            format!(
                "Serving backend policy rejected model {} replica {}: {}",
                base_model_id, replica_id, error
            )
        })?;
    validate_runtime_serving_backend(&policy_manifest, preflight_serving_backend).map_err(
        |error| {
            format!(
                "Serving backend policy rejected model {} replica {}: {}",
                base_model_id, replica_id, error
            )
        },
    )?;
    let mut loader = resolve_package_loader(&absolute_path, base_model_id)?;
    // Deny a request for a CUDA device this process cannot use (hidden by
    // `CUDA_VISIBLE_DEVICES`, or not matched to a CUDA ordinal) here, before any
    // backend, pack download or memory lease. The typed error is not a memory
    // denial, so the arbiter never evicts models trying to make it fit.
    check_requested_device_usable(
        &loader.manifest.hardware_requirements,
        device_info,
        kapsl_llm::cuda_ordinal::installed().as_deref(),
    )?;
    // Default placement: one visible GPU by available room, not a replica on each.
    pin_default_cuda_device(
        &mut loader.manifest.hardware_requirements,
        device_info,
        kapsl_llm::cuda_ordinal::installed().as_deref(),
        base_model_id,
    );
    let model_file_path = loader.get_model_path();
    validate_model_contract(&loader.manifest).map_err(|error| {
        format!(
            "Model contract rejected model {} replica {} after package load: {}",
            base_model_id, replica_id, error
        )
    })?;
    let serving_backend =
        resolve_serving_backend(&loader.manifest, device_info.has_cuda).map_err(|error| {
            format!(
                "Serving backend policy rejected model {} replica {} after package load: {}",
                base_model_id, replica_id, error
            )
        })?;
    validate_runtime_serving_backend(&loader.manifest, serving_backend).map_err(|error| {
        format!(
            "Serving backend policy rejected model {} replica {} after package load: {}",
            base_model_id, replica_id, error
        )
    })?;
    if serving_backend != preflight_serving_backend {
        return Err(format!(
            "Serving backend policy for model {} replica {} changed while the package was loading; refusing a time-of-check/time-of-use mismatch",
            base_model_id, replica_id
        )
        .into());
    }
    let queue_overflow_policy = resolve_queue_overflow_policy(&loader.manifest);
    log_queue_policy_caveat(queue_overflow_policy);
    let (scheduler_max_micro_batch, scheduler_queue_delay_ms) =
        resolve_scheduler_tuning_for_framework(
            &loader.manifest,
            defaults.scheduler_max_micro_batch,
            defaults.scheduler_queue_delay_ms,
        );
    let priority_weight = resolve_model_priority_weight(&loader.manifest, base_model_id);
    let pipeline_stages = manifest_llm_pipeline_stages(&loader.manifest);
    let requested_topology = placement.topology.trim().to_ascii_lowercase();
    let topology_choice = if serving_backend.selected == ResolvedServingBackend::Vllm
        && matches!(
            requested_topology.as_str(),
            "tensor-parallel" | "tensor_parallel"
        ) {
        let degree = placement.tp_degree.max(1);
        EffectiveTopologyChoice {
            mesh_topology: kapsl_hal::device_mesh::MeshTopology::TensorParallel {
                degree,
                mesh_shape: (1, degree),
            },
            worker_topology: "tensor-parallel",
            worker_tp_degree: degree,
            use_pipeline_backend: false,
        }
    } else {
        resolve_effective_topology_choice(
            &loader.manifest,
            &placement.topology,
            placement.tp_degree,
            pipeline_stages.as_deref(),
        )
    };
    let EffectiveTopologyChoice {
        mesh_topology,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend,
    } = topology_choice;
    BackendFactory::validate_requirements(&loader.manifest.hardware_requirements, device_info)
        .map_err(|error| {
            format!(
                "Requirements validation failed for model {} replica {}: {}",
                base_model_id, replica_id, error
            )
        })?;
    export_gguf_auto_sizing_hint(
        &loader.manifest,
        defaults.batch_size,
        Some(model_file_path.as_path()),
    );
    let engine_kind = EngineKind::resolve(&loader.manifest);
    let backend_tuning = tuning_provider.resolve(base_model_id, engine_kind);

    Ok(ModelLoadPlan {
        base_model_id,
        runtime_model_id,
        replica_id,
        absolute_path,
        model_file_path,
        serving_backend,
        batch_size: defaults.batch_size,
        scheduler_queue_size: defaults.scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        queue_overflow_policy,
        priority_weight,
        pipeline_stages,
        mesh_topology,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend,
        backend_tuning: backend_tuning.clone(),
        #[cfg(feature = "gpu-device-pool")]
        onnx_peak_concurrency: onnx_peak_sessions(engine_kind, &backend_tuning),
        isolate_process: resolve_isolate_process(&loader.manifest),
        isolate_strict: resolve_isolate_process_strict(&loader.manifest),
        loader,
    })
}

/// Sessions an ONNX model has busy at peak: the load tuning's concurrency hint,
/// one for a generative model or no tuning. Sizes the arena peak plan, both for
/// an in-process load and for the parent's lease on an isolated worker (the
/// worker gets the same tuning, so it opens the same number of sessions).
pub(crate) fn onnx_peak_sessions(engine_kind: EngineKind, tuning: &BackendLoadTuning) -> usize {
    match tuning {
        BackendLoadTuning::Onnx(_) if engine_kind.is_onnx_generate() => 1,
        BackendLoadTuning::Onnx(tuning) => {
            tuning.peak_concurrency_hint.unwrap_or(1).max(1) as usize
        }
        BackendLoadTuning::None => 1,
    }
}

#[cfg(feature = "gpu-device-pool")]
pub(crate) fn device_memory_bootstrap_plan<'a>(
    plans: impl IntoIterator<Item = &'a ModelLoadPlan>,
    device_info: &DeviceInfo,
) -> Result<DeviceMemoryBootstrapPlan, String> {
    let mut bootstrap = DeviceMemoryBootstrapPlan::default();
    for plan in plans {
        let selection =
            select_mesh_devices(&plan.loader.manifest.hardware_requirements, device_info)?;
        let cuda_device_ids: Vec<_> = selection
            .devices
            .iter()
            .filter(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
            .map(|device| device.id)
            .collect();
        if cuda_device_ids.is_empty() {
            continue;
        }
        if plan.isolate_process {
            // Do not let a new implicit parent allocation starve the child
            // before it loads. Non-strict fallback remains safe: it loads
            // in-process without a runtime-owned pool if worker startup fails.
            for &device_id in &cuda_device_ids {
                bootstrap.mark_isolated_worker(device_id);
            }
            continue;
        }

        let kind = EngineKind::resolve(&plan.loader.manifest);
        let onnx_pool_opt_in = onnx_cuda_pool_opt_in();
        let signed_accelerator_pack = kind.uses_onnx_session()
            && onnx_route_uses_signed_accelerator_pack(&plan.loader.manifest, device_info);
        let wants_pool = plan_pool_route(
            &mut bootstrap,
            &cuda_device_ids,
            route_uses_shared_pool(kind, onnx_pool_opt_in, signed_accelerator_pack),
        );

        if kind.uses_onnx_session() {
            let model_bytes = std::fs::metadata(&plan.model_file_path)
                .map_err(|error| {
                    format!(
                        "stat ONNX model {}: {error}",
                        plan.model_file_path.display()
                    )
                })?
                .len() as usize;
            let per_device_model_bytes = if plan.use_pipeline_backend {
                model_bytes.saturating_add(cuda_device_ids.len().saturating_sub(1))
                    / cuda_device_ids.len().max(1)
            } else {
                model_bytes
            };
            plan_onnx_device_demand(
                &mut bootstrap,
                &cuda_device_ids,
                wants_pool,
                per_device_model_bytes,
                plan.onnx_peak_concurrency,
                (plan.base_model_id, plan.replica_id),
            );
            continue;
        }

        // Native CUDA allocates weights and execution scratch from the runtime
        // pool. A swap uploads the replacement before dropping the active
        // weights, so automatic sizing must include the two-copy peak rather
        // than falling back to the generic 512 MiB demand floor.
        if kind == EngineKind::Native && wants_pool {
            let weight_bytes = planned_external_weight_bytes(kind, &plan.model_file_path)?;
            let pooled_bytes = planned_native_pool_bytes(weight_bytes);
            for &device_id in &cuda_device_ids {
                bootstrap.add_pooled_allocation(
                    device_id,
                    format!(
                        "bootstrap-native:{}:{}:{}",
                        plan.base_model_id, plan.replica_id, device_id
                    ),
                    pooled_bytes,
                );
            }
            continue;
        }

        let bytes = planned_external_weight_bytes(kind, &plan.model_file_path)?;
        if bytes == 0 {
            continue;
        }
        for &device_id in &cuda_device_ids {
            let allocation_id = if kind.is_gguf() && !cfg!(feature = "gguf-native") {
                // llama.cpp shares immutable weights by canonical model path.
                format!("bootstrap-gguf:{}", plan.model_file_path.display())
            } else {
                // Native backends currently allocate one weight copy per engine.
                format!(
                    "bootstrap-{}:{}:{}:{}",
                    kind.label(),
                    plan.base_model_id,
                    plan.replica_id,
                    device_id
                )
            };
            bootstrap.add_external_allocation(device_id, allocation_id, bytes);
            // GGUF/native execution scratch is not served by the KV pool.
            // Reserve a backend-specific load/context workspace estimate so
            // automatic pool sizing cannot consume that headroom first.
            let scratch_bytes = (bytes / 8).max(256 * 1024 * 1024);
            bootstrap.add_external_allocation(
                device_id,
                format!(
                    "bootstrap-scratch:{}:{}:{}",
                    plan.base_model_id, plan.replica_id, device_id
                ),
                scratch_bytes,
            );
        }
    }
    Ok(bootstrap)
}

/// Mark the model's devices as pool consumers when its route allocates from
/// the runtime pool: automatic mode then provisions one there (and only there).
#[cfg(feature = "gpu-device-pool")]
fn plan_pool_route(
    bootstrap: &mut DeviceMemoryBootstrapPlan,
    cuda_device_ids: &[usize],
    uses_pool: bool,
) -> bool {
    if uses_pool {
        for &device_id in cuda_device_ids {
            bootstrap.mark_pool_consumer(device_id);
        }
    }
    uses_pool
}

/// Plan one ONNX model's device memory for automatic pool sizing.
///
/// On ORT's own arena (embedded ORT without the pool opt-in) it is external
/// demand sized to its peak under load. When its allocations come from the
/// runtime pool (the pool opt-in's registered environment allocator, or a
/// signed native accelerator pack's governed allocator), weights plus
/// per-session arenas/workspaces are pooled demand, sized explicitly:
/// serialized model bytes alone are not a peak device-memory plan.
#[cfg(feature = "gpu-device-pool")]
fn plan_onnx_device_demand(
    bootstrap: &mut DeviceMemoryBootstrapPlan,
    cuda_device_ids: &[usize],
    pooled: bool,
    per_device_model_bytes: usize,
    sessions: usize,
    (model_id, replica_id): (u32, u32),
) {
    if !pooled {
        let peak = onnx_arena_peak_bytes(per_device_model_bytes, sessions);
        for &device_id in cuda_device_ids {
            bootstrap.add_external_allocation(
                device_id,
                format!("bootstrap-onnx-arena:{model_id}:{replica_id}:{device_id}"),
                peak,
            );
        }
        return;
    }
    let workspace_per_session = (per_device_model_bytes / 2).max(256 * 1024 * 1024);
    let pooled_bytes = per_device_model_bytes
        .saturating_add(workspace_per_session)
        .saturating_mul(sessions);
    for &device_id in cuda_device_ids {
        bootstrap.add_pooled_allocation(
            device_id,
            format!("bootstrap-onnx:{model_id}:{replica_id}:{device_id}"),
            pooled_bytes,
        );
    }
}

/// Opt-in that puts ONNX CUDA sessions back on the shared Kapsl GPU pool.
///
/// By default ONNX models use ORT's own per-session arenas: run 5 and run 6
/// showed CUDA 700 (illegal memory access) and hangs when concurrent ONNX
/// sessions drew from the pool, and none with the pool off. GGUF and native
/// models keep the pool either way.
#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) const ONNX_CUDA_POOL_ENV: &str = "KAPSL_ONNX_CUDA_POOL";

#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) fn onnx_cuda_pool_opt_in() -> bool {
    std::env::var(ONNX_CUDA_POOL_ENV)
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Whether `kind` draws its device memory from the shared pool.
#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) fn kind_uses_shared_pool(kind: EngineKind, onnx_pool_opt_in: bool) -> bool {
    if kind.uses_onnx_session() {
        return onnx_pool_opt_in;
    }
    (kind.is_gguf()
        && (cfg!(any(
            feature = "gguf-native",
            feature = "gguf-cuda-shared-kv"
        )) || crate::backend::lazy_llama_cpp_packs_enabled()))
        || (kind == EngineKind::Native && cfg!(feature = "native"))
}

/// Whether the route a model loads on draws its device memory from the runtime
/// pool, so automatic mode must provision one on its devices.
/// `signed_accelerator_pack` says an ONNX model resolves to a signed native
/// accelerator pack: such a pack must use the governed device allocator
/// (`BackendManager` rejects one that does not), which allocates from the
/// runtime-owned pool and refuses to start without it.
#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) fn route_uses_shared_pool(
    kind: EngineKind,
    onnx_pool_opt_in: bool,
    signed_accelerator_pack: bool,
) -> bool {
    kind_uses_shared_pool(kind, onnx_pool_opt_in)
        || (kind.uses_onnx_session() && signed_accelerator_pack)
}

/// Host-only and deterministic: whether `ensure_onnx_backend_pack` will route
/// this ONNX model to a signed native CUDA or TensorRT pack on this host.
#[cfg(feature = "gpu-device-pool")]
pub(crate) fn onnx_route_uses_signed_accelerator_pack(
    manifest: &kapsl_core::Manifest,
    device_info: &DeviceInfo,
) -> bool {
    crate::backend::generic_native_backend_packs_enabled().unwrap_or(false)
        && crate::backend::lazy_onnx_packs_enabled()
        && matches!(
            crate::backend::onnx_pack_profile_for_manifest(manifest, device_info),
            Ok(Some(profile)) if profile != crate::backend::OnnxBackendPackProfile::Cpu
        )
}

/// Whether an engine's CUDA claim is ORT's own arena, which the memory
/// authority holds at its peak plan and observed high-water mark rather than
/// following the arena down to its idle size (a second model must not be
/// admitted against memory the first reclaims under load).
#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) fn onnx_arena_claims_held(kind: EngineKind) -> bool {
    arena_claims_held(kind, onnx_cuda_pool_opt_in())
}

#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) fn arena_claims_held(kind: EngineKind, onnx_pool_opt_in: bool) -> bool {
    kind.uses_onnx_session() && !kind_uses_shared_pool(kind, onnx_pool_opt_in)
}

/// Whether the embedded ORT environment allocator is registered with the pool
/// for a provider: only for ONNX on CUDA/TensorRT, and only with the opt-in.
#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) fn ort_pool_registration_wanted(provider: &str, onnx_pool_opt_in: bool) -> bool {
    onnx_pool_opt_in
        && (provider.eq_ignore_ascii_case("cuda") || provider.eq_ignore_ascii_case("tensorrt"))
}

/// Per-session workspace an ONNX CUDA session reaches under concurrent load
/// with ORT's own arena: `max(0.9 x model, 384 MiB)`. The ledger rows measured
/// in runs 5 and 6 (4 sessions all busy) were: YOLOX-nano 130,023,424 B,
/// ResNet-50 1,761,607,680 B, ZFNet-512 2,437,110,191 B, BERT-base
/// 2,810,119,568 B. Beyond the weights that is 29, 337, 255 and 267 MB per
/// session, which is driven by activations and cuDNN workspaces, not file
/// size, so a floor (set by ResNet-50) and a share of the model cover them.
pub(crate) const ONNX_ARENA_WORKSPACE_FLOOR_BYTES: usize = 384 * 1024 * 1024;

pub(crate) fn onnx_arena_workspace_per_session_bytes(model_bytes: usize) -> usize {
    (model_bytes.saturating_mul(9) / 10).max(ONNX_ARENA_WORKSPACE_FLOOR_BYTES)
}

/// Device bytes an ONNX CUDA model on ORT's own arena can reach with every
/// session busy: weights and workspace per session. Admission reserves this,
/// not the load-time figure (ResNet-50 loads at 950 MB and reaches 1758 MB).
#[cfg_attr(not(feature = "gpu-device-pool"), allow(dead_code))]
pub(crate) fn onnx_arena_peak_bytes(model_bytes: usize, sessions: usize) -> usize {
    model_bytes
        .saturating_add(onnx_arena_workspace_per_session_bytes(model_bytes))
        .saturating_mul(sessions.max(1))
}

/// Raise an ONNX backend's CUDA workspace claim to the arena peak.
///
/// The backend plans weights for every session but one workspace of
/// `max(model / 4, 64 MiB)`. Applies only to the backend-managed CUDA claims of
/// a session pool (weights at least one model copy), where sessions are the
/// weights divided by the model size; anything else is left as planned.
pub(crate) fn raise_onnx_arena_claims_to_peak(
    report: &mut kapsl_engine_api::MemoryReport,
    model_bytes: usize,
) {
    use kapsl_engine_api::{MemoryAllocationClass, MemoryAllocationSource, MemoryDomain};
    if model_bytes == 0 {
        return;
    }
    let weights = report.allocations.iter().find(|allocation| {
        matches!(allocation.domain, MemoryDomain::Cuda { .. })
            && allocation.class == MemoryAllocationClass::PersistentWeights
            && allocation.source == MemoryAllocationSource::BackendManaged
    });
    let Some(weights) = weights else { return };
    if weights.bytes < model_bytes {
        return;
    }
    let sessions = weights.bytes / model_bytes;
    let domain = weights.domain.clone();
    let peak_workspace =
        onnx_arena_workspace_per_session_bytes(model_bytes).saturating_mul(sessions.max(1));
    for allocation in report.allocations.iter_mut() {
        if allocation.domain == domain
            && allocation.class == MemoryAllocationClass::TransientWorkspace
            && allocation.source == MemoryAllocationSource::BackendManaged
        {
            allocation.bytes = allocation.bytes.max(peak_workspace);
        }
    }
}

#[cfg(feature = "gpu-device-pool")]
fn planned_native_pool_bytes(weight_bytes: usize) -> usize {
    let scratch_bytes = (weight_bytes / 8).max(256 * 1024 * 1024);
    weight_bytes.saturating_mul(2).saturating_add(scratch_bytes)
}

#[cfg(feature = "gpu-device-pool")]
fn planned_external_weight_bytes(kind: EngineKind, model_path: &Path) -> Result<usize, String> {
    if kind == EngineKind::Native {
        // Mirror NativeBackend::planned_weight_bytes: a manifest may point at
        // one shard, but the backend loads every sibling safetensors shard.
        let model_dir = if model_path.is_dir() {
            model_path
        } else {
            model_path.parent().unwrap_or(model_path)
        };
        let mut bytes = 0usize;
        for entry in std::fs::read_dir(model_dir)
            .map_err(|error| format!("read model directory {}: {error}", model_dir.display()))?
        {
            let entry = entry.map_err(|error| {
                format!("read model entry under {}: {error}", model_dir.display())
            })?;
            if entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors") {
                bytes = bytes.saturating_add(
                    entry
                        .metadata()
                        .map_err(|error| {
                            format!("stat model shard {}: {error}", entry.path().display())
                        })?
                        .len() as usize,
                );
            }
        }
        return Ok(bytes);
    }

    let bytes = std::fs::metadata(model_path)
        .map_err(|error| format!("stat model {}: {error}", model_path.display()))?
        .len() as usize;
    if kind.is_gguf() && cfg!(feature = "gguf-native") {
        // The native GGUF path may expand quantized tensors while uploading.
        // Two times the source size is a conservative host-only bootstrap bound;
        // post-load reconciliation remains authoritative.
        Ok(bytes.saturating_mul(2))
    } else {
        Ok(bytes)
    }
}

/// Spawn the isolated worker, if the plan asks for one. `memory` is the
/// parent's reservation for it (see [`reserve_isolated_worker_memory`]): it is
/// tied to the worker's life, and dropped here when no worker is kept.
pub(super) async fn start_isolated_worker(
    plan: &ModelLoadPlan,
    memory: Option<IsolatedWorkerMemory>,
) -> Result<Option<Arc<WorkerProcess>>, Box<dyn std::error::Error + Send + Sync>> {
    if !plan.isolate_process {
        return Ok(None);
    }

    log::info!(
        "✓ Process isolation enabled for Model ID {} replica #{} (strict={})",
        plan.base_model_id,
        plan.replica_id,
        plan.isolate_strict
    );
    match spawn_worker_process(
        plan.runtime_model_id,
        &plan.absolute_path,
        plan.batch_size,
        plan.scheduler_queue_size,
        plan.scheduler_max_micro_batch,
        plan.scheduler_queue_delay_ms,
        plan.worker_topology,
        plan.worker_tp_degree,
        plan.backend_tuning.onnx(),
    ) {
        Ok(worker) => {
            let worker = Arc::new(worker);
            if let Some(memory) = memory {
                worker.attach_memory_guard(Box::new(memory));
            }
            match wait_for_worker_ready_async(worker.as_ref(), Duration::from_secs(30)).await {
                Ok(()) => Ok(Some(start_worker_with_supervisor(worker))),
                Err(error) => {
                    worker.kill();
                    if plan.isolate_strict {
                        Err(format!(
                            "Model {} replica {} requires process isolation but the worker was not ready: {}",
                            plan.base_model_id, plan.replica_id, error
                        )
                        .into())
                    } else {
                        log::warn!(
                            "Model {} replica {} requested process isolation, but worker was not ready; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                            plan.base_model_id,
                            plan.replica_id,
                            error
                        );
                        Ok(None)
                    }
                }
            }
        }
        Err(error) if plan.isolate_strict => Err(format!(
            "Model {} replica {} requires process isolation but the worker failed to spawn: {}",
            plan.base_model_id, plan.replica_id, error
        )
        .into()),
        Err(error) => {
            log::warn!(
                "Model {} replica {} requested process isolation, but worker spawn failed; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                plan.base_model_id,
                plan.replica_id,
                error
            );
            Ok(None)
        }
    }
}

#[cfg(all(test, feature = "gpu-device-pool"))]
mod device_memory_plan_tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kapsl-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn bootstrap_weight_estimate_uses_the_gguf_file_size() {
        let root = scratch_dir("gguf-bootstrap-plan");
        std::fs::create_dir_all(&root).unwrap();
        let model = root.join("model.gguf");
        std::fs::write(&model, vec![0u8; 4096]).unwrap();

        let bytes = planned_external_weight_bytes(EngineKind::GgufGenerate, &model).unwrap();
        let expected = if cfg!(feature = "gguf-native") {
            8192
        } else {
            4096
        };
        assert_eq!(bytes, expected);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn native_directory_estimate_sums_only_safetensor_shards() {
        let root = scratch_dir("native-bootstrap-plan");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model-1.safetensors"), vec![0u8; 100]).unwrap();
        std::fs::write(root.join("model-2.safetensors"), vec![0u8; 250]).unwrap();
        std::fs::write(root.join("tokenizer.json"), vec![0u8; 999]).unwrap();

        assert_eq!(
            planned_external_weight_bytes(EngineKind::Native, &root).unwrap(),
            350
        );
        assert_eq!(
            planned_external_weight_bytes(EngineKind::Native, &root.join("model-1.safetensors"))
                .unwrap(),
            350,
            "a manifest pointing at one shard must account every sibling shard"
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    const ONNX_KINDS: [EngineKind; 5] = [
        EngineKind::OnnxForward,
        EngineKind::OnnxDetect,
        EngineKind::OnnxClassify,
        EngineKind::OnnxEmbed,
        EngineKind::OnnxGenerate,
    ];

    #[test]
    fn signed_onnx_accelerator_packs_use_the_pool_by_default() {
        for kind in ONNX_KINDS {
            assert!(
                route_uses_shared_pool(kind, false, true),
                "{kind:?} signed GPU pack"
            );
            assert!(
                !route_uses_shared_pool(kind, false, false),
                "{kind:?} embedded ORT keeps its own arena"
            );
            assert!(
                route_uses_shared_pool(kind, true, false),
                "{kind:?} opted in"
            );
        }
        // The signed ONNX route says nothing about GGUF or native models.
        for kind in [EngineKind::GgufGenerate, EngineKind::Native] {
            assert_eq!(
                route_uses_shared_pool(kind, false, true),
                kind_uses_shared_pool(kind, false),
                "{kind:?}"
            );
        }
    }

    /// The v0.3.0 qualification's default ONNX-only starts (signed onnx/cuda12
    /// and onnx/tensorrt10, no explicit pool mode) got no pool, and the pack
    /// refused to start without one.
    #[cfg(feature = "gpu-device-pool")]
    #[test]
    fn a_default_onnx_only_start_on_signed_gpu_packs_plans_a_pool_that_holds_it() {
        const MIB: usize = 1024 * 1024;
        let resnet = 102_576_593;
        let mut bootstrap = DeviceMemoryBootstrapPlan::default();
        let pooled = plan_pool_route(
            &mut bootstrap,
            &[0],
            route_uses_shared_pool(EngineKind::OnnxClassify, false, true),
        );
        plan_onnx_device_demand(&mut bootstrap, &[0], pooled, resnet, 1, (0, 0));
        let (minimum_pooled, external) = bootstrap
            .implicit_pool_demand(0)
            .expect("automatic mode provisions a pool for the governed allocator");
        assert_eq!(minimum_pooled, resnet + 256 * MIB);
        assert_eq!(external, 0, "the pool holds it; no ORT arena outside it");

        // Embedded ORT on the same host: its own arena, external, no pool.
        let mut bootstrap = DeviceMemoryBootstrapPlan::default();
        let pooled = plan_pool_route(
            &mut bootstrap,
            &[0],
            route_uses_shared_pool(EngineKind::OnnxClassify, false, false),
        );
        plan_onnx_device_demand(&mut bootstrap, &[0], pooled, resnet, 1, (0, 0));
        assert_eq!(bootstrap.implicit_pool_demand(0), None);
    }

    /// GGUF on the signed llama.cpp pack already asks for the pool by default:
    /// its weights and scratch stay external, its KV lives in the pool.
    #[cfg(feature = "gpu-device-pool")]
    #[test]
    fn a_default_gguf_start_on_the_signed_pack_plans_a_pool() {
        let mut bootstrap = DeviceMemoryBootstrapPlan::default();
        let uses_pool = kind_uses_shared_pool(EngineKind::GgufGenerate, false);
        plan_pool_route(&mut bootstrap, &[0], uses_pool);
        bootstrap.add_external_allocation(0, "bootstrap-gguf:qwen", 491_400_032);
        bootstrap.add_external_allocation(0, "bootstrap-scratch:1:0:0", 268_435_456);
        let expected = (cfg!(any(
            feature = "gguf-native",
            feature = "gguf-cuda-shared-kv"
        )) || crate::backend::lazy_llama_cpp_packs_enabled())
        .then_some((0, 759_835_488));
        assert_eq!(bootstrap.implicit_pool_demand(0), expected);
    }

    #[test]
    fn onnx_is_off_the_shared_pool_by_default_and_other_kinds_are_unaffected() {
        for kind in [
            EngineKind::OnnxForward,
            EngineKind::OnnxDetect,
            EngineKind::OnnxClassify,
            EngineKind::OnnxEmbed,
            EngineKind::OnnxGenerate,
        ] {
            assert!(!kind_uses_shared_pool(kind, false), "{kind:?} default");
            assert!(kind_uses_shared_pool(kind, true), "{kind:?} opted in");
        }
        // The opt-in is about ONNX only: GGUF and native decide as before.
        for kind in [EngineKind::GgufGenerate, EngineKind::Native] {
            assert_eq!(
                kind_uses_shared_pool(kind, false),
                kind_uses_shared_pool(kind, true),
                "{kind:?} must not depend on the ONNX opt-in"
            );
        }
        assert_eq!(
            kind_uses_shared_pool(EngineKind::Native, false),
            cfg!(feature = "native")
        );
    }

    #[test]
    fn only_onnx_on_its_own_arena_has_its_claim_held() {
        for kind in [
            EngineKind::OnnxForward,
            EngineKind::OnnxDetect,
            EngineKind::OnnxClassify,
        ] {
            assert!(
                arena_claims_held(kind, false),
                "{kind:?} on ORT's arena is held"
            );
            assert!(
                !arena_claims_held(kind, true),
                "{kind:?} on the pool is not held"
            );
        }
        // GGUF and native claims keep following the backend's report.
        for kind in [EngineKind::GgufGenerate, EngineKind::Native] {
            assert!(!arena_claims_held(kind, false), "{kind:?}");
            assert!(!arena_claims_held(kind, true), "{kind:?}");
        }
    }

    #[test]
    fn the_ort_allocator_is_registered_with_the_pool_only_when_opted_in() {
        assert!(!ort_pool_registration_wanted("cuda", false));
        assert!(!ort_pool_registration_wanted("tensorrt", false));
        assert!(ort_pool_registration_wanted("cuda", true));
        assert!(ort_pool_registration_wanted("TensorRT", true));
        assert!(!ort_pool_registration_wanted("cpu", true));
    }

    /// Ledger rows (4 sessions all busy, ORT's own arena) from GPU runs 5 and 6.
    const MEASURED_ARENA_PEAKS: [(&str, usize, usize); 4] = [
        ("yolox-nano", 3_702_598, 130_023_424),
        ("resnet50", 102_576_593, 1_761_607_680),
        ("zfnet512", 349_005_501, 2_437_110_191),
        ("bert-base", 435_852_736, 2_810_119_568),
    ];

    #[test]
    fn the_onnx_arena_peak_plan_covers_every_measured_run_5_and_6_peak() {
        for (name, model_bytes, measured) in MEASURED_ARENA_PEAKS {
            let plan = onnx_arena_peak_bytes(model_bytes, 4);
            assert!(
                plan >= measured / 100 * 110,
                "{name}: plan {plan} must cover the measured peak {measured} with 10% margin"
            );
        }
        // ResNet-50 loads at about 950 MB; the load-time figure is not the plan.
        assert!(onnx_arena_peak_bytes(102_576_593, 4) > 950_009_856 * 2);
        // Larger models scale with 0.9 x model per session beyond the floor.
        let big = 3_000_000_000usize;
        assert_eq!(onnx_arena_workspace_per_session_bytes(big), big / 10 * 9);
        assert_eq!(onnx_arena_peak_bytes(big, 0), big + big / 10 * 9);
    }

    #[test]
    fn admission_claim_for_onnx_on_its_own_arena_is_raised_to_the_peak() {
        use kapsl_engine_api::{
            MemoryAllocation, MemoryAllocationClass, MemoryAllocationSource, MemoryDomain,
            MemoryReport,
        };
        let model = 102_576_593usize;
        let cuda = MemoryDomain::Cuda { device_id: 0 };
        // What the SDK's ONNX backend plans on ORT's own arena: weights for
        // every session, one workspace of max(model / 4, 64 MiB).
        let planned = |source| MemoryReport {
            allocations: vec![
                MemoryAllocation {
                    allocation_id: "m".into(),
                    domain: cuda.clone(),
                    class: MemoryAllocationClass::PersistentWeights,
                    source,
                    bytes: model * 4,
                },
                MemoryAllocation {
                    allocation_id: "m:workspace".into(),
                    domain: cuda.clone(),
                    class: MemoryAllocationClass::TransientWorkspace,
                    source,
                    bytes: (model / 4).max(64 * 1024 * 1024),
                },
            ],
        };

        let mut report = planned(MemoryAllocationSource::BackendManaged);
        raise_onnx_arena_claims_to_peak(&mut report, model);
        let total: usize = report.allocations.iter().map(|a| a.bytes).sum();
        assert_eq!(total, onnx_arena_peak_bytes(model, 4));
        assert!(total >= 1_761_607_680, "covers ResNet-50's measured peak");

        // Pool-managed claims (the opt-in path) are left as the backend planned them.
        let mut pooled = planned(MemoryAllocationSource::RuntimeManaged);
        raise_onnx_arena_claims_to_peak(&mut pooled, model);
        assert_eq!(pooled, planned(MemoryAllocationSource::RuntimeManaged));

        // A split of one model across devices has less than one model copy per device.
        let mut split = planned(MemoryAllocationSource::BackendManaged);
        split.allocations[0].bytes = model / 2;
        let before = split.clone();
        raise_onnx_arena_claims_to_peak(&mut split, model);
        assert_eq!(split, before);
    }

    #[test]
    fn native_pool_estimate_includes_hot_swap_two_copy_peak() {
        const GIB: usize = 1024 * 1024 * 1024;
        let weights = 2 * GIB + 200 * 1024 * 1024;
        let scratch = (weights / 8).max(256 * 1024 * 1024);

        assert_eq!(
            planned_native_pool_bytes(weights),
            weights.saturating_mul(2).saturating_add(scratch)
        );
        assert!(planned_native_pool_bytes(weights) > 4 * GIB);
    }
}
