use super::*;

mod backend;
mod lifecycle;
pub(crate) mod load_plan;
mod replica;
mod scaling;

use backend::*;
pub(crate) use lifecycle::*;
use load_plan::*;
use replica::*;
pub(crate) use scaling::*;

pub(crate) async fn run_worker(
    args: &Args,
    device_info: &DeviceInfo,
    onnx_tuning_profile: &OnnxTuningProfile,
) -> Result<(), DynError> {
    if args.model.len() != 1 {
        return Err("Worker mode expects exactly one --model".into());
    }

    let model_id = args.worker_model_id.unwrap_or(0);
    let model_path = &args.model[0];
    let onnx_tuning = onnx_tuning_profile.resolve(model_id);

    let plan = prepare_model_load(
        model_id,
        model_id,
        0,
        model_path,
        device_info,
        args.batch_size,
        args.scheduler_queue_size,
        args.scheduler_max_micro_batch,
        args.scheduler_queue_delay_ms,
        &args.topology,
        args.tp_degree,
        &onnx_tuning,
    )?;
    #[cfg(feature = "gpu-device-pool")]
    let bootstrap = device_memory_bootstrap_plan(std::iter::once(&plan), device_info)?;

    let registry = Arc::new(Registry::new());
    let model_registry = Arc::new(ModelRegistry::new());
    let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);
    #[cfg(feature = "gpu-device-pool")]
    let resources = RuntimeResources::new_with_device_memory_plan(device_info, &bootstrap)?;
    #[cfg(not(feature = "gpu-device-pool"))]
    let resources = RuntimeResources::new(device_info)?;
    let memory_for_reconciliation = resources.memory().clone();
    let (pool, handles) = load_prepared_model(
        plan,
        device_info,
        resources,
        &model_registry,
        &shared_metrics,
        onnx_tuning,
    )
    .await?;

    // Isolated workers do not run the parent HTTP/system monitor. Keep their
    // process-local authority current as provider arenas and KV allocations
    // change, using the same cadence as the parent sampler.
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            for handle in &handles {
                let _ = handle.actual_memory();
            }
            if let Some(rss) = super::host_memory::process_rss_bytes() {
                memory_for_reconciliation.observe_process_memory(rss);
            }
        }
    });

    let mut schedulers = HashMap::new();
    schedulers.insert(model_id, pool as Arc<dyn ReplicaScheduler + Send + Sync>);

    let server = IpcServer::new(&args.socket, schedulers, None);
    log::info!(
        "Worker process serving model {} via IPC socket {}",
        model_id,
        args.socket
    );
    server.run().await?;
    Ok(())
}

/// Load a model using the ambient Tokio runtime from a blocking lifecycle task.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_model_blocking(
    model_id: u32,
    model_path: &Path,
    device_info: &DeviceInfo,
    resources: Arc<RuntimeResources>,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<(Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>), DynError> {
    tokio::runtime::Handle::current().block_on(load_model(
        model_id,
        model_path,
        device_info,
        resources,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        model_registry,
        shared_metrics,
        topology,
        tp_degree,
        onnx_tuning,
    ))
}

/// Load the primary replica for a model and create its replica pool.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn load_model(
    model_id: u32,
    model_path: &Path,
    device_info: &DeviceInfo,
    resources: Arc<RuntimeResources>,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<(Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>), DynError> {
    let plan = prepare_model_load(
        model_id,
        model_id,
        0,
        model_path,
        device_info,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        topology,
        tp_degree,
        &onnx_tuning,
    )?;
    load_prepared_model(
        plan,
        device_info,
        resources,
        model_registry,
        shared_metrics,
        onnx_tuning,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_model_load(
    base_model_id: u32,
    runtime_model_id: u32,
    replica_id: u32,
    model_path: &Path,
    device_info: &DeviceInfo,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: &OnnxRuntimeTuning,
) -> Result<ModelLoadPlan, DynError> {
    build_model_load_plan(
        base_model_id,
        runtime_model_id,
        replica_id,
        model_path,
        device_info,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        topology,
        tp_degree,
        onnx_tuning,
    )
}

pub(crate) async fn load_prepared_model(
    plan: ModelLoadPlan,
    device_info: &DeviceInfo,
    resources: Arc<RuntimeResources>,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<(Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>), DynError> {
    let model_id = plan.base_model_id();
    #[cfg(feature = "gpu-device-pool")]
    {
        let bootstrap = device_memory_bootstrap_plan(std::iter::once(&plan), device_info)?;
        resources.ensure_device_pools(&bootstrap)?;
        resources.attach_device_memory_metrics(shared_metrics.clone());
    }
    log::info!(
        "Current directory: {:?}",
        std::env::current_dir().unwrap_or_default()
    );

    let LoadedReplica {
        scheduler,
        swap_handles,
        model_info,
    } = load_replica(
        plan,
        ReplicaLoadRole::Primary,
        device_info,
        resources,
        shared_metrics,
        &onnx_tuning,
    )
    .await?;
    model_registry.upsert(model_info);

    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);
    log::info!("✓ Scheduler started for Model ID {}\n", model_id);

    Ok((Arc::new(pool), swap_handles))
}

/// Reserve the aggregate managed-vLLM weight/workspace estimate before Kapsl
/// performs any backend download or starts a backend child. The reservation is
/// intentionally temporary: ordinary per-model load transactions repeat the
/// admission after installation and retain the authoritative leases.
pub(crate) async fn preflight_managed_vllm_admission(
    plans: &[(PathBuf, ModelLoadPlan)],
    device_info: &DeviceInfo,
    resources: &Arc<RuntimeResources>,
) -> Result<Option<MemoryAdmission>, DynError> {
    use kapsl_engine_api::MemoryReport;

    let mut report = MemoryReport::default();
    let mut domains = Vec::new();
    for (_, plan) in plans.iter().filter(|(_, plan)| plan.uses_managed_vllm()) {
        if plan.use_pipeline_backend || plan.worker_tp_degree != 1 {
            return Err(
                "managed vLLM currently supports one CUDA device per Kapsl replica; keep --tp-degree=1"
                    .into(),
            );
        }
        let selection =
            select_mesh_devices(&plan.loader.manifest.hardware_requirements, device_info).map_err(
                |error| {
                    format!(
                "Failed to select a CUDA device for preliminary admission of model {}: {error}",
                plan.base_model_id
            )
                },
            )?;
        let device_id = selection
            .devices
            .iter()
            .find(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
            .map(|device| device.id)
            .ok_or("managed vLLM preliminary admission selected no CUDA device")?;
        let model_report = managed_vllm_memory_report(
            &plan.model_file_path,
            &[device_id],
            plan.base_model_id,
            plan.replica_id,
        )?;
        report.allocations.extend(model_report.allocations);
        let domain = MemoryDomain::Cuda { device_id };
        if !domains.contains(&domain) {
            domains.push(domain);
        }
    }
    if report.allocations.is_empty() {
        return Ok(None);
    }

    // One synthetic owner lets all startup models targeting the same device be
    // checked atomically under one device load lock. The admission is dropped
    // after backend installation, before real model IDs acquire their leases.
    let owner = MemoryOwner::new(u32::MAX, u32::MAX);
    let plan = resources
        .memory()
        .model_load_plan_with_report(&domains, owner, 0, 0, &report)?;
    let admission = resources
        .memory()
        .begin_load(&plan, EngineKind::Native)
        .await
        .map_err(|error| {
            format!(
                "preliminary managed-vLLM memory admission rejected before backend download: {error}"
            )
        })?;
    Ok(Some(admission))
}
