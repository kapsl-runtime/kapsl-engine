use super::*;

mod backend;
mod load_plan;
mod replica;
mod scaling;

use backend::*;
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

    let registry = Arc::new(Registry::new());
    let model_registry = Arc::new(ModelRegistry::new());
    let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);

    let resources = RuntimeResources::new(device_info)?;
    let (pool, _) = load_model(
        model_id,
        model_path,
        device_info,
        resources,
        args.batch_size,
        args.scheduler_queue_size,
        args.scheduler_max_micro_batch,
        args.scheduler_queue_delay_ms,
        &model_registry,
        &shared_metrics,
        &args.topology,
        args.tp_degree,
        onnx_tuning,
    )
    .await?;

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
    #[cfg(feature = "gpu-device-pool")]
    resources.attach_device_memory_metrics(shared_metrics.clone());
    log::info!(
        "Current directory: {:?}",
        std::env::current_dir().unwrap_or_default()
    );

    let plan = build_model_load_plan(
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
    )?;
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
