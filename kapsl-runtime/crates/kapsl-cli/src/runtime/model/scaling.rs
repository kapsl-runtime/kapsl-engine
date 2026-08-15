use super::*;

/// Scale up a model by adding a new replica.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn scale_up_model(
    base_model_id: u32,
    replica_id: u32,
    unique_id: u32,
    model_path: &Path,
    device_info: &DeviceInfo,
    resources: Arc<RuntimeResources>,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<(Arc<Scheduler>, EngineHandle), DynError> {
    #[cfg(feature = "gpu-device-pool")]
    resources.attach_device_memory_metrics(shared_metrics.clone());
    log::info!(
        "Scaling up Model ID {} - Creating replica #{}",
        base_model_id,
        replica_id
    );

    let plan = build_model_load_plan(
        base_model_id,
        unique_id,
        replica_id,
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
    #[cfg(feature = "gpu-device-pool")]
    {
        let bootstrap = device_memory_bootstrap_plan(std::iter::once(&plan), device_info)?;
        resources.ensure_device_pools(&bootstrap)?;
    }
    let LoadedReplica {
        scheduler,
        mut swap_handles,
        model_info,
    } = load_replica(
        plan,
        ReplicaLoadRole::Autoscaled,
        device_info,
        resources,
        shared_metrics,
        &onnx_tuning,
    )
    .await?;
    if swap_handles.len() != 1 {
        return Err(format!(
            "Autoscaled replica {} for model {} produced {} engines; exactly one is required",
            replica_id,
            base_model_id,
            swap_handles.len()
        )
        .into());
    }
    let swap_handle = swap_handles.pop().expect("length checked above");
    model_registry.upsert(model_info);

    log::info!(
        "✓ Replica #{} started for Model ID {}",
        replica_id,
        base_model_id
    );

    Ok((scheduler, swap_handle))
}

/// Scale down a model by removing a replica.
pub(crate) async fn scale_down_model(
    base_model_id: u32,
    replica_id: u32,
    unique_id: u32,
    models: &ModelManager,
) -> Result<(), DynError> {
    log::info!(
        "Scaling down Model ID {} - Removing replica #{}",
        base_model_id,
        replica_id
    );

    if let Err(error) = models
        .registry()
        .set_status(unique_id, ModelStatus::Stopping)
    {
        log::error!("Failed to set status to Stopping for {unique_id}: {error}");
    }

    let pool = models.pool(base_model_id);
    if let Some(pool) = pool {
        let removed = pool.remove_replica(replica_id);
        if !removed {
            let _ = models.registry().set_status(unique_id, ModelStatus::Active);
            return Err(format!(
                "Replica #{} was not present in pool for model {}",
                replica_id, base_model_id
            )
            .into());
        }
    } else {
        let _ = models.registry().set_status(unique_id, ModelStatus::Active);
        return Err(format!("Replica pool not found for model {base_model_id}").into());
    }

    // Scale-up appends one hot-swap handle per replica; scale-down removes
    // replicas in descending ID order, so the matching handle is the tail.
    models.pop_swap_handle(base_model_id);

    if let Err(error) = models
        .registry()
        .set_status(unique_id, ModelStatus::Inactive)
    {
        log::error!("Failed to set status to Inactive for {unique_id}: {error}");
    }

    log::info!(
        "✓ Replica #{} stopped for Model ID {}",
        replica_id,
        base_model_id
    );

    Ok(())
}
