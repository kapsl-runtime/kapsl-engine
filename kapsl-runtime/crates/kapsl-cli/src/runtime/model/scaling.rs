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
    )?;
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

pub(crate) fn force_stop_model_before_remove(
    base_model_id: u32,
    replicas: &[ModelInfo],
    models: &ModelManager,
    resources: &RuntimeResources,
) {
    for replica in replicas {
        if let Err(error) = models
            .registry()
            .set_status(replica.id, ModelStatus::Stopping)
        {
            log::warn!(
                "Failed to set model {} replica {} to Stopping before remove: {}",
                base_model_id,
                replica.replica_id,
                error
            );
        }
    }

    if let Some(pool) = models.pool(base_model_id) {
        for replica in replicas {
            if !pool.remove_replica(replica.replica_id) {
                log::debug!(
                    "Replica {} for model {} was not present in the pool during remove",
                    replica.replica_id,
                    base_model_id
                );
            }
        }
    }

    models.stop_runtime(base_model_id);
    resources.kv().detach_engine_for_model(base_model_id);

    for replica in replicas {
        if let Err(error) = models
            .registry()
            .set_status(replica.id, ModelStatus::Inactive)
        {
            log::warn!(
                "Failed to set model {} replica {} to Inactive before remove: {}",
                base_model_id,
                replica.replica_id,
                error
            );
        }
    }
}

pub(crate) const MEMORY_HEADROOM_FRACTION: f64 = 0.80;

pub(crate) fn cap_scale_up_target_by_memory_headroom(
    current_replicas: u32,
    proposed_target: u32,
    total_model_memory_bytes: usize,
    system_total_memory_kb: u64,
) -> u32 {
    if proposed_target <= current_replicas
        || current_replicas == 0
        || total_model_memory_bytes == 0
        || system_total_memory_kb == 0
    {
        return proposed_target;
    }

    let per_replica_bytes = total_model_memory_bytes as f64 / current_replicas as f64;
    if per_replica_bytes <= 0.0 {
        return proposed_target;
    }

    let budget_bytes = (system_total_memory_kb as f64 * 1024.0 * MEMORY_HEADROOM_FRACTION).max(1.0);
    let max_by_headroom = (budget_bytes / per_replica_bytes).floor() as u32;
    let capped_max = max_by_headroom.max(current_replicas).max(1);
    proposed_target.min(capped_max)
}
