use super::*;

pub(crate) struct AutoScalerTaskConfig {
    pub(crate) auto_scaler: Arc<RwLock<AutoScaler>>,
    pub(crate) models: Arc<ModelManager>,
    pub(crate) device_info: Arc<DeviceInfo>,
    pub(crate) shared_metrics: kapsl_monitor::metrics::KapslMetrics,
    pub(crate) resources: Arc<RuntimeResources>,
    pub(crate) batch_size: usize,
    pub(crate) scheduler_queue_size: usize,
    pub(crate) scheduler_max_micro_batch: usize,
    pub(crate) scheduler_queue_delay_ms: u64,
    pub(crate) topology: String,
    pub(crate) tp_degree: usize,
    pub(crate) onnx_tuning_profile: Arc<OnnxTuningProfile>,
}

pub(crate) fn spawn_auto_scaler_task(config: AutoScalerTaskConfig) {
    let AutoScalerTaskConfig {
        auto_scaler: auto_scaler_clone,
        models: models_for_scaler,
        device_info: device_info_for_scaler,
        shared_metrics: shared_metrics_for_scaler,
        resources: resources_for_scaler,
        batch_size: batch_size_for_scaler,
        scheduler_queue_size: scheduler_queue_size_for_scaler,
        scheduler_max_micro_batch: scheduler_max_micro_batch_for_scaler,
        scheduler_queue_delay_ms: scheduler_queue_delay_ms_for_scaler,
        topology: topology_for_scaler,
        tp_degree: tp_degree_for_scaler,
        onnx_tuning_profile: onnx_tuning_profile_for_scaler,
    } = config;

    tokio::spawn(async move {
        use std::time::Duration;
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        let mut last_check = std::time::Instant::now();

        loop {
            interval.tick().await;
            let elapsed = last_check.elapsed();
            last_check = std::time::Instant::now();

            // Check each model for scaling needs
            for model_info in models_for_scaler.registry().list() {
                let base_model_id = model_info.base_model_id;

                // Only process primary models (not replicas)
                if model_info.replica_id != 0 {
                    continue;
                }

                let current_replicas = models_for_scaler
                    .registry()
                    .count_active_replicas(base_model_id)
                    as u32;

                // Calculate pool state and update metrics.
                let (total_queue_depth, healthy_replicas, metrics_available) =
                    if let Some(pool) = models_for_scaler.pool(base_model_id) {
                        let (high, low) = pool.get_queue_depth();
                        let healthy = pool.get_healthy_replica_count();
                        let metrics = pool.get_metrics();

                        // Update pool metrics
                        let model_id_str = base_model_id.to_string();
                        shared_metrics_for_scaler
                            .pool_active_replicas
                            .with_label_values(&[&model_id_str])
                            .set(current_replicas as i64);
                        shared_metrics_for_scaler
                            .pool_queue_depth_high
                            .with_label_values(&[&model_id_str])
                            .set(high as i64);
                        shared_metrics_for_scaler
                            .pool_queue_depth_low
                            .with_label_values(&[&model_id_str])
                            .set(low as i64);
                        shared_metrics_for_scaler
                            .pool_healthy_replicas
                            .with_label_values(&[&model_id_str])
                            .set(healthy as i64);

                        // kapsl-monitor owns the EngineMetrics -> Prometheus mapping.
                        // Keeping that translation in one place makes newly added
                        // engine fields visible to every runtime caller together.
                        shared_metrics_for_scaler.set_kv_cache_metrics(&model_id_str, &metrics);

                        (high + low, healthy as u32, true)
                    } else {
                        (0, 0, false)
                    };

                // Check for scale-up
                let should_scale_up = auto_scaler_clone.write().should_scale_up(
                    base_model_id,
                    current_replicas,
                    healthy_replicas,
                    total_queue_depth,
                    elapsed,
                    metrics_available,
                );

                if let Some(target_replicas) = should_scale_up {
                    // Queue depth driven by a co-tenant squeezing the GPU is not
                    // load growth: a new replica would land on the same starved
                    // device and thrash. Skip and re-evaluate next tick — the
                    // ceiling's grow-slow recovery provides the hysteresis.
                    let memory_snapshot = resources_for_scaler.memory().snapshot();
                    if memory_snapshot.foreign_pressure_active {
                        log::warn!(
                            "Auto-scaler: model {} queue depth {} exceeds threshold, but a \
                             co-tenant GPU process is limiting the KV ceiling; suppressing \
                             scale-up from {} to {} replicas",
                            base_model_id,
                            total_queue_depth,
                            current_replicas,
                            target_replicas
                        );
                        continue;
                    }
                    let onnx_tuning = onnx_tuning_profile_for_scaler.resolve(base_model_id);
                    let capped_target = memory_snapshot.cap_replica_target(
                        base_model_id,
                        current_replicas,
                        target_replicas,
                    );
                    if capped_target < target_replicas {
                        log::warn!(
                        "Auto-scaler: Capping model {} scale-up target {} -> {} due to memory headroom",
                        base_model_id,
                        target_replicas,
                        capped_target
                    );
                    }

                    if capped_target <= current_replicas {
                        continue;
                    }

                    let replicas_to_add = capped_target.saturating_sub(current_replicas);
                    log::info!(
                    "Auto-scaler: Model {} queue depth {} exceeds threshold, scaling from {} to {} replicas",
                    base_model_id, total_queue_depth, current_replicas, capped_target
                );

                    for _ in 0..replicas_to_add {
                        // Lifecycle operations for one model are serialized so
                        // stop/remove/swap cannot interleave with replica load.
                        let _operation = models_for_scaler.lock_lifecycle(base_model_id).await;
                        let Some(model_path) = models_for_scaler.model_path(base_model_id) else {
                            continue;
                        };
                        if models_for_scaler.pool(base_model_id).is_none() {
                            continue;
                        }

                        // Get existing replica IDs to avoid collision
                        let replicas = models_for_scaler.registry().list_replicas(base_model_id);
                        let existing_replica_ids: Vec<u32> =
                            replicas.iter().map(|r| r.replica_id).collect();

                        let next_replica_id = auto_scaler_clone
                            .read()
                            .get_next_replica_id(base_model_id, &existing_replica_ids);
                        let unique_id = models_for_scaler.next_replica_unique_id();

                        match scale_up_model(
                            base_model_id,
                            next_replica_id,
                            unique_id,
                            &model_path,
                            &device_info_for_scaler,
                            resources_for_scaler.clone(),
                            batch_size_for_scaler,
                            scheduler_queue_size_for_scaler,
                            scheduler_max_micro_batch_for_scaler,
                            scheduler_queue_delay_ms_for_scaler,
                            topology_for_scaler.as_str(),
                            tp_degree_for_scaler,
                            models_for_scaler.registry(),
                            &shared_metrics_for_scaler,
                            onnx_tuning.clone(),
                        )
                        .await
                        {
                            Ok((scheduler, handle)) => {
                                // Add new replica to the pool
                                // Clone the pool to avoid holding the lock across await
                                let pool = models_for_scaler.pool(base_model_id);
                                if let Some(pool) = pool {
                                    pool.add_replica(next_replica_id, scheduler);
                                }
                                // Register engine handle for hot-swap
                                models_for_scaler.add_swap_handle(base_model_id, handle);
                            }
                            Err(e) => {
                                log::error!("Failed to scale up model {}: {}", base_model_id, e);
                            }
                        }
                    }

                    // Do not evaluate scale-down in the same cycle after scale-up.
                    continue;
                }

                // Check for scale-down
                let should_scale_down = auto_scaler_clone.write().should_scale_down(
                    base_model_id,
                    current_replicas,
                    healthy_replicas,
                    total_queue_depth,
                    elapsed,
                    metrics_available,
                );

                if let Some(target_replicas) = should_scale_down {
                    let replicas_to_remove = current_replicas.saturating_sub(target_replicas);
                    log::info!(
                    "Auto-scaler: Model {} queue depth {} below threshold, scaling from {} to {} replicas",
                    base_model_id, total_queue_depth, current_replicas, target_replicas
                );

                    // Remove replicas (highest replica_id first)
                    let replicas = models_for_scaler.registry().list_replicas(base_model_id);
                    let mut replica_ids: Vec<_> = replicas
                        .iter()
                        .filter(|r| r.replica_id > 0 && r.status == ModelStatus::Active)
                        .map(|r| (r.replica_id, r.id))
                        .collect();
                    replica_ids.sort_by(|a, b| b.0.cmp(&a.0)); // Sort descending

                    for (replica_id, unique_id) in
                        replica_ids.iter().take(replicas_to_remove as usize)
                    {
                        let _operation = models_for_scaler.lock_lifecycle(base_model_id).await;
                        if let Err(e) = scale_down_model(
                            base_model_id,
                            *replica_id,
                            *unique_id,
                            &models_for_scaler,
                        )
                        .await
                        {
                            log::error!("Failed to scale down model {}: {}", base_model_id, e);
                        }
                    }
                }
            }
        }
    });
}
