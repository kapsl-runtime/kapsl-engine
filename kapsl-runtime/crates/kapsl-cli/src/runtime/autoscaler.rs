use super::*;

pub(crate) struct AutoScalerTaskConfig {
    pub(crate) auto_scaler: Arc<RwLock<AutoScaler>>,
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) replica_pools: ReplicaPools,
    pub(crate) swap_map: Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
    pub(crate) model_paths: Arc<RwLock<HashMap<u32, PathBuf>>>,
    pub(crate) device_info: Arc<DeviceInfo>,
    pub(crate) unique_id_counter: Arc<AtomicU32>,
    pub(crate) shared_metrics: kapsl_monitor::metrics::KapslMetrics,
    pub(crate) shared_kv: SharedKvState,
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
        model_registry: model_registry_for_scaler,
        replica_pools: replica_pools_for_scaler,
        swap_map: swap_map_for_scaler,
        model_paths: model_paths_for_scaler,
        device_info: device_info_for_scaler,
        unique_id_counter: unique_id_counter_for_scaler,
        shared_metrics: shared_metrics_for_scaler,
        shared_kv: shared_kv_for_scaler,
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
            for model_info in model_registry_for_scaler.list() {
                let base_model_id = model_info.base_model_id;

                // Only process primary models (not replicas)
                if model_info.replica_id != 0 {
                    continue;
                }

                let current_replicas =
                    model_registry_for_scaler.count_active_replicas(base_model_id) as u32;

                // Calculate pool state and update metrics.
                let (
                    total_queue_depth,
                    healthy_replicas,
                    metrics_available,
                    total_model_memory_bytes,
                ) = if let Some(pool) = replica_pools_for_scaler.read().get(&base_model_id) {
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

                    shared_metrics_for_scaler
                        .kv_cache_bytes_used
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_bytes_used as i64);
                    shared_metrics_for_scaler
                        .kv_cache_bytes_capacity
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_bytes_capacity as i64);
                    shared_metrics_for_scaler
                        .kv_cache_blocks_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_blocks_total as i64);
                    shared_metrics_for_scaler
                        .kv_cache_blocks_free
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_blocks_free as i64);
                    shared_metrics_for_scaler
                        .kv_cache_sequences
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_sequences as i64);
                    shared_metrics_for_scaler
                        .kv_cache_evicted_blocks
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_evicted_blocks as i64);
                    shared_metrics_for_scaler
                        .kv_cache_evicted_sequences
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_evicted_sequences as i64);
                    shared_metrics_for_scaler
                        .kv_cache_packed_layers
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_packed_layers as i64);
                    shared_metrics_for_scaler
                        .kv_cache_cpu_offloaded_blocks
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_cpu_offloaded_blocks as i64);
                    shared_metrics_for_scaler
                        .prompt_tokens_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.prompt_tokens_total as i64);
                    shared_metrics_for_scaler
                        .generated_tokens_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.generated_tokens_total as i64);
                    shared_metrics_for_scaler
                        .decode_steps_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.decode_steps_total as i64);
                    shared_metrics_for_scaler
                        .decode_tokens_evaluated_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.decode_tokens_evaluated_total as i64);
                    shared_metrics_for_scaler
                        .kv_partial_reuse_hits_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_partial_reuse_hits_total as i64);
                    shared_metrics_for_scaler
                        .kv_partial_reuse_tokens_saved_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_partial_reuse_tokens_saved_total as i64);
                    shared_metrics_for_scaler
                        .engine_health
                        .with_label_values(&[&model_id_str])
                        .set(metrics.engine_health as i64);
                    shared_metrics_for_scaler
                        .onnx_session_pool_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.onnx_session_pool_total as i64);
                    shared_metrics_for_scaler
                        .onnx_session_pool_idle
                        .with_label_values(&[&model_id_str])
                        .set(metrics.onnx_session_pool_idle as i64);
                    shared_metrics_for_scaler
                        .onnx_session_pool_waits_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.onnx_session_pool_waits_total as i64);
                    shared_metrics_for_scaler
                        .onnx_session_pool_wait_seconds_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.onnx_session_pool_wait_seconds_total);

                    (high + low, healthy as u32, true, metrics.memory_usage)
                } else {
                    (0, 0, false, 0)
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
                    let onnx_tuning = onnx_tuning_profile_for_scaler.resolve(base_model_id);
                    let capped_target = cap_scale_up_target_by_memory_headroom(
                        current_replicas,
                        target_replicas,
                        total_model_memory_bytes,
                        device_info_for_scaler.total_memory,
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
                        let model_path =
                            if let Some(path) = model_paths_for_scaler.read().get(&base_model_id) {
                                path.clone()
                            } else {
                                continue;
                            };

                        // Get existing replica IDs to avoid collision
                        let replicas = model_registry_for_scaler.list_replicas(base_model_id);
                        let existing_replica_ids: Vec<u32> =
                            replicas.iter().map(|r| r.replica_id).collect();

                        let next_replica_id = auto_scaler_clone
                            .read()
                            .get_next_replica_id(base_model_id, &existing_replica_ids);
                        let unique_id = unique_id_counter_for_scaler.fetch_add(1, Ordering::SeqCst);

                        match scale_up_model(
                            base_model_id,
                            next_replica_id,
                            unique_id,
                            &model_path,
                            &device_info_for_scaler,
                            shared_kv_for_scaler.clone(),
                            batch_size_for_scaler,
                            scheduler_queue_size_for_scaler,
                            scheduler_max_micro_batch_for_scaler,
                            scheduler_queue_delay_ms_for_scaler,
                            topology_for_scaler.as_str(),
                            tp_degree_for_scaler,
                            &model_registry_for_scaler,
                            &shared_metrics_for_scaler,
                            onnx_tuning.clone(),
                        )
                        .await
                        {
                            Ok((scheduler, handle)) => {
                                // Add new replica to the pool
                                // Clone the pool to avoid holding the lock across await
                                let pool =
                                    replica_pools_for_scaler.read().get(&base_model_id).cloned();
                                if let Some(pool) = pool {
                                    pool.add_replica(next_replica_id, scheduler);
                                }
                                // Register engine handle for hot-swap
                                swap_map_for_scaler
                                    .write()
                                    .entry(base_model_id)
                                    .or_default()
                                    .push(handle);
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
                    let replicas = model_registry_for_scaler.list_replicas(base_model_id);
                    let mut replica_ids: Vec<_> = replicas
                        .iter()
                        .filter(|r| r.replica_id > 0 && r.status == ModelStatus::Active)
                        .map(|r| (r.replica_id, r.id))
                        .collect();
                    replica_ids.sort_by(|a, b| b.0.cmp(&a.0)); // Sort descending

                    for (replica_id, unique_id) in
                        replica_ids.iter().take(replicas_to_remove as usize)
                    {
                        if let Err(e) = scale_down_model(
                            base_model_id,
                            *replica_id,
                            *unique_id,
                            &model_registry_for_scaler,
                            &replica_pools_for_scaler,
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
