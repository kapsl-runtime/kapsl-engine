use super::*;

pub(crate) struct ModelReaderRoutesConfig {
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) replica_pools: ReplicaPools,
    pub(crate) shared_metrics: kapsl_monitor::metrics::KapslMetrics,
    pub(crate) throughput_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) generated_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) total_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
}

pub(crate) fn build_model_reader_routes(
    config: ModelReaderRoutesConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ModelReaderRoutesConfig {
        model_registry: model_registry_clone,
        replica_pools: replica_pools_clone,
        shared_metrics: shared_metrics_clone,
        throughput_samples: throughput_samples_clone,
        generated_token_samples: generated_token_samples_clone,
        total_token_samples: total_token_samples_clone,
    } = config;

    let model_registry_for_list = model_registry_clone.clone();
    let replica_pools_for_list = replica_pools_clone.clone();
    let metrics_for_list = shared_metrics_clone.clone();
    let throughput_samples_for_list = throughput_samples_clone.clone();
    let generated_token_samples_for_list = generated_token_samples_clone.clone();
    let total_token_samples_for_list = total_token_samples_clone.clone();
    let list_models = warp::path!("api" / "models").and(warp::get()).map(move || {
        #[derive(Serialize)]
        struct ModelStatus {
            #[serde(flatten)]
            info: ModelInfo,
            active_inferences: i64,
            total_inferences: u64,
            queue_depth: (usize, usize),
            memory_usage: usize,
            gpu_utilization: f64,
            throughput: f64,
            prompt_tokens_total: u64,
            generated_tokens_total: u64,
            generated_tokens_per_sec: f64,
            total_tokens_per_sec: f64,
            decode_steps_total: u64,
            decode_tokens_evaluated_total: u64,
            avg_tokens_evaluated_per_decode_step: f64,
            kv_partial_reuse_hits_total: u64,
            kv_partial_reuse_tokens_saved_total: u64,
            onnx_session_pool_total: usize,
            onnx_session_pool_idle: usize,
            onnx_session_pool_waits_total: u64,
            onnx_session_pool_wait_seconds_total: f64,
            healthy: bool,
        }

        let models = model_registry_for_list.list();
        let mut statuses = Vec::new();
        let now = Instant::now();
        let mut seen_ids = HashSet::new();
        let mut throughput_samples = throughput_samples_for_list.write();
        let mut generated_token_samples = generated_token_samples_for_list.write();
        let mut total_token_samples = total_token_samples_for_list.write();

        for model in models {
            seen_ids.insert(model.id);
            let model_id_str = model.id.to_string();
            let active = metrics_for_list
                .active_inferences
                .with_label_values(&[&model_id_str])
                .get();

            let ok_label = "ok".to_string();
            let err_label = "err".to_string();
            let total = metrics_for_list
                .inference_count
                .with_label_values(&[&model_id_str, &ok_label])
                .get()
                + metrics_for_list
                    .inference_count
                    .with_label_values(&[&model_id_str, &err_label])
                    .get();

            let (
                queue_depth,
                healthy,
                engine_memory,
                engine_gpu_util,
                prompt_tokens_total,
                generated_tokens_total,
                decode_steps_total,
                decode_tokens_evaluated_total,
                kv_partial_reuse_hits_total,
                kv_partial_reuse_tokens_saved_total,
                onnx_session_pool_total,
                onnx_session_pool_idle,
                onnx_session_pool_waits_total,
                onnx_session_pool_wait_seconds_total,
            ) = if let Some(pool) = replica_pools_for_list.read().get(&model.id) {
                let metrics = pool.get_metrics();
                (
                    pool.get_queue_depth(),
                    pool.is_healthy(),
                    metrics.memory_usage,
                    metrics.gpu_utilization,
                    metrics.prompt_tokens_total,
                    metrics.generated_tokens_total,
                    metrics.decode_steps_total,
                    metrics.decode_tokens_evaluated_total,
                    metrics.kv_partial_reuse_hits_total,
                    metrics.kv_partial_reuse_tokens_saved_total,
                    metrics.onnx_session_pool_total,
                    metrics.onnx_session_pool_idle,
                    metrics.onnx_session_pool_waits_total,
                    metrics.onnx_session_pool_wait_seconds_total,
                )
            } else {
                ((0, 0), true, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
            };

            // `memory_usage` and `gpu_utilization` are engine-reported metrics only.
            // System-level RSS/GPU stats are exposed separately via GET /api/system/stats.
            let memory_usage = engine_memory;
            let gpu_utilization = engine_gpu_util;
            let throughput = update_throughput(&mut throughput_samples, model.id, total, now);
            let generated_tokens_per_sec = update_throughput(
                &mut generated_token_samples,
                model.id,
                generated_tokens_total,
                now,
            );
            let total_tokens_per_sec = update_throughput(
                &mut total_token_samples,
                model.id,
                prompt_tokens_total.saturating_add(generated_tokens_total),
                now,
            );
            let avg_tokens_evaluated_per_decode_step = if decode_steps_total > 0 {
                decode_tokens_evaluated_total as f64 / decode_steps_total as f64
            } else {
                0.0
            };

            statuses.push(ModelStatus {
                info: model,
                active_inferences: active,
                total_inferences: total,
                queue_depth,
                memory_usage,
                gpu_utilization,
                throughput,
                prompt_tokens_total,
                generated_tokens_total,
                generated_tokens_per_sec,
                total_tokens_per_sec,
                decode_steps_total,
                decode_tokens_evaluated_total,
                avg_tokens_evaluated_per_decode_step,
                kv_partial_reuse_hits_total,
                kv_partial_reuse_tokens_saved_total,
                onnx_session_pool_total,
                onnx_session_pool_idle,
                onnx_session_pool_waits_total,
                onnx_session_pool_wait_seconds_total,
                healthy,
            });
        }

        throughput_samples.retain(|id, _| seen_ids.contains(id));
        generated_token_samples.retain(|id, _| seen_ids.contains(id));
        total_token_samples.retain(|id, _| seen_ids.contains(id));
        warp::reply::json(&statuses)
    });

    let model_registry_for_get = model_registry_clone.clone();
    let replica_pools_for_get = replica_pools_clone.clone();
    let metrics_for_get = shared_metrics_clone.clone();
    let throughput_samples_for_get = throughput_samples_clone.clone();
    let generated_token_samples_for_get = generated_token_samples_clone.clone();
    let total_token_samples_for_get = total_token_samples_clone.clone();
    let get_model =
        warp::path!("api" / "models" / u32)
            .and(warp::get())
            .map(move |model_id: u32| {
                #[derive(Serialize)]
                struct ModelDetailStatus {
                    #[serde(flatten)]
                    info: ModelInfo,
                    active_inferences: i64,
                    total_inferences: u64,
                    successful_inferences: u64,
                    failed_inferences: u64,
                    queue_depth: (usize, usize),
                    memory_usage: usize,
                    gpu_utilization: f64,
                    throughput: f64,
                    prompt_tokens_total: u64,
                    generated_tokens_total: u64,
                    generated_tokens_per_sec: f64,
                    total_tokens_per_sec: f64,
                    decode_steps_total: u64,
                    decode_tokens_evaluated_total: u64,
                    avg_tokens_evaluated_per_decode_step: f64,
                    kv_partial_reuse_hits_total: u64,
                    kv_partial_reuse_tokens_saved_total: u64,
                    onnx_session_pool_total: usize,
                    onnx_session_pool_idle: usize,
                    onnx_session_pool_waits_total: u64,
                    onnx_session_pool_wait_seconds_total: f64,
                    healthy: bool,
                }

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                match model_registry_for_get.get(model_id) {
                    Some(model) => {
                        let model_id_str = model.id.to_string();
                        let active = metrics_for_get
                            .active_inferences
                            .with_label_values(&[&model_id_str])
                            .get();

                        let ok_label = "ok".to_string();
                        let err_label = "err".to_string();
                        let successful = metrics_for_get
                            .inference_count
                            .with_label_values(&[&model_id_str, &ok_label])
                            .get();

                        let failed = metrics_for_get
                            .inference_count
                            .with_label_values(&[&model_id_str, &err_label])
                            .get();

                        let (
                            queue_depth,
                            healthy,
                            engine_memory,
                            engine_gpu_util,
                            prompt_tokens_total,
                            generated_tokens_total,
                            decode_steps_total,
                            decode_tokens_evaluated_total,
                            kv_partial_reuse_hits_total,
                            kv_partial_reuse_tokens_saved_total,
                            onnx_session_pool_total,
                            onnx_session_pool_idle,
                            onnx_session_pool_waits_total,
                            onnx_session_pool_wait_seconds_total,
                        ) = if let Some(pool) = replica_pools_for_get.read().get(&model.id) {
                            let metrics = pool.get_metrics();
                            (
                                pool.get_queue_depth(),
                                pool.is_healthy(),
                                metrics.memory_usage,
                                metrics.gpu_utilization,
                                metrics.prompt_tokens_total,
                                metrics.generated_tokens_total,
                                metrics.decode_steps_total,
                                metrics.decode_tokens_evaluated_total,
                                metrics.kv_partial_reuse_hits_total,
                                metrics.kv_partial_reuse_tokens_saved_total,
                                metrics.onnx_session_pool_total,
                                metrics.onnx_session_pool_idle,
                                metrics.onnx_session_pool_waits_total,
                                metrics.onnx_session_pool_wait_seconds_total,
                            )
                        } else {
                            ((0, 0), true, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
                        };

                        // `memory_usage` and `gpu_utilization` are engine-reported metrics only.
                        // System-level RSS/GPU stats are exposed separately via GET /api/system/stats.
                        let memory_usage = engine_memory;
                        let gpu_utilization = engine_gpu_util;
                        let throughput = {
                            let now = Instant::now();
                            let mut throughput_samples = throughput_samples_for_get.write();
                            update_throughput(
                                &mut throughput_samples,
                                model.id,
                                successful + failed,
                                now,
                            )
                        };
                        let (generated_tokens_per_sec, total_tokens_per_sec) = {
                            let now = Instant::now();
                            let mut generated_token_samples =
                                generated_token_samples_for_get.write();
                            let mut total_token_samples = total_token_samples_for_get.write();
                            (
                                update_throughput(
                                    &mut generated_token_samples,
                                    model.id,
                                    generated_tokens_total,
                                    now,
                                ),
                                update_throughput(
                                    &mut total_token_samples,
                                    model.id,
                                    prompt_tokens_total.saturating_add(generated_tokens_total),
                                    now,
                                ),
                            )
                        };
                        let avg_tokens_evaluated_per_decode_step = if decode_steps_total > 0 {
                            decode_tokens_evaluated_total as f64 / decode_steps_total as f64
                        } else {
                            0.0
                        };

                        let status = ModelDetailStatus {
                            info: model,
                            active_inferences: active,
                            total_inferences: successful + failed,
                            successful_inferences: successful,
                            failed_inferences: failed,
                            queue_depth,
                            memory_usage,
                            gpu_utilization,
                            throughput,
                            prompt_tokens_total,
                            generated_tokens_total,
                            generated_tokens_per_sec,
                            total_tokens_per_sec,
                            decode_steps_total,
                            decode_tokens_evaluated_total,
                            avg_tokens_evaluated_per_decode_step,
                            kv_partial_reuse_hits_total,
                            kv_partial_reuse_tokens_saved_total,
                            onnx_session_pool_total,
                            onnx_session_pool_idle,
                            onnx_session_pool_waits_total,
                            onnx_session_pool_wait_seconds_total,
                            healthy,
                        };

                        warp::reply::json(&status)
                    }
                    None => warp::reply::json(&ErrorResponse {
                        error: format!("Model {} not found", model_id),
                    }),
                }
            });

    list_models.or(get_model).map(reply_into_response).boxed()
}
