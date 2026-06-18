use super::*;

pub(crate) struct ModelRoutes {
    pub(crate) reader: warp::filters::BoxedFilter<(warp::reply::Response,)>,
    pub(crate) admin: warp::filters::BoxedFilter<(warp::reply::Response,)>,
}

pub(crate) struct ModelRoutesConfig {
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) replica_pools: ReplicaPools,
    pub(crate) shared_metrics: kapsl_monitor::metrics::KapslMetrics,
    pub(crate) throughput_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) generated_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) total_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) device_info: Arc<DeviceInfo>,
    pub(crate) batch_size: usize,
    pub(crate) scheduler_queue_size: usize,
    pub(crate) scheduler_max_micro_batch: usize,
    pub(crate) scheduler_queue_delay_ms: u64,
    pub(crate) model_id_counter: Arc<AtomicU32>,
    pub(crate) recycled_model_ids: Arc<Mutex<Vec<u32>>>,
    pub(crate) model_paths: Arc<RwLock<HashMap<u32, PathBuf>>>,
    pub(crate) onnx_tuning_profile: Arc<OnnxTuningProfile>,
    pub(crate) shared_kv: SharedKvState,
    pub(crate) swap_map: Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
    pub(crate) rag_state: RagRuntimeState,
    pub(crate) inter_model_relay_state: Arc<InterModelRelayState>,
    pub(crate) runtime_pressure_state: Arc<AtomicU8>,
    pub(crate) runtime_pressure_config: Arc<RuntimePressureConfig>,
    pub(crate) auto_scaler: Arc<RwLock<AutoScaler>>,
    pub(crate) log_sensitive_ids: bool,
}

pub(crate) fn build_model_routes(config: ModelRoutesConfig) -> ModelRoutes {
    let ModelRoutesConfig {
        model_registry: model_registry_clone,
        replica_pools: replica_pools_clone,
        shared_metrics: shared_metrics_clone,
        throughput_samples: throughput_samples_clone,
        generated_token_samples: generated_token_samples_clone,
        total_token_samples: total_token_samples_clone,
        device_info: device_info_for_api,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        model_id_counter,
        recycled_model_ids,
        model_paths: model_paths_clone,
        onnx_tuning_profile: onnx_tuning_profile_for_api,
        shared_kv,
        swap_map,
        rag_state: rag_state_for_api,
        inter_model_relay_state,
        runtime_pressure_state,
        runtime_pressure_config,
        auto_scaler: auto_scaler_api,
        log_sensitive_ids: log_sensitive_ids_for_api,
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

    // POST /api/models/start - Start a new model
    #[derive(Deserialize)]
    struct StartModelRequest {
        model_path: String,
        model_id: Option<u32>,
        #[serde(default = "default_topology")]
        topology: String,
        #[serde(default = "default_tp_degree")]
        tp_degree: usize,
    }

    fn default_topology() -> String {
        "data-parallel".to_string()
    }

    fn default_tp_degree() -> usize {
        1
    }

    let model_registry_for_start = model_registry_clone.clone();
    let replica_pools_for_start = replica_pools_clone.clone();
    let device_info_for_start = device_info_for_api.clone();
    let batch_size_for_start = batch_size;
    let scheduler_queue_size_for_start = scheduler_queue_size;
    let scheduler_max_micro_batch_for_start = scheduler_max_micro_batch;
    let scheduler_queue_delay_ms_for_start = scheduler_queue_delay_ms;
    let shared_metrics_for_start = shared_metrics_clone.clone();
    let model_id_counter_for_start = model_id_counter.clone();
    let recycled_model_ids_for_start = recycled_model_ids.clone();
    let model_paths_for_start = model_paths_clone.clone();
    let onnx_tuning_profile_for_start = onnx_tuning_profile_for_api.clone();
    let shared_kv_for_start = shared_kv.clone();
    let swap_map_for_start = swap_map.clone();

    let start_model = warp::path!("api" / "models" / "start")
        .and(warp::post())
        .and(warp::body::json())
        .then(move |req: StartModelRequest| {
            let model_registry = model_registry_for_start.clone();
            let replica_pools = replica_pools_for_start.clone();
            let device_info = device_info_for_start.clone();
            let shared_kv = shared_kv_for_start.clone();
            let shared_metrics = shared_metrics_for_start.clone();
            let model_id_counter = model_id_counter_for_start.clone();
            let recycled_model_ids = recycled_model_ids_for_start.clone();
            let model_paths = model_paths_for_start.clone();
            let onnx_tuning_profile = onnx_tuning_profile_for_start.clone();
            let swap_map = swap_map_for_start.clone();

            async move {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct SuccessResponse {
                    message: String,
                    model_id: u32,
                }

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                // Assign ID if missing
                let (model_id, auto_assigned) = match req.model_id {
                    Some(id) => (id, false),
                    None => (
                        allocate_model_id(&model_id_counter, &recycled_model_ids),
                        true,
                    ),
                };
                let onnx_tuning = onnx_tuning_profile.resolve(model_id);

                // Check if model ID already exists
                if replica_pools.read().contains_key(&model_id) {
                    if auto_assigned {
                        recycle_model_id(model_id, &recycled_model_ids);
                    }
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Model ID {} already exists", model_id),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }
                if let Some(info) = model_registry.get(model_id) {
                    match info.status {
                        ModelStatus::Inactive => {}
                        ModelStatus::Starting | ModelStatus::Loading => {
                            if auto_assigned {
                                recycle_model_id(model_id, &recycled_model_ids);
                            }
                            return warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: format!("Model ID {} is already starting", model_id),
                                }),
                                StatusCode::BAD_REQUEST,
                            );
                        }
                        _ => {
                            if auto_assigned {
                                recycle_model_id(model_id, &recycled_model_ids);
                            }
                            return warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: format!(
                                        "Model ID {} already exists (status: {:?})",
                                        model_id, info.status
                                    ),
                                }),
                                StatusCode::BAD_REQUEST,
                            );
                        }
                    }
                }

                // Load the model
                let model_path = PathBuf::from(&req.model_path);
                log::info!(
                    "Attempting to start model {} from path: {:?}",
                    model_id,
                    model_path
                );

                if !model_path.exists() {
                    log::error!("Model path does not exist: {:?}", model_path);
                    if auto_assigned {
                        recycle_model_id(model_id, &recycled_model_ids);
                    }
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Model path does not exist: {:?}", model_path),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }
                let absolute_path = match model_path.canonicalize() {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("Failed to canonicalize model path {:?}: {}", model_path, e);
                        if auto_assigned {
                            recycle_model_id(model_id, &recycled_model_ids);
                        }
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Invalid model path {:?}: {}", model_path, e),
                            }),
                            StatusCode::BAD_REQUEST,
                        );
                    }
                };

                if let Some(info) = model_registry.get(model_id) {
                    if info.status == ModelStatus::Inactive {
                        let _ = model_registry.set_status(model_id, ModelStatus::Starting);
                    }
                }

                let model_name = absolute_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let device_str = device_info.get_best_provider().to_string();
                let optimization_level = "basic".to_string();
                let mut model_info = ModelInfo::new(
                    model_id,
                    model_name,
                    "unknown".to_string(),
                    "unknown".to_string(),
                    device_str,
                    optimization_level,
                    absolute_path.to_string_lossy().to_string(),
                );
                model_info.status = ModelStatus::Starting;
                model_registry.upsert(model_info);
                tokio::spawn({
                    let replica_pools = replica_pools.clone();
                    let model_paths = model_paths.clone();
                    let model_registry = model_registry.clone();
                    let device_info = device_info.clone();
                    let shared_metrics = shared_metrics.clone();
                    let model_path = model_path.clone();
                    let topology = req.topology.clone();
                    let recycled_model_ids = recycled_model_ids.clone();
                    let tp_degree = req.tp_degree;
                    let onnx_tuning = onnx_tuning.clone();
                    let shared_kv = shared_kv.clone();
                    async move {
                        let model_registry_clone = model_registry.clone();
                        let device_info_clone = device_info.clone();
                        let shared_metrics_clone = shared_metrics.clone();
                        let model_path_thread = model_path.clone();
                        let topology_clone = topology.clone();

                        let res = tokio::task::spawn_blocking(move || {
                            load_model_blocking(
                                model_id,
                                &model_path_thread,
                                &device_info_clone,
                                shared_kv,
                                batch_size_for_start,
                                scheduler_queue_size_for_start,
                                scheduler_max_micro_batch_for_start,
                                scheduler_queue_delay_ms_for_start,
                                &model_registry_clone,
                                &shared_metrics_clone,
                                &topology_clone,
                                tp_degree,
                                onnx_tuning.clone(),
                            )
                        })
                        .await;

                        match res {
                            Ok(Err(e)) => {
                                log::error!("Failed to load model {}: {}", model_id, e);
                                let _ = model_registry.set_status(model_id, ModelStatus::Inactive);
                                if auto_assigned {
                                    recycle_model_id(model_id, &recycled_model_ids);
                                }
                            }
                            Ok(Ok((pool, handles))) => {
                                replica_pools.write().insert(model_id, pool);
                                swap_map.write().insert(model_id, handles);
                                model_paths.write().insert(model_id, model_path);
                                let _ = model_registry.set_status(model_id, ModelStatus::Active);
                            }
                            Err(join_err) => {
                                log::error!(
                                    "Loader task panicked/cancelled for {}: {}",
                                    model_id,
                                    join_err
                                );
                                let _ = model_registry.set_status(model_id, ModelStatus::Inactive);
                                if auto_assigned {
                                    recycle_model_id(model_id, &recycled_model_ids);
                                }
                            }
                        }
                    }
                });
                warp::reply::with_status(
                    warp::reply::json(&SuccessResponse {
                        message: "Model load started".to_string(),
                        model_id,
                    }),
                    StatusCode::ACCEPTED,
                )
            }
        });

    // POST /api/models/:id/stop - Stop a model
    let model_registry_for_stop = model_registry_clone.clone();
    let replica_pools_for_stop = replica_pools_clone.clone();
    let swap_map_for_stop = swap_map.clone();
    let shared_kv_for_stop = shared_kv.clone();

    let stop_model = warp::path!("api" / "models" / u32 / "stop")
        .and(warp::post())
        .map(move |model_id: u32| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct SuccessResponse {
                message: String,
            }

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            // Check if model exists
            if !replica_pools_for_stop.read().contains_key(&model_id) {
                return warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: format!("Model ID {} not found", model_id),
                    }),
                    StatusCode::NOT_FOUND,
                );
            }

            // Update status to Stopping
            if let Err(e) = model_registry_for_stop.set_status(model_id, ModelStatus::Stopping) {
                log::warn!(
                    "Failed to set status to Stopping during stop request for {}: {}",
                    model_id,
                    e
                );
                // Proceed anyway? Yes, we want to stop.
            }

            // Remove pool (this will drop it and clean up resources)
            replica_pools_for_stop.write().remove(&model_id);
            swap_map_for_stop.write().remove(&model_id);
            shared_kv_for_stop.detach_engine_for_model(model_id);
            #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
            shared_kv_for_stop.detach_gpu_pool(model_id);

            // Update status to Inactive
            if let Err(e) = model_registry_for_stop.set_status(model_id, ModelStatus::Inactive) {
                log::warn!(
                    "Failed to set status to Inactive after stop for {}: {}",
                    model_id,
                    e
                );
            }

            warp::reply::with_status(
                warp::reply::json(&SuccessResponse {
                    message: format!("Model {} stopped successfully", model_id),
                }),
                StatusCode::OK,
            )
        });

    // POST /api/models/:id/remove - Remove a model and its replicas
    let model_registry_for_remove = model_registry_clone.clone();
    let replica_pools_for_remove = replica_pools_clone.clone();
    let model_paths_for_remove = model_paths_clone.clone();
    let swap_map_for_remove = swap_map.clone();
    let shared_kv_for_remove = shared_kv.clone();

    let remove_model = warp::path!("api" / "models" / u32 / "remove")
        .and(warp::post())
        .map(move |model_id: u32| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct SuccessResponse {
                message: String,
            }

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            let model_info = match model_registry_for_remove.get(model_id) {
                Some(info) => info,
                None => {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Model ID {} not found", model_id),
                        }),
                        StatusCode::NOT_FOUND,
                    );
                }
            };

            let base_model_id = model_info.base_model_id;
            let replicas = model_registry_for_remove.list_replicas(base_model_id);

            force_stop_model_before_remove(
                base_model_id,
                &replicas,
                &model_registry_for_remove,
                &replica_pools_for_remove,
                &swap_map_for_remove,
                &shared_kv_for_remove,
            );
            model_paths_for_remove.write().remove(&base_model_id);

            for replica in replicas {
                model_registry_for_remove.unregister(replica.id);
            }

            warp::reply::with_status(
                warp::reply::json(&SuccessResponse {
                    message: format!("Model {} removed successfully", base_model_id),
                }),
                StatusCode::OK,
            )
        });

    // Shared response types for the three hot-swap endpoints.
    #[derive(Serialize)]
    struct SwapMsgResp {
        message: String,
    }
    #[derive(Serialize)]
    struct SwapErrResp {
        error: String,
    }
    #[derive(Serialize)]
    struct SwapStatusResp {
        model_id: u32,
        supports_swap: bool,
        staged: bool,
    }

    // Returns the engine handles for a model, or an error reply if not found.
    fn lookup_swap_handles(
        swap_map: &Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
        model_id: u32,
    ) -> Result<Vec<EngineHandle>, warp::reply::WithStatus<warp::reply::Json>> {
        let g = swap_map.read();
        match g.get(&model_id) {
            Some(v) => Ok(v.clone()),
            None => Err(warp::reply::with_status(
                warp::reply::json(&SwapErrResp {
                    error: format!("model {} not found", model_id),
                }),
                warp::http::StatusCode::NOT_FOUND,
            )),
        }
    }

    // POST /api/models/:id/stage - Pre-load next model weights into CPU RAM
    let swap_map_for_stage = swap_map.clone();
    let stage_model_route = warp::path!("api" / "models" / u32 / "stage")
        .and(warp::post())
        .and(warp::body::json())
        .then(move |model_id: u32, body: serde_json::Value| {
            let swap_map = swap_map_for_stage.clone();
            async move {
                use warp::http::StatusCode;
                let path_str = match body.get("model_path").and_then(|v| v.as_str()) {
                    Some(s) => s.to_string(),
                    None => {
                        return warp::reply::with_status(
                            warp::reply::json(&SwapErrResp {
                                error: "missing model_path".into(),
                            }),
                            StatusCode::BAD_REQUEST,
                        )
                    }
                };
                let handles = match lookup_swap_handles(&swap_map, model_id) {
                    Ok(h) => h,
                    Err(r) => return r,
                };
                if !handles.iter().any(|e| e.supports_swap()) {
                    return warp::reply::with_status(
                        warp::reply::json(&SwapErrResp {
                            error: "backend does not support hot-swap".into(),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }
                let path = std::path::PathBuf::from(&path_str);
                for engine in &handles {
                    if let Err(e) = engine.stage(&path).await {
                        return warp::reply::with_status(
                            warp::reply::json(&SwapErrResp {
                                error: e.to_string(),
                            }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        );
                    }
                }
                warp::reply::with_status(
                    warp::reply::json(&SwapMsgResp {
                        message: format!(
                            "model {} staged from {}; call /swap when ready",
                            model_id, path_str
                        ),
                    }),
                    StatusCode::OK,
                )
            }
        });

    // POST /api/models/:id/swap - Activate staged weights (PCIe transfer only)
    let swap_map_for_swap = swap_map.clone();
    let swap_model_route = warp::path!("api" / "models" / u32 / "swap")
        .and(warp::post())
        .then(move |model_id: u32| {
            let swap_map = swap_map_for_swap.clone();
            async move {
                use warp::http::StatusCode;
                let handles = match lookup_swap_handles(&swap_map, model_id) {
                    Ok(h) => h,
                    Err(r) => return r,
                };
                for engine in &handles {
                    if let Err(e) = engine.swap().await {
                        return warp::reply::with_status(
                            warp::reply::json(&SwapErrResp {
                                error: e.to_string(),
                            }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        );
                    }
                }
                warp::reply::with_status(
                    warp::reply::json(&SwapMsgResp {
                        message: format!("model {} swapped to staged weights", model_id),
                    }),
                    StatusCode::OK,
                )
            }
        });

    // GET /api/models/:id/swap-status - Check hot-swap staging readiness
    let swap_map_for_status = swap_map.clone();
    let swap_status_route = warp::path!("api" / "models" / u32 / "swap-status")
        .and(warp::get())
        .then(move |model_id: u32| {
            let swap_map = swap_map_for_status.clone();
            async move {
                use warp::http::StatusCode;
                let handles = match lookup_swap_handles(&swap_map, model_id) {
                    Ok(h) => h,
                    Err(r) => return r,
                };
                let supports = handles.iter().any(|e| e.supports_swap());
                // All swap-capable engines must have weights staged.
                let staged = supports
                    && handles
                        .iter()
                        .filter(|e| e.supports_swap())
                        .all(|e| e.is_staged());
                warp::reply::with_status(
                    warp::reply::json(&SwapStatusResp {
                        model_id,
                        supports_swap: supports,
                        staged,
                    }),
                    StatusCode::OK,
                )
            }
        });

    // POST /api/models/:id/infer - Synchronous inference
    let replica_pools_for_infer = replica_pools_clone.clone();
    let model_registry_for_infer = model_registry_clone.clone();
    let request_adapters_for_infer = Arc::new(default_request_adapter_registry());
    let log_sensitive_ids_for_infer = log_sensitive_ids_for_api;
    let rag_state_for_infer = rag_state_for_api.clone();
    let inter_model_relay_for_infer = inter_model_relay_state.clone();
    let runtime_pressure_state_for_infer = runtime_pressure_state.clone();
    let runtime_pressure_config_for_infer = runtime_pressure_config.clone();
    let infer_route = warp::path!("api" / "models" / u32 / "infer")
    .and(warp::post())
    .and(warp::body::bytes())
    .and_then(move |model_id: u32, body: warp::hyper::body::Bytes| {
        let pools = replica_pools_for_infer.clone();
        let model_registry = model_registry_for_infer.clone();
        let request_adapters = request_adapters_for_infer.clone();
        let log_sensitive_ids = log_sensitive_ids_for_infer;
        let rag_state = rag_state_for_infer.clone();
        let inter_model_relay = inter_model_relay_for_infer.clone();
        let runtime_pressure_state = runtime_pressure_state_for_infer.clone();
        let runtime_pressure_config = runtime_pressure_config_for_infer.clone();
        async move {
            use warp::http::StatusCode;
            let scheduler = {
                let p = pools.read();
                p.get(&model_id).cloned()
            };

            match scheduler {
                Some(pool) => {
                    if !pool.is_healthy() {
                        let overload =
                            EngineError::overloaded("Model pool is overloaded".to_string());
                        let status = status_code_for_engine_error(&overload);
                        log::warn!(
                            "Infer request rejected: model_id={} status={} reason={}",
                            model_id,
                            status.as_u16(),
                            overload
                        );
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(
                                &serde_json::json!({ "error": overload.to_string() }),
                            ),
                            status,
                        ));
                    }

                    let source_model_info = model_registry.get(model_id);
                    let model_framework = source_model_info
                        .as_ref()
                        .map(|model| model.framework.clone())
                        .unwrap_or_else(|| "unknown".to_string());
                    let source_model_name = source_model_info
                        .as_ref()
                        .map(|model| model.name.clone())
                        .unwrap_or_else(|| format!("model-{model_id}"));
                    // Fast-path tensor JSON inference: avoid an intermediate `serde_json::Value`
                    // for huge tensors (ex: `[1,3,224,224]` float32 represented as a JSON byte
                    // array). Falling back keeps support for other adapters.
                    let (payload_has_media, rag_options, mut request) =
                        if let Ok(envelope) =
                            serde_json::from_slice::<InferPayloadEnvelope<InferenceRequest>>(
                                body.as_ref(),
                            )
                        {
                            let rag_options = match validate_infer_rag_options(envelope.rag)
                            {
                                Ok(options) => options,
                                Err(RagAugmentError::BadRequest(error)) => {
                                    return Ok::<_, warp::Rejection>(
                                        warp::reply::with_status(
                                            warp::reply::json(
                                                &serde_json::json!({ "error": error }),
                                            ),
                                            StatusCode::BAD_REQUEST,
                                        ),
                                    );
                                }
                                Err(RagAugmentError::Internal(error)) => {
                                    return Ok::<_, warp::Rejection>(
                                        warp::reply::with_status(
                                            warp::reply::json(
                                                &serde_json::json!({ "error": error }),
                                            ),
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                        ),
                                    );
                                }
                            };
                            (false, rag_options, envelope.request)
                        } else {
                            let payload: serde_json::Value =
                                match serde_json::from_slice(body.as_ref()) {
                                    Ok(payload) => payload,
                                    Err(error) => {
                                        log::warn!(
                                            "Infer payload rejected (invalid JSON): model_id={} framework={} error={}",
                                            model_id,
                                            model_framework,
                                            error
                                        );
                                        return Ok::<_, warp::Rejection>(
                                            warp::reply::with_status(
                                                warp::reply::json(&serde_json::json!({
                                                    "error": format!("Invalid infer payload: {}", error)
                                                })),
                                                StatusCode::BAD_REQUEST,
                                            ),
                                        );
                                    }
                                };

                            let payload_has_media = payload.get("media").is_some();
                            let rag_options = match parse_infer_rag_options(&payload) {
                                Ok(options) => options,
                                Err(RagAugmentError::BadRequest(error)) => {
                                    return Ok::<_, warp::Rejection>(
                                        warp::reply::with_status(
                                            warp::reply::json(
                                                &serde_json::json!({ "error": error }),
                                            ),
                                            StatusCode::BAD_REQUEST,
                                        ),
                                    );
                                }
                                Err(RagAugmentError::Internal(error)) => {
                                    return Ok::<_, warp::Rejection>(
                                        warp::reply::with_status(
                                            warp::reply::json(
                                                &serde_json::json!({ "error": error }),
                                            ),
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                        ),
                                    );
                                }
                            };

                            let request = match parse_inference_request_with_registry(
                                payload,
                                &model_framework,
                                &request_adapters,
                            ) {
                                Ok(request) => request,
                                Err(parse_error) => {
                                    let error_message = parse_error.to_string();
                                    let status = if parse_error.is_internal() {
                                        log::error!(
                                            "Infer payload preprocessing failed: model_id={} framework={} has_media={} error={}",
                                            model_id,
                                            model_framework,
                                            payload_has_media,
                                            error_message
                                        );
                                        StatusCode::INTERNAL_SERVER_ERROR
                                    } else {
                                        log::warn!(
                                            "Infer payload rejected: model_id={} framework={} has_media={} error={}",
                                            model_id,
                                            model_framework,
                                            payload_has_media,
                                            error_message
                                        );
                                        StatusCode::BAD_REQUEST
                                    };
                                    let client_error = if status
                                        == StatusCode::INTERNAL_SERVER_ERROR
                                    {
                                        "Failed to preprocess infer payload".to_string()
                                    } else {
                                        error_message
                                    };
                                    return Ok::<_, warp::Rejection>(
                                        warp::reply::with_status(
                                            warp::reply::json(&serde_json::json!({
                                                "error": client_error
                                            })),
                                            status,
                                        ),
                                    );
                                }
                            };

                            (payload_has_media, rag_options, request)
                        };
                    if payload_has_media {
                        if let Some(model_info) = pool.model_info() {
                            if let Err(error) = validate_inference_request_against_model_info(
                                &request,
                                &model_info,
                            ) {
                                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                    warp::reply::json(
                                        &serde_json::json!({ "error": format!("Media tensor validation failed: {}", error) }),
                                    ),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                        }
                    }
                    if let Some(rag_options) = rag_options {
                        match augment_inference_request_with_rag(
                            &mut request,
                            &rag_options,
                            &rag_state,
                        )
                        .await
                        {
                            Ok(chunks_used) => {
                                if chunks_used > 0 {
                                    log::debug!(
                                        "Applied RAG context: model_id={} workspace_id={} chunks_used={}",
                                        model_id,
                                        rag_options.workspace_id,
                                        chunks_used
                                    );
                                }
                            }
                            Err(RagAugmentError::BadRequest(error)) => {
                                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                    warp::reply::json(
                                        &serde_json::json!({ "error": error }),
                                    ),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                            Err(RagAugmentError::Internal(error)) => {
                                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                    warp::reply::json(
                                        &serde_json::json!({ "error": error }),
                                    ),
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                ));
                            }
                        }
                    }
                    let request_id = request
                        .metadata
                        .as_ref()
                        .and_then(|metadata| metadata.request_id.as_deref())
                        .unwrap_or("-");
                    let session_id = request.session_id.as_deref().unwrap_or("-");
                    let request_is_relay = session_id.starts_with(INTER_MODEL_RELAY_SESSION_PREFIX);
                    let request_id_for_log =
                        redact_identifier_for_logs(request_id, log_sensitive_ids);
                    let session_id_for_log =
                        redact_identifier_for_logs(session_id, log_sensitive_ids);
                    let scheduler_priority = scheduler_priority_for_request(&request);
                    let force_cpu = request
                        .metadata
                        .as_ref()
                        .and_then(|metadata| metadata.force_cpu)
                        .unwrap_or(false);
                    let pressure_state = RuntimePressureState::from_u8(
                        runtime_pressure_state.load(Ordering::Relaxed),
                    );
                    if pressure_state == RuntimePressureState::Emergency
                        && matches!(scheduler_priority, kapsl_scheduler::Priority::Throughput)
                    {
                        let error = EngineError::resource_exhausted(format!(
                            "runtime pressure {}: throughput requests are temporarily rejected",
                            pressure_state.as_str()
                        ));
                        let status = status_code_for_engine_error(&error);
                        log::warn!(
                            "Infer execution rejected: model_id={} framework={} request_id={} session_id={} status={} error={}",
                            model_id,
                            model_framework,
                            request_id_for_log,
                            session_id_for_log,
                            status.as_u16(),
                            error
                        );
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({ "error": error.to_string() })),
                            status,
                        ));
                    }
                    if let Some(cap) = runtime_pressure_config.max_new_tokens_cap(pressure_state) {
                        let metadata = request
                            .metadata
                            .get_or_insert_with(kapsl_engine_api::RequestMetadata::default);
                        metadata.max_new_tokens =
                            Some(metadata.max_new_tokens.map(|existing| existing.min(cap)).unwrap_or(cap));
                    }
                    // Attach a cancellation token so that if the HTTP request is dropped
                    // (client disconnects), we can stop any queued/in-flight work.
                    struct CancelOnDrop(kapsl_engine_api::CancellationToken);
                    impl Drop for CancelOnDrop {
                        fn drop(&mut self) {
                            self.0.cancel();
                        }
                    }

                    let cancellation_token = request
                        .cancellation
                        .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
                        .clone();
                    let _cancel_on_drop = CancelOnDrop(cancellation_token.clone());

                    let timeout_ms = request
                        .metadata
                        .as_ref()
                        .and_then(|metadata| metadata.timeout_ms)
                        .filter(|ms| *ms > 0);

                    let infer_fut = pool.infer(&request, scheduler_priority, force_cpu);
                    let infer_result = if let Some(timeout_ms) = timeout_ms {
                        match tokio::time::timeout(
                            Duration::from_millis(timeout_ms),
                            infer_fut,
                        )
                        .await
                        {
                            Ok(result) => result,
                            Err(_) => {
                                cancellation_token.cancel();
                                Err(EngineError::timeout(format!(
                                    "Inference timed out after {timeout_ms}ms"
                                )))
                            }
                        }
                    } else {
                        infer_fut.await
                    };

                    match infer_result {
                        Ok(output) => {
                            maybe_publish_inter_model_relays(
                                &inter_model_relay,
                                model_id,
                                &source_model_name,
                                request_is_relay,
                                &output,
                                &pools,
                                &model_registry,
                            );
                            Ok::<_, warp::Rejection>(warp::reply::with_status(
                                warp::reply::json(&output),
                                StatusCode::OK,
                            ))
                        }
                        Err(e) => {
                            let status = status_code_for_engine_error(&e);
                            if status == StatusCode::INTERNAL_SERVER_ERROR {
                                log::error!(
                                    "Infer execution failed: model_id={} framework={} request_id={} session_id={} status={} error={}",
                                    model_id,
                                    model_framework,
                                    request_id_for_log,
                                    session_id_for_log,
                                    status.as_u16(),
                                    e
                                );
                            } else {
                                log::warn!(
                                    "Infer execution rejected: model_id={} framework={} request_id={} session_id={} status={} error={}",
                                    model_id,
                                    model_framework,
                                    request_id_for_log,
                                    session_id_for_log,
                                    status.as_u16(),
                                    e
                                );
                            }
                            Ok(warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({ "error": e.to_string() })),
                                status,
                            ))
                        }
                    }
                }
                None => {
                    log::warn!("Infer request received for unknown model_id={}", model_id);
                    Ok(warp::reply::with_status(
                        warp::reply::json(
                            &serde_json::json!({ "error": format!("Model {} not found", model_id) }),
                        ),
                        StatusCode::NOT_FOUND,
                    ))
                }
            }
        }
    });

    // GET /api/models/:id/scaling - Get scaling policy
    let auto_scaler_for_get = auto_scaler_api.clone();
    let get_scaling = warp::path!("api" / "models" / u32 / "scaling")
        .and(warp::get())
        .map(move |model_id: u32| {
            let scaler = auto_scaler_for_get.read();
            let policy = scaler.get_policy(model_id);
            warp::reply::json(&policy)
        });

    // POST /api/models/:id/scaling - Update scaling policy
    let auto_scaler_for_post = auto_scaler_api.clone();
    let update_scaling = warp::path!("api" / "models" / u32 / "scaling")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |model_id: u32, policy: ScalingPolicy| {
            use warp::http::StatusCode;

            if let Err(error) = policy.validate() {
                return warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                );
            }

            let mut scaler = auto_scaler_for_post.write();
            scaler.register_policy(model_id, policy);
            warp::reply::with_status(
                warp::reply::json(&serde_json::json!({ "status": "ok" })),
                StatusCode::OK,
            )
        });

    let reader = list_models
        .or(get_model)
        .or(infer_route)
        .or(get_scaling)
        .map(reply_into_response)
        .boxed();
    let admin = start_model
        .or(stop_model)
        .or(remove_model)
        .or(stage_model_route)
        .or(swap_model_route)
        .or(swap_status_route)
        .or(update_scaling)
        .map(reply_into_response)
        .boxed();

    ModelRoutes { reader, admin }
}
