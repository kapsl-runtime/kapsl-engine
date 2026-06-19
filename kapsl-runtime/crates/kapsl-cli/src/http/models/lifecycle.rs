use super::*;

pub(crate) struct ModelLifecycleRoutesConfig {
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) replica_pools: ReplicaPools,
    pub(crate) device_info: Arc<DeviceInfo>,
    pub(crate) batch_size: usize,
    pub(crate) scheduler_queue_size: usize,
    pub(crate) scheduler_max_micro_batch: usize,
    pub(crate) scheduler_queue_delay_ms: u64,
    pub(crate) shared_metrics: kapsl_monitor::metrics::KapslMetrics,
    pub(crate) model_id_counter: Arc<AtomicU32>,
    pub(crate) recycled_model_ids: Arc<Mutex<Vec<u32>>>,
    pub(crate) model_paths: Arc<RwLock<HashMap<u32, PathBuf>>>,
    pub(crate) onnx_tuning_profile: Arc<OnnxTuningProfile>,
    pub(crate) shared_kv: SharedKvState,
    pub(crate) swap_map: Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
}

pub(crate) fn build_model_lifecycle_routes(
    config: ModelLifecycleRoutesConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ModelLifecycleRoutesConfig {
        model_registry: model_registry_clone,
        replica_pools: replica_pools_clone,
        device_info: device_info_for_api,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        shared_metrics: shared_metrics_clone,
        model_id_counter,
        recycled_model_ids,
        model_paths: model_paths_clone,
        onnx_tuning_profile: onnx_tuning_profile_for_api,
        shared_kv,
        swap_map,
    } = config;

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

    start_model
        .or(stop_model)
        .or(remove_model)
        .map(reply_into_response)
        .boxed()
}
