use super::*;

pub(crate) struct ModelLifecycleRoutesConfig {
    pub(crate) model_runtime: Arc<ModelRuntime>,
}

pub(crate) fn build_model_lifecycle_routes(
    config: ModelLifecycleRoutesConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ModelLifecycleRoutesConfig { model_runtime } = config;
    let models = model_runtime.models().clone();

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

    let models_for_start = models.clone();
    let model_runtime_for_start = model_runtime.clone();

    let start_model = warp::path!("api" / "models" / "start")
        .and(warp::post())
        .and(warp::body::json())
        .then(move |req: StartModelRequest| {
            let models = models_for_start.clone();
            let model_runtime = model_runtime_for_start.clone();

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
                    None => (models.allocate_model_id(), true),
                };
                let lifecycle_guard = models.lock_lifecycle(model_id).await;

                // Check if model ID already exists
                if models.contains_pool(model_id) {
                    if auto_assigned {
                        models.release_model_id(model_id);
                    }
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Model ID {} already exists", model_id),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }
                if let Some(info) = models.registry().get(model_id) {
                    match info.status {
                        ModelStatus::Inactive => {}
                        ModelStatus::Starting | ModelStatus::Loading => {
                            if auto_assigned {
                                models.release_model_id(model_id);
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
                                models.release_model_id(model_id);
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
                        models.release_model_id(model_id);
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
                            models.release_model_id(model_id);
                        }
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Invalid model path {:?}: {}", model_path, e),
                            }),
                            StatusCode::BAD_REQUEST,
                        );
                    }
                };

                if let Some(info) = models.registry().get(model_id) {
                    if info.status == ModelStatus::Inactive {
                        let _ = models
                            .registry()
                            .set_status(model_id, ModelStatus::Starting);
                    }
                }

                let model_name = absolute_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                // Provider selection happens inside the asynchronous load. Do
                // not claim that the detected fastest device is the selected
                // provider while the model is still starting.
                let device_str = "pending".to_string();
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
                models.registry().upsert(model_info);
                tokio::spawn({
                    let models = models.clone();
                    let model_registry = models.registry().clone();
                    let model_path = model_path.clone();
                    let placement = ModelLoadPlacement::new(req.topology.clone(), req.tp_degree);
                    let model_runtime = model_runtime.clone();
                    async move {
                        let _lifecycle_guard = lifecycle_guard;
                        // A typed authority rejection may reclaim one idle,
                        // strictly lower-weight victim at a time and retry from
                        // a clean load transaction. Ordinary backend failures
                        // never enter arbitration.
                        let mut reclaim_attempts = 0usize;
                        let res = loop {
                            let model_path_for_attempt = model_path.clone();
                            let placement_for_attempt = placement.clone();
                            let model_runtime_for_attempt = model_runtime.clone();
                            let attempt = tokio::task::spawn_blocking(move || {
                                model_runtime_for_attempt.load_model_blocking(
                                    model_id,
                                    &model_path_for_attempt,
                                    &placement_for_attempt,
                                )
                            })
                            .await;

                            let Ok(Err(error)) = &attempt else {
                                break attempt;
                            };
                            let Some(failure) = error.downcast_ref::<MemoryAdmissionFailure>() else {
                                break attempt;
                            };
                            if reclaim_attempts >= 32 {
                                log::warn!(
                                    "Priority arbitration for model {} reached its bounded retry limit",
                                    model_id
                                );
                                break attempt;
                            }
                            let Some(_reclaimed) = model_runtime
                                .reclaim_one_lower_priority_model(failure)
                                .await
                            else {
                                break attempt;
                            };
                            reclaim_attempts += 1;
                        };

                        match res {
                            Ok(Err(e)) => {
                                log::error!("Failed to load model {}: {}", model_id, e);
                                let _ = model_registry.set_status(model_id, ModelStatus::Inactive);
                                if auto_assigned {
                                    model_registry.unregister(model_id);
                                    models.release_model_id(model_id);
                                }
                            }
                            Ok(Ok((pool, handles))) => {
                                models.install_loaded(model_id, model_path, pool, handles);
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
                                    model_registry.unregister(model_id);
                                    models.release_model_id(model_id);
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
    let models_for_stop = models.clone();
    let model_runtime_for_stop = model_runtime.clone();

    let stop_model = warp::path!("api" / "models" / u32 / "stop")
        .and(warp::post())
        .then(move |model_id: u32| {
            let models = models_for_stop.clone();
            let model_runtime = model_runtime_for_stop.clone();
            async move {
                use warp::http::StatusCode;
                let _lifecycle_guard = models.lock_lifecycle(model_id).await;

                #[derive(Serialize)]
                struct SuccessResponse {
                    message: String,
                }

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                // Check if model exists
                if !models.contains_pool(model_id) {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Model ID {} not found", model_id),
                        }),
                        StatusCode::NOT_FOUND,
                    );
                }

                if let Err(error) = model_runtime.stop_model_and_replicas(model_id) {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error }),
                        StatusCode::INTERNAL_SERVER_ERROR,
                    );
                }

                warp::reply::with_status(
                    warp::reply::json(&SuccessResponse {
                        message: format!("Model {} stopped successfully", model_id),
                    }),
                    StatusCode::OK,
                )
            }
        });

    // POST /api/models/:id/remove - Remove a model and its replicas
    let models_for_remove = models.clone();
    let model_runtime_for_remove = model_runtime.clone();

    let remove_model = warp::path!("api" / "models" / u32 / "remove")
        .and(warp::post())
        .then(move |model_id: u32| {
            let models = models_for_remove.clone();
            let model_runtime = model_runtime_for_remove.clone();
            async move {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct SuccessResponse {
                    message: String,
                }

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let model_info = match models.registry().get(model_id) {
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
                let _lifecycle_guard = models.lock_lifecycle(base_model_id).await;
                let replicas = match model_runtime.stop_model_and_replicas(base_model_id) {
                    Ok(replicas) => replicas,
                    Err(error) => {
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse { error }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        );
                    }
                };
                models.remove(base_model_id);

                for replica in replicas {
                    models.registry().unregister(replica.id);
                }

                warp::reply::with_status(
                    warp::reply::json(&SuccessResponse {
                        message: format!("Model {} removed successfully", base_model_id),
                    }),
                    StatusCode::OK,
                )
            }
        });

    start_model
        .or(stop_model)
        .or(remove_model)
        .map(reply_into_response)
        .boxed()
}
