use super::*;

pub(crate) struct ModelInferRouteConfig {
    pub(crate) replica_pools: ReplicaPools,
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) latency_samples: Arc<RwLock<HashMap<u32, LatencyWindow>>>,
    pub(crate) log_sensitive_ids: bool,
    pub(crate) rag_state: RagRuntimeState,
    pub(crate) inter_model_relay_state: Arc<InterModelRelayState>,
    pub(crate) runtime_pressure_state: Arc<AtomicU8>,
    pub(crate) runtime_pressure_config: Arc<RuntimePressureConfig>,
}

pub(crate) fn build_model_infer_route(
    config: ModelInferRouteConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ModelInferRouteConfig {
        replica_pools: replica_pools_clone,
        model_registry: model_registry_clone,
        latency_samples: latency_samples_clone,
        log_sensitive_ids: log_sensitive_ids_for_api,
        rag_state: rag_state_for_api,
        inter_model_relay_state,
        runtime_pressure_state,
        runtime_pressure_config,
    } = config;

    // POST /api/models/:id/infer - Synchronous inference
    let replica_pools_for_infer = replica_pools_clone.clone();
    let model_registry_for_infer = model_registry_clone.clone();
    let latency_samples_for_infer = latency_samples_clone.clone();
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
        let latency_samples = latency_samples_for_infer.clone();
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

                    // End-to-end inference latency: scheduler queue wait + compute.
                    // This is the signal the enterprise autoscaler scales SLOs on.
                    let inference_started = std::time::Instant::now();
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
                            let latency_ms = inference_started.elapsed().as_secs_f64() * 1000.0;
                            latency_samples
                                .write()
                                .entry(model_id)
                                .or_default()
                                .record(latency_ms);
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

    infer_route.map(reply_into_response).boxed()
}
