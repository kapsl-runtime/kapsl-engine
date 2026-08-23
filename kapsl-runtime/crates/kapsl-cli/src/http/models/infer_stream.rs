use super::*;
use futures::StreamExt;

pub(crate) struct ModelInferStreamRouteConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) log_sensitive_ids: bool,
    pub(crate) rag_state: RagRuntimeState,
}

/// `POST /api/models/:id/infer/stream` — Server-Sent Events token streaming.
///
/// This is the HTTP/SSE sibling of the binary `OP_INFER_STREAM` transport: it
/// drives the same `ReplicaPool::infer_stream` scheduler path and emits each
/// generated token as an SSE `data:` event of the form
/// `{"text": "<token>", "shape": [...], "dtype": "..."}`, terminated by
/// `data: [DONE]`. `text` is the decoded token; the kapsl-enterprise gateway
/// reads it to back OpenAI streaming chat completions.
pub(crate) fn build_model_infer_stream_route(
    config: ModelInferStreamRouteConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ModelInferStreamRouteConfig {
        models,
        inference,
        log_sensitive_ids,
        rag_state,
    } = config;

    let request_adapters = Arc::new(default_request_adapter_registry());

    warp::path!("api" / "models" / u32 / "infer" / "stream")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::bytes())
        .and_then(move |model_id: u32, authorization: Option<String>, body: warp::hyper::body::Bytes| {
            let models = models.clone();
            let inference = inference.clone();
            let request_adapters = request_adapters.clone();
            let rag_state = rag_state.clone();
            async move {
                use warp::http::StatusCode;

                // Build a JSON error response as a concrete `warp::reply::Response`
                // so every branch shares the and_then return type.
                let json_error = |status: StatusCode, message: String| {
                    reply_into_response(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": message })),
                        status,
                    ))
                };

                if models.pool(model_id).is_none() {
                    log::warn!("Infer-stream request received for unknown model_id={}", model_id);
                    return Ok::<_, warp::Rejection>(json_error(
                        StatusCode::NOT_FOUND,
                        format!("Model {model_id} not found"),
                    ));
                }

                let model_framework = models
                    .registry()
                    .get(model_id)
                    .map(|model| model.framework.clone())
                    .unwrap_or_else(|| "unknown".to_string());

                // Parse the request, preferring the fast-path envelope (what the
                // gateway and SDK send) and falling back to the framework adapter.
                let (rag_options, mut request) =
                    if let Ok(envelope) = serde_json::from_slice::<
                        InferPayloadEnvelope<InferenceRequest>,
                    >(body.as_ref())
                    {
                        let rag_options = match validate_infer_rag_options(envelope.rag) {
                            Ok(options) => options,
                            Err(RagAugmentError::BadRequest(error)) => {
                                return Ok(json_error(StatusCode::BAD_REQUEST, error));
                            }
                            Err(RagAugmentError::Internal(error)) => {
                                return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, error));
                            }
                        };
                        (rag_options, envelope.request)
                    } else {
                        let payload: serde_json::Value = match serde_json::from_slice(body.as_ref())
                        {
                            Ok(payload) => payload,
                            Err(error) => {
                                log::warn!(
                                    "Infer-stream payload rejected (invalid JSON): model_id={} framework={} error={}",
                                    model_id,
                                    model_framework,
                                    error
                                );
                                return Ok(json_error(
                                    StatusCode::BAD_REQUEST,
                                    format!("Invalid infer payload: {error}"),
                                ));
                            }
                        };
                        let rag_options = match parse_infer_rag_options(&payload) {
                            Ok(options) => options,
                            Err(RagAugmentError::BadRequest(error)) => {
                                return Ok(json_error(StatusCode::BAD_REQUEST, error));
                            }
                            Err(RagAugmentError::Internal(error)) => {
                                return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, error));
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
                                let (status, client_error) = if parse_error.is_internal() {
                                    log::error!(
                                        "Infer-stream payload preprocessing failed: model_id={} framework={} error={}",
                                        model_id,
                                        model_framework,
                                        error_message
                                    );
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        "Failed to preprocess infer payload".to_string(),
                                    )
                                } else {
                                    log::warn!(
                                        "Infer-stream payload rejected: model_id={} framework={} error={}",
                                        model_id,
                                        model_framework,
                                        error_message
                                    );
                                    (StatusCode::BAD_REQUEST, error_message)
                                };
                                return Ok(json_error(status, client_error));
                            }
                        };
                        (rag_options, request)
                    };

                if let Some(rag_options) = rag_options {
                    match augment_inference_request_with_rag(&mut request, &rag_options, &rag_state)
                        .await
                    {
                        Ok(_) => {}
                        Err(RagAugmentError::BadRequest(error)) => {
                            return Ok(json_error(StatusCode::BAD_REQUEST, error));
                        }
                        Err(RagAugmentError::Internal(error)) => {
                            return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, error));
                        }
                    }
                }

                let client_session_id = request.session_id.clone();
                request.session_id = scope_session_id_for_authorization(
                    client_session_id.as_deref(),
                    authorization.as_deref(),
                );
                let request_id = request
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.request_id.as_deref())
                    .unwrap_or("-");
                let session_id = client_session_id.as_deref().unwrap_or("-");
                let request_id_for_log = redact_identifier_for_logs(request_id, log_sensitive_ids);
                let session_id_for_log = redact_identifier_for_logs(session_id, log_sensitive_ids);

                let scheduler_priority = inference.priority_for_request(&request);
                let force_cpu = request
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.force_cpu)
                    .unwrap_or(false);

                let stream = match inference
                    .infer_stream(model_id, request, scheduler_priority, force_cpu)
                    .await
                {
                    Ok(stream) => stream,
                    Err(error) => {
                        let status = status_code_for_engine_error(&error);
                        log::warn!(
                            "Infer-stream start failed: model_id={} framework={} request_id={} session_id={} status={} error={}",
                            model_id,
                            model_framework,
                            request_id_for_log,
                            session_id_for_log,
                            status.as_u16(),
                            error
                        );
                        return Ok(json_error(status, error.to_string()));
                    }
                };

                let worker_label = format!("model-{model_id}");
                let sse = stream
                    .map(move |item| {
                        let payload = match item {
                            // Emit the decoded token text under `text` rather than the
                            // raw packet: `BinaryTensorPacket`'s JSON form carries bytes
                            // as `data_base64`, which the gateway's text extractor can't
                            // read. `text` is its first-choice key and is what any plain
                            // SSE client wants anyway. `shape`/`dtype` are kept for
                            // clients that need the tensor metadata.
                            Ok(packet) => {
                                let text = String::from_utf8_lossy(&packet.data);
                                serde_json::json!({
                                    "text": text,
                                    "shape": packet.shape,
                                    "dtype": packet.dtype,
                                })
                                .to_string()
                            }
                            Err(error) => {
                                // Surface as an SSE comment (`:`), which streaming
                                // clients and the gateway both skip, then let the
                                // stream end rather than emitting a bogus token.
                                log::warn!(
                                    "Infer-stream chunk error on {}: {}",
                                    worker_label,
                                    error
                                );
                                let message = error.to_string().replace(['\n', '\r'], " ");
                                return Ok::<_, std::convert::Infallible>(
                                    warp::hyper::body::Bytes::from(format!(": error {message}\n\n")),
                                );
                            }
                        };
                        Ok(warp::hyper::body::Bytes::from(format!("data: {payload}\n\n")))
                    })
                    .chain(futures::stream::once(async {
                        Ok::<_, std::convert::Infallible>(warp::hyper::body::Bytes::from(
                            "data: [DONE]\n\n",
                        ))
                    }));

                let body = warp::hyper::Body::wrap_stream(sse);
                let response = warp::http::Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream; charset=utf-8")
                    .header("cache-control", "no-cache")
                    // Disable proxy buffering so tokens flush to the client live.
                    .header("x-accel-buffering", "no")
                    .body(body)
                    .expect("SSE response should build");
                Ok(response)
            }
        })
        .map(reply_into_response)
        .boxed()
}
