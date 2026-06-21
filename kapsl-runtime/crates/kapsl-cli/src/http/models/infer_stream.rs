use super::*;
use futures::StreamExt;
use warp::Filter;
use warp::Reply;

pub(crate) struct ModelInferStreamRouteConfig {
    pub(crate) replica_pools: ReplicaPools,
}

/// `POST /api/models/:id/infer/stream` — streams generated tokens incrementally
/// as Server-Sent Events. Each `data:` frame is a JSON-encoded engine output
/// packet; the stream terminates with `data: [DONE]`. Backed by the replica
/// pool's `infer_stream`, so the client sees tokens as they are produced rather
/// than waiting for the full completion.
pub(crate) fn build_model_infer_stream_route(
    config: ModelInferStreamRouteConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ModelInferStreamRouteConfig { replica_pools } = config;

    let route = warp::path!("api" / "models" / u32 / "infer" / "stream")
        .and(warp::post())
        .and(warp::body::bytes())
        .and_then(move |model_id: u32, body: warp::hyper::body::Bytes| {
            let pools = replica_pools.clone();
            async move {
                use warp::http::StatusCode;

                let request = match parse_stream_request(body.as_ref()) {
                    Ok(request) => request,
                    Err(error) => {
                        return Ok::<_, warp::Rejection>(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({ "error": error })),
                                StatusCode::BAD_REQUEST,
                            )
                            .into_response(),
                        );
                    }
                };

                let pool = pools.read().get(&model_id).cloned();
                let Some(pool) = pool else {
                    return Ok(warp::reply::with_status(
                        warp::reply::json(
                            &serde_json::json!({ "error": format!("Model {model_id} not found") }),
                        ),
                        StatusCode::NOT_FOUND,
                    )
                    .into_response());
                };

                let priority = scheduler_priority_for_request(&request);
                let force_cpu = request
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.force_cpu)
                    .unwrap_or(false);

                match pool.infer_stream(request, priority, force_cpu).await {
                    Ok(stream) => {
                        // Map each engine packet to an SSE frame, then close with [DONE].
                        let events = stream
                            .map(|item| match item {
                                Ok(packet) => serde_json::to_string(&packet)
                                    .map(|json| warp::sse::Event::default().data(json)),
                                Err(engine_error) => Ok(warp::sse::Event::default()
                                    .event("error")
                                    .data(engine_error.to_string())),
                            })
                            .chain(futures::stream::once(async {
                                Ok(warp::sse::Event::default().data("[DONE]"))
                            }));
                        Ok(warp::sse::reply(warp::sse::keep_alive().stream(events)).into_response())
                    }
                    Err(engine_error) => {
                        let status = status_code_for_engine_error(&engine_error);
                        Ok(warp::reply::with_status(
                            warp::reply::json(
                                &serde_json::json!({ "error": engine_error.to_string() }),
                            ),
                            status,
                        )
                        .into_response())
                    }
                }
            }
        });

    route.map(reply_into_response).boxed()
}

/// Accept either the binary inference envelope or a bare `InferenceRequest`.
fn parse_stream_request(body: &[u8]) -> Result<InferenceRequest, String> {
    if let Ok(envelope) = serde_json::from_slice::<InferPayloadEnvelope<InferenceRequest>>(body) {
        return Ok(envelope.request);
    }
    serde_json::from_slice::<InferenceRequest>(body)
        .map_err(|error| format!("Invalid inference request: {error}"))
}
