use super::*;

pub(crate) fn build_metrics_route(
    registry_arc: Arc<Registry>,
    api_auth_state_for_api: Arc<RwLock<ApiAuthState>>,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    // Metrics endpoint (admin scope when auth is enabled; loopback only when disabled)
    let metrics_route =
        warp::path("metrics")
            .and(warp::get())
            .map(move || -> warp::reply::Response {
                let encoder = TextEncoder::new();
                let metric_families = registry_arc.gather();
                let mut buffer = vec![];
                if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
                    log::error!("Failed to encode Prometheus metrics: {e}");
                    return warp::http::Response::builder()
                        .status(warp::http::StatusCode::INTERNAL_SERVER_ERROR)
                        .body(warp::hyper::Body::from("metrics encoding error"))
                        .unwrap_or_default();
                }
                match String::from_utf8(buffer) {
                    Ok(text) => warp::http::Response::builder()
                        .status(warp::http::StatusCode::OK)
                        .header(warp::http::header::CONTENT_TYPE, encoder.format_type())
                        .body(warp::hyper::Body::from(text))
                        .unwrap_or_default(),
                    Err(e) => {
                        log::error!("Prometheus metrics output is not valid UTF-8: {e}");
                        warp::http::Response::builder()
                            .status(warp::http::StatusCode::INTERNAL_SERVER_ERROR)
                            .body(warp::hyper::Body::from("metrics encoding error"))
                            .unwrap_or_default()
                    }
                }
            });
    let metrics_route = api_auth_filter(
        ApiRole::Admin,
        ApiScope::Admin,
        api_auth_state_for_api.clone(),
    )
    .and(metrics_route)
    .map(|response: warp::reply::Response| response)
    .or_else(map_api_auth_rejection);

    metrics_route.boxed()
}
