use super::*;

pub(crate) fn build_metrics_route(
    registry_arc: Arc<Registry>,
    api_auth_state_for_api: Arc<RwLock<ApiAuthState>>,
    resources: Arc<RuntimeResources>,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    // Metrics endpoint (admin scope when auth is enabled; loopback only when disabled)
    // Match the endpoint before authenticating it. If auth runs first, this
    // branch can turn a reader/writer rejection into a 401/403 response for an
    // unrelated path (including the public login route), preventing Warp from
    // trying the route that actually owns the request.
    let metrics_auth = api_auth_filter(
        ApiRole::Admin,
        ApiScope::Admin,
        api_auth_state_for_api.clone(),
    );
    let metrics_route = warp::path("metrics")
        .and(warp::path::end())
        .and(warp::get())
        .and(metrics_auth)
        .map(move || -> warp::reply::Response {
            resources.refresh_device_pool_metrics();
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
        })
        .or_else(map_api_auth_rejection);

    metrics_route.boxed()
}
