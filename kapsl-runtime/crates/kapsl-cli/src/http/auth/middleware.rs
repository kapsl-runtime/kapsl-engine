//! Warp authentication middleware and rejection mapping.

use super::*;
use warp::{Filter, Reply};

#[derive(Debug)]
pub(crate) struct ApiUnauthorized;

impl warp::reject::Reject for ApiUnauthorized {}

#[derive(Debug)]
pub(crate) struct ApiForbidden;

impl warp::reject::Reject for ApiForbidden {}

#[derive(Debug)]
pub(crate) struct ApiLocalOnly;

impl warp::reject::Reject for ApiLocalOnly {}

pub(crate) fn is_loopback_remote(remote: Option<std::net::SocketAddr>) -> bool {
    remote.is_some_and(|addr| addr.ip().is_loopback())
}

pub(crate) fn api_auth_filter(
    required_role: ApiRole,
    required_scope: ApiScope,
    auth_state: Arc<RwLock<ApiAuthState>>,
) -> impl Filter<Extract = (), Error = warp::Rejection> + Clone {
    warp::header::optional::<String>("authorization")
        .and(warp::addr::remote())
        .and_then(
            move |authorization: Option<String>, remote: Option<std::net::SocketAddr>| {
                let auth_state = auth_state.clone();
                async move {
                    authorize_api_request(
                        &auth_state,
                        required_role,
                        required_scope,
                        authorization.as_deref(),
                        remote.map(|address| address.ip()),
                    )
                    .map_err(|error| match error {
                        ApiAuthorizationError::Unauthorized => {
                            warp::reject::custom(ApiUnauthorized)
                        }
                        ApiAuthorizationError::Forbidden => warp::reject::custom(ApiForbidden),
                        ApiAuthorizationError::LocalOnly => warp::reject::custom(ApiLocalOnly),
                    })
                }
            },
        )
        .untuple_one()
}

pub(crate) async fn map_api_auth_rejection(
    rejection: warp::Rejection,
) -> Result<(warp::reply::Response,), warp::Rejection> {
    if rejection.find::<ApiForbidden>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Forbidden"
            })),
            warp::http::StatusCode::FORBIDDEN,
        )
        .into_response(),));
    }
    if rejection.find::<ApiUnauthorized>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Unauthorized"
            })),
            warp::http::StatusCode::UNAUTHORIZED,
        )
        .into_response(),));
    }
    if rejection.find::<ApiLocalOnly>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Unauthorized",
                "detail": "Authentication is disabled; this endpoint is restricted to loopback clients only."
            })),
            warp::http::StatusCode::FORBIDDEN,
        )
        .into_response(),));
    }
    Err(rejection)
}
