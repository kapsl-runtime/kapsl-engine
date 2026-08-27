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
                    let grant_match = {
                        let state = auth_state.read();
                        if !state.auth_enabled() {
                            if is_loopback_remote(remote) {
                                return Ok::<(), warp::Rejection>(());
                            }
                            return Err(warp::reject::custom(ApiLocalOnly));
                        }

                        state.grant_from_authorization_header_read(authorization.as_deref())
                    };

                    let Some(grant_match) = grant_match else {
                        return Err(warp::reject::custom(ApiUnauthorized));
                    };

                    if !grant_match.grant.role.allows(required_role) {
                        return Err(warp::reject::custom(ApiForbidden));
                    }

                    if let Some(scopes) = grant_match.grant.scopes.as_ref() {
                        if !key_scopes_allow(scopes, required_scope) {
                            return Err(warp::reject::custom(ApiForbidden));
                        }
                    }

                    if let Some(key_index) = grant_match.matched_key_index {
                        if let Some(mut state) = auth_state.try_write() {
                            state.touch_key_last_used_by_index(key_index, now_unix_seconds());
                        }
                    }

                    Ok(())
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
