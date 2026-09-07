//! Public authentication login route.

use super::*;

pub(super) fn build_login_route(
    auth_state: Arc<RwLock<ApiAuthState>>,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    warp::path!("api" / "auth" / "login")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::addr::remote())
        .and(warp::body::json::<ApiAuthLoginRequest>())
        .map(
            move |authorization: Option<String>,
                  remote: Option<std::net::SocketAddr>,
                  payload: ApiAuthLoginRequest| {
                use warp::http::StatusCode;

                let token_from_body = normalize_optional_text(payload.token);
                let normalized_authorization = authorization
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .or(token_from_body);

                let access = match authorize_api_request(
                    &auth_state,
                    ApiRole::Reader,
                    ApiScope::Read,
                    normalized_authorization.as_deref(),
                    remote.map(|address| address.ip()),
                ) {
                    Ok(access) => access,
                    Err(error) => {
                        let (status, message, detail) = match error {
                            ApiAuthorizationError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized", "Invalid or missing API token."),
                            ApiAuthorizationError::Forbidden => (StatusCode::FORBIDDEN, "Forbidden", "Token does not grant reader access."),
                            ApiAuthorizationError::LocalOnly => (StatusCode::FORBIDDEN, "Forbidden", "Authentication is disabled; this endpoint is restricted to loopback clients only."),
                        };
                        return warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({ "error": message, "detail": detail })),
                            status,
                        );
                    }
                };
                let status = auth_state.read().status_response();
                let write_allowed = access.grant.allows(ApiRole::Writer, ApiScope::Write);
                let admin_allowed = access.grant.allows(ApiRole::Admin, ApiScope::Admin);

                let response = ApiAuthLoginResponse {
                    authenticated: true,
                    auth_enabled: status.auth_enabled,
                    role_token_auth_enabled: status.role_token_auth_enabled,
                    role: access.grant.role,
                    scopes: access.grant.scopes.unwrap_or_default(),
                    mode: access.mode.to_string(),
                    access: ApiAuthLoginAccess {
                        read: true,
                        write: write_allowed,
                        admin: admin_allowed,
                    },
                };
                warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
            },
        )
        .map(reply_into_response)
        .boxed()
}
