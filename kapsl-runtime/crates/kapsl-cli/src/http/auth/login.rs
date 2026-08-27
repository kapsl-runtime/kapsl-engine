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

                let mut auth_state = auth_state.write();
                let status = auth_state.status_response();

                if !status.auth_enabled {
                    if is_loopback_remote(remote) {
                        let response = ApiAuthLoginResponse {
                            authenticated: true,
                            auth_enabled: status.auth_enabled,
                            role_token_auth_enabled: status.role_token_auth_enabled,
                            role: ApiRole::Admin,
                            scopes: Vec::new(),
                            mode: "local-loopback".to_string(),
                            access: ApiAuthLoginAccess {
                                read: true,
                                write: true,
                                admin: true,
                            },
                        };
                        return warp::reply::with_status(
                            warp::reply::json(&response),
                            StatusCode::OK,
                        );
                    }
                    return warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "error": "Forbidden",
                            "detail": "Authentication is disabled; this endpoint is restricted to loopback clients only."
                        })),
                        StatusCode::FORBIDDEN,
                    );
                }

                let token_from_body = payload.token.and_then(|token| {
                    let trimmed = token.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                });
                let normalized_authorization = authorization
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .or(token_from_body);

                let Some(grant_match) = auth_state
                    .grant_from_authorization_header_read(normalized_authorization.as_deref())
                else {
                    return warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "error": "Unauthorized",
                            "detail": "Invalid or missing API token."
                        })),
                        StatusCode::UNAUTHORIZED,
                    );
                };

                let ApiAuthGrantMatch {
                    grant,
                    matched_key_index,
                } = grant_match;
                let role = grant.role;
                let scopes = grant.scopes.unwrap_or_default();

                let read_allowed =
                    role.allows(ApiRole::Reader) && key_scopes_allow(&scopes, ApiScope::Read);
                if !read_allowed {
                    return warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "error": "Forbidden",
                            "detail": "Token does not grant reader access."
                        })),
                        StatusCode::FORBIDDEN,
                    );
                }

                let write_allowed =
                    role.allows(ApiRole::Writer) && key_scopes_allow(&scopes, ApiScope::Write);
                let admin_allowed =
                    role.allows(ApiRole::Admin) && key_scopes_allow(&scopes, ApiScope::Admin);

                if let Some(key_index) = matched_key_index {
                    auth_state.touch_key_last_used_by_index(key_index, now_unix_seconds());
                }

                let response = ApiAuthLoginResponse {
                    authenticated: true,
                    auth_enabled: status.auth_enabled,
                    role_token_auth_enabled: status.role_token_auth_enabled,
                    role,
                    scopes,
                    mode: if matched_key_index.is_some() {
                        "api-key".to_string()
                    } else {
                        "role-token".to_string()
                    },
                    access: ApiAuthLoginAccess {
                        read: read_allowed,
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
