use super::*;

pub(crate) struct AuthRoutes {
    pub(crate) login: warp::filters::BoxedFilter<(warp::reply::Response,)>,
    pub(crate) admin: warp::filters::BoxedFilter<(warp::reply::Response,)>,
}

pub(crate) fn build_auth_routes(api_auth_state_for_api: Arc<RwLock<ApiAuthState>>) -> AuthRoutes {
    let api_auth_state_for_get_roles = api_auth_state_for_api.clone();
    let get_role_tokens = warp::path!("api" / "auth" / "roles")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let config = api_auth_state_for_get_roles.read().legacy_token_config();
            warp::reply::with_status(warp::reply::json(&config), StatusCode::OK)
        });

    let api_auth_state_for_set_roles = api_auth_state_for_api.clone();
    let set_role_tokens = warp::path!("api" / "auth" / "roles")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |payload: ApiRoleTokenConfig| {
            use warp::http::StatusCode;

            let mut auth_state = api_auth_state_for_set_roles.write();
            match auth_state.update_legacy_token_config(payload) {
                Ok(config) => warp::reply::with_status(warp::reply::json(&config), StatusCode::OK),
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let api_auth_state_for_login = api_auth_state_for_api.clone();
    let login = warp::path!("api" / "auth" / "login")
    .and(warp::post())
    .and(warp::header::optional::<String>("authorization"))
    .and(warp::addr::remote())
    .and(warp::body::json::<ApiAuthLoginRequest>())
    .map(
        move |authorization: Option<String>,
              remote: Option<std::net::SocketAddr>,
              payload: ApiAuthLoginRequest| {
            use warp::http::StatusCode;

            let mut auth_state = api_auth_state_for_login.write();
            let status = auth_state.status_response();

            if !status.auth_enabled {
                if is_loopback_remote(remote) {
                    let response = ApiAuthLoginResponse {
                        authenticated: true,
                        auth_enabled: status.auth_enabled,
                        legacy_auth_enabled: status.legacy_auth_enabled,
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
                legacy_auth_enabled: status.legacy_auth_enabled,
                role,
                scopes,
                mode: if matched_key_index.is_some() {
                    "api-key".to_string()
                } else {
                    "legacy-token".to_string()
                },
                access: ApiAuthLoginAccess {
                    read: read_allowed,
                    write: write_allowed,
                    admin: admin_allowed,
                },
            };
            warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
        },
    );

    let api_auth_state_for_status = api_auth_state_for_api.clone();
    let get_access_status = warp::path!("api" / "auth" / "access" / "status")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let auth_state = api_auth_state_for_status.read();
            warp::reply::with_status(
                warp::reply::json(&auth_state.status_response()),
                StatusCode::OK,
            )
        });

    let api_auth_state_for_access_roles = api_auth_state_for_api.clone();
    let get_access_roles = warp::path!("api" / "auth" / "access" / "roles")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let auth_state = api_auth_state_for_access_roles.read();
            warp::reply::with_status(
                warp::reply::json(&auth_state.role_summaries()),
                StatusCode::OK,
            )
        });

    let api_auth_state_for_list_users = api_auth_state_for_api.clone();
    let list_access_users = warp::path!("api" / "auth" / "access" / "users")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let auth_state = api_auth_state_for_list_users.read();
            warp::reply::with_status(warp::reply::json(&auth_state.list_users()), StatusCode::OK)
        });

    let api_auth_state_for_create_user = api_auth_state_for_api.clone();
    let create_access_user = warp::path!("api" / "auth" / "access" / "users")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |payload: CreateAuthUserRequest| {
            use warp::http::StatusCode;
            let mut auth_state = api_auth_state_for_create_user.write();
            match auth_state.create_user(payload) {
                Ok(user) => warp::reply::with_status(warp::reply::json(&user), StatusCode::CREATED),
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let api_auth_state_for_update_user = api_auth_state_for_api.clone();
    let update_access_user = warp::path!("api" / "auth" / "access" / "users" / String)
        .and(warp::patch())
        .and(warp::body::json())
        .map(move |user_id: String, payload: UpdateAuthUserRequest| {
            use warp::http::StatusCode;
            let mut auth_state = api_auth_state_for_update_user.write();
            match auth_state.update_user(&user_id, payload) {
                Ok(user) => warp::reply::with_status(warp::reply::json(&user), StatusCode::OK),
                Err(error) if error.contains("not found") => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::NOT_FOUND,
                ),
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let api_auth_state_for_list_keys = api_auth_state_for_api.clone();
    let list_access_keys = warp::path!("api" / "auth" / "access" / "keys")
        .and(warp::get())
        .and(warp::query::<HashMap<String, String>>())
        .map(move |query: HashMap<String, String>| {
            use warp::http::StatusCode;
            let user_id = query.get("user_id").map(String::as_str);
            let auth_state = api_auth_state_for_list_keys.read();
            warp::reply::with_status(
                warp::reply::json(&auth_state.list_keys(user_id)),
                StatusCode::OK,
            )
        });

    let api_auth_state_for_create_key = api_auth_state_for_api.clone();
    let create_access_key = warp::path!("api" / "auth" / "access" / "users" / String / "keys")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |user_id: String, payload: CreateApiKeyRequest| {
            use warp::http::StatusCode;
            let mut auth_state = api_auth_state_for_create_key.write();
            match auth_state.create_api_key(&user_id, payload) {
                Ok(response) => {
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::CREATED)
                }
                Err(error) if error.contains("not found") => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::NOT_FOUND,
                ),
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let api_auth_state_for_revoke_key = api_auth_state_for_api.clone();
    let revoke_access_key = warp::path!("api" / "auth" / "access" / "keys" / String / "revoke")
        .and(warp::post())
        .map(move |key_id: String| {
            use warp::http::StatusCode;
            let mut auth_state = api_auth_state_for_revoke_key.write();
            match auth_state.revoke_api_key(&key_id) {
                Ok(response) => {
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                }
                Err(error) if error.contains("not found") => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::NOT_FOUND,
                ),
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let login = login.map(reply_into_response).boxed();
    let admin = get_role_tokens
        .or(set_role_tokens)
        .or(get_access_status)
        .or(get_access_roles)
        .or(list_access_users)
        .or(create_access_user)
        .or(update_access_user)
        .or(list_access_keys)
        .or(create_access_key)
        .or(revoke_access_key)
        .map(reply_into_response)
        .boxed();

    AuthRoutes { login, admin }
}
