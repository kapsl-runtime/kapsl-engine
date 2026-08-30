//! Administrative user and API-key management routes.

use super::*;

pub(super) fn build_access_routes(
    auth_state: Arc<RwLock<ApiAuthState>>,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let auth_state_for_status = auth_state.clone();
    let get_access_status = warp::path!("api" / "auth" / "access" / "status")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let auth_state = auth_state_for_status.read();
            warp::reply::with_status(
                warp::reply::json(&auth_state.status_response()),
                StatusCode::OK,
            )
        });

    let auth_state_for_roles = auth_state.clone();
    let get_access_roles = warp::path!("api" / "auth" / "access" / "roles")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let auth_state = auth_state_for_roles.read();
            warp::reply::with_status(
                warp::reply::json(&auth_state.role_summaries()),
                StatusCode::OK,
            )
        });

    let auth_state_for_list_users = auth_state.clone();
    let list_access_users = warp::path!("api" / "auth" / "access" / "users")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let auth_state = auth_state_for_list_users.read();
            warp::reply::with_status(warp::reply::json(&auth_state.list_users()), StatusCode::OK)
        });

    let auth_state_for_create_user = auth_state.clone();
    let create_access_user = warp::path!("api" / "auth" / "access" / "users")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |payload: CreateAuthUserRequest| {
            use warp::http::StatusCode;
            let mut auth_state = auth_state_for_create_user.write();
            match auth_state.create_user(payload) {
                Ok(user) => warp::reply::with_status(warp::reply::json(&user), StatusCode::CREATED),
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let auth_state_for_update_user = auth_state.clone();
    let update_access_user = warp::path!("api" / "auth" / "access" / "users" / String)
        .and(warp::patch())
        .and(warp::body::json())
        .map(move |user_id: String, payload: UpdateAuthUserRequest| {
            use warp::http::StatusCode;
            let mut auth_state = auth_state_for_update_user.write();
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

    let auth_state_for_list_keys = auth_state.clone();
    let list_access_keys = warp::path!("api" / "auth" / "access" / "keys")
        .and(warp::get())
        .and(warp::query::<HashMap<String, String>>())
        .map(move |query: HashMap<String, String>| {
            use warp::http::StatusCode;
            let user_id = query.get("user_id").map(String::as_str);
            let auth_state = auth_state_for_list_keys.read();
            warp::reply::with_status(
                warp::reply::json(&auth_state.list_keys(user_id)),
                StatusCode::OK,
            )
        });

    let auth_state_for_create_key = auth_state.clone();
    let create_access_key = warp::path!("api" / "auth" / "access" / "users" / String / "keys")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |user_id: String, payload: CreateApiKeyRequest| {
            use warp::http::StatusCode;
            let mut auth_state = auth_state_for_create_key.write();
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

    let auth_state_for_revoke_key = auth_state;
    let revoke_access_key = warp::path!("api" / "auth" / "access" / "keys" / String / "revoke")
        .and(warp::post())
        .map(move |key_id: String| {
            use warp::http::StatusCode;
            let mut auth_state = auth_state_for_revoke_key.write();
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

    get_access_status
        .or(get_access_roles)
        .or(list_access_users)
        .or(create_access_user)
        .or(update_access_user)
        .or(list_access_keys)
        .or(create_access_key)
        .or(revoke_access_key)
        .map(reply_into_response)
        .boxed()
}
