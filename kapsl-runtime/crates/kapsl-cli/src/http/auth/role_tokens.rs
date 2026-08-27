//! Administrative routes for the legacy role-token configuration.

use super::*;

pub(super) fn build_role_token_routes(
    auth_state: Arc<RwLock<ApiAuthState>>,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let auth_state_for_get = auth_state.clone();
    let get_role_tokens = warp::path!("api" / "auth" / "roles")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;
            let config = auth_state_for_get.read().role_token_config();
            warp::reply::with_status(warp::reply::json(&config), StatusCode::OK)
        });

    let set_role_tokens = warp::path!("api" / "auth" / "roles")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |payload: ApiRoleTokenConfig| {
            use warp::http::StatusCode;

            let mut auth_state = auth_state.write();
            match auth_state.update_role_token_config(payload) {
                Ok(config) => warp::reply::with_status(warp::reply::json(&config), StatusCode::OK),
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    get_role_tokens
        .or(set_role_tokens)
        .map(reply_into_response)
        .boxed()
}
