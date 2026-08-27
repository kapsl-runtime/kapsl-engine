//! Authentication HTTP facade and route composition.

use super::*;

mod access;
mod login;
mod middleware;
mod role_tokens;

use access::build_access_routes;
use login::build_login_route;
pub(crate) use middleware::*;
use role_tokens::build_role_token_routes;

pub(crate) struct AuthRoutes {
    pub(crate) login: warp::filters::BoxedFilter<(warp::reply::Response,)>,
    pub(crate) admin: warp::filters::BoxedFilter<(warp::reply::Response,)>,
}

pub(crate) fn build_auth_routes(auth_state: Arc<RwLock<ApiAuthState>>) -> AuthRoutes {
    let login = build_login_route(auth_state.clone());
    let admin = build_role_token_routes(auth_state.clone())
        .or(build_access_routes(auth_state))
        .unify()
        .boxed();

    AuthRoutes { login, admin }
}
