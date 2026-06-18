use super::*;

pub(crate) struct ModelScalingRoutes {
    pub(crate) reader: warp::filters::BoxedFilter<(warp::reply::Response,)>,
    pub(crate) admin: warp::filters::BoxedFilter<(warp::reply::Response,)>,
}

pub(crate) struct ModelScalingRoutesConfig {
    pub(crate) auto_scaler: Arc<RwLock<AutoScaler>>,
}

pub(crate) fn build_model_scaling_routes(config: ModelScalingRoutesConfig) -> ModelScalingRoutes {
    let auto_scaler_api = config.auto_scaler;

    // GET /api/models/:id/scaling - Get scaling policy
    let auto_scaler_for_get = auto_scaler_api.clone();
    let get_scaling = warp::path!("api" / "models" / u32 / "scaling")
        .and(warp::get())
        .map(move |model_id: u32| {
            let scaler = auto_scaler_for_get.read();
            let policy = scaler.get_policy(model_id);
            warp::reply::json(&policy)
        });

    // POST /api/models/:id/scaling - Update scaling policy
    let auto_scaler_for_post = auto_scaler_api.clone();
    let update_scaling = warp::path!("api" / "models" / u32 / "scaling")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |model_id: u32, policy: ScalingPolicy| {
            use warp::http::StatusCode;

            if let Err(error) = policy.validate() {
                return warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": error })),
                    StatusCode::BAD_REQUEST,
                );
            }

            let mut scaler = auto_scaler_for_post.write();
            scaler.register_policy(model_id, policy);
            warp::reply::with_status(
                warp::reply::json(&serde_json::json!({ "status": "ok" })),
                StatusCode::OK,
            )
        });

    ModelScalingRoutes {
        reader: get_scaling.map(reply_into_response).boxed(),
        admin: update_scaling.map(reply_into_response).boxed(),
    }
}
