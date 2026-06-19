use super::*;

pub(crate) struct ModelSwapRoutesConfig {
    pub(crate) swap_map: Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
}

pub(crate) fn build_model_swap_routes(
    config: ModelSwapRoutesConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let swap_map = config.swap_map;

    // Shared response types for the three hot-swap endpoints.
    #[derive(Serialize)]
    struct SwapMsgResp {
        message: String,
    }
    #[derive(Serialize)]
    struct SwapErrResp {
        error: String,
    }
    #[derive(Serialize)]
    struct SwapStatusResp {
        model_id: u32,
        supports_swap: bool,
        staged: bool,
    }

    // Returns the engine handles for a model, or an error reply if not found.
    fn lookup_swap_handles(
        swap_map: &Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
        model_id: u32,
    ) -> Result<Vec<EngineHandle>, warp::reply::WithStatus<warp::reply::Json>> {
        let g = swap_map.read();
        match g.get(&model_id) {
            Some(v) => Ok(v.clone()),
            None => Err(warp::reply::with_status(
                warp::reply::json(&SwapErrResp {
                    error: format!("model {} not found", model_id),
                }),
                warp::http::StatusCode::NOT_FOUND,
            )),
        }
    }

    // POST /api/models/:id/stage - Pre-load next model weights into CPU RAM
    let swap_map_for_stage = swap_map.clone();
    let stage_model_route = warp::path!("api" / "models" / u32 / "stage")
        .and(warp::post())
        .and(warp::body::json())
        .then(move |model_id: u32, body: serde_json::Value| {
            let swap_map = swap_map_for_stage.clone();
            async move {
                use warp::http::StatusCode;
                let path_str = match body.get("model_path").and_then(|v| v.as_str()) {
                    Some(s) => s.to_string(),
                    None => {
                        return warp::reply::with_status(
                            warp::reply::json(&SwapErrResp {
                                error: "missing model_path".into(),
                            }),
                            StatusCode::BAD_REQUEST,
                        )
                    }
                };
                let handles = match lookup_swap_handles(&swap_map, model_id) {
                    Ok(h) => h,
                    Err(r) => return r,
                };
                if !handles.iter().any(|e| e.supports_swap()) {
                    return warp::reply::with_status(
                        warp::reply::json(&SwapErrResp {
                            error: "backend does not support hot-swap".into(),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }
                let path = std::path::PathBuf::from(&path_str);
                for engine in &handles {
                    if let Err(e) = engine.stage(&path).await {
                        return warp::reply::with_status(
                            warp::reply::json(&SwapErrResp {
                                error: e.to_string(),
                            }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        );
                    }
                }
                warp::reply::with_status(
                    warp::reply::json(&SwapMsgResp {
                        message: format!(
                            "model {} staged from {}; call /swap when ready",
                            model_id, path_str
                        ),
                    }),
                    StatusCode::OK,
                )
            }
        });

    // POST /api/models/:id/swap - Activate staged weights (PCIe transfer only)
    let swap_map_for_swap = swap_map.clone();
    let swap_model_route = warp::path!("api" / "models" / u32 / "swap")
        .and(warp::post())
        .then(move |model_id: u32| {
            let swap_map = swap_map_for_swap.clone();
            async move {
                use warp::http::StatusCode;
                let handles = match lookup_swap_handles(&swap_map, model_id) {
                    Ok(h) => h,
                    Err(r) => return r,
                };
                for engine in &handles {
                    if let Err(e) = engine.swap().await {
                        return warp::reply::with_status(
                            warp::reply::json(&SwapErrResp {
                                error: e.to_string(),
                            }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        );
                    }
                }
                warp::reply::with_status(
                    warp::reply::json(&SwapMsgResp {
                        message: format!("model {} swapped to staged weights", model_id),
                    }),
                    StatusCode::OK,
                )
            }
        });

    // GET /api/models/:id/swap-status - Check hot-swap staging readiness
    let swap_map_for_status = swap_map.clone();
    let swap_status_route = warp::path!("api" / "models" / u32 / "swap-status")
        .and(warp::get())
        .then(move |model_id: u32| {
            let swap_map = swap_map_for_status.clone();
            async move {
                use warp::http::StatusCode;
                let handles = match lookup_swap_handles(&swap_map, model_id) {
                    Ok(h) => h,
                    Err(r) => return r,
                };
                let supports = handles.iter().any(|e| e.supports_swap());
                // All swap-capable engines must have weights staged.
                let staged = supports
                    && handles
                        .iter()
                        .filter(|e| e.supports_swap())
                        .all(|e| e.is_staged());
                warp::reply::with_status(
                    warp::reply::json(&SwapStatusResp {
                        model_id,
                        supports_swap: supports,
                        staged,
                    }),
                    StatusCode::OK,
                )
            }
        });

    stage_model_route
        .or(swap_model_route)
        .or(swap_status_route)
        .map(reply_into_response)
        .boxed()
}
