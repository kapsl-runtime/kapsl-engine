use super::*;

pub(crate) fn build_rag_routes(
    rag_state_for_api: RagRuntimeState,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let rag_state_for_query = rag_state_for_api.clone();
    let query_rag = warp::path!("api" / "rag" / "query")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |req: RagQueryRequest| {
            let rag_state = rag_state_for_query.clone();
            async move {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let workspace_id = req.workspace_id.trim().to_string();
                if workspace_id.is_empty() {
                    return Ok::<_, warp::Rejection>(warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: "workspace_id is required".to_string(),
                        }),
                        StatusCode::BAD_REQUEST,
                    ));
                }

                match query_rag_chunks(
                    &rag_state,
                    &workspace_id,
                    req.tenant_id.as_deref(),
                    &req.query,
                    req.source_id,
                    req.source_ids,
                    req.top_k,
                    req.min_score,
                    req.allowed_users,
                    req.allowed_groups,
                )
                .await
                {
                    Ok(matches) => {
                        let count = matches.len();
                        Ok(warp::reply::with_status(
                            warp::reply::json(&json!({
                                "status": "ok",
                                "workspace_id": workspace_id,
                                "matches": matches,
                                "count": count
                            })),
                            StatusCode::OK,
                        ))
                    }
                    Err(RagAugmentError::BadRequest(error)) => Ok(warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error }),
                        StatusCode::BAD_REQUEST,
                    )),
                    Err(RagAugmentError::Internal(error)) => Ok(warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error }),
                        StatusCode::INTERNAL_SERVER_ERROR,
                    )),
                }
            }
        });

    query_rag.map(reply_into_response).boxed()
}
