use super::*;

pub(crate) fn build_rag_routes(
    rag: RagService,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let query_rag = warp::path!("api" / "rag" / "query")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |req: RagQueryRequest| {
            let rag = rag.clone();
            async move {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let query = RagQuery::from_request(req);
                let workspace_id = query.workspace_id().to_string();
                match rag.query_chunks(query).await {
                    Ok(matches) => {
                        let count = matches.len();
                        Ok::<_, warp::Rejection>(warp::reply::with_status(
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
