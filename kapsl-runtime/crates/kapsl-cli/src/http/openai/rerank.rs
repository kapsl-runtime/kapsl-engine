//! `POST /v1/rerank`: score documents against a query with a cross-encoder.
//!
//! The request and response follow the Jina/Cohere rerank shape that
//! llama-server also serves, so existing rerank clients work unchanged:
//!
//! ```json
//! {"model": "bge-reranker", "query": "what is a panda?",
//!  "documents": ["hi", "The giant panda is a bear."], "top_n": 1}
//! ```
//!
//! returns `results` sorted by descending `relevance_score`, each carrying the
//! document's `index` in the request (and its text with `return_documents`).
//! The model must be packaged with `task=rerank`. The engine receives the
//! query as its primary UTF-8 input and the documents as `document.<n>`
//! inputs, scores them in one call, and returns one score per document.

use super::types::openai_error;
use super::*;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use kapsl_engine_api::{NamedTensor, RequestMetadata};
use rand::RngCore;

const MAX_DOCUMENTS: usize = 2_048;

fn rerank_request_id() -> String {
    let mut bytes = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    format!("rerank_{}", URL_SAFE_NO_PAD.encode(bytes))
}

pub(crate) struct RerankConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
}

pub(crate) fn build_rerank_route(
    config: RerankConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let RerankConfig { models, inference } = config;
    warp::path!("v1" / "rerank")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(warp::header::optional::<String>("x-kapsl-session"))
        .and(warp::header::optional::<String>("authorization"))
        .and_then(
            move |body: warp::hyper::body::Bytes,
                  session_header: Option<String>,
                  authorization: Option<String>| {
                let models = models.clone();
                let inference = inference.clone();
                async move {
                    Ok::<_, warp::Rejection>(
                        handle_rerank(body, session_header, authorization, &models, &inference)
                            .await,
                    )
                }
            },
        )
        .map(reply_into_response)
        .boxed()
}

#[derive(Debug, Deserialize)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<RerankDocument>,
    #[serde(default)]
    top_n: Option<usize>,
    #[serde(default)]
    return_documents: bool,
}

/// A document as a bare string or as `{"text": "..."}` (Cohere's form).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RerankDocument {
    Text(String),
    Object { text: String },
}

impl RerankDocument {
    fn text(&self) -> &str {
        match self {
            Self::Text(text) | Self::Object { text } => text,
        }
    }
}

fn supports_rerank(model: &ModelInfo) -> bool {
    model
        .task
        .as_deref()
        .is_some_and(|task| task.eq_ignore_ascii_case("rerank"))
}

fn validate_request(request: &RerankRequest) -> Result<(), String> {
    if request.query.is_empty() {
        return Err("query must not be empty".to_string());
    }
    if request.documents.is_empty() {
        return Err("documents must contain at least one document".to_string());
    }
    if request.documents.len() > MAX_DOCUMENTS {
        return Err(format!(
            "documents contains {} items; at most {MAX_DOCUMENTS} are supported",
            request.documents.len()
        ));
    }
    if let Some(index) = request.documents.iter().position(|d| d.text().is_empty()) {
        return Err(format!("documents[{index}] must not be empty"));
    }
    if request.top_n == Some(0) {
        return Err("top_n must be at least 1".to_string());
    }
    Ok(())
}

fn text_tensor(text: &str) -> Result<BinaryTensorPacket, String> {
    BinaryTensorPacket::new(
        vec![text.len() as i64],
        TensorDtype::Utf8,
        text.as_bytes().to_vec(),
    )
    .map_err(|error| format!("Failed to build rerank input tensor: {error}"))
}

fn build_rerank_request(
    request: &RerankRequest,
    session_id: Option<String>,
) -> Result<InferenceRequest, String> {
    let mut inference_request =
        InferenceRequest::new(text_tensor(&request.query)?).with_metadata(RequestMetadata {
            request_id: Some(rerank_request_id()),
            ..RequestMetadata::default()
        });
    inference_request.session_id = session_id;
    inference_request.additional_inputs = request
        .documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            Ok(NamedTensor {
                name: format!("document.{index}"),
                tensor: text_tensor(document.text())?,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(inference_request)
}

/// One finite float32 score per document.
fn decode_scores(packet: &BinaryTensorPacket, expected: usize) -> Result<Vec<f32>, String> {
    if packet.dtype != TensorDtype::Float32 {
        return Err(format!(
            "Rerank backend returned {}, expected float32",
            packet.dtype
        ));
    }
    if packet.shape != [expected as i64] {
        return Err(format!(
            "Rerank backend returned shape {:?}, expected [{expected}]",
            packet.shape
        ));
    }
    let scores: Vec<f32> = packet
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    if scores.len() != expected || scores.iter().any(|score| !score.is_finite()) {
        return Err("Rerank backend returned an invalid score".to_string());
    }
    Ok(scores)
}

/// Results sorted by descending score (ties keep request order), cut to
/// `top_n`.
fn ranked_results(request: &RerankRequest, scores: &[f32]) -> Vec<serde_json::Value> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order.truncate(request.top_n.unwrap_or(order.len()));
    order
        .into_iter()
        .map(|index| {
            let mut result = serde_json::json!({
                "index": index,
                "relevance_score": scores[index],
            });
            if request.return_documents {
                result["document"] = serde_json::json!({ "text": request.documents[index].text() });
            }
            result
        })
        .collect()
}

async fn handle_rerank(
    body: warp::hyper::body::Bytes,
    session_header: Option<String>,
    authorization: Option<String>,
    models: &Arc<ModelManager>,
    inference: &Arc<InferenceService>,
) -> warp::reply::Response {
    use warp::http::StatusCode;

    let request: RerankRequest = match serde_json::from_slice(body.as_ref()) {
        Ok(request) => request,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("Invalid rerank payload: {error}"),
                "invalid_request_error",
            );
        }
    };
    if let Err(message) = validate_request(&request) {
        return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
    }

    let resolved = match resolve_model(models, &request.model) {
        Ok(resolved) => resolved,
        Err(message) => {
            return openai_error(StatusCode::NOT_FOUND, message, "invalid_request_error");
        }
    };
    let Some(model) = models.registry().get(resolved.id) else {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("The model '{}' does not exist", request.model),
            "invalid_request_error",
        );
    };
    if !supports_rerank(&model) {
        let task = model.task.as_deref().unwrap_or("unspecified");
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The model '{}' cannot rerank (task={task}); load a GGUF cross-encoder packaged with model_type=seq-classifier and task=rerank",
                resolved.name
            ),
            "invalid_request_error",
        );
    }
    if let Err(response) = ensure_model_loaded(inference, &resolved).await {
        return response;
    }
    if models.pool(resolved.id).is_none() {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("The model '{}' is not currently loaded", resolved.name),
            "server_error",
        );
    }

    let client_session_id = session_header
        .map(|session| session.trim().to_string())
        .filter(|session| !session.is_empty());
    let session_id =
        scope_session_id_for_authorization(client_session_id.as_deref(), authorization.as_deref());
    let inference_request = match build_rerank_request(&request, session_id) {
        Ok(inference_request) => inference_request,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };

    let priority = inference.priority_for_request(&inference_request);
    let output = match inference
        .infer(resolved.id, inference_request, priority, false)
        .await
    {
        Ok(output) => output,
        Err(error) => {
            let status = status_code_for_engine_error(&error);
            if status == StatusCode::INTERNAL_SERVER_ERROR {
                log::error!("Rerank failed: model_id={} error={}", resolved.id, error);
            } else {
                log::warn!("Rerank rejected: model_id={} error={}", resolved.id, error);
            }
            let error_type = if status == StatusCode::BAD_REQUEST {
                "invalid_request_error"
            } else {
                "server_error"
            };
            return openai_error(status, error.to_string(), error_type);
        }
    };
    let scores = match decode_scores(&output, request.documents.len()) {
        Ok(scores) => scores,
        Err(message) => {
            log::error!(
                "Invalid rerank output: model_id={} error={}",
                resolved.id,
                message
            );
            return openai_error(StatusCode::INTERNAL_SERVER_ERROR, message, "server_error");
        }
    };

    // Text is tokenized inside the engine, which does not report token counts.
    reply_into_response(warp::reply::json(&serde_json::json!({
        "object": "list",
        "model": resolved.name,
        "results": ranked_results(&request, &scores),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: serde_json::Value) -> RerankRequest {
        serde_json::from_value(json).unwrap()
    }

    fn scores_packet(scores: &[f32]) -> BinaryTensorPacket {
        BinaryTensorPacket::new(
            vec![scores.len() as i64],
            TensorDtype::Float32,
            scores
                .iter()
                .flat_map(|score| score.to_le_bytes())
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn accepts_string_and_object_documents() {
        let parsed = request(serde_json::json!({
            "model": "m", "query": "q",
            "documents": ["plain", {"text": "cohere style"}]
        }));
        assert_eq!(parsed.documents[0].text(), "plain");
        assert_eq!(parsed.documents[1].text(), "cohere style");
        assert!(validate_request(&parsed).is_ok());
    }

    #[test]
    fn rejects_empty_and_oversized_requests() {
        for json in [
            serde_json::json!({"model": "m", "query": "", "documents": ["d"]}),
            serde_json::json!({"model": "m", "query": "q", "documents": []}),
            serde_json::json!({"model": "m", "query": "q", "documents": ["d", ""]}),
            serde_json::json!({"model": "m", "query": "q", "documents": ["d"], "top_n": 0}),
        ] {
            assert!(validate_request(&request(json)).is_err());
        }
        let many = vec!["d"; MAX_DOCUMENTS + 1];
        let error = validate_request(&request(serde_json::json!({
            "model": "m", "query": "q", "documents": many
        })))
        .unwrap_err();
        assert!(error.contains("at most"), "{error}");
    }

    #[test]
    fn engine_request_carries_query_and_indexed_documents() {
        let parsed = request(serde_json::json!({
            "model": "m", "query": "what is a panda?", "documents": ["hi", "a bear"]
        }));
        let built = build_rerank_request(&parsed, Some("s".to_string())).unwrap();
        assert_eq!(built.input.dtype, TensorDtype::Utf8);
        assert_eq!(built.input.data, b"what is a panda?");
        let names: Vec<_> = built
            .additional_inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect();
        assert_eq!(names, vec!["document.0", "document.1"]);
        assert_eq!(built.additional_inputs[1].tensor.data, b"a bear");
        assert_eq!(built.session_id.as_deref(), Some("s"));
    }

    #[test]
    fn results_are_sorted_cut_and_optionally_carry_documents() {
        let parsed = request(serde_json::json!({
            "model": "m", "query": "q", "documents": ["a", "b", "c"],
            "top_n": 2, "return_documents": true
        }));
        let results = ranked_results(&parsed, &[0.1, 0.9, 0.1]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["index"], 1);
        assert_eq!(results[0]["document"]["text"], "b");
        // Equal scores keep request order.
        assert_eq!(results[1]["index"], 0);

        let parsed = request(serde_json::json!({
            "model": "m", "query": "q", "documents": ["a", "b"]
        }));
        let results = ranked_results(&parsed, &[-1.0, 2.0]);
        assert_eq!(results.len(), 2);
        assert!(results[0].get("document").is_none());
    }

    #[test]
    fn scores_must_match_the_document_count() {
        assert_eq!(
            decode_scores(&scores_packet(&[0.5, -1.0]), 2).unwrap(),
            vec![0.5, -1.0]
        );
        assert!(decode_scores(&scores_packet(&[0.5]), 2).is_err());
        assert!(decode_scores(&scores_packet(&[f32::NAN]), 1).is_err());
    }
}
