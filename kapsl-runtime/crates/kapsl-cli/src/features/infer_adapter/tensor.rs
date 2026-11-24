use super::error::{InferRequestError, InferResult};
use super::ModelRequestAdapter;
use kapsl_engine_api::{InferenceRequest, RequestMetadata as EngineRequestMetadata};

pub(super) struct TensorRequestAdapter;

impl ModelRequestAdapter for TensorRequestAdapter {
    fn name(&self) -> &'static str {
        "tensor_json"
    }

    fn supports_payload(&self, payload: &serde_json::Value) -> bool {
        payload.get("input").is_some()
    }

    fn adapt(&self, payload: serde_json::Value) -> InferResult<InferenceRequest> {
        // Extract top-level generation params before consuming the payload.
        // Callers often send {"input": ..., "max_tokens": 256, "min_tokens": 256}
        // which serde would silently drop since they're not InferenceRequest fields.
        let top_max_tokens = payload.get("max_tokens").and_then(|value| value.as_u64());
        let top_min_tokens = payload
            .get("min_tokens")
            .or_else(|| payload.get("min_new_tokens"))
            .and_then(|value| value.as_u64());
        let top_temperature = payload.get("temperature").and_then(|value| value.as_f64());

        let mut request = serde_json::from_value::<InferenceRequest>(payload).map_err(|error| {
            InferRequestError::bad_request(format!("Invalid tensor infer payload: {}", error))
        })?;
        apply_top_level_generation_params(
            &mut request,
            top_max_tokens,
            top_min_tokens,
            top_temperature,
        );
        Ok(request)
    }
}

/// Puts top-level `max_tokens`, `min_tokens` (or `min_new_tokens`) and `temperature` of an
/// infer payload into the request's metadata, where the backends read them. A value the
/// payload already carries in its `metadata` object wins. Shared by this adapter and by the
/// infer handlers' fast path, which deserializes an `{"input": ...}` body straight into an
/// `InferenceRequest` without ever reaching the adapter (run 9b found the fast path silently
/// dropping these: a native request with `max_tokens: 3` still generated the full answer).
pub(crate) fn apply_top_level_generation_params(
    request: &mut InferenceRequest,
    max_tokens: Option<u64>,
    min_tokens: Option<u64>,
    temperature: Option<f64>,
) {
    if max_tokens.is_none() && min_tokens.is_none() && temperature.is_none() {
        return;
    }
    let metadata = request
        .metadata
        .get_or_insert_with(EngineRequestMetadata::default);
    if metadata.max_new_tokens.is_none() {
        metadata.max_new_tokens = max_tokens.map(|value| value as u32);
    }
    if metadata.min_new_tokens.is_none() {
        metadata.min_new_tokens = min_tokens.map(|value| value as u32);
    }
    if metadata.temperature.is_none() {
        metadata.temperature = temperature.map(|value| value as f32);
    }
}
