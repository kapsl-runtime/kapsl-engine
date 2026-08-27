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
        let top_max_tokens = payload
            .get("max_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as u32);
        let top_min_tokens = payload
            .get("min_tokens")
            .or_else(|| payload.get("min_new_tokens"))
            .and_then(|value| value.as_u64())
            .map(|value| value as u32);
        let top_temperature = payload
            .get("temperature")
            .and_then(|value| value.as_f64())
            .map(|value| value as f32);

        let mut request = serde_json::from_value::<InferenceRequest>(payload).map_err(|error| {
            InferRequestError::bad_request(format!("Invalid tensor infer payload: {}", error))
        })?;

        if top_max_tokens.is_some() || top_min_tokens.is_some() || top_temperature.is_some() {
            let metadata = request
                .metadata
                .get_or_insert_with(EngineRequestMetadata::default);
            if metadata.max_new_tokens.is_none() {
                metadata.max_new_tokens = top_max_tokens;
            }
            if metadata.min_new_tokens.is_none() {
                metadata.min_new_tokens = top_min_tokens;
            }
            if metadata.temperature.is_none() {
                metadata.temperature = top_temperature;
            }
        }

        Ok(request)
    }
}
