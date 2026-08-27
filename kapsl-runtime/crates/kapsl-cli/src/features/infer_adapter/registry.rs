use super::error::{InferRequestError, InferResult};
use super::media::MediaRequestAdapter;
use super::tensor::TensorRequestAdapter;
use super::ModelRequestAdapter;
use kapsl_engine_api::InferenceRequest;

/// Ordered collection of request adapters used to select a payload converter.
pub(crate) struct RequestAdapterRegistry {
    adapters: Vec<Box<dyn ModelRequestAdapter>>,
}

impl RequestAdapterRegistry {
    /// Creates an empty registry for explicit adapter injection.
    pub(crate) fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// Adds an adapter after the adapters already registered.
    pub(crate) fn register(&mut self, adapter: Box<dyn ModelRequestAdapter>) {
        self.adapters.push(adapter);
    }

    fn register_if_missing(&mut self, adapter: Box<dyn ModelRequestAdapter>) -> bool {
        if self.has_adapter_name(adapter.name()) {
            return false;
        }
        self.register(adapter);
        true
    }

    fn has_adapter_name(&self, name: &str) -> bool {
        self.adapters.iter().any(|adapter| adapter.name() == name)
    }

    fn new_default() -> Self {
        let mut registry = Self::new();
        registry.register_if_missing(Box::new(TensorRequestAdapter));
        registry.register_if_missing(Box::new(MediaRequestAdapter));
        registry
    }

    fn adapt(&self, framework: &str, payload: serde_json::Value) -> InferResult<InferenceRequest> {
        let framework = framework.trim().to_ascii_lowercase();
        let framework = if framework.is_empty() {
            "unknown".to_string()
        } else {
            framework
        };

        let mut payload_matched_any = false;
        let mut framework_filtered = Vec::new();
        let mut selected_adapter_index = None;
        for (index, adapter) in self.adapters.iter().enumerate() {
            if !adapter.supports_payload(&payload) {
                continue;
            }
            payload_matched_any = true;
            if !adapter.supports_framework(&framework) {
                framework_filtered.push(adapter.name());
                continue;
            }
            selected_adapter_index = Some(index);
            break;
        }

        if let Some(index) = selected_adapter_index {
            return self.adapters[index].adapt(payload);
        }

        if payload_matched_any {
            if framework_filtered.is_empty() {
                return Err(InferRequestError::bad_request(
                    "Infer payload matched adapter shape but could not be adapted",
                ));
            }
            return Err(InferRequestError::bad_request(format!(
                "No infer adapter supports framework `{}` for this payload. Matching adapters: {}",
                framework,
                framework_filtered.join(", ")
            )));
        }

        let adapters = self
            .adapters
            .iter()
            .map(|adapter| format!("`{}`", adapter.name()))
            .collect::<Vec<_>>()
            .join(", ");
        Err(InferRequestError::bad_request(format!(
            "Invalid infer payload for framework `{}`. Known adapters: {}",
            framework, adapters
        )))
    }
}

/// Creates the production registry with tensor and base64-media adapters.
pub(crate) fn default_request_adapter_registry() -> RequestAdapterRegistry {
    RequestAdapterRegistry::new_default()
}

pub(super) fn parse_inference_request_with_registry(
    body: serde_json::Value,
    model_framework: &str,
    registry: &RequestAdapterRegistry,
) -> InferResult<InferenceRequest> {
    registry.adapt(model_framework, body)
}
