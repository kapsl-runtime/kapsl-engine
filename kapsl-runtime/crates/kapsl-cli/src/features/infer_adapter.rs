//! Conversion of HTTP inference payloads into engine requests.
//!
//! The facade keeps adapter selection separate from the tensor and media
//! implementations. Callers only depend on the registry API exposed here.

mod error;
mod media;
mod registry;
mod tensor;

pub(crate) use error::InferRequestError;
use kapsl_engine_api::InferenceRequest;
pub(crate) use registry::{default_request_adapter_registry, RequestAdapterRegistry};

/// Converts one supported HTTP payload shape into an engine request.
pub(crate) trait ModelRequestAdapter: Send + Sync {
    /// Stable name used when describing adapter selection failures.
    fn name(&self) -> &'static str;

    /// Model frameworks accepted by this adapter. A wildcard accepts all
    /// frameworks.
    fn supported_frameworks(&self) -> &'static [&'static str] {
        &["*"]
    }

    /// Returns whether the payload has the shape handled by this adapter.
    fn supports_payload(&self, payload: &serde_json::Value) -> bool;

    /// Converts a matching payload into an engine request.
    fn adapt(&self, payload: serde_json::Value) -> Result<InferenceRequest, InferRequestError>;

    /// Returns whether this adapter supports the selected model framework.
    fn supports_framework(&self, framework: &str) -> bool {
        self.supported_frameworks()
            .iter()
            .any(|candidate| *candidate == "*" || candidate.eq_ignore_ascii_case(framework))
    }
}

/// Converts a payload using the first matching adapter in the injected
/// registry.
pub(crate) fn parse_inference_request_with_registry(
    body: serde_json::Value,
    model_framework: &str,
    registry: &RequestAdapterRegistry,
) -> Result<InferenceRequest, InferRequestError> {
    registry::parse_inference_request_with_registry(body, model_framework, registry)
}

#[cfg(test)]
use error::InferResult;

#[cfg(test)]
#[path = "../tests/infer_adapter_tests.rs"]
mod tests;
