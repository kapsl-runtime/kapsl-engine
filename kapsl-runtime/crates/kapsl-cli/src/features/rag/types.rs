use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct RagQueryRequest {
    pub(crate) workspace_id: String,
    pub(crate) query: String,
    pub(crate) source_id: Option<String>,
    pub(crate) source_ids: Option<Vec<String>>,
    pub(crate) top_k: Option<usize>,
    pub(crate) min_score: Option<f32>,
    pub(crate) tenant_id: Option<String>,
    #[serde(default)]
    pub(crate) allowed_users: Vec<String>,
    #[serde(default)]
    pub(crate) allowed_groups: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct InferRagOptions {
    pub(crate) enabled: Option<bool>,
    pub(crate) workspace_id: String,
    pub(crate) source_id: Option<String>,
    pub(crate) source_ids: Option<Vec<String>>,
    pub(crate) top_k: Option<usize>,
    pub(crate) min_score: Option<f32>,
    pub(crate) tenant_id: Option<String>,
    pub(crate) max_context_tokens: Option<usize>,
    pub(crate) max_chunks: Option<usize>,
    pub(crate) max_per_source: Option<usize>,
}

#[derive(Debug)]
pub(crate) enum RagAugmentError {
    BadRequest(String),
    Internal(String),
}

impl RagAugmentError {
    pub(super) fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub(super) fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct InferPayloadEnvelope<T> {
    #[serde(default)]
    pub(crate) rag: Option<InferRagOptions>,
    #[serde(flatten)]
    pub(crate) request: T,
}

/// Storage-independent query passed to [`super::RagService::query_chunks`].
#[derive(Debug, Clone)]
pub(crate) struct RagQuery {
    pub(super) workspace_id: String,
    pub(super) tenant_id: Option<String>,
    pub(super) text: String,
    pub(super) source_id: Option<String>,
    pub(super) source_ids: Option<Vec<String>>,
    pub(super) top_k: Option<usize>,
    pub(super) min_score: Option<f32>,
    pub(super) allowed_users: Vec<String>,
    pub(super) allowed_groups: Vec<String>,
}

impl RagQuery {
    pub(crate) fn from_request(request: RagQueryRequest) -> Self {
        Self {
            workspace_id: request.workspace_id.trim().to_string(),
            tenant_id: request.tenant_id,
            text: request.query,
            source_id: request.source_id,
            source_ids: request.source_ids,
            top_k: request.top_k,
            min_score: request.min_score,
            allowed_users: request.allowed_users,
            allowed_groups: request.allowed_groups,
        }
    }

    pub(super) fn for_inference(prompt: String, options: &InferRagOptions) -> Self {
        Self {
            workspace_id: options.workspace_id.trim().to_string(),
            tenant_id: options.tenant_id.clone(),
            text: prompt,
            source_id: options.source_id.clone(),
            source_ids: options.source_ids.clone(),
            top_k: options.top_k,
            min_score: options.min_score,
            allowed_users: Vec::new(),
            allowed_groups: Vec::new(),
        }
    }

    pub(crate) fn workspace_id(&self) -> &str {
        &self.workspace_id
    }
}
