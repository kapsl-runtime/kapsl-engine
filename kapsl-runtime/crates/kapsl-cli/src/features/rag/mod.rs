//! Retrieval-augmented generation services.
//!
//! Concrete storage adapters are supplied once through [`RagService`]. The
//! remaining modules own document ingestion, retrieval, embedding, and prompt
//! augmentation without reaching into runtime startup configuration.

mod documents;
mod embedding;
mod inference;
mod query;
mod service;
mod types;

pub(crate) use inference::{parse_infer_rag_options, validate_infer_rag_options};
pub(crate) use query::normalize_tenant_id;
pub(crate) use service::RagService;
pub(crate) use types::{
    InferPayloadEnvelope, InferRagOptions, RagAugmentError, RagQuery, RagQueryRequest,
};
