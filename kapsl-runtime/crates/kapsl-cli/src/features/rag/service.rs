use kapsl_rag::{DocStore, VectorStore};
use std::sync::Arc;

/// Process-owned RAG service with injected persistence dependencies.
///
/// Routes and connector synchronization clone this lightweight facade instead
/// of receiving concrete SQLite and filesystem adapters independently.
#[derive(Clone)]
pub(crate) struct RagService {
    vector_store: Arc<dyn VectorStore>,
    doc_store: Arc<dyn DocStore>,
}

impl RagService {
    /// Creates a RAG service from independently replaceable storage adapters.
    pub(crate) fn new(vector_store: Arc<dyn VectorStore>, doc_store: Arc<dyn DocStore>) -> Self {
        Self {
            vector_store,
            doc_store,
        }
    }

    pub(super) fn vector_store(&self) -> &dyn VectorStore {
        self.vector_store.as_ref()
    }

    pub(super) fn doc_store(&self) -> &dyn DocStore {
        self.doc_store.as_ref()
    }
}
