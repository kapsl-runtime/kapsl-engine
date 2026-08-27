use super::embedding::embed_text;
use super::RagService;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use kapsl_rag::storage::DocKey;
use kapsl_rag::{AccessControl, EmbeddedChunk};
use kapsl_rag_sdk::types::DocumentPayload;
use std::collections::HashMap;

const CHUNK_SIZE: usize = 200;
const CHUNK_OVERLAP: usize = 40;

fn chunk_document_text_with_limits(
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Vec<(i64, String)> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return Vec::new();
    }

    let chunk_size = chunk_size.max(1);
    let overlap = overlap.min(chunk_size.saturating_sub(1));
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut index = 0i64;
    while start < tokens.len() {
        let end = (start + chunk_size).min(tokens.len());
        chunks.push((index, tokens[start..end].join(" ")));
        if end >= tokens.len() {
            break;
        }
        start = end.saturating_sub(overlap);
        index += 1;
    }
    chunks
}

fn chunk_document_text(text: &str) -> Vec<(i64, String)> {
    chunk_document_text_with_limits(text, CHUNK_SIZE, CHUNK_OVERLAP)
}

fn is_textual_content_type(content_type: &str) -> bool {
    let lowered = content_type.trim().to_ascii_lowercase();
    lowered.starts_with("text/")
        || lowered.contains("json")
        || lowered.contains("xml")
        || lowered.contains("yaml")
        || lowered.contains("markdown")
        || lowered.contains("csv")
}

fn decode_text_document_payload(payload: &DocumentPayload) -> Result<(Vec<u8>, String), String> {
    let bytes = BASE64
        .decode(payload.bytes_b64.as_bytes())
        .map_err(|error| format!("invalid base64 document payload: {error}"))?;
    if bytes.is_empty() {
        return Err("document payload is empty".to_string());
    }

    match std::str::from_utf8(&bytes) {
        Ok(text) if text.trim().is_empty() => {
            Err("decoded document has no text content".to_string())
        }
        Ok(text) => Ok((bytes.clone(), text.to_string())),
        Err(_) if is_textual_content_type(&payload.content_type) => {
            let text = String::from_utf8_lossy(&bytes).into_owned();
            if text.trim().is_empty() {
                Err("decoded document has no text content".to_string())
            } else {
                Ok((bytes, text))
            }
        }
        Err(_) => Err(format!(
            "unsupported non-text content type `{}`",
            payload.content_type
        )),
    }
}

fn merged_document_metadata(
    payload: &DocumentPayload,
    source_id: &str,
    document_id: &str,
) -> HashMap<String, String> {
    let mut metadata = payload.metadata.clone();
    metadata.insert("source".to_string(), source_id.to_string());
    metadata.insert("doc_id".to_string(), document_id.to_string());
    metadata.insert("document_id".to_string(), document_id.to_string());
    metadata
}

impl RagService {
    pub(crate) async fn delete_document(
        &self,
        tenant_id: &str,
        workspace_id: &str,
        source_id: &str,
        document_id: &str,
    ) -> Result<(), String> {
        self.vector_store()
            .delete_by_doc(tenant_id, workspace_id, source_id, document_id)
            .await
            .map_err(|error| format!("failed to delete document from vector store: {error}"))?;
        self.doc_store()
            .delete(&document_key(
                tenant_id,
                workspace_id,
                source_id,
                document_id,
            ))
            .map_err(|error| format!("failed to delete document from doc store: {error}"))
    }

    pub(crate) async fn ingest_document(
        &self,
        tenant_id: &str,
        workspace_id: &str,
        source_id: &str,
        payload: &DocumentPayload,
    ) -> Result<usize, String> {
        let document_id = payload.id.trim();
        if document_id.is_empty() {
            return Err("document id is empty".to_string());
        }

        // Complete all fallible decoding and allocation before mutating either
        // persistence adapter. Cross-store commits are not atomic, so document
        // bytes are restored on a subsequent vector-store failure.
        let (bytes, text) = decode_text_document_payload(payload)?;
        let chunks = chunk_document_text(&text);
        let chunk_count = chunks.len();
        let base_metadata = merged_document_metadata(payload, source_id, document_id);
        let access_control = AccessControl {
            allow_users: payload.acl.allow_users.clone(),
            allow_groups: payload.acl.allow_groups.clone(),
            deny_users: payload.acl.deny_users.clone(),
            deny_groups: payload.acl.deny_groups.clone(),
        };
        let embedded_chunks = chunks
            .into_iter()
            .map(|(chunk_index, text)| {
                let mut metadata = base_metadata.clone();
                metadata.insert("chunk_index".to_string(), chunk_index.to_string());
                EmbeddedChunk {
                    id: format!("{document_id}:{chunk_index}"),
                    tenant_id: tenant_id.to_string(),
                    workspace_id: workspace_id.to_string(),
                    source_id: source_id.to_string(),
                    doc_id: document_id.to_string(),
                    chunk_index,
                    embedding: embed_text(&text),
                    text,
                    metadata,
                    acl: access_control.clone(),
                }
            })
            .collect::<Vec<_>>();

        let key = document_key(tenant_id, workspace_id, source_id, document_id);
        let previous_document = self.doc_store().get(&key).ok();
        if let Err(error) = self.doc_store().put(&key, &bytes) {
            let message = format!("failed to persist document bytes: {error}");
            return Err(self.with_document_rollback(&key, previous_document.as_deref(), message));
        }

        if let Err(error) = self
            .vector_store()
            .delete_by_doc(tenant_id, workspace_id, source_id, document_id)
            .await
        {
            let message = format!("failed to replace document in vector store: {error}");
            return Err(self.with_document_rollback(&key, previous_document.as_deref(), message));
        }

        if let Err(error) = self.vector_store().upsert(embedded_chunks).await {
            let message = format!(
                "failed to upsert vector chunks: {error}; previous vector chunks may require resynchronization"
            );
            return Err(self.with_document_rollback(&key, previous_document.as_deref(), message));
        }

        Ok(chunk_count)
    }

    fn with_document_rollback(
        &self,
        key: &DocKey,
        previous_document: Option<&[u8]>,
        message: String,
    ) -> String {
        let rollback = match previous_document {
            Some(bytes) => self.doc_store().put(key, bytes).map(|_| ()),
            None => self.doc_store().delete(key),
        };
        match rollback {
            Ok(()) => message,
            Err(error) => format!("{message}; failed to roll back document bytes: {error}"),
        }
    }
}

fn document_key(tenant_id: &str, workspace_id: &str, source_id: &str, document_id: &str) -> DocKey {
    DocKey {
        tenant_id: tenant_id.to_string(),
        workspace_id: workspace_id.to_string(),
        source_id: source_id.to_string(),
        doc_id: document_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        chunk_document_text_with_limits, decode_text_document_payload, document_key, RagService,
    };
    use async_trait::async_trait;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use base64::Engine as _;
    use kapsl_rag::storage::{DocKey, DocStoreError};
    use kapsl_rag::{
        DocStore, EmbeddedChunk, VectorQuery, VectorSearchResult, VectorStore, VectorStoreError,
    };
    use kapsl_rag_sdk::types::{DocumentPayload, ExternalAcl};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    fn payload(bytes: &[u8], content_type: &str) -> DocumentPayload {
        DocumentPayload {
            id: "doc-1".to_string(),
            content_type: content_type.to_string(),
            bytes_b64: BASE64.encode(bytes),
            metadata: HashMap::new(),
            acl: ExternalAcl::default(),
        }
    }

    #[test]
    fn chunks_overlap_without_repeating_forever() {
        let chunks = chunk_document_text_with_limits("a b c d e", 3, 1);
        assert_eq!(
            chunks,
            vec![(0, "a b c".to_string()), (1, "c d e".to_string())]
        );
    }

    #[test]
    fn decodes_utf8_and_rejects_binary_payloads() {
        let (_, text) = decode_text_document_payload(&payload(b"hello", "text/plain"))
            .expect("decode text payload");
        assert_eq!(text, "hello");

        let error = decode_text_document_payload(&payload(&[0xff, 0xfe], "image/png"))
            .expect_err("reject binary payload");
        assert!(error.contains("unsupported non-text content type"));
    }

    #[derive(Default)]
    struct MemoryDocStore {
        documents: Mutex<HashMap<String, Vec<u8>>>,
    }

    impl MemoryDocStore {
        fn key(key: &DocKey) -> String {
            format!(
                "{}/{}/{}/{}",
                key.tenant_id, key.workspace_id, key.source_id, key.doc_id
            )
        }
    }

    impl DocStore for MemoryDocStore {
        fn put(&self, key: &DocKey, bytes: &[u8]) -> Result<PathBuf, DocStoreError> {
            self.documents
                .lock()
                .expect("lock document store")
                .insert(Self::key(key), bytes.to_vec());
            Ok(PathBuf::from(&key.doc_id))
        }

        fn get(&self, key: &DocKey) -> Result<Vec<u8>, DocStoreError> {
            self.documents
                .lock()
                .expect("lock document store")
                .get(&Self::key(key))
                .cloned()
                .ok_or_else(|| DocStoreError::Io("not found".to_string()))
        }

        fn delete(&self, key: &DocKey) -> Result<(), DocStoreError> {
            self.documents
                .lock()
                .expect("lock document store")
                .remove(&Self::key(key));
            Ok(())
        }
    }

    struct FailingVectorStore;

    #[async_trait]
    impl VectorStore for FailingVectorStore {
        async fn upsert(&self, _chunks: Vec<EmbeddedChunk>) -> Result<(), VectorStoreError> {
            Err(VectorStoreError::Db("injected failure".to_string()))
        }

        async fn delete_by_doc(
            &self,
            _tenant_id: &str,
            _workspace_id: &str,
            _source_id: &str,
            _document_id: &str,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }

        async fn query(
            &self,
            _request: VectorQuery,
        ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn injected_store_failure_restores_previous_document_bytes() {
        let documents = Arc::new(MemoryDocStore::default());
        let key = document_key("tenant", "workspace", "source", "doc-1");
        documents.put(&key, b"previous").expect("seed document");
        let rag = RagService::new(Arc::new(FailingVectorStore), documents.clone());

        let error = rag
            .ingest_document(
                "tenant",
                "workspace",
                "source",
                &payload(b"replacement", "text/plain"),
            )
            .await
            .expect_err("injected vector failure");

        assert!(error.contains("previous vector chunks may require resynchronization"));
        assert_eq!(documents.get(&key).expect("restored document"), b"previous");
    }
}
