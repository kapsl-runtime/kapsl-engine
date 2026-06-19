use super::*;

pub(crate) const RAG_DEFAULT_TOP_K: usize = 4;
pub(crate) const RAG_MAX_TOP_K: usize = 32;
pub(crate) const RAG_EMBEDDING_DIM: usize = 256;
pub(crate) const RAG_CHUNK_SIZE: usize = 200;
pub(crate) const RAG_CHUNK_OVERLAP: usize = 40;
pub(crate) const RAG_CONTEXT_MAX_TOKENS: usize = 768;

#[derive(Clone)]
pub(crate) struct RagRuntimeState {
    pub(crate) vector_store: Arc<SqliteVectorStore>,
    pub(crate) doc_store: FsDocStore,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SyncExtensionRequest {
    pub(crate) workspace_id: String,
    pub(crate) source_id: Option<String>,
    pub(crate) cursor: Option<String>,
    pub(crate) tenant_id: Option<String>,
}

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
    fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

pub(crate) fn extension_key(workspace_id: &str, extension_id: &str) -> String {
    format!("{workspace_id}:{extension_id}")
}

pub(crate) fn normalize_tenant_id(tenant_id: Option<&str>) -> String {
    tenant_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("default")
        .to_string()
}

pub(crate) fn normalize_source_ids(
    source_id: Option<String>,
    source_ids: Option<Vec<String>>,
) -> Option<Vec<String>> {
    let mut combined = Vec::new();

    if let Some(source_id) = source_id {
        let trimmed = source_id.trim();
        if !trimmed.is_empty() {
            combined.push(trimmed.to_string());
        }
    }

    if let Some(source_ids) = source_ids {
        for source_id in source_ids {
            let trimmed = source_id.trim();
            if !trimmed.is_empty() {
                combined.push(trimmed.to_string());
            }
        }
    }

    if combined.is_empty() {
        return None;
    }

    combined.sort();
    combined.dedup();
    Some(combined)
}

pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub(crate) fn embed_text_for_rag_with_dim(text: &str, dimension: usize) -> Vec<f32> {
    if dimension == 0 {
        return Vec::new();
    }
    let mut embedding = vec![0.0f32; dimension];
    let mut token_count = 0usize;

    for token in text
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric()))
        .filter(|token| !token.is_empty())
    {
        let normalized = token.to_ascii_lowercase();
        let hash = fnv1a_64(normalized.as_bytes());
        let index = (hash % dimension as u64) as usize;
        let sign = if (hash & 1) == 0 { 1.0 } else { -1.0 };
        embedding[index] += sign;
        token_count += 1;
    }

    if token_count == 0 {
        return embedding;
    }

    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

pub(crate) fn embed_text_for_rag(text: &str) -> Vec<f32> {
    embed_text_for_rag_with_dim(text, RAG_EMBEDDING_DIM)
}

pub(crate) fn chunk_document_text(text: &str) -> Vec<(i64, String)> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return Vec::new();
    }

    let chunk_size = RAG_CHUNK_SIZE.max(1);
    let overlap = RAG_CHUNK_OVERLAP.min(chunk_size.saturating_sub(1));
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut index = 0i64;

    while start < tokens.len() {
        let end = (start + chunk_size).min(tokens.len());
        let chunk = tokens[start..end].join(" ");
        chunks.push((index, chunk));
        if end >= tokens.len() {
            break;
        }
        start = end.saturating_sub(overlap);
        index += 1;
    }

    chunks
}

pub(crate) fn is_textual_content_type(content_type: &str) -> bool {
    let lowered = content_type.trim().to_ascii_lowercase();
    lowered.starts_with("text/")
        || lowered.contains("json")
        || lowered.contains("xml")
        || lowered.contains("yaml")
        || lowered.contains("markdown")
        || lowered.contains("csv")
}

pub(crate) fn decode_text_document_payload(
    payload: &DocumentPayload,
) -> Result<(Vec<u8>, String), String> {
    let bytes = BASE64
        .decode(payload.bytes_b64.as_bytes())
        .map_err(|error| format!("invalid base64 document payload: {}", error))?;

    if bytes.is_empty() {
        return Err("document payload is empty".to_string());
    }

    match String::from_utf8(bytes.clone()) {
        Ok(text) => {
            if text.trim().is_empty() {
                Err("decoded document has no text content".to_string())
            } else {
                Ok((bytes, text))
            }
        }
        Err(_) if is_textual_content_type(&payload.content_type) => {
            let text = String::from_utf8_lossy(&bytes).to_string();
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

pub(crate) fn merged_document_metadata(
    payload: &DocumentPayload,
    source_id: &str,
    doc_id: &str,
) -> HashMap<String, String> {
    let mut metadata = payload.metadata.clone();
    metadata.insert("source".to_string(), source_id.to_string());
    metadata.insert("doc_id".to_string(), doc_id.to_string());
    metadata.insert("document_id".to_string(), doc_id.to_string());
    metadata
}

pub(crate) async fn delete_document_from_rag(
    rag_state: &RagRuntimeState,
    tenant_id: &str,
    workspace_id: &str,
    source_id: &str,
    doc_id: &str,
) -> Result<(), String> {
    rag_state
        .vector_store
        .delete_by_doc(tenant_id, workspace_id, source_id, doc_id)
        .await
        .map_err(|error| format!("failed to delete document from vector store: {}", error))?;

    rag_state
        .doc_store
        .delete(&kapsl_rag::storage::DocKey {
            tenant_id: tenant_id.to_string(),
            workspace_id: workspace_id.to_string(),
            source_id: source_id.to_string(),
            doc_id: doc_id.to_string(),
        })
        .map_err(|error| format!("failed to delete document from doc store: {}", error))?;

    Ok(())
}

pub(crate) async fn ingest_document_payload_into_rag(
    rag_state: &RagRuntimeState,
    tenant_id: &str,
    workspace_id: &str,
    source_id: &str,
    payload: &DocumentPayload,
) -> Result<usize, String> {
    let doc_id = payload.id.trim();
    if doc_id.is_empty() {
        return Err("document id is empty".to_string());
    }

    let (bytes, text) = decode_text_document_payload(payload)?;

    delete_document_from_rag(rag_state, tenant_id, workspace_id, source_id, doc_id).await?;

    rag_state
        .doc_store
        .put(
            &kapsl_rag::storage::DocKey {
                tenant_id: tenant_id.to_string(),
                workspace_id: workspace_id.to_string(),
                source_id: source_id.to_string(),
                doc_id: doc_id.to_string(),
            },
            &bytes,
        )
        .map_err(|error| format!("failed to persist document bytes: {}", error))?;

    let chunks = chunk_document_text(&text);
    if chunks.is_empty() {
        return Ok(0);
    }
    let chunk_count = chunks.len();

    let base_metadata = merged_document_metadata(payload, source_id, doc_id);
    let acl = AccessControl {
        allow_users: payload.acl.allow_users.clone(),
        allow_groups: payload.acl.allow_groups.clone(),
        deny_users: payload.acl.deny_users.clone(),
        deny_groups: payload.acl.deny_groups.clone(),
    };

    let mut embedded_chunks = Vec::with_capacity(chunks.len());
    for (chunk_index, chunk_text) in chunks {
        let mut metadata = base_metadata.clone();
        metadata.insert("chunk_index".to_string(), chunk_index.to_string());
        let embedding = embed_text_for_rag(&chunk_text);
        embedded_chunks.push(EmbeddedChunk {
            id: format!("{doc_id}:{chunk_index}"),
            tenant_id: tenant_id.to_string(),
            workspace_id: workspace_id.to_string(),
            source_id: source_id.to_string(),
            doc_id: doc_id.to_string(),
            chunk_index,
            text: chunk_text,
            embedding,
            metadata,
            acl: acl.clone(),
        });
    }

    rag_state
        .vector_store
        .upsert(embedded_chunks)
        .await
        .map_err(|error| format!("failed to upsert vector chunks: {}", error))?;

    Ok(chunk_count)
}

pub(crate) fn select_sync_source_id(
    explicit_source_id: Option<String>,
    connector_config: serde_json::Value,
    client: &mut ConnectorClient<ConnectorRuntimeHandle>,
) -> Result<String, String> {
    if let Some(source_id) = explicit_source_id {
        let trimmed = source_id.trim();
        if trimmed.is_empty() {
            return Err("source_id cannot be empty".to_string());
        }
        return Ok(trimmed.to_string());
    }

    let sources_response = client
        .request(ConnectorRequestKind::ListSources {
            config: connector_config,
        })
        .map_err(|error| format!("failed to list connector sources: {}", error))?;

    match sources_response.kind {
        ConnectorResponseKind::Err(error) => Err(error.message),
        ConnectorResponseKind::Ok(ConnectorResult::Sources(sources)) => {
            pick_default_source_id(&sources)
        }
        _ => Err("connector returned unexpected response for ListSources".to_string()),
    }
}

pub(crate) fn pick_default_source_id(sources: &[SourceDescriptor]) -> Result<String, String> {
    let source = sources
        .first()
        .ok_or_else(|| "connector returned no sources".to_string())?;
    let source_id = source.id.trim();
    if source_id.is_empty() {
        return Err("connector returned an empty source id".to_string());
    }
    Ok(source_id.to_string())
}

pub(crate) fn parse_infer_rag_options(
    payload: &serde_json::Value,
) -> Result<Option<InferRagOptions>, RagAugmentError> {
    let Some(raw_rag) = payload.get("rag") else {
        return Ok(None);
    };

    let options: InferRagOptions = serde_json::from_value(raw_rag.clone()).map_err(|error| {
        RagAugmentError::bad_request(format!("Invalid `rag` infer options: {}", error))
    })?;

    validate_infer_rag_options(Some(options))
}

pub(crate) fn validate_infer_rag_options(
    options: Option<InferRagOptions>,
) -> Result<Option<InferRagOptions>, RagAugmentError> {
    let Some(options) = options else {
        return Ok(None);
    };

    if options.enabled == Some(false) {
        return Ok(None);
    }

    if options.workspace_id.trim().is_empty() {
        return Err(RagAugmentError::bad_request(
            "`rag.workspace_id` is required",
        ));
    }

    if matches!(options.top_k, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.top_k` must be greater than 0",
        ));
    }

    if matches!(options.max_context_tokens, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.max_context_tokens` must be greater than 0",
        ));
    }

    if matches!(options.max_chunks, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.max_chunks` must be greater than 0",
        ));
    }

    if matches!(options.max_per_source, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.max_per_source` must be greater than 0",
        ));
    }

    Ok(Some(options))
}

#[derive(Debug, Deserialize)]
pub(crate) struct InferPayloadEnvelope<T> {
    #[serde(default)]
    pub(crate) rag: Option<InferRagOptions>,
    #[serde(flatten)]
    pub(crate) request: T,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn query_rag_chunks(
    rag_state: &RagRuntimeState,
    workspace_id: &str,
    tenant_id: Option<&str>,
    query: &str,
    source_id: Option<String>,
    source_ids: Option<Vec<String>>,
    top_k: Option<usize>,
    min_score: Option<f32>,
    allowed_users: Vec<String>,
    allowed_groups: Vec<String>,
) -> Result<Vec<RagChunk>, RagAugmentError> {
    let query = query.trim();
    if query.is_empty() {
        return Err(RagAugmentError::bad_request("RAG query cannot be empty"));
    }

    let source_ids = normalize_source_ids(source_id, source_ids);
    let top_k = top_k.unwrap_or(RAG_DEFAULT_TOP_K).clamp(1, RAG_MAX_TOP_K);
    let min_score = min_score.unwrap_or(0.0);
    let query_embedding = embed_text_for_rag(query);
    if query_embedding.is_empty() {
        return Ok(Vec::new());
    }

    let query_request = VectorQuery {
        query_embedding,
        top_k,
        tenant_id: normalize_tenant_id(tenant_id),
        workspace_id: workspace_id.to_string(),
        source_ids,
        allowed_users,
        allowed_groups,
        min_score,
    };

    let results = rag_state
        .vector_store
        .query(query_request)
        .await
        .map_err(|error| {
            RagAugmentError::internal(format!("Failed to query vector store: {}", error))
        })?;

    Ok(results
        .into_iter()
        .map(|result| RagChunk {
            id: result.chunk.id,
            text: result.chunk.text,
            score: result.score,
            metadata: result.chunk.metadata,
        })
        .collect())
}

pub(crate) fn inject_rag_context_into_prompt(prompt: &str, context: &str) -> String {
    let user_marker = "<start_of_turn>user\n";
    let end_marker = "<end_of_turn>";
    if let Some(user_start) = prompt.rfind(user_marker) {
        let content_start = user_start + user_marker.len();
        if prompt[content_start..].contains(end_marker) {
            let mut output = String::with_capacity(prompt.len() + context.len() + 160);
            output.push_str(&prompt[..content_start]);
            output.push_str("Use the retrieved context below when relevant.\n\n");
            output.push_str("[Retrieved Context]\n");
            output.push_str(context);
            output.push_str("\n[/Retrieved Context]\n\n");
            output.push_str(&prompt[content_start..]);
            return output;
        }
    }

    format!(
        "Use the retrieved context below when relevant.\n\n[Retrieved Context]\n{}\n[/Retrieved Context]\n\n{}",
        context, prompt
    )
}

pub(crate) async fn augment_inference_request_with_rag(
    request: &mut InferenceRequest,
    rag_options: &InferRagOptions,
    rag_state: &RagRuntimeState,
) -> Result<usize, RagAugmentError> {
    if request.input.dtype != TensorDtype::Utf8 {
        return Err(RagAugmentError::bad_request(
            "`rag` is currently supported only for `string` infer inputs",
        ));
    }

    let prompt = String::from_utf8(request.input.data.clone()).map_err(|error| {
        RagAugmentError::bad_request(format!("failed to decode UTF-8 prompt: {}", error))
    })?;

    let retrieved_chunks = query_rag_chunks(
        rag_state,
        &rag_options.workspace_id,
        rag_options.tenant_id.as_deref(),
        &prompt,
        rag_options.source_id.clone(),
        rag_options.source_ids.clone(),
        rag_options.top_k,
        rag_options.min_score,
        Vec::new(),
        Vec::new(),
    )
    .await?;

    if retrieved_chunks.is_empty() {
        return Ok(0);
    }

    let mut prompt_config = RagPromptConfig {
        max_context_tokens: RAG_CONTEXT_MAX_TOKENS,
        citation_style: CitationStyle::BracketedNumber,
        ..RagPromptConfig::default()
    };
    if let Some(max_context_tokens) = rag_options.max_context_tokens {
        prompt_config.max_context_tokens = max_context_tokens;
    }
    if let Some(max_chunks) = rag_options.max_chunks {
        prompt_config.max_chunks = max_chunks;
    }
    if let Some(max_per_source) = rag_options.max_per_source {
        prompt_config.max_per_source = max_per_source;
    }
    if let Some(min_score) = rag_options.min_score {
        prompt_config.min_score = min_score;
    }

    let rag_prompt = build_rag_prompt(&retrieved_chunks, &prompt_config, &WhitespaceTokenCounter);
    if rag_prompt.context.trim().is_empty() {
        return Ok(0);
    }

    let augmented_prompt = inject_rag_context_into_prompt(&prompt, &rag_prompt.context);
    request.input.data = augmented_prompt.into_bytes();
    request.input.shape = vec![1, request.input.data.len() as i64];

    Ok(rag_prompt.used_chunks.len())
}
