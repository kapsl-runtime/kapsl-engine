use super::embedding::embed_text;
use super::{RagAugmentError, RagQuery, RagService};
use kapsl_llm::rag::RagChunk;
use kapsl_rag::VectorQuery;

const DEFAULT_TOP_K: usize = 4;
const MAX_TOP_K: usize = 32;

pub(crate) fn normalize_tenant_id(tenant_id: Option<&str>) -> String {
    tenant_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("default")
        .to_string()
}

fn normalize_source_ids(
    source_id: Option<String>,
    source_ids: Option<Vec<String>>,
) -> Option<Vec<String>> {
    let mut combined = source_id
        .into_iter()
        .chain(source_ids.into_iter().flatten())
        .filter_map(|source_id| {
            let trimmed = source_id.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect::<Vec<_>>();
    if combined.is_empty() {
        return None;
    }
    combined.sort();
    combined.dedup();
    Some(combined)
}

impl RagService {
    pub(crate) async fn query_chunks(
        &self,
        query: RagQuery,
    ) -> Result<Vec<RagChunk>, RagAugmentError> {
        if query.workspace_id.is_empty() {
            return Err(RagAugmentError::bad_request("workspace_id is required"));
        }
        let text = query.text.trim();
        if text.is_empty() {
            return Err(RagAugmentError::bad_request("RAG query cannot be empty"));
        }

        let request = VectorQuery {
            query_embedding: embed_text(text),
            top_k: query.top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K),
            tenant_id: normalize_tenant_id(query.tenant_id.as_deref()),
            workspace_id: query.workspace_id,
            source_ids: normalize_source_ids(query.source_id, query.source_ids),
            allowed_users: query.allowed_users,
            allowed_groups: query.allowed_groups,
            min_score: query.min_score.unwrap_or(0.0),
        };
        let results = self.vector_store().query(request).await.map_err(|error| {
            RagAugmentError::internal(format!("Failed to query vector store: {error}"))
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
}

#[cfg(test)]
mod tests {
    use super::{normalize_source_ids, normalize_tenant_id};

    #[test]
    fn source_ids_are_trimmed_sorted_and_deduplicated() {
        assert_eq!(
            normalize_source_ids(
                Some(" source-b ".to_string()),
                Some(vec![
                    "source-a".to_string(),
                    "source-b".to_string(),
                    " ".to_string(),
                ]),
            ),
            Some(vec!["source-a".to_string(), "source-b".to_string()])
        );
    }

    #[test]
    fn tenant_defaults_only_for_missing_or_blank_values() {
        assert_eq!(normalize_tenant_id(None), "default");
        assert_eq!(normalize_tenant_id(Some("  ")), "default");
        assert_eq!(normalize_tenant_id(Some(" tenant-a ")), "tenant-a");
    }
}
