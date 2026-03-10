#[cfg(test)]
mod tests {
    use super::super::{
        build_rag_prompt, CitationStyle, RagChunk, RagPromptConfig, WhitespaceTokenCounter,
    };
    use std::collections::HashMap;

    fn chunk(id: &str, text: &str, score: f32, source: Option<&str>) -> RagChunk {
        let mut metadata = HashMap::new();
        if let Some(src) = source {
            metadata.insert("source".to_string(), src.to_string());
        }
        RagChunk {
            id: id.to_string(),
            text: text.to_string(),
            score,
            metadata,
        }
    }

    #[test]
    fn empty_chunks_produces_fallback() {
        let config = RagPromptConfig::default();
        let prompt = build_rag_prompt(&[], &config, &WhitespaceTokenCounter);
        assert!(prompt.context.is_empty());
        assert!(prompt.used_chunks.is_empty());
        assert!(prompt.fallback_message.is_some());
    }

    #[test]
    fn dedupe_and_per_source_limits_are_applied() {
        let chunks = vec![
            chunk("a", "Same text", 0.9, Some("doc1")),
            chunk("b", "Same text", 0.8, Some("doc1")),
            chunk("c", "Other text", 0.7, Some("doc2")),
        ];
        let config = RagPromptConfig {
            max_per_source: 1,
            max_chunks: 5,
            dedupe: true,
            ..Default::default()
        };

        let prompt = build_rag_prompt(&chunks, &config, &WhitespaceTokenCounter);
        assert_eq!(prompt.used_chunks.len(), 2);
    }

    #[test]
    fn truncation_and_inline_citations_work() {
        let chunks = vec![chunk("a", "one two three four", 0.5, Some("doc"))];
        let config = RagPromptConfig {
            max_context_tokens: 3,
            max_chunks: 1,
            truncate: true,
            citation_style: CitationStyle::Inline,
            ..Default::default()
        };

        let prompt = build_rag_prompt(&chunks, &config, &WhitespaceTokenCounter);
        assert!(prompt.truncated);
        assert_eq!(prompt.total_context_tokens, 3);
        assert!(prompt.context.contains("(1)"));
    }
}
