use super::{InferRagOptions, RagAugmentError, RagQuery, RagService};
use kapsl_engine_api::{InferenceRequest, TensorDtype};
use kapsl_llm::rag::{build_rag_prompt, CitationStyle, RagPromptConfig, WhitespaceTokenCounter};

const DEFAULT_CONTEXT_MAX_TOKENS: usize = 768;

pub(crate) fn parse_infer_rag_options(
    payload: &serde_json::Value,
) -> Result<Option<InferRagOptions>, RagAugmentError> {
    let Some(raw_rag) = payload.get("rag") else {
        return Ok(None);
    };
    let options = serde_json::from_value(raw_rag.clone()).map_err(|error| {
        RagAugmentError::bad_request(format!("Invalid `rag` infer options: {error}"))
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

fn inject_context_into_prompt(prompt: &str, context: &str) -> String {
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
        "Use the retrieved context below when relevant.\n\n[Retrieved Context]\n{context}\n[/Retrieved Context]\n\n{prompt}"
    )
}

impl RagService {
    pub(crate) async fn augment_inference_request(
        &self,
        request: &mut InferenceRequest,
        options: &InferRagOptions,
    ) -> Result<usize, RagAugmentError> {
        if request.input.dtype != TensorDtype::Utf8 {
            return Err(RagAugmentError::bad_request(
                "`rag` is currently supported only for `string` infer inputs",
            ));
        }
        let prompt = String::from_utf8(request.input.data.clone()).map_err(|error| {
            RagAugmentError::bad_request(format!("failed to decode UTF-8 prompt: {error}"))
        })?;
        let retrieved_chunks = self
            .query_chunks(RagQuery::for_inference(prompt.clone(), options))
            .await?;
        if retrieved_chunks.is_empty() {
            return Ok(0);
        }

        let mut prompt_config = RagPromptConfig {
            max_context_tokens: DEFAULT_CONTEXT_MAX_TOKENS,
            citation_style: CitationStyle::BracketedNumber,
            ..RagPromptConfig::default()
        };
        if let Some(max_context_tokens) = options.max_context_tokens {
            prompt_config.max_context_tokens = max_context_tokens;
        }
        if let Some(max_chunks) = options.max_chunks {
            prompt_config.max_chunks = max_chunks;
        }
        if let Some(max_per_source) = options.max_per_source {
            prompt_config.max_per_source = max_per_source;
        }
        if let Some(min_score) = options.min_score {
            prompt_config.min_score = min_score;
        }

        let rag_prompt =
            build_rag_prompt(&retrieved_chunks, &prompt_config, &WhitespaceTokenCounter);
        if rag_prompt.context.trim().is_empty() {
            return Ok(0);
        }
        request.input.data = inject_context_into_prompt(&prompt, &rag_prompt.context).into_bytes();
        request.input.shape = vec![1, request.input.data.len() as i64];
        Ok(rag_prompt.used_chunks.len())
    }
}

#[cfg(test)]
mod tests {
    use super::{inject_context_into_prompt, parse_infer_rag_options};

    #[test]
    fn injects_context_into_the_last_gemma_user_turn() {
        let prompt = "<start_of_turn>user\nquestion<end_of_turn>";
        let augmented = inject_context_into_prompt(prompt, "fact");
        assert!(augmented.starts_with("<start_of_turn>user\nUse the retrieved context"));
        assert!(augmented.contains("[Retrieved Context]\nfact\n[/Retrieved Context]"));
        assert!(augmented.ends_with("question<end_of_turn>"));
    }

    #[test]
    fn rejects_zero_inference_limits() {
        let payload = serde_json::json!({
            "rag": {
                "workspace_id": "workspace-a",
                "top_k": 0
            }
        });
        let error = parse_infer_rag_options(&payload).expect_err("reject zero top_k");
        assert!(matches!(error, super::RagAugmentError::BadRequest(_)));
    }
}
