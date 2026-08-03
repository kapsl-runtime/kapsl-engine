//! Wire types for the OpenAI-compatible surface, plus the translation between
//! them and the runtime's native `InferenceRequest`.

use super::*;
use kapsl_llm::prompt_adapter::{chat_template_from_model_identifiers, ChatTurn};

/// `POST /v1/chat/completions` request body.
///
/// Unknown fields are ignored rather than rejected: OpenAI clients routinely
/// send parameters this runtime has no equivalent for, and failing the request
/// over them would break the drop-in replacement property this surface exists
/// for. Parameters that *are* understood but cannot be honored are reported
/// explicitly (see [`ChatCompletionRequest::unsupported`]).
#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(crate) max_tokens: Option<u32>,
    /// The newer OpenAI spelling of `max_tokens`.
    #[serde(default)]
    pub(crate) max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) top_p: Option<f32>,
    /// Not part of the OpenAI schema, but widely accepted by compatible
    /// servers and directly supported by the runtime sampler.
    #[serde(default)]
    pub(crate) top_k: Option<u32>,
    #[serde(default)]
    pub(crate) repetition_penalty: Option<f32>,
    #[serde(default)]
    pub(crate) seed: Option<u64>,
    #[serde(default)]
    pub(crate) stream: bool,
    #[serde(default)]
    pub(crate) stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub(crate) stop: Option<StopField>,
    #[serde(default)]
    pub(crate) n: Option<u32>,
    /// End-user identifier. Used as the session key for KV affinity when no
    /// `X-Kapsl-Session` header is present.
    #[serde(default)]
    pub(crate) user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: serde_json::Value,
}

/// OpenAI accepts `stop` as either a single string or an array of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn into_sequences(self) -> Vec<String> {
        let raw = match self {
            Self::One(stop) => vec![stop],
            Self::Many(stops) => stops,
        };
        raw.into_iter().filter(|stop| !stop.is_empty()).collect()
    }
}

impl ChatCompletionRequest {
    pub(crate) fn max_new_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub(crate) fn stop_sequences(&self) -> Vec<String> {
        match &self.stop {
            Some(StopField::One(stop)) => StopField::One(stop.clone()).into_sequences(),
            Some(StopField::Many(stops)) => StopField::Many(stops.clone()).into_sequences(),
            None => Vec::new(),
        }
    }

    /// Reject the parameters this runtime understands but cannot honor, rather
    /// than accepting them and quietly returning something different from what
    /// was asked for.
    pub(crate) fn unsupported(&self) -> Option<String> {
        match self.n {
            Some(n) if n > 1 => Some(format!(
                "n={n} is not supported: this runtime returns a single choice per request"
            )),
            _ => None,
        }
    }

    pub(crate) fn to_request_metadata(&self) -> kapsl_engine_api::RequestMetadata {
        kapsl_engine_api::RequestMetadata {
            max_new_tokens: self.max_new_tokens(),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            ..Default::default()
        }
    }
}

/// Flatten OpenAI message content, which is either a plain string or an array
/// of typed content parts, into text.
pub(crate) fn message_content_to_text(value: &serde_json::Value) -> Result<String, String> {
    if value.is_null() {
        return Ok(String::new());
    }
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }
    if let Some(parts) = value.as_array() {
        let mut texts = Vec::new();
        for part in parts {
            match part.get("type").and_then(|value| value.as_str()) {
                Some("text") => {
                    if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                        texts.push(text.to_string());
                    }
                }
                Some(other) => {
                    return Err(format!(
                        "Unsupported chat message content part '{other}': only text is supported"
                    ));
                }
                None => {}
            }
        }
        return Ok(texts.join("\n"));
    }
    Err("Chat message content must be a string or an array of content parts".to_string())
}

/// Turn a conversation into the prompt string handed to the backend.
///
/// A lone user message is passed through untouched so the backend can apply
/// the model's *real* chat template — for GGUF that is the Jinja template
/// embedded in the file, which is always more faithful than anything inferable
/// from the model's name.
///
/// Multi-turn conversations have no such path today: `InferenceRequest` carries
/// a single prompt, so the turns have to be rendered here, using the template
/// family guessed from the model name. The rendered text contains chat markers,
/// which the backend detects and passes through rather than double-wrapping.
pub(crate) fn build_prompt(messages: &[ChatMessage], model_name: &str) -> Result<String, String> {
    if messages.is_empty() {
        return Err("messages must contain at least one item".to_string());
    }

    let turns = messages
        .iter()
        .map(|message| {
            message_content_to_text(&message.content)
                .map(|content| ChatTurn::new(message.role.clone(), content))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let only_user_turn = turns.len() == 1 && turns[0].role.trim().eq_ignore_ascii_case("user");
    if only_user_turn {
        return Ok(turns[0].content.clone());
    }

    match chat_template_from_model_identifiers([model_name]) {
        Some(template) => Ok(template.render_chat(&turns)),
        // No recognizable family: a plain transcript is the honest fallback.
        // It is what the model sees, and it is better than dropping history.
        None => {
            let mut prompt = String::new();
            for turn in &turns {
                prompt.push_str(turn.role.trim());
                prompt.push_str(": ");
                prompt.push_str(turn.content.trim());
                prompt.push('\n');
            }
            prompt.push_str("assistant: ");
            Ok(prompt)
        }
    }
}

/// Where a stop sequence was found in generated text.
pub(crate) struct StopHit {
    pub(crate) cut: usize,
}

/// Find the earliest stop sequence in `text`.
///
/// The runtime's backends only honor their built-in chat turn markers, so
/// caller-supplied `stop` strings are enforced here instead: the completion is
/// truncated at the marker and, when streaming, generation is cancelled.
pub(crate) fn find_stop_sequence(text: &str, stops: &[String]) -> Option<StopHit> {
    stops
        .iter()
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
        .map(|cut| StopHit { cut })
}

pub(crate) fn completion_id() -> String {
    let mut bytes = [0u8; 12];
    OsRng.fill_bytes(&mut bytes);
    format!("chatcmpl-{}", BASE64_URL_SAFE_NO_PAD.encode(bytes))
}

/// Approximate token count.
///
/// The scheduler returns generated text, not token counts, so `usage` on the
/// non-streaming path is an estimate. The streaming path counts emitted tokens
/// exactly and uses that instead. Roughly four characters per token matches
/// common BPE vocabularies far better than counting whitespace-delimited words.
pub(crate) fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    ((text.chars().count() as f64) / 4.0).ceil() as u64
}

pub(crate) fn chat_completion_response(
    id: &str,
    created: u64,
    model: &str,
    content: &str,
    finish_reason: &str,
    usage: Usage,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "logprobs": null,
            "finish_reason": finish_reason,
        }],
        "usage": usage.to_json(),
    })
}

pub(crate) struct Usage {
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
}

impl Usage {
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        })
    }
}

pub(crate) fn chat_completion_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> String {
    let event = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "logprobs": null,
            "finish_reason": finish_reason,
        }],
    });
    format!("data: {event}\n\n")
}

/// A terminal `chat.completion.chunk` carrying only usage, as requested by
/// `stream_options.include_usage`. Per the OpenAI spec this chunk has an empty
/// `choices` array.
pub(crate) fn chat_completion_usage_chunk(
    id: &str,
    created: u64,
    model: &str,
    usage: &Usage,
) -> String {
    let event = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage.to_json(),
    });
    format!("data: {event}\n\n")
}

/// OpenAI-shaped error body. Clients (and the official SDKs) parse the nested
/// `error` object, not a bare `{"error": "..."}` string.
pub(crate) fn openai_error(
    status: warp::http::StatusCode,
    message: impl Into<String>,
    error_type: &str,
) -> warp::reply::Response {
    let body = serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
            "param": serde_json::Value::Null,
            "code": serde_json::Value::Null,
        }
    });
    reply_into_response(warp::reply::with_status(warp::reply::json(&body), status))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
        }
    }

    #[test]
    fn lone_user_message_is_left_for_the_backend_template() {
        let prompt = build_prompt(&[message("user", "hello")], "qwen2.5-7b-instruct")
            .expect("prompt should build");
        assert_eq!(prompt, "hello");
    }

    #[test]
    fn multi_turn_uses_the_family_template_guessed_from_the_model_name() {
        let prompt = build_prompt(
            &[message("system", "Be terse."), message("user", "hello")],
            "qwen2.5-7b-instruct",
        )
        .expect("prompt should build");
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe terse.<|im_end|>\n\
             <|im_start|>user\nhello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn unknown_model_family_falls_back_to_a_transcript() {
        let prompt = build_prompt(
            &[message("system", "Be terse."), message("user", "hello")],
            "some-unknown-model",
        )
        .expect("prompt should build");
        assert_eq!(prompt, "system: Be terse.\nuser: hello\nassistant: ");
    }

    #[test]
    fn empty_messages_are_rejected() {
        assert!(build_prompt(&[], "qwen").is_err());
    }

    #[test]
    fn content_parts_are_flattened_to_text() {
        let content = serde_json::json!([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]);
        assert_eq!(
            message_content_to_text(&content).expect("text parts should flatten"),
            "first\nsecond"
        );
    }

    #[test]
    fn non_text_content_parts_are_rejected() {
        let content = serde_json::json!([
            {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
        ]);
        assert!(message_content_to_text(&content).is_err());
    }

    #[test]
    fn stop_accepts_a_string_or_an_array() {
        let one: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [], "stop": "END"
        }))
        .expect("request should parse");
        assert_eq!(one.stop_sequences(), vec!["END".to_string()]);

        let many: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [], "stop": ["A", "", "B"]
        }))
        .expect("request should parse");
        assert_eq!(
            many.stop_sequences(),
            vec!["A".to_string(), "B".to_string()]
        );
    }

    #[test]
    fn unknown_openai_fields_are_ignored() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 0.5,
            "logit_bias": {"1": 2},
            "response_format": {"type": "text"},
        }))
        .expect("unknown fields should not fail the request");
        assert_eq!(request.model, "m");
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [], "max_tokens": 10, "max_completion_tokens": 20
        }))
        .expect("request should parse");
        assert_eq!(request.max_new_tokens(), Some(20));
    }

    #[test]
    fn multiple_choices_are_rejected_rather_than_silently_ignored() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [], "n": 3
        }))
        .expect("request should parse");
        assert!(request.unsupported().is_some());
    }

    #[test]
    fn stop_sequence_cuts_at_the_earliest_match() {
        let hit = find_stop_sequence("abcSTOPdefEND", &["END".to_string(), "STOP".to_string()])
            .expect("stop should be found");
        assert_eq!(hit.cut, 3);
        assert!(find_stop_sequence("abc", &["END".to_string()]).is_none());
    }
}
