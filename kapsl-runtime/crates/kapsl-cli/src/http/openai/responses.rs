//! Text-only `POST /v1/responses` compatibility.
//!
//! The Responses API has a different object and streaming event model from
//! Chat Completions, but both ultimately use the same Kapsl scheduler. This
//! adapter intentionally rejects state, tools, structured output, and
//! multimodal inputs until those semantics can be honoured end to end.

use super::types::*;
use super::*;
use futures::StreamExt;

pub(crate) struct ResponsesConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) log_sensitive_ids: bool,
}

pub(crate) fn build_responses_route(
    config: ResponsesConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ResponsesConfig {
        models,
        inference,
        log_sensitive_ids,
    } = config;

    warp::path!("v1" / "responses")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(warp::header::optional::<String>("x-kapsl-session"))
        .and(warp::header::optional::<String>("authorization"))
        .and_then(
            move |body: warp::hyper::body::Bytes,
                  session_header: Option<String>,
                  authorization: Option<String>| {
                let models = models.clone();
                let inference = inference.clone();
                async move {
                    Ok::<_, warp::Rejection>(
                        handle_response(
                            body,
                            session_header,
                            authorization,
                            &models,
                            &inference,
                            log_sensitive_ids,
                        )
                        .await,
                    )
                }
            },
        )
        .map(reply_into_response)
        .boxed()
}

#[derive(Debug, Deserialize)]
struct CreateResponseRequest {
    model: String,
    #[serde(default)]
    input: serde_json::Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    /// Kapsl extension, also accepted by several compatible servers.
    #[serde(default)]
    top_k: Option<u32>,
    /// Kapsl extension, also accepted by several compatible servers.
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    background: Option<bool>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    conversation: Option<serde_json::Value>,
    #[serde(default)]
    prompt: Option<serde_json::Value>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    text: Option<ResponseTextConfig>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default)]
    include: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    top_logprobs: Option<u32>,
    #[serde(default)]
    truncation: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    prompt_cache_key: Option<String>,
    #[serde(default)]
    safety_identifier: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseTextConfig {
    #[serde(default)]
    format: Option<ResponseFormatConfig>,
    #[serde(default)]
    verbosity: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseFormatConfig {
    #[serde(rename = "type")]
    kind: String,
}

impl CreateResponseRequest {
    fn unsupported(&self) -> Option<String> {
        if self.store == Some(true) {
            return Some(
                "store=true is not supported: Kapsl Responses are currently stateless".to_string(),
            );
        }
        if self.background == Some(true) {
            return Some("background responses are not supported".to_string());
        }
        if self
            .previous_response_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        {
            return Some(
                "previous_response_id is not supported: send prior text as input messages"
                    .to_string(),
            );
        }
        if self.conversation.is_some() {
            return Some("conversation state is not supported".to_string());
        }
        if self.prompt.is_some() {
            return Some("stored prompt templates are not supported".to_string());
        }
        if self.tools.as_ref().is_some_and(|tools| !tools.is_empty()) {
            return Some("tools are not supported by the Kapsl Responses endpoint".to_string());
        }
        if let Some(choice) = self.tool_choice.as_ref().filter(|value| !value.is_null()) {
            let harmless = choice
                .as_str()
                .is_some_and(|choice| matches!(choice, "auto" | "none"));
            if !harmless {
                return Some("forced tool_choice is not supported".to_string());
            }
        }
        if let Some(text) = &self.text {
            if let Some(format) = &text.format {
                if format.kind != "text" {
                    return Some(format!(
                        "text.format.type={} is not supported: only plain text output is available",
                        format.kind
                    ));
                }
            }
            if text.verbosity.is_some() {
                return Some("text.verbosity is not supported".to_string());
            }
        }
        if self.reasoning.is_some() {
            return Some("reasoning controls are not supported".to_string());
        }
        if self.include.as_ref().is_some_and(|items| !items.is_empty()) {
            return Some("include expansions are not supported".to_string());
        }
        if self.top_logprobs.is_some_and(|count| count > 0) {
            return Some("top_logprobs is not supported".to_string());
        }
        if self
            .truncation
            .as_deref()
            .is_some_and(|strategy| strategy != "disabled")
        {
            return Some("truncation=auto is not supported".to_string());
        }
        if self
            .service_tier
            .as_deref()
            .is_some_and(|tier| !matches!(tier, "auto" | "default"))
        {
            return Some("only service_tier=auto or default is supported".to_string());
        }
        if let Some(metadata) = &self.metadata {
            if !metadata.is_object() {
                return Some("metadata must be a JSON object".to_string());
            }
        }
        None
    }

    fn to_request_metadata(&self) -> kapsl_engine_api::RequestMetadata {
        kapsl_engine_api::RequestMetadata {
            max_new_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            ..Default::default()
        }
    }

    fn session_key(&self) -> Option<String> {
        [&self.prompt_cache_key, &self.user, &self.safety_identifier]
            .into_iter()
            .flatten()
            .map(|value| value.trim())
            .find(|value| !value.is_empty())
            .map(str::to_string)
    }

    fn response_metadata(&self) -> serde_json::Value {
        self.metadata
            .clone()
            .unwrap_or_else(|| serde_json::json!({}))
    }
}

#[derive(Clone)]
struct ResponseContext {
    response_id: String,
    message_id: String,
    created_at: u64,
    model: String,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    metadata: serde_json::Value,
    user: Option<String>,
    temperature: f32,
    top_p: f32,
    parallel_tool_calls: bool,
}

#[allow(clippy::too_many_arguments)]
async fn handle_response(
    body: warp::hyper::body::Bytes,
    session_header: Option<String>,
    authorization: Option<String>,
    models: &Arc<ModelManager>,
    inference: &Arc<InferenceService>,
    log_sensitive_ids: bool,
) -> warp::reply::Response {
    use warp::http::StatusCode;

    let response_request: CreateResponseRequest = match serde_json::from_slice(body.as_ref()) {
        Ok(request) => request,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("Invalid response payload: {error}"),
                "invalid_request_error",
            );
        }
    };

    if let Some(message) = response_request.unsupported() {
        return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
    }

    let resolved = match resolve_model(models, &response_request.model) {
        Ok(resolved) => resolved,
        Err(message) => {
            return openai_error(StatusCode::NOT_FOUND, message, "invalid_request_error");
        }
    };

    let prompt = match build_response_prompt(&response_request, &resolved.name) {
        Ok(prompt) => prompt,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };

    let data = prompt.as_bytes().to_vec();
    let input = match BinaryTensorPacket::new(
        vec![1, data.len() as i64],
        kapsl_engine_api::TensorDtype::Utf8,
        data,
    ) {
        Ok(input) => input,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("Failed to build prompt tensor: {error}"),
                "invalid_request_error",
            );
        }
    };

    let response_id = random_id("resp_");
    let message_id = random_id("msg_");
    let mut metadata = response_request.to_request_metadata();
    metadata.request_id = Some(response_id.clone());
    let mut request = InferenceRequest::new(input).with_metadata(metadata);
    let client_session_id = session_header
        .map(|session| session.trim().to_string())
        .filter(|session| !session.is_empty())
        .or_else(|| response_request.session_key());
    request.session_id =
        scope_session_id_for_authorization(client_session_id.as_deref(), authorization.as_deref());

    let scheduler_priority = inference.priority_for_request(&request);
    let session_id_for_log = redact_identifier_for_logs(
        client_session_id.as_deref().unwrap_or("-"),
        log_sensitive_ids,
    );
    let prompt_tokens = estimate_tokens(&prompt);
    let context = ResponseContext {
        response_id,
        message_id,
        created_at: now_unix_seconds(),
        model: resolved.name,
        instructions: response_request.instructions.clone(),
        max_output_tokens: response_request.max_output_tokens,
        metadata: response_request.response_metadata(),
        user: response_request.user.clone(),
        temperature: response_request.temperature.unwrap_or(1.0),
        top_p: response_request.top_p.unwrap_or(1.0),
        parallel_tool_calls: response_request.parallel_tool_calls.unwrap_or(true),
    };

    if response_request.stream {
        return stream_response(StreamResponseArgs {
            inference: inference.clone(),
            request,
            scheduler_priority,
            model_id: resolved.id,
            context,
            prompt_tokens,
            session_id_for_log,
        })
        .await;
    }

    match inference
        .infer(resolved.id, request, scheduler_priority, false)
        .await
    {
        Ok(output) => {
            let content = String::from_utf8_lossy(&output.data).to_string();
            let usage = response_usage(prompt_tokens, estimate_tokens(&content));
            reply_into_response(warp::reply::json(&response_object(
                &context,
                "completed",
                Some(now_unix_seconds()),
                Some(&content),
                Some(usage),
                None,
            )))
        }
        Err(error) => {
            let status = status_code_for_engine_error(&error);
            if status == StatusCode::INTERNAL_SERVER_ERROR {
                log::error!(
                    "Response failed: model_id={} session_id={} status={} error={}",
                    resolved.id,
                    session_id_for_log,
                    status.as_u16(),
                    error
                );
            } else {
                log::warn!(
                    "Response rejected: model_id={} session_id={} status={} error={}",
                    resolved.id,
                    session_id_for_log,
                    status.as_u16(),
                    error
                );
            }
            openai_error(status, error.to_string(), "server_error")
        }
    }
}

fn build_response_prompt(
    request: &CreateResponseRequest,
    model_name: &str,
) -> Result<String, String> {
    let mut messages = Vec::new();
    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
    {
        messages.push(ChatMessage {
            // Local chat templates consistently understand `system`; the
            // Responses contract defines `instructions` as a system or
            // developer message, so this preserves its instruction priority.
            role: "system".to_string(),
            content: serde_json::Value::String(instructions.to_string()),
        });
    }

    match &request.input {
        serde_json::Value::String(text) => messages.push(ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String(text.clone()),
        }),
        serde_json::Value::Array(items) => {
            for item in items {
                messages.push(response_input_item_to_message(item)?);
            }
        }
        serde_json::Value::Null => {}
        _ => return Err("input must be a string or an array of text messages".to_string()),
    }

    if messages.is_empty() {
        return Err("input or instructions must contain text".to_string());
    }
    build_prompt(&messages, model_name)
}

fn response_input_item_to_message(item: &serde_json::Value) -> Result<ChatMessage, String> {
    let object = item
        .as_object()
        .ok_or_else(|| "response input items must be JSON objects".to_string())?;
    let item_type = object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("message");

    if item_type == "input_text" {
        let text = object
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "input_text items require a string text field".to_string())?;
        return Ok(ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String(text.to_string()),
        });
    }
    if item_type != "message" {
        return Err(format!(
            "Unsupported response input item '{item_type}': only text messages are supported"
        ));
    }

    let role = object
        .get("role")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "response message items require a role".to_string())?;
    if !matches!(role, "user" | "assistant" | "system" | "developer") {
        return Err(format!("Unsupported response message role '{role}'"));
    }
    let content = object
        .get("content")
        .ok_or_else(|| "response message items require content".to_string())?;
    let text = response_content_to_text(content)?;
    Ok(ChatMessage {
        role: role.to_string(),
        content: serde_json::Value::String(text),
    })
}

fn response_content_to_text(value: &serde_json::Value) -> Result<String, String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }
    let parts = value
        .as_array()
        .ok_or_else(|| "response message content must be a string or an array".to_string())?;
    let mut texts = Vec::new();
    for part in parts {
        let part_type = part
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "response content parts require a type".to_string())?;
        if !matches!(part_type, "input_text" | "output_text" | "text") {
            return Err(format!(
                "Unsupported response content part '{part_type}': only text is supported"
            ));
        }
        let text = part
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{part_type} content parts require a string text field"))?;
        texts.push(text.to_string());
    }
    Ok(texts.join("\n"))
}

fn random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 12];
    OsRng.fill_bytes(&mut bytes);
    format!("{prefix}{}", BASE64_URL_SAFE_NO_PAD.encode(bytes))
}

fn output_message(
    context: &ResponseContext,
    status: &str,
    content: Option<&str>,
) -> serde_json::Value {
    let content = match content {
        Some(text) => vec![serde_json::json!({
            "type": "output_text",
            "text": text,
            "annotations": [],
        })],
        None => Vec::new(),
    };
    serde_json::json!({
        "id": context.message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    })
}

fn response_usage(input_tokens: u64, output_tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": 0,
        },
        "total_tokens": input_tokens + output_tokens,
    })
}

fn response_object(
    context: &ResponseContext,
    status: &str,
    completed_at: Option<u64>,
    content: Option<&str>,
    usage: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
) -> serde_json::Value {
    let output = content
        .map(|text| vec![output_message(context, "completed", Some(text))])
        .unwrap_or_default();
    serde_json::json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": completed_at,
        "status": status,
        "error": error,
        "incomplete_details": serde_json::Value::Null,
        "instructions": context.instructions,
        "max_output_tokens": context.max_output_tokens,
        "model": context.model,
        "output": output,
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": serde_json::Value::Null,
        "reasoning": {
            "effort": serde_json::Value::Null,
            "summary": serde_json::Value::Null,
        },
        "store": false,
        "temperature": context.temperature,
        "text": {
            "format": { "type": "text" },
        },
        "tool_choice": "auto",
        "tools": [],
        "top_p": context.top_p,
        "truncation": "disabled",
        "usage": usage,
        "user": context.user,
        "metadata": context.metadata,
    })
}

struct StreamResponseArgs {
    inference: Arc<InferenceService>,
    request: InferenceRequest,
    scheduler_priority: kapsl_scheduler::Priority,
    model_id: u32,
    context: ResponseContext,
    prompt_tokens: u64,
    session_id_for_log: String,
}

async fn stream_response(args: StreamResponseArgs) -> warp::reply::Response {
    let StreamResponseArgs {
        inference,
        request,
        scheduler_priority,
        model_id,
        context,
        prompt_tokens,
        session_id_for_log,
    } = args;

    let stream = match inference
        .infer_stream(model_id, request, scheduler_priority, false)
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            let status = status_code_for_engine_error(&error);
            log::warn!(
                "Response stream start failed: model_id={} session_id={} status={} error={}",
                model_id,
                session_id_for_log,
                status.as_u16(),
                error
            );
            return openai_error(status, error.to_string(), "server_error");
        }
    };

    struct StreamState {
        opened: bool,
        finished: bool,
        sequence_number: u64,
        output_text: String,
        output_tokens: u64,
    }

    let state = StreamState {
        opened: false,
        finished: false,
        sequence_number: 0,
        output_text: String::new(),
        output_tokens: 0,
    };
    let context_for_events = context.clone();

    // `InferenceService` wraps this scheduler stream in a cancel-on-drop guard.
    // Keeping the stream inside the HTTP body state therefore cancels backend
    // generation when a client disconnects and drops the response body.
    let events = futures::stream::unfold((stream, state), move |(mut stream, mut state)| {
        let context = context_for_events.clone();
        async move {
            if state.finished {
                return None;
            }

            if !state.opened {
                state.opened = true;
                let payload = opening_events(&context, &mut state.sequence_number);
                return Some((
                    Ok::<_, std::convert::Infallible>(warp::hyper::body::Bytes::from(payload)),
                    (stream, state),
                ));
            }

            match stream.next().await {
                Some(Ok(packet)) => {
                    let delta = String::from_utf8_lossy(&packet.data).to_string();
                    state.output_tokens += 1;
                    state.output_text.push_str(&delta);
                    let event = serde_json::json!({
                        "type": "response.output_text.delta",
                        "item_id": context.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                    });
                    let payload = sse_event(event, &mut state.sequence_number);
                    Some((
                        Ok::<_, std::convert::Infallible>(warp::hyper::body::Bytes::from(payload)),
                        (stream, state),
                    ))
                }
                Some(Err(error)) => {
                    log::warn!(
                        "Response stream error: model_id={} error={}",
                        model_id,
                        error
                    );
                    let event = serde_json::json!({
                        "type": "error",
                        "code": "server_error",
                        "message": error.to_string(),
                        "param": serde_json::Value::Null,
                    });
                    let payload = sse_event(event, &mut state.sequence_number);
                    state.finished = true;
                    Some((
                        Ok::<_, std::convert::Infallible>(warp::hyper::body::Bytes::from(payload)),
                        (stream, state),
                    ))
                }
                None => {
                    let usage = response_usage(prompt_tokens, state.output_tokens);
                    let payload = closing_events(
                        &context,
                        &state.output_text,
                        usage,
                        &mut state.sequence_number,
                    );
                    state.finished = true;
                    Some((
                        Ok::<_, std::convert::Infallible>(warp::hyper::body::Bytes::from(payload)),
                        (stream, state),
                    ))
                }
            }
        }
    });

    let body = warp::hyper::Body::wrap_stream(events);
    warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(body)
        .expect("Responses SSE response should build")
}

fn opening_events(context: &ResponseContext, sequence_number: &mut u64) -> String {
    let response = response_object(context, "in_progress", None, None, None, None);
    let mut output = sse_event(
        serde_json::json!({
            "type": "response.created",
            "response": response,
        }),
        sequence_number,
    );
    output.push_str(&sse_event(
        serde_json::json!({
            "type": "response.in_progress",
            "response": response_object(context, "in_progress", None, None, None, None),
        }),
        sequence_number,
    ));
    output.push_str(&sse_event(
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": output_message(context, "in_progress", None),
        }),
        sequence_number,
    ));
    output.push_str(&sse_event(
        serde_json::json!({
            "type": "response.content_part.added",
            "item_id": context.message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
            },
        }),
        sequence_number,
    ));
    output
}

fn closing_events(
    context: &ResponseContext,
    text: &str,
    usage: serde_json::Value,
    sequence_number: &mut u64,
) -> String {
    let mut output = sse_event(
        serde_json::json!({
            "type": "response.output_text.done",
            "item_id": context.message_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
        }),
        sequence_number,
    );
    output.push_str(&sse_event(
        serde_json::json!({
            "type": "response.content_part.done",
            "item_id": context.message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": text,
                "annotations": [],
            },
        }),
        sequence_number,
    ));
    output.push_str(&sse_event(
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": output_message(context, "completed", Some(text)),
        }),
        sequence_number,
    ));
    output.push_str(&sse_event(
        serde_json::json!({
            "type": "response.completed",
            "response": response_object(
                context,
                "completed",
                Some(now_unix_seconds()),
                Some(text),
                Some(usage),
                None,
            ),
        }),
        sequence_number,
    ));
    output
}

fn sse_event(mut event: serde_json::Value, sequence_number: &mut u64) -> String {
    let event_type = event
        .get("type")
        .and_then(serde_json::Value::as_str)
        .expect("Responses stream events always have a type")
        .to_string();
    event
        .as_object_mut()
        .expect("Responses stream events are objects")
        .insert(
            "sequence_number".to_string(),
            serde_json::Value::from(*sequence_number),
        );
    *sequence_number += 1;
    format!("event: {event_type}\ndata: {event}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(input: serde_json::Value) -> CreateResponseRequest {
        serde_json::from_value(serde_json::json!({
            "model": "qwen2.5-7b-instruct",
            "input": input,
        }))
        .expect("response request should parse")
    }

    fn context() -> ResponseContext {
        ResponseContext {
            response_id: "resp_test".to_string(),
            message_id: "msg_test".to_string(),
            created_at: 1,
            model: "qwen2.5-7b-instruct".to_string(),
            instructions: None,
            max_output_tokens: Some(10),
            metadata: serde_json::json!({}),
            user: None,
            temperature: 1.0,
            top_p: 1.0,
            parallel_tool_calls: true,
        }
    }

    fn sse_data(payload: &str) -> Vec<serde_json::Value> {
        payload
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("SSE data should be JSON"))
            .collect()
    }

    #[test]
    fn string_input_is_a_plain_user_prompt() {
        let prompt = build_response_prompt(&request(serde_json::json!("hello")), "qwen")
            .expect("prompt should build");
        assert_eq!(prompt, "hello");
    }

    #[test]
    fn message_input_parts_are_flattened_and_templated() {
        let prompt = build_response_prompt(
            &request(serde_json::json!([
                {"role":"system","content":[{"type":"input_text","text":"Be terse."}]},
                {"role":"user","content":"hello"}
            ])),
            "qwen2.5-7b-instruct",
        )
        .expect("prompt should build");
        assert!(prompt.contains("<|im_start|>system\nBe terse."));
        assert!(prompt.contains("<|im_start|>user\nhello"));
    }

    #[test]
    fn multimodal_content_is_rejected() {
        let error = build_response_prompt(
            &request(serde_json::json!([{
                "role":"user",
                "content":[{"type":"input_image","image_url":"https://example.invalid/a.png"}]
            }])),
            "qwen",
        )
        .expect_err("image input must be rejected");
        assert!(error.contains("only text is supported"));
    }

    #[test]
    fn stateful_and_structured_requests_are_rejected() {
        let stateful: CreateResponseRequest = serde_json::from_value(serde_json::json!({
            "model":"m", "input":"hi", "store":true
        }))
        .expect("request should parse");
        assert!(stateful.unsupported().unwrap().contains("stateless"));

        let structured: CreateResponseRequest = serde_json::from_value(serde_json::json!({
            "model":"m", "input":"hi", "text":{"format":{"type":"json_schema"}}
        }))
        .expect("request should parse");
        assert!(structured.unsupported().unwrap().contains("plain text"));
    }

    #[test]
    fn response_sampler_fields_map_to_engine_metadata() {
        let request: CreateResponseRequest = serde_json::from_value(serde_json::json!({
            "model":"m",
            "input":"hi",
            "max_output_tokens":42,
            "temperature":0.25,
            "top_p":0.8,
            "seed":7,
            "top_k":12,
            "repetition_penalty":1.1
        }))
        .expect("request should parse");
        let metadata = request.to_request_metadata();
        assert_eq!(metadata.max_new_tokens, Some(42));
        assert_eq!(metadata.temperature, Some(0.25));
        assert_eq!(metadata.top_p, Some(0.8));
        assert_eq!(metadata.seed, Some(7));
        assert_eq!(metadata.top_k, Some(12));
        assert_eq!(metadata.repetition_penalty, Some(1.1));
    }

    #[test]
    fn response_object_has_sdk_output_text_shape() {
        let response = response_object(
            &context(),
            "completed",
            Some(2),
            Some("hello"),
            Some(response_usage(2, 1)),
            None,
        );
        assert_eq!(response["object"], "response");
        assert_eq!(response["status"], "completed");
        assert_eq!(response["store"], false);
        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(response["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(response["output"][0]["content"][0]["text"], "hello");
        assert_eq!(response["usage"]["total_tokens"], 3);
    }

    #[test]
    fn stream_events_follow_the_responses_lifecycle() {
        let mut sequence = 0;
        let mut payload = opening_events(&context(), &mut sequence);
        payload.push_str(&sse_event(
            serde_json::json!({
                "type":"response.output_text.delta",
                "item_id":"msg_test",
                "output_index":0,
                "content_index":0,
                "delta":"hello"
            }),
            &mut sequence,
        ));
        payload.push_str(&closing_events(
            &context(),
            "hello",
            response_usage(2, 1),
            &mut sequence,
        ));

        let events = sse_data(&payload);
        let types: Vec<&str> = events
            .iter()
            .map(|event| event["type"].as_str().expect("event type"))
            .collect();
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (expected, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], expected as u64);
        }
        assert!(!payload.contains("[DONE]"));
    }
}
