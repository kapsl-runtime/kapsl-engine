//! `POST /v1/chat/completions`.

use super::types::*;
use super::*;
use futures::StreamExt;
use kapsl_engine_api::{
    OpenAiWireEndpoint, OpenAiWireFormat, OpenAiWireHeader, OpenAiWireMetadata, OpenAiWireRequest,
    OpenAiWireResponse, OpenAiWireResponseHead, OpenAiWireStreamResponse,
};
use warp::hyper::body::Buf;

const VLLM_BRIDGE_MODE_ENV: &str = "KAPSL_VLLM_BRIDGE_MODE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManagedVllmOpenAiMode {
    Translated,
    Wire,
}

impl ManagedVllmOpenAiMode {
    fn from_environment() -> Self {
        Self::from_value(std::env::var(VLLM_BRIDGE_MODE_ENV).ok().as_deref())
    }

    fn from_value(value: Option<&str>) -> Self {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            None | Some("") | Some("wire") => Self::Wire,
            Some("legacy" | "async-translated" | "translated") => Self::Translated,
            Some(other) => {
                log::warn!(
                    "Ignoring unsupported {VLLM_BRIDGE_MODE_ENV}={other:?}; expected wire, async-translated, or legacy"
                );
                Self::Wire
            }
        }
    }

    fn metric_mode(self, streaming: bool) -> &'static str {
        match (self, streaming) {
            (Self::Wire, _) => "wire",
            (Self::Translated, true) => "async_translated",
            (Self::Translated, false) => "legacy",
        }
    }
}

pub(crate) struct ChatCompletionsConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) log_sensitive_ids: bool,
}

enum BoundedChatBody {
    Bytes(warp::hyper::body::Bytes),
    TooLarge,
    ReadError(String),
}

fn bounded_chat_body(
) -> impl warp::Filter<Extract = (BoundedChatBody,), Error = warp::Rejection> + Clone {
    warp::body::stream().then(collect_bounded_chat_body)
}

async fn collect_bounded_chat_body<S, B>(body: S) -> BoundedChatBody
where
    S: futures::Stream<Item = Result<B, warp::Error>>,
    B: Buf,
{
    futures::pin_mut!(body);
    let limit = kapsl_transport::protocol::MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES;
    let mut collected = Vec::new();
    while let Some(chunk) = body.next().await {
        let mut chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => return BoundedChatBody::ReadError(error.to_string()),
        };
        let Some(next_len) = collected.len().checked_add(chunk.remaining()) else {
            return BoundedChatBody::TooLarge;
        };
        if next_len > limit {
            return BoundedChatBody::TooLarge;
        }
        collected.reserve(chunk.remaining());
        while chunk.has_remaining() {
            let bytes = chunk.chunk();
            let length = bytes.len();
            collected.extend_from_slice(bytes);
            chunk.advance(length);
        }
    }
    BoundedChatBody::Bytes(warp::hyper::body::Bytes::from(collected))
}

pub(crate) fn build_chat_completions_route(
    config: ChatCompletionsConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ChatCompletionsConfig {
        models,
        inference,
        log_sensitive_ids,
    } = config;
    let managed_vllm_mode = ManagedVllmOpenAiMode::from_environment();

    warp::path!("v1" / "chat" / "completions")
        .and(warp::post())
        .and(warp::any().map(std::time::Instant::now))
        .and(bounded_chat_body())
        .and(warp::header::optional::<String>("x-kapsl-session"))
        .and(warp::header::optional::<String>("authorization"))
        .and_then(
            move |ingress_started: std::time::Instant,
                  body: BoundedChatBody,
                  session_header: Option<String>,
                  authorization: Option<String>| {
                let models = models.clone();
                let inference = inference.clone();
                async move {
                    Ok::<_, warp::Rejection>(
                        handle_chat_completion(
                            body,
                            ingress_started,
                            session_header,
                            authorization,
                            &models,
                            &inference,
                            log_sensitive_ids,
                            managed_vllm_mode,
                        )
                        .await,
                    )
                }
            },
        )
        .map(reply_into_response)
        .boxed()
}

#[allow(clippy::too_many_arguments)]
async fn handle_chat_completion(
    body: BoundedChatBody,
    ingress_started: std::time::Instant,
    session_header: Option<String>,
    authorization: Option<String>,
    models: &Arc<ModelManager>,
    inference: &Arc<InferenceService>,
    log_sensitive_ids: bool,
    managed_vllm_mode: ManagedVllmOpenAiMode,
) -> warp::reply::Response {
    use warp::http::StatusCode;

    let body = match body {
        BoundedChatBody::Bytes(body) => body,
        BoundedChatBody::TooLarge => {
            return openai_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "Chat completion payload exceeds the {}-byte limit",
                    kapsl_transport::protocol::MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES
                ),
                "invalid_request_error",
            );
        }
        BoundedChatBody::ReadError(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("Failed to read chat completion payload: {error}"),
                "invalid_request_error",
            );
        }
    };

    let mut normalized_body: serde_json::Value = match serde_json::from_slice(body.as_ref()) {
        Ok(value) => value,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("Invalid chat completion payload: {error}"),
                "invalid_request_error",
            );
        }
    };
    let chat: ChatCompletionRequest = match serde_json::from_value(normalized_body.clone()) {
        Ok(chat) => chat,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("Invalid chat completion payload: {error}"),
                "invalid_request_error",
            );
        }
    };

    if let Some(message) = chat.unsupported() {
        return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
    }

    let resolved = match resolve_model(models, &chat.model) {
        Ok(resolved) => resolved,
        Err(message) => {
            return openai_error(StatusCode::NOT_FOUND, message, "invalid_request_error");
        }
    };

    if models.pool(resolved.id).is_none() {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("The model '{}' is not currently loaded", chat.model),
            "invalid_request_error",
        );
    }

    let is_managed_vllm = models
        .registry()
        .get(resolved.id)
        .is_some_and(|model| model.device == "vllm");
    if is_managed_vllm {
        inference.observe_managed_vllm_ingress(
            &resolved.name,
            managed_vllm_mode.metric_mode(chat.stream),
            ingress_started.elapsed(),
        );
    }

    let request_id = completion_id();
    // KV affinity: an explicit session header wins, else the OpenAI end-user id.
    // The internal key is credential-scoped because both values are controlled
    // by the client and must not collide across authentication principals.
    let client_session_id = session_header
        .map(|session| session.trim().to_string())
        .filter(|session| !session.is_empty())
        .or_else(|| {
            chat.user
                .as_ref()
                .map(|user| user.trim().to_string())
                .filter(|user| !user.is_empty())
        });
    let scoped_session_id =
        scope_session_id_for_authorization(client_session_id.as_deref(), authorization.as_deref());
    let session_id_for_log = redact_identifier_for_logs(
        client_session_id.as_deref().unwrap_or("-"),
        log_sensitive_ids,
    );

    if is_managed_vllm && managed_vllm_mode == ManagedVllmOpenAiMode::Wire {
        if let Some(object) = normalized_body.as_object_mut() {
            object.insert(
                "model".to_string(),
                serde_json::Value::String(resolved.name.clone()),
            );
        } else {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "Chat completion payload must be a JSON object",
                "invalid_request_error",
            );
        }
        return relay_managed_vllm_wire(ManagedVllmWireArgs {
            inference: inference.clone(),
            model_id: resolved.id,
            normalized_body,
            original_body_bytes: body.len(),
            stream: chat.stream,
            request_id,
            session_id: scoped_session_id,
            session_id_for_log,
        })
        .await;
    }

    let prompt = match build_prompt(&chat.messages, &resolved.name) {
        Ok(prompt) => prompt,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };
    let stops = chat.stop_sequences();

    // A UTF-8 prompt tensor is shaped `[1, byte_len]`, matching the RAG and
    // native inference path. `[1]` only validates for a one-byte prompt.
    let data = if is_managed_vllm {
        match managed_vllm_chat_input(&chat.messages, &stops) {
            Ok(data) => data,
            Err(error) => {
                return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error");
            }
        }
    } else {
        prompt.as_bytes().to_vec()
    };
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

    let mut metadata = chat.to_request_metadata();
    metadata.request_id = Some(request_id.clone());

    let mut request = InferenceRequest::new(input).with_metadata(metadata);
    request.session_id = scoped_session_id;

    let scheduler_priority = inference.priority_for_request(&request);

    let cancellation = request
        .cancellation
        .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
        .clone();

    let created = now_unix_seconds();
    let prompt_tokens = estimate_tokens(&prompt);

    if chat.stream {
        let include_usage = chat
            .stream_options
            .as_ref()
            .is_some_and(|options| options.include_usage);
        return stream_chat_completion(StreamChatArgs {
            inference: inference.clone(),
            request,
            scheduler_priority,
            cancellation,
            model_id: resolved.id,
            model_name: resolved.name,
            completion_id: request_id,
            created,
            stops,
            prompt_tokens,
            include_usage,
            session_id_for_log,
        })
        .await;
    }

    let result = inference
        .infer(resolved.id, request, scheduler_priority, false)
        .await;

    match result {
        Ok(output) => {
            let raw = String::from_utf8_lossy(&output.data).to_string();
            let (content, finish_reason) = match find_stop_sequence(&raw, &stops) {
                Some(hit) => (raw[..hit.cut].to_string(), "stop"),
                None => (raw, "stop"),
            };
            let usage = Usage {
                prompt_tokens,
                completion_tokens: estimate_tokens(&content),
            };
            reply_into_response(warp::reply::json(&chat_completion_response(
                &request_id,
                created,
                &resolved.name,
                &content,
                finish_reason,
                usage,
            )))
        }
        Err(error) => {
            let status = status_code_for_engine_error(&error);
            if status == StatusCode::INTERNAL_SERVER_ERROR {
                log::error!(
                    "Chat completion failed: model_id={} session_id={} status={} error={}",
                    resolved.id,
                    session_id_for_log,
                    status.as_u16(),
                    error
                );
            } else {
                log::warn!(
                    "Chat completion rejected: model_id={} session_id={} status={} error={}",
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

struct ManagedVllmWireArgs {
    inference: Arc<InferenceService>,
    model_id: u32,
    normalized_body: serde_json::Value,
    original_body_bytes: usize,
    stream: bool,
    request_id: String,
    session_id: Option<String>,
    session_id_for_log: String,
}

async fn relay_managed_vllm_wire(args: ManagedVllmWireArgs) -> warp::reply::Response {
    let ManagedVllmWireArgs {
        inference,
        model_id,
        mut normalized_body,
        original_body_bytes,
        stream,
        request_id,
        session_id,
        session_id_for_log,
    } = args;

    let metadata = OpenAiWireMetadata {
        request_id: Some(request_id),
        timeout_ms: None,
        priority: None,
    };
    let priority = inference.priority_for_openai_wire_parts(original_body_bytes, Some(&metadata));
    let generation_cap = match inference.openai_wire_generation_cap(priority) {
        Ok(cap) => cap,
        Err(error) => {
            return logged_wire_error(model_id, &session_id_for_log, error, "admission");
        }
    };
    if let Some(cap) = generation_cap {
        apply_wire_generation_cap(&mut normalized_body, cap);
    }

    let serialized = match serde_json::to_vec(&normalized_body) {
        Ok(body) => body,
        Err(error) => {
            return openai_error(
                warp::http::StatusCode::BAD_REQUEST,
                format!("Failed to normalize chat completion payload: {error}"),
                "invalid_request_error",
            );
        }
    };
    let format = if stream {
        OpenAiWireFormat::ServerSentEvents
    } else {
        OpenAiWireFormat::Json
    };
    let mut request =
        OpenAiWireRequest::new(OpenAiWireEndpoint::ChatCompletions, format, serialized)
            .with_metadata(metadata);
    request.session_id = session_id;
    request.cancellation = Some(kapsl_engine_api::CancellationToken::new());
    if let Err(error) =
        request.validate(kapsl_transport::protocol::MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES)
    {
        return openai_error(
            status_code_for_engine_error(&error),
            error.to_string(),
            "invalid_request_error",
        );
    }

    if stream {
        match inference
            .infer_openai_wire_stream(model_id, request, priority, false)
            .await
        {
            Ok(response) => wire_stream_response(response).unwrap_or_else(|error| {
                openai_error(warp::http::StatusCode::BAD_GATEWAY, error, "server_error")
            }),
            Err(error) => logged_wire_error(model_id, &session_id_for_log, error, "stream start"),
        }
    } else {
        match inference
            .infer_openai_wire(model_id, request, priority, false)
            .await
        {
            Ok(response) => wire_buffered_response(response).unwrap_or_else(|error| {
                openai_error(warp::http::StatusCode::BAD_GATEWAY, error, "server_error")
            }),
            Err(error) => logged_wire_error(model_id, &session_id_for_log, error, "request"),
        }
    }
}

fn apply_wire_generation_cap(body: &mut serde_json::Value, cap: u32) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    let mut found = false;
    for field in ["max_tokens", "max_completion_tokens"] {
        if let Some(value) = object.get(field) {
            found = true;
            let value = value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .map_or(cap, |value| value.min(cap));
            object.insert(field.to_string(), serde_json::Value::from(value));
        }
    }
    if !found {
        object.insert(
            "max_completion_tokens".to_string(),
            serde_json::Value::from(cap),
        );
    }
}

fn logged_wire_error(
    model_id: u32,
    session_id_for_log: &str,
    error: EngineError,
    stage: &str,
) -> warp::reply::Response {
    let status = status_code_for_engine_error(&error);
    if status == warp::http::StatusCode::INTERNAL_SERVER_ERROR {
        log::error!(
            "Managed vLLM wire {stage} failed: model_id={model_id} session_id={session_id_for_log} status={} error={error}",
            status.as_u16()
        );
    } else {
        log::warn!(
            "Managed vLLM wire {stage} rejected: model_id={model_id} session_id={session_id_for_log} status={} error={error}",
            status.as_u16()
        );
    }
    openai_error(status, error.to_string(), "server_error")
}

fn wire_response_builder(
    head: &OpenAiWireResponseHead,
) -> Result<warp::http::response::Builder, String> {
    head.validate().map_err(|error| error.to_string())?;
    let status = warp::http::StatusCode::from_u16(head.status)
        .map_err(|error| format!("Invalid managed vLLM response status: {error}"))?;
    let mut builder = warp::http::Response::builder().status(status);
    for OpenAiWireHeader { name, value } in &head.headers {
        let value = warp::http::HeaderValue::from_bytes(value).map_err(|error| {
            format!(
                "Invalid managed vLLM response header '{}': {error}",
                name.as_str()
            )
        })?;
        builder = builder.header(name.as_str(), value);
    }
    Ok(builder)
}

fn wire_buffered_response(response: OpenAiWireResponse) -> Result<warp::reply::Response, String> {
    wire_response_builder(&response.head)?
        .body(warp::hyper::Body::from(response.body))
        .map_err(|error| format!("Failed to build managed vLLM response: {error}"))
}

fn wire_stream_response(
    response: OpenAiWireStreamResponse,
) -> Result<warp::reply::Response, String> {
    let body = response.body.map(|item| {
        item.map(warp::hyper::body::Bytes::from).map_err(|error| {
            std::io::Error::other(format!("managed vLLM response stream failed: {error}"))
        })
    });
    wire_response_builder(&response.head)?
        .body(warp::hyper::Body::wrap_stream(body))
        .map_err(|error| format!("Failed to build managed vLLM stream response: {error}"))
}

struct StreamChatArgs {
    inference: Arc<InferenceService>,
    request: InferenceRequest,
    scheduler_priority: kapsl_scheduler::Priority,
    cancellation: kapsl_engine_api::CancellationToken,
    model_id: u32,
    model_name: String,
    completion_id: String,
    created: u64,
    stops: Vec<String>,
    prompt_tokens: u64,
    include_usage: bool,
    session_id_for_log: String,
}

/// Stream real token deltas as `chat.completion.chunk` events.
///
/// Each token the scheduler emits becomes one delta, so time-to-first-token is
/// what the client actually observes. Caller-supplied stop sequences are
/// enforced here: when one appears the text before it is emitted, the stream is
/// closed, and the underlying generation is cancelled rather than left running.
async fn stream_chat_completion(args: StreamChatArgs) -> warp::reply::Response {
    let StreamChatArgs {
        inference,
        request,
        scheduler_priority,
        cancellation,
        model_id,
        model_name,
        completion_id,
        created,
        stops,
        prompt_tokens,
        include_usage,
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
                "Chat completion stream start failed: model_id={} session_id={} status={} error={}",
                model_id,
                session_id_for_log,
                status.as_u16(),
                error
            );
            return openai_error(status, error.to_string(), "server_error");
        }
    };

    // The longest stop sequence bounds how much tail text has to be withheld:
    // a stop marker can straddle token boundaries, so text that could still
    // turn out to be a prefix of one is not emitted until it is ruled out.
    let max_stop_len = stops.iter().map(|stop| stop.len()).max().unwrap_or(0);

    struct StreamState {
        cancellation: kapsl_engine_api::CancellationToken,
        /// Text generated so far that has not yet been emitted as a delta.
        pending: String,
        emitted_tokens: u64,
        finished: bool,
        opened: bool,
    }

    let state = StreamState {
        cancellation,
        pending: String::new(),
        emitted_tokens: 0,
        finished: false,
        opened: false,
    };

    let id_for_chunks = completion_id.clone();
    let model_for_chunks = model_name.clone();
    let stops_for_chunks = stops.clone();

    let events = futures::stream::unfold((stream, state), move |(mut stream, mut state)| {
        let id = id_for_chunks.clone();
        let model = model_for_chunks.clone();
        let stops = stops_for_chunks.clone();
        async move {
            if state.finished {
                return None;
            }

            let mut out = String::new();
            if !state.opened {
                state.opened = true;
                out.push_str(&chat_completion_chunk(
                    &id,
                    created,
                    &model,
                    serde_json::json!({ "role": "assistant", "content": "" }),
                    None,
                ));
            }

            loop {
                match stream.next().await {
                    Some(Ok(packet)) => {
                        state
                            .pending
                            .push_str(&String::from_utf8_lossy(&packet.data));
                        state.emitted_tokens += 1;

                        if let Some(hit) = find_stop_sequence(&state.pending, &stops) {
                            let content = state.pending[..hit.cut].to_string();
                            if !content.is_empty() {
                                out.push_str(&chat_completion_chunk(
                                    &id,
                                    created,
                                    &model,
                                    serde_json::json!({ "content": content }),
                                    None,
                                ));
                            }
                            // Nothing after the stop marker is wanted, so
                            // stop paying to generate it.
                            state.cancellation.cancel();
                            out.push_str(&finish(&id, created, &model, "stop"));
                            out.push_str(&usage_tail(
                                include_usage,
                                &id,
                                created,
                                &model,
                                prompt_tokens,
                                state.emitted_tokens,
                            ));
                            out.push_str("data: [DONE]\n\n");
                            state.finished = true;
                            break;
                        }

                        // Hold back only what could still become a stop
                        // marker; emit the rest immediately.
                        let safe_len = state.pending.len().saturating_sub(max_stop_len);
                        let safe_len = floor_char_boundary(&state.pending, safe_len);
                        if safe_len > 0 {
                            let content: String = state.pending.drain(..safe_len).collect();
                            out.push_str(&chat_completion_chunk(
                                &id,
                                created,
                                &model,
                                serde_json::json!({ "content": content }),
                                None,
                            ));
                            break;
                        }
                        // Nothing safe to emit yet: keep reading rather
                        // than flushing an empty delta.
                        continue;
                    }
                    Some(Err(error)) => {
                        log::warn!(
                            "Chat completion stream error: model_id={} error={}",
                            model_id,
                            error
                        );
                        if !state.pending.is_empty() {
                            let content = std::mem::take(&mut state.pending);
                            out.push_str(&chat_completion_chunk(
                                &id,
                                created,
                                &model,
                                serde_json::json!({ "content": content }),
                                None,
                            ));
                        }
                        // The status line is already 200 by the time the
                        // stream is live, so the failure has to travel in
                        // band. Emitting `finish_reason: "stop"` here would
                        // tell the client the completion succeeded, which
                        // is how a backend error turns into a silently
                        // empty answer.
                        out.push_str(&stream_error_event(&error.to_string()));
                        out.push_str("data: [DONE]\n\n");
                        state.finished = true;
                        break;
                    }
                    None => {
                        if !state.pending.is_empty() {
                            let content = std::mem::take(&mut state.pending);
                            out.push_str(&chat_completion_chunk(
                                &id,
                                created,
                                &model,
                                serde_json::json!({ "content": content }),
                                None,
                            ));
                        }
                        out.push_str(&finish(&id, created, &model, "stop"));
                        out.push_str(&usage_tail(
                            include_usage,
                            &id,
                            created,
                            &model,
                            prompt_tokens,
                            state.emitted_tokens,
                        ));
                        out.push_str("data: [DONE]\n\n");
                        state.finished = true;
                        break;
                    }
                }
            }

            // `state._guard` rides along in the unfold state, so it stays
            // alive exactly as long as the response stream does: if the
            // client disconnects mid-generation, dropping the stream drops
            // the guard and cancels the work.
            Some((
                Ok::<_, std::convert::Infallible>(warp::hyper::body::Bytes::from(out)),
                (stream, state),
            ))
        }
    });

    let body = warp::hyper::Body::wrap_stream(events);
    warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        // Stop proxies from buffering the stream into a single response.
        .header("x-accel-buffering", "no")
        .body(body)
        .expect("SSE response should build")
}

/// An in-band SSE error, for failures that surface after the 200 status line
/// has already gone out. Compatible servers emit the same `{"error": {...}}`
/// event, and the OpenAI SDKs raise on it instead of returning empty content.
fn stream_error_event(message: &str) -> String {
    let event = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "param": serde_json::Value::Null,
            "code": serde_json::Value::Null,
        }
    });
    format!("data: {event}\n\n")
}

fn finish(id: &str, created: u64, model: &str, reason: &str) -> String {
    chat_completion_chunk(id, created, model, serde_json::json!({}), Some(reason))
}

fn usage_tail(
    include_usage: bool,
    id: &str,
    created: u64,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> String {
    if !include_usage {
        return String::new();
    }
    chat_completion_usage_chunk(
        id,
        created,
        model,
        &Usage {
            prompt_tokens,
            completion_tokens,
        },
    )
}

/// Largest index `<= index` that is a UTF-8 character boundary, so draining
/// the buffer never splits a multi-byte character across two deltas.
fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut index = index;
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_vllm_wire_is_default_with_explicit_translated_rollback() {
        assert_eq!(
            ManagedVllmOpenAiMode::from_value(None),
            ManagedVllmOpenAiMode::Wire
        );
        assert_eq!(
            ManagedVllmOpenAiMode::from_value(Some("wire")),
            ManagedVllmOpenAiMode::Wire
        );
        assert_eq!(
            ManagedVllmOpenAiMode::from_value(Some("async-translated")),
            ManagedVllmOpenAiMode::Translated
        );
        assert_eq!(
            ManagedVllmOpenAiMode::from_value(Some("legacy")),
            ManagedVllmOpenAiMode::Translated
        );
        assert_eq!(
            ManagedVllmOpenAiMode::from_value(Some("invalid")),
            ManagedVllmOpenAiMode::Wire
        );
    }

    #[tokio::test]
    async fn chat_body_filter_rejects_before_buffering_above_the_wire_limit() {
        let body = vec![b'x'; kapsl_transport::protocol::MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES + 1];
        let result = warp::test::request()
            .method("POST")
            .body(body)
            .filter(&bounded_chat_body())
            .await
            .expect("body filter should return a typed result");
        assert!(matches!(result, BoundedChatBody::TooLarge));
    }

    #[test]
    fn pressure_cap_bounds_both_token_spellings_and_inserts_a_default() {
        let mut both = serde_json::json!({
            "max_tokens": 100,
            "max_completion_tokens": 80,
        });
        apply_wire_generation_cap(&mut both, 12);
        assert_eq!(both["max_tokens"], 12);
        assert_eq!(both["max_completion_tokens"], 12);

        let mut absent = serde_json::json!({"model": "test"});
        apply_wire_generation_cap(&mut absent, 7);
        assert_eq!(absent["max_completion_tokens"], 7);
    }

    #[tokio::test]
    async fn buffered_wire_response_preserves_status_body_and_allowlisted_headers() {
        let response = OpenAiWireResponse {
            head: OpenAiWireResponseHead::new(
                429,
                vec![OpenAiWireHeader::new(
                    kapsl_engine_api::OpenAiWireHeaderName::RetryAfter,
                    b"3".to_vec(),
                )
                .unwrap()],
            )
            .unwrap(),
            body: br#"{"error":{"message":"busy"}}"#.to_vec(),
        };
        let response = wire_buffered_response(response).unwrap();
        assert_eq!(response.status(), warp::http::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "3");
        let body = warp::hyper::body::to_bytes(response.into_body()).await;
        assert_eq!(body.unwrap().as_ref(), br#"{"error":{"message":"busy"}}"#);
    }

    #[test]
    fn prompt_tensor_shape_validates_for_a_realistic_prompt() {
        // Regression: `[1]` was accepted by the type but rejected by
        // `validate()` for any prompt longer than a single byte, so every
        // real chat request failed with a length mismatch.
        let prompt = "Say hello in five words.";
        let data = prompt.as_bytes().to_vec();
        let packet = BinaryTensorPacket::new(
            vec![1, data.len() as i64],
            kapsl_engine_api::TensorDtype::Utf8,
            data,
        )
        .expect("utf8 prompt tensor should validate");
        assert_eq!(
            String::from_utf8_lossy(&packet.data),
            prompt,
            "round-tripping the prompt should preserve it exactly"
        );
    }

    #[test]
    fn char_boundary_never_splits_a_multibyte_character() {
        let text = "aé漢";
        assert_eq!(floor_char_boundary(text, 0), 0);
        assert_eq!(floor_char_boundary(text, 1), 1);
        // Byte 2 lands inside 'é' (bytes 1..3).
        assert_eq!(floor_char_boundary(text, 2), 1);
        assert_eq!(floor_char_boundary(text, 3), 3);
        // Byte 4 lands inside '漢' (bytes 3..6).
        assert_eq!(floor_char_boundary(text, 4), 3);
        assert_eq!(floor_char_boundary(text, 99), text.len());
    }
}
