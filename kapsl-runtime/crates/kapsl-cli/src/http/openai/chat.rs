//! `POST /v1/chat/completions`.

use super::types::*;
use super::*;
use futures::StreamExt;

pub(crate) struct ChatCompletionsConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) log_sensitive_ids: bool,
}

pub(crate) fn build_chat_completions_route(
    config: ChatCompletionsConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let ChatCompletionsConfig {
        models,
        inference,
        log_sensitive_ids,
    } = config;

    warp::path!("v1" / "chat" / "completions")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(warp::header::optional::<String>("x-kapsl-session"))
        .and_then(
            move |body: warp::hyper::body::Bytes, session_header: Option<String>| {
                let models = models.clone();
                let inference = inference.clone();
                async move {
                    Ok::<_, warp::Rejection>(
                        handle_chat_completion(
                            body,
                            session_header,
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

#[allow(clippy::too_many_arguments)]
async fn handle_chat_completion(
    body: warp::hyper::body::Bytes,
    session_header: Option<String>,
    models: &Arc<ModelManager>,
    inference: &Arc<InferenceService>,
    log_sensitive_ids: bool,
) -> warp::reply::Response {
    use warp::http::StatusCode;

    let chat: ChatCompletionRequest = match serde_json::from_slice(body.as_ref()) {
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

    let prompt = match build_prompt(&chat.messages, &resolved.name) {
        Ok(prompt) => prompt,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };

    // A UTF-8 prompt tensor is shaped `[1, byte_len]`, matching the RAG and
    // native inference path. `[1]` only validates for a one-byte prompt.
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

    let request_id = completion_id();
    let mut metadata = chat.to_request_metadata();
    metadata.request_id = Some(request_id.clone());

    let mut request = InferenceRequest::new(input).with_metadata(metadata);
    // KV affinity: an explicit session header wins, else the OpenAI end-user id.
    if let Some(session) = session_header
        .map(|session| session.trim().to_string())
        .filter(|session| !session.is_empty())
        .or_else(|| {
            chat.user
                .as_ref()
                .map(|user| user.trim().to_string())
                .filter(|user| !user.is_empty())
        })
    {
        request.session_id = Some(session);
    }

    let scheduler_priority = inference.priority_for_request(&request);
    let session_id_for_log = redact_identifier_for_logs(
        request.session_id.as_deref().unwrap_or("-"),
        log_sensitive_ids,
    );

    let cancellation = request
        .cancellation
        .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
        .clone();

    let stops = chat.stop_sequences();
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
