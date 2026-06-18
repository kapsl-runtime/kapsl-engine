use super::*;

pub(crate) fn execute_add_model_command(args: AddModelCommandArgs) -> Result<(), DynError> {
    if args.model.is_empty() {
        return Err(dyn_error_from_message(
            "At least one --model PATH is required.",
        ));
    }

    let base_url = match &args.http_url {
        Some(url) => url.trim_end_matches('/').to_string(),
        None => format!("http://{}:{}", args.http_host, args.http_port),
    };

    let timeout = std::time::Duration::from_millis(args.timeout_ms.max(1));
    let agent_config = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .build();
    let agent: ureq::Agent = agent_config.into();

    let start_url = format!("{}/api/models/start", base_url);

    let a = Ansi::new();
    let mut any_error = false;
    for model_path in &args.model {
        let display = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_else(|| model_path.to_str().unwrap_or("?"));

        let absolute_path = match model_path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "  {}  {}  {}",
                    a.red("✗"),
                    display,
                    a.dim(&format!("({})", e))
                );
                any_error = true;
                continue;
            }
        };

        let payload = serde_json::json!({
            "model_path": absolute_path.to_string_lossy(),
            "topology": args.topology,
            "tp_degree": args.tp_degree,
        });

        let payload_str = serde_json::to_string(&payload)
            .map_err(|e| dyn_error_from_message(format!("Failed to serialize request: {}", e)))?;

        let mut request = agent
            .post(&start_url)
            .header("Content-Type", "application/json");
        if let Some(token) = &args.auth_token {
            request = request.header("Authorization", &format!("Bearer {}", token));
        }

        match request.send(payload_str) {
            Ok(mut response) => {
                let body = response
                    .body_mut()
                    .read_to_string()
                    .unwrap_or_else(|_| String::new());
                // Extract model_id from JSON if present for a nicer summary line.
                let model_id = serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|json| json.get("model_id").and_then(|v| v.as_u64()))
                    .map(|id| format!(" (id={})", id))
                    .unwrap_or_default();
                eprintln!("  {}  {}{}", a.green("✓"), display, a.dim(&model_id));
            }
            Err(e) => {
                eprintln!(
                    "  {}  {}  {}",
                    a.red("✗"),
                    display,
                    a.dim(&format!("({})", format_remote_http_error(e)))
                );
                any_error = true;
            }
        }
    }

    if any_error {
        Err(dyn_error_from_message(
            "One or more models could not be added.",
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn env_flag(name: &str) -> bool {
    optional_env_var(name)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub(crate) fn provider_policy() -> String {
    optional_env_var(PROVIDER_POLICY_ENV)
        .unwrap_or_else(|| "fastest".to_string())
        .trim()
        .to_ascii_lowercase()
}

pub(crate) fn parse_bind_ip(raw: &str, fallback: IpAddr, field_name: &str) -> IpAddr {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return fallback;
    }
    match trimmed.parse::<IpAddr>() {
        Ok(addr) => addr,
        Err(error) => {
            log::warn!(
                "Invalid {} value `{}`: {}. Falling back to {}",
                field_name,
                trimmed,
                error,
                fallback
            );
            fallback
        }
    }
}

pub(crate) fn preflight_http_bind(http_bind: IpAddr, port: u16) -> Result<(), DynError> {
    use std::net::{SocketAddr, TcpListener};

    let addr = SocketAddr::new(http_bind, port);
    match TcpListener::bind(addr) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(error) => {
            let mut message = format!("Failed to bind HTTP API on {}: {}", addr, error);
            if matches!(error.kind(), std::io::ErrorKind::AddrInUse) {
                message.push_str(
                    ". Another process is already using this port. Stop the other runtime or pick a different port with --metrics-port.",
                );
            }
            Err(message.into())
        }
    }
}

#[cfg(unix)]
pub(crate) fn preflight_ipc_socket(socket_path: &str) -> Result<(), DynError> {
    use std::os::unix::net::UnixStream;
    use std::path::Path;

    if !Path::new(socket_path).exists() {
        return Ok(());
    }

    if UnixStream::connect(socket_path).is_ok() {
        return Err(format!(
            "IPC socket path {} is already in use. Stop the other runtime or choose a different path with --socket.",
            socket_path
        )
        .into());
    }

    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn preflight_ipc_socket(_socket_path: &str) -> Result<(), DynError> {
    Ok(())
}

pub(crate) fn redact_identifier_for_logs(raw: &str, expose_sensitive: bool) -> String {
    if expose_sensitive || raw == "-" || raw.is_empty() {
        return raw.to_string();
    }
    let prefix: String = raw.chars().take(4).collect();
    format!("{}...[redacted]", prefix)
}

pub(crate) fn reply_into_response<R: Reply>(reply: R) -> warp::reply::Response {
    reply.into_response()
}

pub(crate) fn status_code_for_engine_error(error: &EngineError) -> warp::http::StatusCode {
    use warp::http::StatusCode;

    match error {
        EngineError::InvalidInput { .. } => StatusCode::BAD_REQUEST,
        EngineError::ModelNotLoaded => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::Overloaded { .. } | EngineError::ResourceExhausted { .. } => {
            StatusCode::TOO_MANY_REQUESTS
        }
        EngineError::TimeoutError { .. } => StatusCode::GATEWAY_TIMEOUT,
        EngineError::Cancelled { .. } => StatusCode::REQUEST_TIMEOUT,
        EngineError::Backend { .. }
        | EngineError::ModelLoadError { .. }
        | EngineError::InferenceError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

pub(crate) fn inferred_batch_size(shape: &[i64]) -> usize {
    shape
        .first()
        .copied()
        .filter(|dim| *dim > 0)
        .map(|dim| dim as usize)
        .unwrap_or(1)
}

pub(crate) fn scheduler_priority_for_request(
    request: &InferenceRequest,
) -> kapsl_scheduler::Priority {
    let scheduler_metadata = SchedulerRequestMetadata {
        priority: request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.priority)
            .unwrap_or(1),
        sla_deadline: request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.timeout_ms),
        batch_size: inferred_batch_size(&request.input.shape),
        input_size_bytes: Some(request.input.data.len()),
        estimated_flops: None,
    };

    determine_priority(&scheduler_metadata)
}
