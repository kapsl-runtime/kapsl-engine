use super::*;

#[derive(Debug, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct ListedModel {
    pub(crate) id: u32,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) version: String,
    #[serde(default)]
    pub(crate) format: Option<String>,
    #[serde(default)]
    pub(crate) framework: String,
    #[serde(default)]
    pub(crate) device: String,
    #[serde(default)]
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) healthy: Option<bool>,
}

fn runtime_base_url(
    http_url: Option<&str>,
    http_host: &str,
    http_port: u16,
) -> Result<String, DynError> {
    if let Some(url) = http_url {
        let url = url.trim().trim_end_matches('/');
        if url.is_empty() {
            return Err(dyn_error_from_message("--http-url cannot be empty."));
        }
        return Ok(url.to_string());
    }

    let host = http_host.trim();
    if host.is_empty() {
        return Err(dyn_error_from_message("--http-host cannot be empty."));
    }
    Ok(format!("http://{}:{}", host, http_port))
}

fn runtime_http_agent(timeout_ms: u64) -> ureq::Agent {
    let timeout = std::time::Duration::from_millis(timeout_ms.max(1));
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .build()
        .into()
}

fn remove_model_error_detail(model_id: u32, error: ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(401) | ureq::Error::StatusCode(403) => {
            "the running engine rejected the request; pass an admin token with --auth-token"
                .to_string()
        }
        ureq::Error::StatusCode(404) => format!("model {} was not found", model_id),
        ureq::Error::StatusCode(status) => {
            format!("the running engine returned HTTP {}", status)
        }
        other => other.to_string(),
    }
}

fn parse_listed_models(body: &str) -> Result<Vec<ListedModel>, DynError> {
    serde_json::from_str(body).map_err(|error| {
        dyn_error_from_message(format!(
            "The running engine returned an invalid model list: {}",
            error
        ))
    })
}

fn table_cell(value: &str) -> &str {
    if value.trim().is_empty() {
        "-"
    } else {
        value
    }
}

fn render_model_table(models: &[ListedModel]) -> String {
    if models.is_empty() {
        return "No models are loaded.\n".to_string();
    }

    let headers = [
        "ID", "NAME", "VERSION", "FORMAT", "DEVICE", "STATUS", "HEALTH",
    ];
    let mut rows: Vec<[String; 7]> = models
        .iter()
        .map(|model| {
            let format = model.format.as_deref().unwrap_or(&model.framework);
            [
                model.id.to_string(),
                table_cell(&model.name).to_string(),
                table_cell(&model.version).to_string(),
                table_cell(format).to_string(),
                table_cell(&model.device).to_string(),
                table_cell(&model.status).to_string(),
                match model.healthy {
                    Some(true) => "healthy",
                    Some(false) => "unhealthy",
                    None => "-",
                }
                .to_string(),
            ]
        })
        .collect();
    rows.sort_by_key(|row| row[0].parse::<u32>().unwrap_or(u32::MAX));

    let mut widths = headers.map(str::len);
    for row in &rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let render_row = |row: &[String; 7]| {
        row.iter()
            .enumerate()
            .map(|(index, cell)| {
                if index + 1 == row.len() {
                    cell.clone()
                } else {
                    format!("{:<width$}", cell, width = widths[index])
                }
            })
            .collect::<Vec<_>>()
            .join("  ")
    };

    let header = headers.map(str::to_string);
    let mut output = String::new();
    output.push_str(&render_row(&header));
    output.push('\n');
    for row in &rows {
        output.push_str(&render_row(row));
        output.push('\n');
    }
    output
}

pub(crate) fn execute_list_command(args: ListCommandArgs) -> Result<(), DynError> {
    let base_url = runtime_base_url(args.http_url.as_deref(), &args.http_host, args.http_port)?;
    let models_url = format!("{}/api/models", base_url);
    let agent = runtime_http_agent(args.timeout_ms);
    let mut request = agent.get(&models_url).header("Accept", "application/json");
    if let Some(token) = &args.auth_token {
        request = request.header("Authorization", &format!("Bearer {}", token));
    }

    let mut response = request.call().map_err(|error| {
        let detail = match error {
            ureq::Error::StatusCode(status) => {
                format!("the running engine returned HTTP {}", status)
            }
            other => other.to_string(),
        };
        dyn_error_from_message(format!(
            "Failed to list models from {}: {}",
            models_url, detail
        ))
    })?;
    let body = response.body_mut().read_to_string().map_err(|error| {
        dyn_error_from_message(format!(
            "Failed to read the model list from {}: {}",
            models_url, error
        ))
    })?;

    let models = parse_listed_models(&body)?;
    if args.json {
        let value: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
            dyn_error_from_message(format!(
                "The running engine returned invalid JSON: {}",
                error
            ))
        })?;
        println!(
            "{}",
            serde_json::to_string_pretty(&value).map_err(|error| {
                dyn_error_from_message(format!("Failed to format the model list: {}", error))
            })?
        );
    } else {
        print!("{}", render_model_table(&models));
    }
    Ok(())
}

pub(crate) fn execute_remove_model_command(args: RemoveModelCommandArgs) -> Result<(), DynError> {
    let base_url = runtime_base_url(args.http_url.as_deref(), &args.http_host, args.http_port)?;
    let remove_url = format!("{}/api/models/{}/remove", base_url, args.model_id);
    let agent = runtime_http_agent(args.timeout_ms);
    let mut request = agent.post(&remove_url).header("Accept", "application/json");
    if let Some(token) = &args.auth_token {
        request = request.header("Authorization", &format!("Bearer {}", token));
    }

    request.send_empty().map_err(|error| {
        let detail = remove_model_error_detail(args.model_id, error);
        dyn_error_from_message(format!(
            "Failed to remove model {} from {}: {}",
            args.model_id, base_url, detail
        ))
    })?;

    let a = Ansi::new();
    eprintln!("  {}  Model {} removed", a.green("✓"), args.model_id);
    Ok(())
}

pub(crate) fn execute_add_model_command(args: AddModelCommandArgs) -> Result<(), DynError> {
    if args.model.is_empty() {
        return Err(dyn_error_from_message(
            "At least one --model PATH is required.",
        ));
    }

    let base_url = runtime_base_url(args.http_url.as_deref(), &args.http_host, args.http_port)?;

    let agent = runtime_http_agent(args.timeout_ms);

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

pub(crate) fn validate_native_tcp_exposure(
    bind_ip: IpAddr,
    auth_token: Option<&str>,
) -> Result<(), String> {
    if bind_ip.is_loopback() || auth_token.is_some_and(|token| !token.trim().is_empty()) {
        return Ok(());
    }

    Err(format!(
        "Refusing unauthenticated TCP inference on non-loopback address {bind_ip}. Set {TCP_AUTH_TOKEN_ENV} to a dedicated native-transport token, or bind --bind to a loopback address. Raw TCP is plaintext; use a trusted network or TLS tunnel for cross-host serving."
    ))
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

pub(crate) fn scheduler_priority_for_openai_wire_parts(
    body_bytes: usize,
    metadata: Option<&kapsl_engine_api::OpenAiWireMetadata>,
) -> kapsl_scheduler::Priority {
    let scheduler_metadata = SchedulerRequestMetadata {
        priority: metadata.and_then(|metadata| metadata.priority).unwrap_or(1),
        sla_deadline: metadata.and_then(|metadata| metadata.timeout_ms),
        batch_size: 1,
        input_size_bytes: Some(body_bytes),
        estimated_flops: None,
    };

    determine_priority(&scheduler_metadata)
}

#[cfg(test)]
mod command_tests {
    use super::*;
    use clap::Parser;

    fn listed_model(id: u32, name: &str) -> ListedModel {
        ListedModel {
            id,
            name: name.to_string(),
            version: "1.0.0".to_string(),
            format: Some("gguf".to_string()),
            framework: "llm".to_string(),
            device: "CUDA".to_string(),
            status: "active".to_string(),
            healthy: Some(true),
        }
    }

    #[test]
    fn list_command_parses_runtime_connection_options() {
        let cli = Cli::try_parse_from([
            "kapsl",
            "list",
            "--http-url",
            "http://engine.example:9195/",
            "--auth-token",
            "secret",
            "--timeout-ms",
            "2500",
            "--json",
        ])
        .expect("parse list command");

        assert!(matches!(
            cli.command,
            Some(KapslCommand::List(ListCommandArgs {
                http_url: Some(url),
                auth_token: Some(token),
                timeout_ms: 2500,
                json: true,
                ..
            })) if url == "http://engine.example:9195/" && token == "secret"
        ));
    }

    #[test]
    fn list_command_uses_local_engine_defaults() {
        let cli = Cli::try_parse_from(["kapsl", "list"]).expect("parse list command");

        assert!(matches!(
            cli.command,
            Some(KapslCommand::List(ListCommandArgs {
                http_host,
                http_port: 9095,
                http_url: None,
                auth_token: None,
                timeout_ms: 30000,
                json: false,
            })) if http_host == "127.0.0.1"
        ));
    }

    #[test]
    fn remove_model_command_parses_id_and_runtime_options() {
        let cli = Cli::try_parse_from([
            "kapsl",
            "remove-model",
            "42",
            "--http-host",
            "engine.local",
            "--http-port",
            "9195",
            "--auth-token",
            "admin-secret",
            "--timeout-ms",
            "2500",
        ])
        .expect("parse remove-model command");

        assert!(matches!(
            cli.command,
            Some(KapslCommand::RemoveModel(RemoveModelCommandArgs {
                model_id: 42,
                http_host,
                http_port: 9195,
                http_url: None,
                auth_token: Some(token),
                timeout_ms: 2500,
            })) if http_host == "engine.local" && token == "admin-secret"
        ));
    }

    #[test]
    fn remove_model_command_requires_an_id() {
        let error = Cli::try_parse_from(["kapsl", "remove-model"])
            .expect_err("model id should be required");

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn remove_model_errors_explain_not_found_and_admin_auth() {
        assert_eq!(
            remove_model_error_detail(42, ureq::Error::StatusCode(404)),
            "model 42 was not found"
        );
        assert!(remove_model_error_detail(42, ureq::Error::StatusCode(403)).contains("admin token"));
    }

    #[test]
    fn runtime_base_url_prefers_and_normalizes_full_url() {
        assert_eq!(
            runtime_base_url(Some("  http://engine.example:9195///  "), "ignored", 1)
                .expect("valid URL"),
            "http://engine.example:9195"
        );
        assert_eq!(
            runtime_base_url(None, "engine.local", 9195).expect("valid host"),
            "http://engine.local:9195"
        );
    }

    #[test]
    fn parses_model_list_and_ignores_additional_metrics() {
        let models = parse_listed_models(
            r#"[{"id":7,"name":"qwen","version":"2","format":"gguf","framework":"llm","device":"CUDA","status":"active","healthy":true,"active_inferences":3}]"#,
        )
        .expect("parse model list");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, 7);
        assert_eq!(models[0].name, "qwen");
        assert_eq!(models[0].healthy, Some(true));
    }

    #[test]
    fn model_table_is_sorted_and_uses_framework_as_legacy_format() {
        let mut second = listed_model(9, "vision");
        second.format = None;
        second.framework = "onnx".to_string();
        second.healthy = Some(false);
        let first = listed_model(2, "qwen");

        let output = render_model_table(&[second, first]);
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("ID") && lines[0].contains("HEALTH"));
        assert!(lines[1].starts_with('2') && lines[1].contains("gguf"));
        assert!(lines[2].starts_with('9') && lines[2].contains("onnx"));
        assert!(lines[2].ends_with("unhealthy"));
    }

    #[test]
    fn empty_model_table_has_a_clear_message() {
        assert_eq!(render_model_table(&[]), "No models are loaded.\n");
    }
}
