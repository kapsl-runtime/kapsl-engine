//! Runtime command clients and shared serving-boundary helpers.

use super::*;
use crate::features::http_client::format_remote_http_error;
use std::collections::HashMap;

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

/// Memory-authority view from `GET /api/system/stats`, reduced to what the
/// model commands print. Every field is optional so older runtimes still parse.
#[derive(Debug, Default, serde::Deserialize)]
struct StatsBody {
    #[serde(default)]
    memory_authority: Option<AuthorityBody>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct AuthorityBody {
    #[serde(default)]
    models: Vec<AuthorityModel>,
    #[serde(default)]
    domains: Vec<AuthorityDomain>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct AuthorityModel {
    model_id: u32,
    #[serde(default)]
    reserved_bytes: usize,
    #[serde(default)]
    committed_bytes: usize,
    #[serde(default)]
    observed_bytes: usize,
    #[serde(default)]
    used_bytes: usize,
}

#[derive(Debug, Default, serde::Deserialize)]
struct AuthorityDomain {
    domain: String,
    #[serde(default)]
    available_bytes: usize,
}

#[derive(Debug, Default)]
struct MemoryView {
    /// model_id -> bytes held, using the runtime's own rule: the largest of
    /// reserved, committed, observed and used.
    model_bytes: HashMap<u32, usize>,
    /// domain name ("cuda:0", "host", ...) -> bytes still available
    available: HashMap<String, usize>,
}

fn parse_memory_view(body: &str) -> Option<MemoryView> {
    let stats: StatsBody = serde_json::from_str(body).ok()?;
    let authority = stats.memory_authority?;
    let mut view = MemoryView::default();
    for model in authority.models {
        let bytes = model
            .reserved_bytes
            .max(model.committed_bytes)
            .max(model.observed_bytes)
            .max(model.used_bytes);
        let entry = view.model_bytes.entry(model.model_id).or_insert(0);
        *entry = entry.saturating_add(bytes);
    }
    for domain in authority.domains {
        view.available.insert(domain.domain, domain.available_bytes);
    }
    Some(view)
}

/// Best effort: a missing or old stats endpoint just means no memory figures.
fn fetch_memory_view(
    agent: &ureq::Agent,
    base_url: &str,
    auth_token: Option<&str>,
) -> Option<MemoryView> {
    let mut request = agent
        .get(&format!("{}/api/system/stats", base_url))
        .header("Accept", "application/json");
    if let Some(token) = auth_token {
        request = request.header("Authorization", &format!("Bearer {}", token));
    }
    let mut response = request.call().ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body = response.body_mut().read_to_string().ok()?;
    parse_memory_view(&body)
}

/// Decimal units: 980_000_000 -> "0.98 GB", below 0.1 GB -> "42 MB".
fn format_memory(bytes: usize) -> String {
    if bytes >= 100_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", (bytes as f64 / 1e6).round() as u64)
    }
}

fn is_host_memory_domain(domain: &str) -> bool {
    domain.starts_with("host")
}

fn render_model_table(models: &[ListedModel], memory: &HashMap<u32, usize>) -> String {
    if models.is_empty() {
        return "No models are loaded.\n".to_string();
    }

    let headers = [
        "ID", "NAME", "VERSION", "FORMAT", "DEVICE", "MEMORY", "STATUS", "HEALTH",
    ];
    let mut rows: Vec<[String; 8]> = models
        .iter()
        .map(|model| {
            let format = model.format.as_deref().unwrap_or(&model.framework);
            [
                model.id.to_string(),
                table_cell(&model.name).to_string(),
                table_cell(&model.version).to_string(),
                table_cell(format).to_string(),
                table_cell(&model.device).to_string(),
                memory
                    .get(&model.id)
                    .filter(|bytes| **bytes > 0)
                    .map(|bytes| format_memory(*bytes))
                    .unwrap_or_else(|| "-".to_string()),
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

    let render_row = |row: &[String; 8]| {
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
        let memory = fetch_memory_view(&agent, &base_url, args.auth_token.as_deref())
            .map(|view| view.model_bytes)
            .unwrap_or_default();
        print!("{}", render_model_table(&models, &memory));
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

/// How one `add-model` request ended, as printed on its result line.
#[derive(Debug, PartialEq)]
enum AddModelResult {
    /// Admitted and active. `bytes` is its lease when the stats endpoint reports it.
    Granted { model_id: u32, bytes: Option<usize> },
    /// Queued only (`--no-wait`).
    Queued { model_id: u32 },
    /// Rejected by the memory authority. `free` is only set when the request
    /// really exceeds the domain's free memory; when a narrower budget (for
    /// example a per-class cap) blocked it, `free` is None so the line never
    /// reads "needs 0.27 GB, 0.48 GB free".
    Denied {
        needed: Option<usize>,
        free: Option<usize>,
        domain: Option<String>,
    },
    /// Any other failure, with a short reason.
    Failed { reason: String },
}

#[derive(Debug, Default, serde::Deserialize)]
struct StartModelReply {
    #[serde(default)]
    model_id: Option<u32>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    requested: Vec<RequestedMemory>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct RequestedMemory {
    domain: String,
    #[serde(default)]
    bytes: usize,
}

/// Picks the accelerator domain that actually blocked the load: the first
/// non-host domain where the request exceeds what is free, else the first
/// non-host domain, else whatever was requested.
fn denial_from_reply(reply: &StartModelReply, memory: Option<&MemoryView>) -> AddModelResult {
    let free_for = |domain: &str| memory.and_then(|view| view.available.get(domain).copied());
    let mut candidates: Vec<&RequestedMemory> = reply
        .requested
        .iter()
        .filter(|r| !is_host_memory_domain(&r.domain))
        .collect();
    if candidates.is_empty() {
        candidates = reply.requested.iter().collect();
    }
    let chosen = candidates
        .iter()
        .find(|r| free_for(&r.domain).is_some_and(|free| r.bytes > free))
        .or_else(|| candidates.first())
        .copied();
    match chosen {
        Some(r) => AddModelResult::Denied {
            needed: Some(r.bytes).filter(|bytes| *bytes > 0),
            free: free_for(&r.domain).filter(|free| r.bytes > *free),
            domain: Some(r.domain.clone()),
        },
        None => AddModelResult::Denied {
            needed: None,
            free: None,
            domain: None,
        },
    }
}

/// Maps a start reply (any status) onto a result. `memory` is a stats snapshot
/// taken after the reply, used for the granted lease or the free figure.
fn add_model_result(status: u16, body: &str, memory: Option<&MemoryView>) -> AddModelResult {
    let reply: StartModelReply = serde_json::from_str(body).unwrap_or_default();
    match (status, reply.status.as_deref()) {
        (200, Some("active")) => {
            let model_id = reply.model_id.unwrap_or_default();
            AddModelResult::Granted {
                model_id,
                bytes: memory.and_then(|view| view.model_bytes.get(&model_id).copied()),
            }
        }
        (409, _) if reply.reason.as_deref() == Some("memory_admission") => {
            denial_from_reply(&reply, memory)
        }
        (200..=299, _) => match reply.model_id {
            Some(model_id) => AddModelResult::Queued { model_id },
            None => AddModelResult::Failed {
                reason: "runtime returned no model id".to_string(),
            },
        },
        _ => AddModelResult::Failed {
            reason: reply
                .error
                .unwrap_or_else(|| format!("the running engine returned HTTP {}", status)),
        },
    }
}

/// Older runtimes ignore `?wait=true` and answer 202. Poll the model until it
/// settles so the result line still tells the truth (without denial details).
fn poll_until_settled(
    agent: &ureq::Agent,
    base_url: &str,
    auth_token: Option<&str>,
    model_id: u32,
    timeout: std::time::Duration,
) -> AddModelResult {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let mut request = agent
            .get(&format!("{}/api/models/{}", base_url, model_id))
            .header("Accept", "application/json");
        if let Some(token) = auth_token {
            request = request.header("Authorization", &format!("Bearer {}", token));
        }
        if let Ok(mut response) = request.call() {
            let code = response.status().as_u16();
            let body = response.body_mut().read_to_string().unwrap_or_default();
            let value: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
            if code == 404 || value.get("error").is_some() {
                return AddModelResult::Failed {
                    reason: "not loaded; this runtime predates admission results, see its log"
                        .to_string(),
                };
            }
            match value.get("status").and_then(|s| s.as_str()) {
                Some("starting") | Some("loading") | None => {}
                Some("active") => {
                    let bytes = fetch_memory_view(agent, base_url, auth_token)
                        .and_then(|view| view.model_bytes.get(&model_id).copied());
                    return AddModelResult::Granted { model_id, bytes };
                }
                Some(other) => {
                    return AddModelResult::Failed {
                        reason: format!("status {}", other),
                    }
                }
            }
        }
        if std::time::Instant::now() >= deadline {
            return AddModelResult::Failed {
                reason: "timed out waiting for the load".to_string(),
            };
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn add_model_result_line(display: &str, result: &AddModelResult, a: &Ansi) -> String {
    match result {
        AddModelResult::Granted { model_id, bytes } => {
            let lease = bytes
                .filter(|b| *b > 0)
                .map(|b| format!(" · {}", format_memory(b)))
                .unwrap_or_default();
            format!(
                "  {}  {} {}  {}{}",
                a.green("✓"),
                display,
                a.dim(&format!("(id={})", model_id)),
                a.green("granted"),
                a.dim(&lease)
            )
        }
        AddModelResult::Queued { model_id } => format!(
            "  {}  {} {}  {}",
            a.green("✓"),
            display,
            a.dim(&format!("(id={})", model_id)),
            a.dim("load started")
        ),
        AddModelResult::Denied {
            needed,
            free,
            domain,
        } => {
            let place = domain.as_deref().filter(|d| !d.is_empty());
            let detail = match (needed, free) {
                (Some(n), Some(f)) => {
                    format!(" · needs {}, {} free", format_memory(*n), format_memory(*f))
                }
                (Some(n), None) => match place {
                    Some(d) => format!(" · needs {}, over the {} budget", format_memory(*n), d),
                    None => format!(" · needs {}", format_memory(*n)),
                },
                _ => String::new(),
            };
            // "free" lines name the GPU only when it isn't the default one.
            let on = match (free, place) {
                (Some(_), Some(d)) if d != "cuda:0" => format!(" on {}", d),
                _ => String::new(),
            };
            format!(
                "  {}  {}  {}{}",
                a.red("✗"),
                display,
                a.red("denied"),
                a.dim(&format!("{}{}", detail, on))
            )
        }
        AddModelResult::Failed { reason } => {
            format!(
                "  {}  {}  {}",
                a.red("✗"),
                display,
                a.dim(&format!("({})", reason))
            )
        }
    }
}

pub(crate) fn execute_add_model_command(args: AddModelCommandArgs) -> Result<(), DynError> {
    if args.model.is_empty() {
        return Err(dyn_error_from_message(
            "At least one --model PATH is required.",
        ));
    }

    let base_url = runtime_base_url(args.http_url.as_deref(), &args.http_host, args.http_port)?;
    // Status codes are results here (409 = denied), so read every body.
    let timeout = std::time::Duration::from_millis(args.timeout_ms.max(1));
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into();
    let start_url = if args.no_wait {
        format!("{}/api/models/start", base_url)
    } else {
        format!("{}/api/models/start?wait=true", base_url)
    };
    let token = args.auth_token.as_deref();

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
                let result = AddModelResult::Failed {
                    reason: e.to_string(),
                };
                eprintln!("{}", add_model_result_line(display, &result, &a));
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
        if let Some(token) = token {
            request = request.header("Authorization", &format!("Bearer {}", token));
        }

        let result = match request.send(payload_str) {
            Ok(mut response) => {
                let status = response.status().as_u16();
                let body = response.body_mut().read_to_string().unwrap_or_default();
                let memory = fetch_memory_view(&agent, &base_url, token);
                match add_model_result(status, &body, memory.as_ref()) {
                    AddModelResult::Queued { model_id } if !args.no_wait => {
                        poll_until_settled(&agent, &base_url, token, model_id, timeout)
                    }
                    other => other,
                }
            }
            Err(e) => AddModelResult::Failed {
                reason: format_remote_http_error(e),
            },
        };
        if matches!(
            result,
            AddModelResult::Denied { .. } | AddModelResult::Failed { .. }
        ) {
            any_error = true;
        }
        eprintln!("{}", add_model_result_line(display, &result, &a));
    }

    if any_error {
        // The result lines already explain what happened; exit non-zero
        // without the Debug-formatted error main() would print.
        std::process::exit(1);
    }
    Ok(())
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

        let output = render_model_table(&[second, first], &HashMap::new());
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("ID") && lines[0].contains("HEALTH"));
        assert!(lines[1].starts_with('2') && lines[1].contains("gguf"));
        assert!(lines[2].starts_with('9') && lines[2].contains("onnx"));
        assert!(lines[2].ends_with("unhealthy"));
    }

    #[test]
    fn empty_model_table_has_a_clear_message() {
        assert_eq!(
            render_model_table(&[], &HashMap::new()),
            "No models are loaded.\n"
        );
    }

    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    const STATS: &str = r#"{"memory_authority":{"models":[
        {"model_id":1,"name":"chat","reserved_bytes":980000000,"committed_bytes":970000000,"observed_bytes":0,"used_bytes":975000000},
        {"model_id":2,"name":"summarizer","reserved_bytes":0,"committed_bytes":960000000,"observed_bytes":966000000,"used_bytes":0}],
        "domains":[{"domain":"host","available_bytes":50000000000},{"domain":"cuda:0","available_bytes":1090000000}]}}"#;

    #[test]
    fn memory_view_takes_the_largest_accounting_state() {
        let view = parse_memory_view(STATS).expect("parse stats");
        assert_eq!(view.model_bytes[&1], 980_000_000);
        assert_eq!(view.model_bytes[&2], 966_000_000);
        assert_eq!(view.available["cuda:0"], 1_090_000_000);
        assert!(parse_memory_view(r#"{"pressure_state":"normal"}"#).is_none());
    }

    #[test]
    fn memory_is_formatted_in_decimal_units() {
        assert_eq!(format_memory(980_000_000), "0.98 GB");
        assert_eq!(format_memory(2_450_000_000), "2.45 GB");
        assert_eq!(format_memory(42_000_000), "42 MB");
    }

    #[test]
    fn model_table_shows_memory_per_model() {
        let view = parse_memory_view(STATS).expect("parse stats");
        let output = render_model_table(
            &[listed_model(1, "chat"), listed_model(3, "no-lease")],
            &view.model_bytes,
        );
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines[0].contains("MEMORY"));
        assert!(lines[1].contains("0.98 GB"));
        assert!(lines[2].contains(" - "));
    }

    #[test]
    fn add_model_waits_by_default_with_room_for_large_loads() {
        let cli = Cli::try_parse_from(["kapsl", "add-model", "--model", "a.aimod"])
            .expect("parse add-model");
        assert!(matches!(
            cli.command,
            Some(KapslCommand::AddModel(AddModelCommandArgs {
                no_wait: false,
                timeout_ms: 600000,
                ..
            }))
        ));
        let cli = Cli::try_parse_from(["kapsl", "add-model", "--model", "a.aimod", "--no-wait"])
            .expect("parse add-model --no-wait");
        assert!(matches!(
            cli.command,
            Some(KapslCommand::AddModel(AddModelCommandArgs {
                no_wait: true,
                ..
            }))
        ));
    }

    #[test]
    fn waited_start_reports_the_granted_lease() {
        let view = parse_memory_view(STATS).expect("parse stats");
        let result = add_model_result(
            200,
            r#"{"message":"Model loaded","model_id":2,"status":"active"}"#,
            Some(&view),
        );
        assert_eq!(
            result,
            AddModelResult::Granted {
                model_id: 2,
                bytes: Some(966_000_000)
            }
        );
        let line = strip_ansi(&add_model_result_line(
            "summarizer.aimod",
            &result,
            &Ansi::new(),
        ));
        assert_eq!(line, "  ✓  summarizer.aimod (id=2)  granted · 0.97 GB");
    }

    #[test]
    fn admission_denial_names_the_blocking_accelerator_domain() {
        let view = parse_memory_view(STATS).expect("parse stats");
        let result = add_model_result(
            409,
            r#"{"error":"memory admission failed","model_id":3,"status":"denied","reason":"memory_admission",
                "requested":[{"domain":"host","bytes":120000000},{"domain":"cuda:0","bytes":2450000000}]}"#,
            Some(&view),
        );
        assert_eq!(
            result,
            AddModelResult::Denied {
                needed: Some(2_450_000_000),
                free: Some(1_090_000_000),
                domain: Some("cuda:0".to_string())
            }
        );
        let line = strip_ansi(&add_model_result_line(
            "assistant-3b.aimod",
            &result,
            &Ansi::new(),
        ));
        assert_eq!(
            line,
            "  ✗  assistant-3b.aimod  denied · needs 2.45 GB, 1.09 GB free"
        );
    }

    #[test]
    fn denial_on_another_gpu_says_which_one() {
        let result = AddModelResult::Denied {
            needed: Some(2_000_000_000),
            free: Some(500_000_000),
            domain: Some("cuda:1".to_string()),
        };
        let line = strip_ansi(&add_model_result_line("big.aimod", &result, &Ansi::new()));
        assert!(line.ends_with("needs 2.00 GB, 0.50 GB free on cuda:1"));
    }

    #[test]
    fn unwaited_or_legacy_start_is_queued_and_other_errors_fail() {
        assert_eq!(
            add_model_result(
                202,
                r#"{"message":"Model load started","model_id":4}"#,
                None
            ),
            AddModelResult::Queued { model_id: 4 }
        );
        assert_eq!(
            add_model_result(400, r#"{"error":"Model path does not exist"}"#, None),
            AddModelResult::Failed {
                reason: "Model path does not exist".to_string()
            }
        );
        assert_eq!(
            add_model_result(500, "", None),
            AddModelResult::Failed {
                reason: "the running engine returned HTTP 500".to_string()
            }
        );
    }

    #[test]
    fn denial_under_a_narrower_budget_never_claims_enough_free_memory() {
        let view = parse_memory_view(STATS).expect("parse stats");
        // 0.30 GB requested, 1.09 GB free on cuda:0: a class budget blocked it.
        let result = add_model_result(
            409,
            r#"{"status":"denied","reason":"memory_admission","requested":[{"domain":"cuda:0","bytes":300000000}]}"#,
            Some(&view),
        );
        assert_eq!(
            result,
            AddModelResult::Denied {
                needed: Some(300_000_000),
                free: None,
                domain: Some("cuda:0".to_string())
            }
        );
        let line = strip_ansi(&add_model_result_line("small.aimod", &result, &Ansi::new()));
        assert_eq!(
            line,
            "  ✗  small.aimod  denied · needs 0.30 GB, over the cuda:0 budget"
        );
    }
}
