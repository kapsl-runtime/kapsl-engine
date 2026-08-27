//! Remote registry authentication used by package push, pull, and login.

use super::*;
use crate::features::http_client::{format_remote_http_error, native_tls_http_agent};

mod callback;
mod token_store;

use callback::{open_browser, percent_encode_query_component, wait_for_login_callback_token};
use token_store::read_last_remote_url_from_store;
pub(crate) use token_store::{resolved_remote_token, store_remote_token_for_remote};

#[derive(Debug, Serialize)]
pub(crate) struct LoginResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) auth_base_url: String,
    pub(crate) provider: String,
    pub(crate) login_method: String,
    pub(crate) callback_url: String,
    pub(crate) token_store_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) verification_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user_code: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeStartResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    expires_in: Option<u64>,
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodePollResponse {
    status: String,
    token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
    interval: Option<u64>,
}

/// The remote URL sources both resolvers share, in precedence order: an
/// explicit non-blank argument, then `REMOTE_URL_ENV`.
fn remote_url_from_arg_or_env(custom_url: Option<&str>) -> Option<String> {
    if let Some(url) = custom_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    optional_env_var(REMOTE_URL_ENV)
}

pub(crate) fn resolved_remote_url(custom_url: Option<&str>) -> String {
    remote_url_from_arg_or_env(custom_url).unwrap_or_else(|| DEFAULT_REMOTE_URL.to_string())
}

/// As [`resolved_remote_url`], but falls back to the last remote recorded by a
/// previous login before giving up on the default.
pub(crate) fn resolved_login_remote_url(custom_url: Option<&str>) -> String {
    remote_url_from_arg_or_env(custom_url)
        .or_else(read_last_remote_url_from_store)
        .unwrap_or_else(|| DEFAULT_REMOTE_URL.to_string())
}

pub(crate) fn auth_base_url_from_remote_url(remote_url: &str) -> Result<String, String> {
    let trimmed = remote_url.trim().trim_end_matches('/');
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return Err(format!(
            "Remote URL must start with http:// or https:// for login flow (got '{}')",
            remote_url
        ));
    }

    if let Some(stripped) = trimmed.strip_suffix("/api/v1") {
        if stripped.is_empty() {
            return Err(format!("Invalid remote URL '{}'", remote_url));
        }
        return Ok(stripped.to_string());
    }
    if let Some(stripped) = trimmed.strip_suffix("/v1") {
        if stripped.is_empty() {
            return Err(format!("Invalid remote URL '{}'", remote_url));
        }
        return Ok(stripped.to_string());
    }
    Ok(trimmed.to_string())
}

pub(crate) fn perform_browser_login_flow(
    remote_url: &str,
    provider: OAuthProvider,
    callback_host: &str,
    callback_port: u16,
    timeout_seconds: u64,
    no_browser: bool,
) -> Result<LoginResponse, String> {
    let auth_base_url = auth_base_url_from_remote_url(remote_url)?;
    let callback_addr = format!("{}:{}", callback_host.trim(), callback_port);
    let listener = TcpListener::bind(&callback_addr).map_err(|e| {
        format!(
            "Failed to bind local login callback listener at {}: {}",
            callback_addr, e
        )
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("Failed to configure callback listener: {}", e))?;

    let local_addr = listener
        .local_addr()
        .map_err(|e| format!("Failed to read callback address: {}", e))?;
    let callback_url = format!("http://{}/callback", local_addr);
    let login_url = format!(
        "{}/auth/{}/login?redirect_uri={}",
        auth_base_url,
        provider.route_segment(),
        percent_encode_query_component(&callback_url)
    );

    let a = Ansi::new();
    if no_browser {
        eprintln!("  {}  {}", a.dim("Sign in at:"), a.teal(&login_url));
    } else if !open_browser(&login_url) {
        eprintln!(
            "  {}  {}",
            a.dim("Browser could not open. Sign in at:"),
            a.teal(&login_url)
        );
    }

    let timeout = Duration::from_secs(timeout_seconds.max(1));
    let token = wait_for_login_callback_token(listener, timeout)
        .map_err(|e| format!("Login callback failed: {}", e))?;

    let token_store_path = store_remote_token_for_remote(remote_url, &token)?;

    Ok(LoginResponse {
        status: "ok".to_string(),
        remote_url: remote_url.to_string(),
        auth_base_url,
        provider: provider.route_segment().to_string(),
        login_method: "browser-callback".to_string(),
        callback_url,
        token_store_path: token_store_path.to_string_lossy().to_string(),
        verification_uri: None,
        user_code: None,
    })
}

pub(crate) fn perform_device_code_login_flow(
    remote_url: &str,
    provider: OAuthProvider,
    timeout_seconds: u64,
    no_browser: bool,
) -> Result<LoginResponse, String> {
    if provider != OAuthProvider::GitHub {
        return Err("Device code flow currently supports only --provider github.".to_string());
    }

    let auth_base_url = auth_base_url_from_remote_url(remote_url)?;
    let start_url = format!(
        "{}/auth/{}/device/start",
        auth_base_url,
        provider.route_segment()
    );
    let poll_url = format!(
        "{}/auth/{}/device/poll",
        auth_base_url,
        provider.route_segment()
    );

    let agent = native_tls_http_agent();

    let mut start_response = agent
        .post(&start_url)
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .send("{}")
        .map_err(|error| match error {
            ureq::Error::StatusCode(404) => format!(
                "Remote backend does not support device code login at {} (missing endpoint /auth/{}/device/start).",
                auth_base_url,
                provider.route_segment()
            ),
            other => format!(
                "Failed to start device code login at {}: {}",
                start_url,
                format_remote_http_error(other)
            ),
        })?;
    let start_body = start_response
        .body_mut()
        .read_to_string()
        .map_err(|error| {
            format!(
                "Failed to read device code start response from {}: {}",
                start_url, error
            )
        })?;
    let start: DeviceCodeStartResponse = serde_json::from_str(&start_body).map_err(|error| {
        format!(
            "Failed to decode device code start response from {}: {}",
            start_url, error
        )
    })?;

    let device_code = start.device_code.trim();
    if device_code.is_empty() {
        return Err("Remote backend returned an empty device_code.".to_string());
    }
    let verification_uri = start.verification_uri.trim();
    if verification_uri.is_empty() {
        return Err("Remote backend returned an empty verification_uri.".to_string());
    }
    let user_code = start.user_code.trim();
    if user_code.is_empty() {
        return Err("Remote backend returned an empty user_code.".to_string());
    }

    let a = Ansi::new();
    if let Some(complete_url) = start.verification_uri_complete.as_deref() {
        let trimmed = complete_url.trim();
        if !trimmed.is_empty() {
            if no_browser {
                eprintln!("  {}  {}", a.dim("Authorize at:"), a.teal(trimmed));
            } else if !open_browser(trimmed) {
                eprintln!(
                    "  {}  {}",
                    a.dim("Browser could not open. Authorize at:"),
                    a.teal(trimmed)
                );
            }
        }
    }
    eprintln!(
        "  {}  {}  {}  {}",
        a.dim("Enter code"),
        a.bold(user_code),
        a.dim("at"),
        a.teal(verification_uri)
    );
    eprintln!("  {}", a.dim("Waiting for authorization approval..."));

    let started_at = Instant::now();
    let timeout = Duration::from_secs(timeout_seconds.max(1));
    let mut interval_secs = start.interval.unwrap_or(5).max(1);
    let expires_in = start.expires_in.unwrap_or(timeout.as_secs()).max(1);
    let flow_deadline = started_at + Duration::from_secs(expires_in);
    let timeout_deadline = started_at + timeout;

    loop {
        let now = Instant::now();
        if now >= timeout_deadline {
            return Err("Timed out waiting for device authorization approval.".to_string());
        }
        if now >= flow_deadline {
            return Err("Device authorization code expired. Start login again.".to_string());
        }
        std::thread::sleep(Duration::from_secs(interval_secs));

        let poll_payload = serde_json::json!({
            "device_code": device_code
        });
        let mut poll_response = agent
            .post(&poll_url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .send(
                serde_json::to_string(&poll_payload)
                    .map_err(|error| format!("Failed to encode device poll payload: {}", error))?,
            )
            .map_err(|error| match error {
                ureq::Error::StatusCode(404) => format!(
                    "Remote backend does not support device code polling at {} (missing endpoint /auth/{}/device/poll).",
                    auth_base_url,
                    provider.route_segment()
                ),
                other => format!(
                    "Failed to poll device code login at {}: {}",
                    poll_url,
                    format_remote_http_error(other)
                ),
            })?;
        let poll_body = poll_response.body_mut().read_to_string().map_err(|error| {
            format!(
                "Failed to read device poll response from {}: {}",
                poll_url, error
            )
        })?;
        let poll: DeviceCodePollResponse = serde_json::from_str(&poll_body).map_err(|error| {
            format!(
                "Failed to decode device poll response from {}: {}",
                poll_url, error
            )
        })?;

        match poll.status.trim() {
            "approved" => {
                let token = poll.token.unwrap_or_default();
                let trimmed = token.trim();
                if trimmed.is_empty() {
                    return Err("Device authorization completed without token.".to_string());
                }
                let token_store_path = store_remote_token_for_remote(remote_url, trimmed)?;
                return Ok(LoginResponse {
                    status: "ok".to_string(),
                    remote_url: remote_url.to_string(),
                    auth_base_url,
                    provider: provider.route_segment().to_string(),
                    login_method: "device-code".to_string(),
                    callback_url: String::new(),
                    token_store_path: token_store_path.to_string_lossy().to_string(),
                    verification_uri: Some(verification_uri.to_string()),
                    user_code: Some(user_code.to_string()),
                });
            }
            "pending" => {
                interval_secs = poll.interval.unwrap_or(interval_secs).max(1);
                continue;
            }
            "denied" => {
                return Err("Device authorization was denied by the user.".to_string());
            }
            "expired" => {
                return Err("Device authorization code expired. Start login again.".to_string());
            }
            "error" => {
                let err = poll
                    .error
                    .unwrap_or_else(|| "device_code_error".to_string());
                let description = poll.error_description.unwrap_or_default();
                if description.trim().is_empty() {
                    return Err(format!("Device authorization failed: {}", err));
                }
                return Err(format!(
                    "Device authorization failed: {} ({})",
                    err, description
                ));
            }
            other => {
                return Err(format!("Unexpected device authorization status: {}", other));
            }
        }
    }
}

pub(crate) fn looks_like_auth_transport_failure(http_error: &RemoteHttpRequestError) -> bool {
    if http_error.status_code.is_some() {
        return false;
    }

    let message = http_error.message.to_ascii_lowercase();
    message.contains("broken pipe")
        || message.contains("connection reset")
        || message.contains("connection closed")
}

pub(crate) fn maybe_auto_login_for_remote(
    remote_url: &str,
    request_has_explicit_token: bool,
    interactive_login: bool,
    remote_token: &mut Option<String>,
    http_error: &RemoteHttpRequestError,
) -> Result<bool, String> {
    if !interactive_login || request_has_explicit_token || remote_token.is_some() {
        return Ok(false);
    }
    if http_error.status_code != Some(401) && !looks_like_auth_transport_failure(http_error) {
        return Ok(false);
    }

    let a = Ansi::new();
    eprintln!("  {}  {}", a.dim("Authenticating with"), a.teal(remote_url));
    let browser_login = perform_browser_login_flow(
        remote_url,
        OAuthProvider::GitHub,
        "127.0.0.1",
        0,
        180,
        false,
    );
    if let Err(error) = browser_login {
        println!(
            "Browser login flow failed ({}). Falling back to device code flow...",
            error
        );
        let _ = perform_device_code_login_flow(remote_url, OAuthProvider::GitHub, 600, true)?;
    }
    *remote_token = resolved_remote_token(remote_url, None);
    Ok(remote_token.is_some())
}

pub(crate) fn is_likely_headless_session() -> bool {
    std::env::var_os("SSH_CONNECTION").is_some()
        || std::env::var_os("SSH_CLIENT").is_some()
        || std::env::var_os("SSH_TTY").is_some()
}
