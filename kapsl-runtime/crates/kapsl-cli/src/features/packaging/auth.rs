use super::*;

pub(crate) fn resolved_remote_url(custom_url: Option<&str>) -> String {
    if let Some(url) = custom_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(url) = optional_env_var(REMOTE_URL_ENV) {
        return url;
    }

    if let Some(url) = optional_env_var(REMOTE_PLACEHOLDER_URL_ENV) {
        return url;
    }

    DEFAULT_REMOTE_URL.to_string()
}

pub(crate) fn resolved_login_remote_url(custom_url: Option<&str>) -> String {
    if let Some(url) = custom_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(url) = optional_env_var(REMOTE_URL_ENV) {
        return url;
    }

    if let Some(url) = optional_env_var(REMOTE_PLACEHOLDER_URL_ENV) {
        return url;
    }

    if let Some(url) = read_last_remote_url_from_store() {
        return url;
    }

    DEFAULT_REMOTE_URL.to_string()
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
    if is_oci_remote_url(remote_url) {
        return Err(
            "Login is only supported for HTTP(S) remote backends, not oci:// remotes.".to_string(),
        );
    }

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
    if is_oci_remote_url(remote_url) {
        return Err(
            "Login is only supported for HTTP(S) remote backends, not oci:// remotes.".to_string(),
        );
    }
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
        a.bold(&user_code),
        a.dim("at"),
        a.teal(&verification_uri)
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

#[derive(Debug)]
pub(crate) struct RemoteHttpRequestError {
    pub(crate) status_code: Option<u16>,
    pub(crate) message: String,
}

impl RemoteHttpRequestError {}

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

pub(crate) fn remote_token_store_key(remote_url: &str) -> String {
    auth_base_url_from_remote_url(remote_url).unwrap_or_else(|_| remote_url.trim().to_string())
}

pub(crate) fn resolve_remote_token_store_path() -> PathBuf {
    if let Some(path) = optional_env_var(REMOTE_TOKEN_STORE_PATH_ENV) {
        return PathBuf::from(path);
    }

    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".kapsl").join("remote-token-store.json")
}

pub(crate) fn load_remote_token_store(path: &Path) -> RemoteTokenStoreFile {
    let Ok(raw) = fs::read_to_string(path) else {
        return RemoteTokenStoreFile::default();
    };

    serde_json::from_str::<RemoteTokenStoreFile>(&raw).unwrap_or_default()
}

pub(crate) fn save_remote_token_store(
    path: &Path,
    store: &RemoteTokenStoreFile,
) -> Result<(), String> {
    let parent = path.parent().ok_or_else(|| {
        format!(
            "Invalid token store path (missing parent directory): {}",
            path.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|e| {
        format!(
            "Failed to create token store directory {}: {}",
            parent.display(),
            e
        )
    })?;

    let raw = serde_json::to_string_pretty(store)
        .map_err(|e| format!("Failed to serialize token store: {}", e))?;
    fs::write(path, raw)
        .map_err(|e| format!("Failed to write token store {}: {}", path.display(), e))
}

pub(crate) fn read_stored_remote_token_for_remote(remote_url: &str) -> Option<String> {
    let path = resolve_remote_token_store_path();
    let store = load_remote_token_store(&path);
    let key = remote_token_store_key(remote_url);
    store.tokens_by_remote.get(&key).cloned()
}

pub(crate) fn read_last_remote_url_from_store() -> Option<String> {
    let path = resolve_remote_token_store_path();
    let store = load_remote_token_store(&path);
    store
        .last_remote_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(crate) fn store_remote_token_for_remote(
    remote_url: &str,
    token: &str,
) -> Result<PathBuf, String> {
    let path = resolve_remote_token_store_path();
    let mut store = load_remote_token_store(&path);
    let trimmed_remote_url = remote_url.trim();
    if !trimmed_remote_url.is_empty() {
        store.last_remote_url = Some(trimmed_remote_url.to_string());
    }
    store
        .tokens_by_remote
        .insert(remote_token_store_key(remote_url), token.trim().to_string());
    save_remote_token_store(&path, &store)?;
    Ok(path)
}

pub(crate) fn resolved_remote_token(
    remote_url: &str,
    custom_token: Option<&str>,
) -> Option<String> {
    if let Some(explicit) = format_authorization_header(custom_token) {
        return Some(explicit);
    }

    let env_token = optional_env_var(REMOTE_TOKEN_ENV);
    if let Some(env_header) = format_authorization_header(env_token.as_deref()) {
        return Some(env_header);
    }

    format_authorization_header(read_stored_remote_token_for_remote(remote_url).as_deref())
}

pub(crate) fn percent_encode_query_component(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len());
    for byte in input.bytes() {
        let ch = byte as char;
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
            encoded.push(ch);
        } else {
            encoded.push('%');
            encoded.push_str(&format!("{:02X}", byte));
        }
    }
    encoded
}

pub(crate) fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        return Command::new("open").arg(url).status().is_ok();
    }
    #[cfg(target_os = "windows")]
    {
        return Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()
            .is_ok();
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Command::new("xdg-open").arg(url).status().is_ok()
    }
}

pub(crate) fn wait_for_login_callback_token(
    listener: TcpListener,
    timeout: Duration,
) -> Result<String, String> {
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                return handle_login_callback_stream(&mut stream);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err("timed out waiting for login callback".to_string());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                return Err(format!("failed to accept callback connection: {}", err));
            }
        }
    }
}

pub(crate) fn handle_login_callback_stream(stream: &mut TcpStream) -> Result<String, String> {
    let mut buffer = [0u8; 8192];
    let bytes_read = stream
        .read(&mut buffer)
        .map_err(|e| format!("failed to read callback request: {}", e))?;
    if bytes_read == 0 {
        return Err("empty callback request".to_string());
    }

    let request = String::from_utf8_lossy(&buffer[..bytes_read]);
    let request_line = request
        .lines()
        .next()
        .ok_or_else(|| "missing callback request line".to_string())?;
    let path = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| "malformed callback request line".to_string())?;

    let token = extract_query_value_from_path(path, "token");
    if let Some(raw_token) = token {
        let trimmed = raw_token.trim();
        if !trimmed.is_empty() {
            let body =
                "<html><body><h3>Login complete</h3><p>You can close this tab.</p></body></html>";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
            return Ok(trimmed.to_string());
        }
    }

    let body = "<html><body><h3>Login failed</h3><p>Token not found in callback.</p></body></html>";
    let response = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
    Err("callback did not include token".to_string())
}

pub(crate) fn extract_query_value_from_path(path: &str, key: &str) -> Option<String> {
    let (_, query) = path.split_once('?')?;
    for pair in query.split('&') {
        let (raw_key, raw_value) = pair.split_once('=').unwrap_or((pair, ""));
        if raw_key == key {
            return Some(percent_decode(raw_value));
        }
    }
    None
}

pub(crate) fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = &value[i + 1..i + 3];
                if let Ok(decoded) = u8::from_str_radix(hex, 16) {
                    out.push(decoded);
                    i += 3;
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            ch => {
                out.push(ch);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}
