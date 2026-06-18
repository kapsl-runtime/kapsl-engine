use super::*;

#[derive(Debug, Deserialize)]
pub(crate) struct PackageKapslRequest {
    pub(crate) model_path: String,
    pub(crate) output_path: Option<String>,
    pub(crate) project_name: Option<String>,
    pub(crate) framework: Option<String>,
    #[serde(default)]
    pub(crate) format: Option<String>,
    #[serde(default)]
    pub(crate) model_type: Option<String>,
    #[serde(default)]
    pub(crate) task: Option<String>,
    pub(crate) version: Option<String>,
    pub(crate) metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackageKapslResponse {
    pub(crate) status: String,
    pub(crate) kapsl_path: String,
    pub(crate) project_name: String,
    pub(crate) framework: String,
    pub(crate) version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PushKapslRequest {
    pub(crate) kapsl_path: String,
    pub(crate) target: String,
    pub(crate) remote_url: Option<String>,
    pub(crate) remote_token: Option<String>,
    #[serde(default)]
    pub(crate) interactive_login: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PushKapslResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) artifact_url: String,
    pub(crate) mirrored_path: String,
    pub(crate) bytes_uploaded: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) manifest_digest: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PullKapslRequest {
    pub(crate) target: String,
    pub(crate) reference: Option<String>,
    pub(crate) destination_dir: Option<String>,
    pub(crate) remote_url: Option<String>,
    pub(crate) remote_token: Option<String>,
    #[serde(default)]
    pub(crate) interactive_login: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PullKapslResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) artifact_url: String,
    pub(crate) kapsl_path: String,
    pub(crate) bytes_downloaded: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RemoteArtifactLabelSummary {
    pub(crate) label: String,
    pub(crate) reference: String,
    pub(crate) size_bytes: u64,
    pub(crate) updated_at: String,
    pub(crate) download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RemoteArtifactModelSummary {
    pub(crate) name: String,
    pub(crate) latest_label: Option<String>,
    pub(crate) latest_reference: Option<String>,
    pub(crate) artifact_count: usize,
    #[serde(default)]
    pub(crate) labels: Vec<RemoteArtifactLabelSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RemoteArtifactInventoryResponse {
    pub(crate) status: String,
    pub(crate) repo: String,
    #[serde(default)]
    pub(crate) available_repos: Vec<String>,
    #[serde(default)]
    pub(crate) models: Vec<RemoteArtifactModelSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimeRemoteArtifactInventoryResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) repo: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) available_repos: Vec<String>,
    #[serde(default)]
    pub(crate) models: Vec<RemoteArtifactModelSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct RemoteTokenStoreFile {
    #[serde(default)]
    pub(crate) tokens_by_remote: HashMap<String, String>,
    #[serde(default)]
    pub(crate) last_remote_url: Option<String>,
}

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
pub(crate) struct DeviceCodeStartResponse {
    pub(crate) device_code: String,
    pub(crate) user_code: String,
    pub(crate) verification_uri: String,
    pub(crate) verification_uri_complete: Option<String>,
    pub(crate) expires_in: Option<u64>,
    pub(crate) interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DeviceCodePollResponse {
    pub(crate) status: String,
    pub(crate) token: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) error_description: Option<String>,
    pub(crate) interval: Option<u64>,
}

pub(crate) fn format_human_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0usize;
    while value >= 1024.0 && unit_idx + 1 < units.len() {
        value /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{}{}", bytes, units[unit_idx])
    } else if (value - value.round()).abs() < 0.05 {
        format!("{:.0}{}", value, units[unit_idx])
    } else {
        format!("{:.1}{}", value, units[unit_idx])
    }
}

pub(crate) fn print_build_summary(kapsl_path: &str, metadata_path: Option<&str>) {
    let a = Ansi::new();
    let display_name = Path::new(kapsl_path)
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or(kapsl_path);
    match fs::metadata(kapsl_path) {
        Ok(metadata) => eprintln!(
            "  {}  {} {}",
            a.green("✓"),
            display_name,
            a.dim(&format!("({})", format_human_bytes(metadata.len())))
        ),
        Err(_) => eprintln!("  {}  {}", a.green("✓"), display_name),
    }
    if let Some(metadata_path) = metadata_path {
        eprintln!("  {}  created {}", a.green("✓"), metadata_path);
    }
}

pub(crate) fn context_metadata_missing(context_path: &Path) -> bool {
    context_path.is_dir() && !context_path.join("metadata.json").exists()
}

pub(crate) fn execute_context_build(
    context_path: &Path,
    model_override: Option<&Path>,
    output_override: Option<&Path>,
    project_name_override: Option<&str>,
    framework_override: Option<&str>,
    version_override: Option<&str>,
    metadata_override: Option<&serde_json::Value>,
    axes: AxisOverrides,
) -> Result<PackageKapslResponse, DynError> {
    let interactive_metadata_setup = cli_stdin_is_tty() && context_metadata_missing(context_path);
    let build = || {
        create_kapsl_package_from_context(
            context_path,
            model_override,
            output_override,
            project_name_override,
            framework_override,
            version_override,
            metadata_override,
            axes,
            interactive_metadata_setup,
        )
        .map_err(dyn_error_from_message)
    };

    if interactive_metadata_setup {
        build()
    } else {
        run_with_loading("Building package", build)
    }
}

pub(crate) fn model_file_metadata_missing(model_path: &Path) -> bool {
    let metadata_dir = match model_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    !metadata_dir.join("metadata.json").exists()
}

pub(crate) fn execute_model_file_build(
    request: &PackageKapslRequest,
) -> Result<PackageKapslResponse, DynError> {
    let model_path = PathBuf::from(request.model_path.trim());
    let interactive_metadata_setup = cli_stdin_is_tty() && model_file_metadata_missing(&model_path);
    let build = || {
        create_kapsl_package(request, interactive_metadata_setup).map_err(dyn_error_from_message)
    };

    if interactive_metadata_setup {
        build()
    } else {
        run_with_loading("Building package", build)
    }
}

pub(crate) fn format_elapsed(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 1.0 {
        format!("{}ms", duration.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.2}s")
    } else {
        let minutes = (secs / 60.0).floor() as u64;
        let rem_secs = secs - (minutes as f64 * 60.0);
        format!("{minutes}m {rem_secs:.1}s")
    }
}

pub(crate) fn transfer_backend_label(remote_url: &str) -> &'static str {
    if is_oci_remote_url(remote_url) {
        "oci"
    } else if is_default_placeholder_remote(remote_url) {
        "placeholder"
    } else {
        "http"
    }
}

pub(crate) fn print_transfer_summary(
    action: &str,
    remote_url: &str,
    bytes: u64,
    elapsed: Duration,
    path_or_target: &str,
) {
    let a = Ansi::new();
    let elapsed_secs = elapsed.as_secs_f64().max(0.001);
    let bytes_per_sec = (bytes as f64 / elapsed_secs).round() as u64;
    eprintln!(
        "  {}  {} {}  {}  {}",
        a.green("✓"),
        action,
        format_human_bytes(bytes),
        a.dim(&format!("via {}", transfer_backend_label(remote_url))),
        a.dim(&format!(
            "in {} ({}/s)",
            format_elapsed(elapsed),
            format_human_bytes(bytes_per_sec)
        )),
    );
    eprintln!("     {}", a.teal(path_or_target));
}

pub(crate) fn discover_kapsl_in_current_dir() -> Result<PathBuf, String> {
    let cwd =
        std::env::current_dir().map_err(|e| format!("Failed to read current directory: {}", e))?;
    let entries = fs::read_dir(&cwd)
        .map_err(|e| format!("Failed to read current directory {}: {}", cwd.display(), e))?;

    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("aimod"))
            .unwrap_or(false)
        {
            candidates.push(path);
        }
    }

    candidates.sort();
    if candidates.is_empty() {
        return Err("No .aimod files found in the current directory. Pass an explicit package path via `kapsl push <repo>/<model>:<label> <PATH>` or `--model <PATH>`.".to_string());
    }
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }

    if let Some(dir_name) = cwd.file_name().and_then(|v| v.to_str()) {
        let expected = cwd.join(format!("{}.aimod", dir_name));
        if expected.exists() && expected.is_file() {
            return Ok(expected);
        }
    }

    Err(format!(
        "Multiple .aimod files found in the current directory. Pass an explicit path.\nFound: {}",
        candidates
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

pub(crate) fn execute_build_command(args: BuildCommandArgs) -> Result<(), DynError> {
    let metadata = match args.metadata_json.as_deref() {
        Some(raw) => Some(
            serde_json::from_str::<serde_json::Value>(raw)
                .map_err(|e| dyn_error_from_message(format!("Invalid --metadata-json: {}", e)))?,
        ),
        None => None,
    };

    let response = match args.context.as_ref() {
        Some(context_or_model_path) => {
            if context_or_model_path.is_dir() {
                execute_context_build(
                    context_or_model_path,
                    args.model.as_deref(),
                    args.output.as_deref(),
                    args.project_name.as_deref(),
                    args.framework.as_deref(),
                    args.version.as_deref(),
                    metadata.as_ref(),
                    AxisOverrides {
                        format: args.format.as_deref(),
                        model_type: args.model_type.as_deref(),
                        task: args.task.as_deref(),
                    },
                )?
            } else if looks_like_model_file_path(context_or_model_path)
                || context_or_model_path.is_file()
            {
                if args.model.is_some() {
                    return Err(dyn_error_from_message(
                        "When CONTEXT is a model file, do not also pass --model.",
                    ));
                }
                let request = PackageKapslRequest {
                    model_path: context_or_model_path.to_string_lossy().to_string(),
                    output_path: args.output.map(|p| p.to_string_lossy().to_string()),
                    project_name: args.project_name.clone(),
                    framework: args.framework.clone(),
                    format: args.format.clone(),
                    model_type: args.model_type.clone(),
                    task: args.task.clone(),
                    version: args.version.clone(),
                    metadata: metadata.clone(),
                };
                execute_model_file_build(&request)?
            } else {
                execute_context_build(
                    context_or_model_path,
                    args.model.as_deref(),
                    args.output.as_deref(),
                    args.project_name.as_deref(),
                    args.framework.as_deref(),
                    args.version.as_deref(),
                    metadata.as_ref(),
                    AxisOverrides {
                        format: args.format.as_deref(),
                        model_type: args.model_type.as_deref(),
                        task: args.task.as_deref(),
                    },
                )?
            }
        }
        None => {
            if let Some(model_path) = args.model.as_ref() {
                let request = PackageKapslRequest {
                    model_path: model_path.to_string_lossy().to_string(),
                    output_path: args.output.map(|p| p.to_string_lossy().to_string()),
                    project_name: args.project_name,
                    framework: args.framework,
                    format: args.format,
                    model_type: args.model_type,
                    task: args.task,
                    version: args.version,
                    metadata,
                };
                execute_model_file_build(&request)?
            } else {
                // Docker-style default: `kapsl build` means "build from the current directory".
                let context_dir = PathBuf::from(".");
                execute_context_build(
                    &context_dir,
                    None,
                    args.output.as_deref(),
                    args.project_name.as_deref(),
                    args.framework.as_deref(),
                    args.version.as_deref(),
                    metadata.as_ref(),
                    AxisOverrides {
                        format: args.format.as_deref(),
                        model_type: args.model_type.as_deref(),
                        task: args.task.as_deref(),
                    },
                )?
            }
        }
    };
    print_build_summary(&response.kapsl_path, response.metadata_path.as_deref());
    Ok(())
}

pub(crate) fn execute_push_command(args: PushCommandArgs) -> Result<(), DynError> {
    if args.kapsl.is_some() && args.model.is_some() {
        return Err(dyn_error_from_message(
            "Push expects a single `.aimod` argument. Use either `kapsl push <repo>/<model>:<label> <KAPSL>` or `kapsl push <repo>/<model>:<label> --model <PATH>`.",
        ));
    }
    let target = parse_model_target(&args.target).map_err(dyn_error_from_message)?;

    let kapsl_path = match args.kapsl.as_ref().or(args.model.as_ref()) {
        Some(path) => path.clone(),
        None => discover_kapsl_in_current_dir().map_err(dyn_error_from_message)?,
    };

    let request = PushKapslRequest {
        kapsl_path: kapsl_path.to_string_lossy().to_string(),
        target: target.as_string(),
        remote_url: args.remote_url,
        remote_token: args.remote_token,
        interactive_login: true,
    };

    let started_at = Instant::now();
    let response = run_with_loading("Uploading package", || {
        push_kapsl_to_placeholder_remote(&request).map_err(dyn_error_from_message)
    })?;
    print_transfer_summary(
        "Uploaded",
        &response.remote_url,
        response.bytes_uploaded,
        started_at.elapsed(),
        &response.artifact_url,
    );
    Ok(())
}

pub(crate) fn execute_pull_command(args: PullCommandArgs) -> Result<(), DynError> {
    if args.target.is_some() && args.model.is_some() {
        return Err(dyn_error_from_message(
            "Pull expects a single target argument. Use either `kapsl pull <repo>/<model>:<label>` or `kapsl pull --model <repo>/<model>:<label>`.",
        ));
    }
    let target = args.target.or(args.model).ok_or_else(|| {
        dyn_error_from_message("Target is required. Usage: `kapsl pull <repo>/<model>:<label>`.")
    })?;
    let target = parse_model_target(&target).map_err(dyn_error_from_message)?;

    let request = PullKapslRequest {
        target: target.as_string(),
        reference: args.reference,
        destination_dir: args
            .destination_dir
            .map(|p| p.to_string_lossy().to_string()),
        remote_url: args.remote_url,
        remote_token: args.remote_token,
        interactive_login: true,
    };

    let started_at = Instant::now();
    let response = run_with_loading("Downloading package", || {
        pull_kapsl_from_placeholder_remote(&request).map_err(dyn_error_from_message)
    })?;
    print_transfer_summary(
        "Downloaded",
        &response.remote_url,
        response.bytes_downloaded,
        started_at.elapsed(),
        &response.kapsl_path,
    );
    Ok(())
}

pub(crate) fn execute_login_command(args: LoginCommandArgs) -> Result<(), DynError> {
    let remote_url = resolved_login_remote_url(args.remote_url.as_deref());
    let auto_headless = args.no_browser || is_likely_headless_session();
    let use_device_code =
        args.device_code || (auto_headless && args.provider == OAuthProvider::GitHub);

    let response = if use_device_code {
        perform_device_code_login_flow(
            &remote_url,
            args.provider,
            args.timeout_seconds,
            args.no_browser,
        )
    } else {
        perform_browser_login_flow(
            &remote_url,
            args.provider,
            args.callback_host.trim(),
            args.callback_port,
            args.timeout_seconds,
            args.no_browser,
        )
    }
    .map_err(dyn_error_from_message)?;

    let a = Ansi::new();
    eprintln!();
    eprintln!(
        "  {}  {}",
        a.green("✓"),
        a.bold("Authenticated successfully")
    );
    eprintln!("     {}  {}", a.dim("Provider"), response.provider);
    eprintln!(
        "     {}    {}",
        a.dim("Remote"),
        a.teal(&response.remote_url)
    );
    eprintln!(
        "     {}    {}",
        a.dim("Token"),
        a.dim(&response.token_store_path)
    );
    eprintln!();

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelTargetRef {
    pub(crate) repo: String,
    pub(crate) model: String,
    pub(crate) label: String,
}

impl ModelTargetRef {
    pub(crate) fn as_string(&self) -> String {
        format!("{}/{}:{}", self.repo, self.model, self.label)
    }
}

pub(crate) fn is_valid_target_part(part: &str) -> bool {
    !part.is_empty()
        && part
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-')
}

pub(crate) fn parse_model_target(raw: &str) -> Result<ModelTargetRef, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(
            "Target cannot be empty. Expected format: <repo_name>/<model>:<label>.".to_string(),
        );
    }

    let (repo, rest) = trimmed.split_once('/').ok_or_else(|| {
        format!(
            "Invalid target `{}`. Expected format: <repo_name>/<model>:<label>.",
            trimmed
        )
    })?;

    if rest.contains('/') {
        return Err(format!(
            "Invalid target `{}`. Only one `/` is allowed (between repo and model).",
            trimmed
        ));
    }

    let (model, label) = rest.split_once(':').ok_or_else(|| {
        format!(
            "Invalid target `{}`. Expected format: <repo_name>/<model>:<label>.",
            trimmed
        )
    })?;

    if label.contains(':') {
        return Err(format!(
            "Invalid target `{}`. Label must not contain `:`.",
            trimmed
        ));
    }

    if !is_valid_target_part(repo) {
        return Err(format!(
            "Invalid repo `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            repo, trimmed
        ));
    }
    if !is_valid_target_part(model) {
        return Err(format!(
            "Invalid model `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            model, trimmed
        ));
    }
    if !is_valid_target_part(label) {
        return Err(format!(
            "Invalid label `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            label, trimmed
        ));
    }

    Ok(ModelTargetRef {
        repo: repo.to_string(),
        model: model.to_string(),
        label: label.to_string(),
    })
}

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

pub(crate) fn placeholder_remote_storage_dir() -> PathBuf {
    if let Some(path) = optional_env_var(REMOTE_PLACEHOLDER_DIR_ENV) {
        return PathBuf::from(path);
    }

    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(REMOTE_PLACEHOLDER_DIR)
}

pub(crate) fn is_default_placeholder_remote(remote_url: &str) -> bool {
    remote_url.trim_end_matches('/') == REMOTE_PLACEHOLDER_URL.trim_end_matches('/')
}

pub(crate) fn artifact_url_for_remote(remote_url: &str, target: &ModelTargetRef) -> String {
    format!(
        "{}/aimod/{}/{}:{}",
        remote_url.trim_end_matches('/'),
        target.repo,
        target.model,
        target.label
    )
}

pub(crate) fn remote_inventory_url_for_remote(remote_url: &str) -> String {
    format!(
        "{}/kapsl/repositories/current/models",
        remote_url.trim_end_matches('/')
    )
}

pub(crate) fn placeholder_remote_artifact_path(target: &ModelTargetRef) -> PathBuf {
    placeholder_remote_storage_dir()
        .join(&target.repo)
        .join(&target.model)
        .join(format!("{}.aimod", target.label))
}

pub(crate) fn format_remote_http_error(error: ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(status) => format!("Remote backend returned HTTP {}", status),
        other => other.to_string(),
    }
}

pub(crate) fn native_tls_http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .build(),
        )
        .build()
        .into()
}

/// Agent with no global timeout, suitable for large file uploads/downloads.
/// Uses rustls to avoid macOS native-tls rejecting certs with long validity periods.
pub(crate) fn http_agent_for_transfer() -> ureq::Agent {
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .build(),
        )
        .timeout_global(None)
        .build()
        .into()
}

pub(crate) fn fetch_remote_artifact_inventory(
    custom_remote_url: Option<&str>,
) -> Result<RuntimeRemoteArtifactInventoryResponse, String> {
    let remote_url = resolved_login_remote_url(custom_remote_url);
    if is_oci_remote_url(&remote_url) {
        return Err(
            "Remote artifact browsing is not available for oci:// remotes yet.".to_string(),
        );
    }

    let inventory_url = remote_inventory_url_for_remote(&remote_url);
    let authorization_header = resolved_remote_token(&remote_url, None);
    let agent = native_tls_http_agent();
    let mut request = agent
        .get(&inventory_url)
        .header("Accept", "application/json");
    if let Some(header) = authorization_header.as_deref() {
        request = request.header("Authorization", header);
    }

    let mut response = request.call().map_err(|error| match error {
        ureq::Error::StatusCode(401) | ureq::Error::StatusCode(403) => format!(
            "Remote artifact inventory requires authentication for {}. Run `kapsl login --remote-url {}` first.",
            remote_url, remote_url
        ),
        other => format!(
            "Failed to fetch remote artifact inventory from {}: {}",
            inventory_url,
            format_remote_http_error(other)
        ),
    })?;

    let body = response.body_mut().read_to_string().map_err(|error| {
        format!(
            "Failed to read remote artifact inventory response from {}: {}",
            inventory_url, error
        )
    })?;

    let payload: RemoteArtifactInventoryResponse =
        serde_json::from_str(&body).map_err(|error| {
            format!(
                "Failed to parse remote artifact inventory response from {}: {}",
                inventory_url, error
            )
        })?;

    Ok(RuntimeRemoteArtifactInventoryResponse {
        status: payload.status,
        remote_url,
        repo: payload.repo,
        available_repos: payload.available_repos,
        models: payload.models,
    })
}

/// Try to get a presigned upload URL from the server. Returns None if the
/// server does not support presigned uploads (404 or 501).
pub(crate) fn request_presigned_upload_url(
    artifact_url: &str,
    authorization_header: Option<&str>,
) -> Result<Option<PresignedUploadResponse>, RemoteHttpRequestError> {
    let upload_endpoint = format!("{}/upload", artifact_url);
    let agent = http_agent_for_transfer();
    let mut request = agent
        .post(&upload_endpoint)
        .header("Content-Type", "application/json");
    if let Some(header) = authorization_header {
        request = request.header("Authorization", header);
    }
    match request.send_empty() {
        Ok(resp) => {
            let body: String =
                resp.into_body()
                    .read_to_string()
                    .map_err(|e| RemoteHttpRequestError {
                        status_code: None,
                        message: format!("Failed to read presigned upload response: {}", e),
                    })?;
            let parsed: serde_json::Value =
                serde_json::from_str(&body).map_err(|e| RemoteHttpRequestError {
                    status_code: None,
                    message: format!("Invalid presigned upload response JSON: {}", e),
                })?;
            let upload_url = parsed["upload_url"]
                .as_str()
                .ok_or_else(|| RemoteHttpRequestError {
                    status_code: None,
                    message: "Presigned upload response missing 'upload_url' field".to_string(),
                })?
                .to_string();
            let method = parsed["method"].as_str().unwrap_or("PUT").to_string();
            let headers: Vec<(String, String)> = parsed["headers"]
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default();
            Ok(Some(PresignedUploadResponse {
                upload_url,
                method,
                headers,
            }))
        }
        Err(ureq::Error::StatusCode(code)) if code == 404 || code == 501 => Ok(None),
        Err(e) => {
            // Other errors (auth, server error) — propagate.
            let status = match e {
                ureq::Error::StatusCode(code) => Some(code),
                _ => None,
            };
            Err(RemoteHttpRequestError {
                status_code: status,
                message: format!(
                    "Failed to request presigned upload URL from {}: {}",
                    upload_endpoint,
                    format_remote_http_error(e)
                ),
            })
        }
    }
}

pub(crate) struct PresignedUploadResponse {
    pub(crate) upload_url: String,
    pub(crate) method: String,
    pub(crate) headers: Vec<(String, String)>,
}

pub(crate) fn push_kapsl_to_http_remote(
    artifact_url: &str,
    source_path: &Path,
    authorization_header: Option<&str>,
) -> Result<u64, RemoteHttpRequestError> {
    let file_size = fs::metadata(source_path).map_err(|e| RemoteHttpRequestError {
        status_code: None,
        message: format!(
            "Failed to read .aimod metadata for upload {}: {}",
            source_path.display(),
            e
        ),
    })?;

    // Try presigned URL flow first.
    if let Some(presigned) = request_presigned_upload_url(artifact_url, authorization_header)? {
        eprintln!(
            "  {}",
            Ansi::new().dim(&format!("Uploading {} bytes...", file_size.len()))
        );
        let file = File::open(source_path).map_err(|e| RemoteHttpRequestError {
            status_code: None,
            message: format!(
                "Failed to open .aimod for upload {}: {}",
                source_path.display(),
                e
            ),
        })?;

        let agent = http_agent_for_transfer();
        let mut request = if presigned.method == "PUT" {
            agent.put(&presigned.upload_url)
        } else {
            agent.post(&presigned.upload_url)
        };
        request = request
            .header("Content-Type", "application/octet-stream")
            .header("Content-Length", &file_size.len().to_string());
        for (key, value) in &presigned.headers {
            request = request.header(key, value);
        }
        // No Authorization header — the SAS token is in the URL.
        request.send(file).map_err(|e| {
            let status = match e {
                ureq::Error::StatusCode(code) => Some(code),
                _ => None,
            };
            RemoteHttpRequestError {
                status_code: status,
                message: format!(
                    "Failed to upload .aimod to presigned URL: {}",
                    format_remote_http_error(e)
                ),
            }
        })?;

        return Ok(file_size.len());
    }

    // Fallback: direct upload to the API server.
    let file = File::open(source_path).map_err(|e| RemoteHttpRequestError {
        status_code: None,
        message: format!(
            "Failed to open .aimod for upload {}: {}",
            source_path.display(),
            e
        ),
    })?;

    let agent = http_agent_for_transfer();
    let mut request = agent
        .put(artifact_url)
        .header("Content-Type", "application/octet-stream")
        .header("Content-Length", &file_size.len().to_string());
    if let Some(header) = authorization_header {
        request = request.header("Authorization", header);
    }
    request.send(file).map_err(|e| {
        let status = match e {
            ureq::Error::StatusCode(code) => Some(code),
            _ => None,
        };
        RemoteHttpRequestError {
            status_code: status,
            message: format!(
                "Failed to upload .aimod to remote backend {}: {}",
                artifact_url,
                format_remote_http_error(e)
            ),
        }
    })?;

    Ok(file_size.len())
}

/// Try to get a presigned download URL from the server. Returns None if the
/// server does not support presigned downloads (404 or 501).
pub(crate) fn request_presigned_download_url(
    artifact_url: &str,
    authorization_header: Option<&str>,
) -> Result<Option<String>, RemoteHttpRequestError> {
    let download_endpoint = format!("{}/download", artifact_url);
    let agent = http_agent_for_transfer();
    let mut request = agent
        .post(&download_endpoint)
        .header("Content-Type", "application/json");
    if let Some(header) = authorization_header {
        request = request.header("Authorization", header);
    }
    match request.send_empty() {
        Ok(resp) => {
            let body: String =
                resp.into_body()
                    .read_to_string()
                    .map_err(|e| RemoteHttpRequestError {
                        status_code: None,
                        message: format!("Failed to read presigned download response: {}", e),
                    })?;
            let parsed: serde_json::Value =
                serde_json::from_str(&body).map_err(|e| RemoteHttpRequestError {
                    status_code: None,
                    message: format!("Invalid presigned download response JSON: {}", e),
                })?;
            let download_url = parsed["download_url"]
                .as_str()
                .ok_or_else(|| RemoteHttpRequestError {
                    status_code: None,
                    message: "Presigned download response missing 'download_url' field".to_string(),
                })?
                .to_string();
            Ok(Some(download_url))
        }
        Err(ureq::Error::StatusCode(code)) if code == 404 || code == 501 => Ok(None),
        Err(e) => {
            let status = match e {
                ureq::Error::StatusCode(code) => Some(code),
                _ => None,
            };
            Err(RemoteHttpRequestError {
                status_code: status,
                message: format!(
                    "Failed to request presigned download URL from {}: {}",
                    download_endpoint,
                    format_remote_http_error(e)
                ),
            })
        }
    }
}

pub(crate) fn download_to_file(
    url: &str,
    output_path: &Path,
    authorization_header: Option<&str>,
) -> Result<u64, RemoteHttpRequestError> {
    let agent = http_agent_for_transfer();
    let mut request = agent.get(url);
    if let Some(header) = authorization_header {
        request = request.header("Authorization", header);
    }
    let mut response = request.call().map_err(|e| {
        let status = match e {
            ureq::Error::StatusCode(code) => Some(code),
            _ => None,
        };
        RemoteHttpRequestError {
            status_code: status,
            message: format!(
                "Failed to download .aimod from {}: {}",
                url,
                format_remote_http_error(e)
            ),
        }
    })?;

    let staged_path = staged_output_path(output_path, "download");
    let write_result = (|| -> Result<u64, RemoteHttpRequestError> {
        let file = File::create(&staged_path).map_err(|e| RemoteHttpRequestError {
            status_code: None,
            message: format!(
                "Failed to create staging file for pull {} -> {}: {}",
                url,
                staged_path.display(),
                e
            ),
        })?;
        let mut writer = BufWriter::new(file);
        let mut reader = response.body_mut().as_reader();
        let bytes_downloaded =
            std::io::copy(&mut reader, &mut writer).map_err(|e| RemoteHttpRequestError {
                status_code: None,
                message: format!(
                    "Failed to stream .aimod response body from {} to {}: {}",
                    url,
                    staged_path.display(),
                    e
                ),
            })?;
        writer.flush().map_err(|e| RemoteHttpRequestError {
            status_code: None,
            message: format!(
                "Failed to flush pulled .aimod to {}: {}",
                staged_path.display(),
                e
            ),
        })?;
        Ok(bytes_downloaded)
    })();
    let bytes_downloaded = match write_result {
        Ok(bytes_downloaded) => bytes_downloaded,
        Err(error) => {
            let _ = fs::remove_file(&staged_path);
            return Err(error);
        }
    };

    replace_output_file(&staged_path, output_path).map_err(|e| {
        let _ = fs::remove_file(&staged_path);
        RemoteHttpRequestError {
            status_code: None,
            message: format!(
                "Failed to finalize pulled .aimod {} -> {}: {}",
                staged_path.display(),
                output_path.display(),
                e
            ),
        }
    })?;

    Ok(bytes_downloaded)
}

pub(crate) fn pull_kapsl_from_http_remote(
    artifact_url: &str,
    authorization_header: Option<&str>,
    output_path: &Path,
) -> Result<u64, RemoteHttpRequestError> {
    // Try presigned URL flow first.
    if let Some(download_url) = request_presigned_download_url(artifact_url, authorization_header)?
    {
        eprintln!("  {}", Ansi::new().dim("Downloading..."));
        // No auth header needed — the SAS token is in the URL.
        return download_to_file(&download_url, output_path, None);
    }

    // Fallback: direct download from the API server.
    download_to_file(artifact_url, output_path, authorization_header)
}

#[derive(Debug, Clone)]
pub(crate) struct OrasAuth {
    pub(crate) username: String,
    pub(crate) password: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct KapslOciConfig {
    pub(crate) artifact_type: String,
    pub(crate) filename: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sha256: Option<String>,
    pub(crate) size: u64,
}

pub(crate) struct TempDirGuard {
    pub(crate) path: PathBuf,
}

impl TempDirGuard {
    pub(crate) fn new(prefix: &str) -> Result<Self, String> {
        Self::new_in(&std::env::temp_dir(), prefix)
    }

    pub(crate) fn new_in(parent: &Path, prefix: &str) -> Result<Self, String> {
        let dir = parent.join(format!(
            "{}-{}-{}",
            prefix,
            std::process::id(),
            temp_nonce()
        ));
        fs::create_dir_all(&dir).map_err(|e| {
            format!(
                "Failed to create temporary directory {}: {}",
                dir.display(),
                e
            )
        })?;
        Ok(Self { path: dir })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(crate) fn temp_nonce() -> String {
    let mut nonce_bytes = [0u8; 8];
    OsRng.fill_bytes(&mut nonce_bytes);
    hex_encode(&nonce_bytes)
}

pub(crate) fn staged_output_path(output_path: &Path, prefix: &str) -> PathBuf {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact.aimod");
    parent.join(format!(
        ".{}.{}-{}-{}.part",
        file_name,
        prefix,
        std::process::id(),
        temp_nonce()
    ))
}

pub(crate) fn replace_output_file(staged_path: &Path, output_path: &Path) -> std::io::Result<()> {
    if output_path.exists() {
        fs::remove_file(output_path)?;
    }
    fs::rename(staged_path, output_path)
}

pub(crate) fn stage_link_or_copy_file(
    source_path: &Path,
    output_path: &Path,
    prefix: &str,
) -> Result<u64, String> {
    if source_path == output_path {
        return fs::metadata(source_path)
            .map(|meta| meta.len())
            .map_err(|e| format!("Failed to stat {}: {}", source_path.display(), e));
    }

    let staged_path = staged_output_path(output_path, prefix);
    let stage_result = match fs::hard_link(source_path, &staged_path) {
        Ok(()) => fs::metadata(source_path)
            .map(|meta| meta.len())
            .map_err(|e| {
                format!(
                    "Failed to stat staged linked artifact {}: {}",
                    source_path.display(),
                    e
                )
            }),
        Err(_) => fs::copy(source_path, &staged_path).map_err(|e| {
            format!(
                "Failed to copy artifact {} to staging path {}: {}",
                source_path.display(),
                staged_path.display(),
                e
            )
        }),
    };

    let bytes = match stage_result {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_file(&staged_path);
            return Err(error);
        }
    };

    replace_output_file(&staged_path, output_path).map_err(|e| {
        let _ = fs::remove_file(&staged_path);
        format!(
            "Failed to finalize staged artifact {} -> {}: {}",
            staged_path.display(),
            output_path.display(),
            e
        )
    })?;

    Ok(bytes)
}

pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub(crate) fn is_oci_remote_url(remote_url: &str) -> bool {
    remote_url.trim().starts_with(OCI_REMOTE_PREFIX)
}

pub(crate) fn parse_oci_remote_prefix(remote_url: &str) -> Result<String, String> {
    let trimmed = remote_url.trim();
    if !trimmed.starts_with(OCI_REMOTE_PREFIX) {
        return Err(format!(
            "Invalid OCI remote URL (expected {}<registry>[/prefix]): {}",
            OCI_REMOTE_PREFIX, trimmed
        ));
    }
    let prefix = trimmed
        .trim_start_matches(OCI_REMOTE_PREFIX)
        .trim()
        .trim_end_matches('/')
        .to_string();
    if prefix.is_empty() {
        return Err(format!(
            "OCI remote URL is missing registry (expected {}<registry>[/prefix])",
            OCI_REMOTE_PREFIX
        ));
    }
    if prefix.contains("://") {
        return Err(format!(
            "OCI remote URL must be an OCI reference, not a URL: {}",
            prefix
        ));
    }
    if prefix.contains('@') {
        return Err(format!(
            "OCI remote URL must be a registry/prefix without digest, got: {}",
            prefix
        ));
    }

    let segments: Vec<&str> = prefix.split('/').collect();
    if segments.iter().any(|segment| segment.is_empty()) {
        return Err(format!(
            "OCI remote URL contains empty path segments, got: {}",
            prefix
        ));
    }

    for segment in segments.iter().skip(1) {
        if segment.contains(':') {
            return Err(format!(
                "OCI remote URL prefix path must not contain tags, got: {}",
                prefix
            ));
        }
    }

    Ok(prefix)
}

pub(crate) fn build_oci_repo_for_target(
    remote_url: &str,
    target: &ModelTargetRef,
) -> Result<String, String> {
    let prefix = parse_oci_remote_prefix(remote_url)?;
    Ok(format!("{}/{}/{}", prefix, target.repo, target.model))
}

pub(crate) fn oci_registry_for_repo(repo: &str) -> Result<String, String> {
    let registry = repo
        .split('/')
        .next()
        .ok_or_else(|| format!("Invalid OCI repository: {}", repo))?;
    if registry.trim().is_empty() {
        return Err(format!("Invalid OCI repository: {}", repo));
    }
    Ok(registry.to_string())
}

pub(crate) fn oras_bin() -> String {
    optional_env_var(ORAS_BIN_ENV).unwrap_or_else(|| "oras".to_string())
}

pub(crate) fn load_oras_auth_from_env() -> Result<Option<OrasAuth>, String> {
    let username = optional_env_var(OCI_USERNAME_ENV);
    let password = optional_env_var(OCI_PASSWORD_ENV);
    match (username, password) {
        (None, None) => Ok(None),
        (Some(username), Some(password)) => Ok(Some(OrasAuth { username, password })),
        _ => Err(format!(
            "OCI auth requires both {} and {} to be set (or neither).",
            OCI_USERNAME_ENV, OCI_PASSWORD_ENV
        )),
    }
}

pub(crate) fn ensure_oras_support(oras_bin: &str) -> Result<(), String> {
    let version = Command::new(oras_bin)
        .arg("version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            format!(
                "Failed to execute `{}`: {}. Install ORAS and ensure it is on PATH (or set {}).",
                oras_bin, e, ORAS_BIN_ENV
            )
        })?;
    if !version.status.success() {
        let stderr = String::from_utf8_lossy(&version.stderr);
        return Err(format!(
            "`{} version` failed (exit {}): {}",
            oras_bin,
            version.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }

    Ok(())
}

pub(crate) fn sha256_file_hex(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open file for sha256 {}: {}", path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("Failed to read file for sha256 {}: {}", path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(hex_encode(digest.as_slice()))
}

pub(crate) fn read_manifest_from_kapsl_archive(package_path: &Path) -> Result<Manifest, String> {
    let file = File::open(package_path).map_err(|e| {
        format!(
            "Failed to open .aimod to read metadata.json {}: {}",
            package_path.display(),
            e
        )
    })?;
    let tar = GzDecoder::new(file);
    let mut archive = Archive::new(tar);
    let entries = archive.entries().map_err(|e| {
        format!(
            "Failed to list .aimod archive entries for {}: {}",
            package_path.display(),
            e
        )
    })?;

    for entry in entries {
        let mut entry = entry.map_err(|e| {
            format!(
                "Failed to read .aimod archive entry for {}: {}",
                package_path.display(),
                e
            )
        })?;
        let path = entry.path().map_err(|e| {
            format!(
                "Failed to read .aimod entry path for {}: {}",
                package_path.display(),
                e
            )
        })?;
        if path.as_ref() != Path::new("metadata.json") {
            continue;
        }

        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|e| {
            format!(
                "Failed to read metadata.json from {}: {}",
                package_path.display(),
                e
            )
        })?;
        let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| {
            format!(
                "Failed to parse metadata.json in {}: {}",
                package_path.display(),
                e
            )
        })?;
        return Ok(manifest);
    }

    Err(format!(
        "Manifest not found in package (metadata.json missing): {}",
        package_path.display()
    ))
}

pub(crate) fn parse_oras_manifest_digest(output: &str) -> Option<String> {
    let mut last_digest = None;
    for line in output.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("digest:") {
            if let Some(idx) = lower.find("sha256:") {
                let candidate = &trimmed[idx..];
                if candidate.len() >= "sha256:".len() + 64
                    && candidate["sha256:".len()..]
                        .chars()
                        .take(64)
                        .all(|ch| ch.is_ascii_hexdigit())
                {
                    return Some(candidate[..("sha256:".len() + 64)].to_string());
                }
            }
        }
        if let Some(idx) = lower.find("sha256:") {
            let candidate = &trimmed[idx..];
            if candidate.len() >= "sha256:".len() + 64
                && candidate["sha256:".len()..]
                    .chars()
                    .take(64)
                    .all(|ch| ch.is_ascii_hexdigit())
            {
                last_digest = Some(candidate[..("sha256:".len() + 64)].to_string());
            }
        }
    }
    last_digest
}

pub(crate) fn oras_login(
    oras_bin: &str,
    registry: &str,
    auth: &OrasAuth,
    docker_config_dir: Option<&Path>,
) -> Result<(), String> {
    let mut cmd = Command::new(oras_bin);
    cmd.arg("login")
        .arg(registry)
        .arg("--username")
        .arg(&auth.username)
        .arg("--password-stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = docker_config_dir {
        cmd.env("DOCKER_CONFIG", dir);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start `oras login`: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(auth.password.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .map_err(|e| format!("Failed to write `oras login` password: {}", e))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for `oras login`: {}", e))?;
    if !output.status.success() {
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(format!(
            "`oras login` failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            combined.trim()
        ));
    }

    Ok(())
}

pub(crate) fn oras_push_kapsl(
    oras_bin: &str,
    reference: &str,
    kapsl_path: &Path,
    config_path: &Path,
    annotations: &[(String, String)],
    docker_config_dir: Option<&Path>,
) -> Result<Option<String>, String> {
    let config_spec = format!("{}:{}", config_path.display(), KAPSL_OCI_CONFIG_TYPE);
    let layer_spec = format!("{}:{}", kapsl_path.display(), KAPSL_OCI_LAYER_TYPE);

    let mut cmd = Command::new(oras_bin);
    cmd.arg("push")
        .arg("--artifact-type")
        .arg(KAPSL_OCI_ARTIFACT_TYPE)
        .arg("--config")
        .arg(&config_spec);
    for (key, value) in annotations {
        cmd.arg("--annotation").arg(format!("{}={}", key, value));
    }
    cmd.arg(reference)
        .arg(&layer_spec)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = docker_config_dir {
        cmd.env("DOCKER_CONFIG", dir);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute `oras push`: {}", e))?;
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return Err(format!(
            "`oras push` failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            combined.trim()
        ));
    }
    Ok(parse_oras_manifest_digest(&combined))
}

pub(crate) fn oras_pull(
    oras_bin: &str,
    reference: &str,
    output_dir: &Path,
    docker_config_dir: Option<&Path>,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|e| {
        format!(
            "Failed to create OCI pull output dir {}: {}",
            output_dir.display(),
            e
        )
    })?;
    let mut cmd = Command::new(oras_bin);
    cmd.arg("pull")
        .arg(reference)
        .arg("--output")
        .arg(output_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = docker_config_dir {
        cmd.env("DOCKER_CONFIG", dir);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute `oras pull`: {}", e))?;
    if !output.status.success() {
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(format!(
            "`oras pull` failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            combined.trim()
        ));
    }
    Ok(())
}

pub(crate) fn build_oci_reference(
    repo: &str,
    tag: &str,
    reference_override: Option<&str>,
) -> Result<String, String> {
    if let Some(reference_override) = reference_override {
        let trimmed = reference_override.trim();
        if trimmed.is_empty() {
            return Err("OCI reference override cannot be empty".to_string());
        }

        if trimmed.starts_with("sha256:") {
            return Ok(format!("{}@{}", repo, trimmed));
        }
        if trimmed.starts_with("@sha256:") {
            return Ok(format!("{}{}", repo, trimmed));
        }
        if trimmed.contains("@sha256:") {
            return Ok(trimmed.to_string());
        }

        return Err(
            "Invalid OCI reference override. Expected `sha256:<digest>`, `@sha256:<digest>`, or `<repo>@sha256:<digest>`."
                .to_string(),
        );
    }

    Ok(format!("{}:{}", repo, tag))
}

pub(crate) fn push_kapsl_to_oci_remote(
    remote_url: &str,
    absolute_path: &Path,
    target: &ModelTargetRef,
    filename: &str,
) -> Result<PushKapslResponse, String> {
    let oras = oras_bin();
    ensure_oras_support(&oras)?;

    let remote_url = remote_url.trim().trim_end_matches('/').to_string();
    let repo = build_oci_repo_for_target(&remote_url, target)?;
    let reference = format!("{}:{}", repo, target.label);

    let bytes_uploaded = fs::metadata(absolute_path)
        .map_err(|e| format!("Failed to stat .aimod {}: {}", absolute_path.display(), e))?
        .len();
    let sha256 = if parse_env_bool(OCI_PRECOMPUTE_SHA256_ENV).unwrap_or(false) {
        Some(sha256_file_hex(absolute_path)?)
    } else {
        None
    };

    let manifest = read_manifest_from_kapsl_archive(absolute_path).ok();

    let mut annotations = Vec::new();
    annotations.push(("io.kapsl.aimod.target".to_string(), target.as_string()));
    annotations.push(("io.kapsl.aimod.repo".to_string(), target.repo.clone()));
    annotations.push(("io.kapsl.aimod.model".to_string(), target.model.clone()));
    annotations.push(("io.kapsl.aimod.label".to_string(), target.label.clone()));
    annotations.push(("io.kapsl.aimod.filename".to_string(), filename.to_string()));
    annotations.push((
        "io.kapsl.aimod.size".to_string(),
        bytes_uploaded.to_string(),
    ));
    if let Some(sha256) = sha256.as_ref() {
        annotations.push(("io.kapsl.aimod.sha256".to_string(), sha256.clone()));
    }
    if let Some(manifest) = manifest.as_ref() {
        annotations.push((
            "io.kapsl.aimod.project_name".to_string(),
            manifest.project_name.clone(),
        ));
        annotations.push((
            "io.kapsl.aimod.framework".to_string(),
            manifest.framework.clone(),
        ));
        annotations.push((
            "io.kapsl.aimod.version".to_string(),
            manifest.version.clone(),
        ));
        annotations.push((
            "io.kapsl.aimod.created_at".to_string(),
            manifest.created_at.clone(),
        ));
        annotations.push((
            "org.opencontainers.image.title".to_string(),
            manifest.project_name.clone(),
        ));
        annotations.push((
            "org.opencontainers.image.version".to_string(),
            manifest.version.clone(),
        ));
        annotations.push((
            "org.opencontainers.image.created".to_string(),
            manifest.created_at.clone(),
        ));
    } else {
        annotations.push((
            "org.opencontainers.image.title".to_string(),
            filename.to_string(),
        ));
    }

    let config = KapslOciConfig {
        artifact_type: KAPSL_OCI_ARTIFACT_TYPE.to_string(),
        filename: filename.to_string(),
        sha256,
        size: bytes_uploaded,
    };

    let temp_dir = TempDirGuard::new("kapsl-oci")?;
    let config_path = temp_dir.path().join("kapsl-config.json");
    let config_bytes = serde_json::to_vec(&config)
        .map_err(|e| format!("Failed to encode OCI config JSON: {}", e))?;
    fs::write(&config_path, &config_bytes).map_err(|e| {
        format!(
            "Failed to write OCI config JSON to {}: {}",
            config_path.display(),
            e
        )
    })?;

    let auth = load_oras_auth_from_env()?;
    let docker_config_dir = if auth.is_some() {
        let docker_dir = temp_dir.path().join("docker-config");
        fs::create_dir_all(&docker_dir).map_err(|e| {
            format!(
                "Failed to create docker config directory {}: {}",
                docker_dir.display(),
                e
            )
        })?;
        Some(docker_dir)
    } else {
        None
    };

    if let Some(auth) = auth.as_ref() {
        let registry = oci_registry_for_repo(&repo)?;
        oras_login(&oras, &registry, auth, docker_config_dir.as_deref())?;
    }

    let manifest_digest = oras_push_kapsl(
        &oras,
        &reference,
        absolute_path,
        &config_path,
        &annotations,
        docker_config_dir.as_deref(),
    )?;

    let artifact_url = format!("{}{}", OCI_REMOTE_PREFIX, reference);

    Ok(PushKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url: artifact_url.clone(),
        mirrored_path: artifact_url,
        bytes_uploaded,
        manifest_digest,
    })
}

pub(crate) fn pull_kapsl_from_oci_remote(
    remote_url: &str,
    target: &ModelTargetRef,
    reference_override: Option<&str>,
    destination_dir: &Path,
) -> Result<PullKapslResponse, String> {
    let oras = oras_bin();
    ensure_oras_support(&oras)?;

    let remote_url = remote_url.trim().trim_end_matches('/').to_string();
    let repo = build_oci_repo_for_target(&remote_url, target)?;
    let reference = build_oci_reference(&repo, &target.label, reference_override)?;
    let filename = format!("{}.aimod", target.model);

    fs::create_dir_all(destination_dir).map_err(|e| {
        format!(
            "Failed to create destination directory {}: {}",
            destination_dir.display(),
            e
        )
    })?;
    let temp_dir = TempDirGuard::new_in(destination_dir, ".kapsl-oci-pull")?;
    let auth = load_oras_auth_from_env()?;
    let docker_config_dir = if auth.is_some() {
        let docker_dir = temp_dir.path().join("docker-config");
        fs::create_dir_all(&docker_dir).map_err(|e| {
            format!(
                "Failed to create docker config directory {}: {}",
                docker_dir.display(),
                e
            )
        })?;
        Some(docker_dir)
    } else {
        None
    };

    if let Some(auth) = auth.as_ref() {
        let registry = oci_registry_for_repo(&repo)?;
        oras_login(&oras, &registry, auth, docker_config_dir.as_deref())?;
    }

    oras_pull(
        &oras,
        &reference,
        temp_dir.path(),
        docker_config_dir.as_deref(),
    )?;

    let expected = temp_dir.path().join(&filename);
    let pulled_path = if expected.exists() {
        expected
    } else {
        let mut kapsls = Vec::new();
        let entries = fs::read_dir(temp_dir.path()).map_err(|e| {
            format!(
                "Failed to read OCI pull output dir {}: {}",
                temp_dir.path().display(),
                e
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read OCI pull dir entry: {}", e))?;
            let path = entry.path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("aimod"))
                .unwrap_or(false)
            {
                kapsls.push(path);
            }
        }

        kapsls.sort();
        if kapsls.len() == 1 {
            kapsls.remove(0)
        } else if kapsls.is_empty() {
            return Err(
                "OCI pull succeeded but no .aimod file was found in the pulled artifact."
                    .to_string(),
            );
        } else {
            return Err(format!(
                "OCI pull produced multiple .aimod files; expected one. Files: {}",
                kapsls
                    .iter()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    };

    let output_path = destination_dir.join(&filename);
    replace_output_file(&pulled_path, &output_path).map_err(|e| {
        format!(
            "Failed to move pulled .aimod to {}: {}",
            output_path.display(),
            e
        )
    })?;

    let bytes_downloaded = fs::metadata(&output_path)
        .map_err(|e| {
            format!(
                "Failed to stat pulled .aimod {}: {}",
                output_path.display(),
                e
            )
        })?
        .len();

    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);
    let artifact_url = format!("{}{}", OCI_REMOTE_PREFIX, reference);

    Ok(PullKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        bytes_downloaded,
    })
}

pub(crate) fn infer_framework_from_model_path(model_path: &Path) -> String {
    let ext = model_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "onnx" => "onnx".to_string(),
        "gguf" => "gguf".to_string(),
        "safetensors" => "pytorch".to_string(),
        "pt" | "pth" => "pytorch".to_string(),
        "pb" => "tensorflow".to_string(),
        _ => "onnx".to_string(),
    }
}

pub(crate) fn looks_like_model_file_path(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "onnx" | "gguf" | "safetensors" | "pt" | "pth" | "pb"
    )
}

/// Resolve a model path to a `PackageLoader`.
///
/// Priority:
/// 1. Known single-file extension (`.gguf`, `.onnx`, `.safetensors` …) →
///    `from_raw_file`
/// 2. Directory with safetensors shards → `from_directory`
/// 3. `.aimod` archive → `PackageLoader::load`
pub(crate) fn resolve_package_loader(
    path: &Path,
    model_id: impl std::fmt::Display,
) -> Result<PackageLoader, Box<dyn std::error::Error + Send + Sync>> {
    if looks_like_model_file_path(path) {
        return PackageLoader::from_raw_file(path)
            .map_err(|e| format!("Failed to load raw model {model_id}: {e}").into());
    }
    if path.is_dir() {
        return PackageLoader::from_directory(path)
            .map_err(|e| format!("Failed to load model directory {model_id}: {e}").into());
    }
    PackageLoader::load(path).map_err(|e| format!("Failed to load model {model_id}: {e}").into())
}

pub(crate) fn append_tar_bytes_entry<W: Write>(
    builder: &mut Builder<W>,
    entry_path: &str,
    bytes: &[u8],
) -> Result<(), String> {
    let mut header = tar::Header::new_gnu();
    header
        .set_path(entry_path)
        .map_err(|e| format!("Failed to set tar path {}: {}", entry_path, e))?;
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append(&header, bytes)
        .map_err(|e| format!("Failed to append {} to archive: {}", entry_path, e))
}

pub(crate) fn create_kapsl_package(
    request: &PackageKapslRequest,
    interactive_metadata_setup: bool,
) -> Result<PackageKapslResponse, String> {
    let input_model_path = PathBuf::from(request.model_path.trim());
    if !input_model_path.exists() {
        return Err(format!(
            "Model file does not exist: {}",
            input_model_path.display()
        ));
    }

    if !input_model_path.is_file() {
        return Err(format!(
            "Model path must be a file: {}",
            input_model_path.display()
        ));
    }

    let model_path = input_model_path.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve model file path {}: {}",
            input_model_path.display(),
            e
        )
    })?;

    let model_file_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid model filename: {}", model_path.display()))?
        .to_string();

    let mut project_name = request
        .project_name
        .as_ref()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| {
            model_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "kapsl-model".to_string());

    let mut framework = request
        .framework
        .as_ref()
        .map(|framework| framework.trim())
        .filter(|framework| !framework.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| infer_framework_from_model_path(&model_path));

    let mut version = request
        .version
        .as_ref()
        .map(|version| version.trim())
        .filter(|version| !version.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "1.0.0".to_string());

    let mut hardware_requirements = kapsl_core::HardwareRequirements::default();

    // Orthogonal model axes (format / model_type / task). Populated from CLI
    // overrides and/or the interactive flow; left None otherwise so the loader
    // infers them from `framework`. See kapsl_core::EngineKind.
    let mut format_axis: Option<String> = None;
    let mut model_type_axis: Option<String> = None;
    let mut task_axis: Option<String> = None;

    // Non-interactive: --format / --model-type / --task fill the axes (and a
    // consistent legacy `framework`) without prompting.
    let axes = AxisOverrides {
        format: request.format.as_deref(),
        model_type: request.model_type.as_deref(),
        task: request.task.as_deref(),
    };
    if axes.any() {
        let (fmt, mt, tk, fw) =
            resolve_axis_triple(infer_format_from_model_path(&model_path), axes);
        framework = fw;
        format_axis = Some(fmt);
        model_type_axis = Some(mt);
        task_axis = Some(tk);
    }

    // The model file's directory is where we persist a generated metadata.json so
    // a subsequent build can be reproduced without re-prompting.
    let source_metadata_path = model_path
        .parent()
        .map(|parent| parent.join("metadata.json"))
        .unwrap_or_else(|| PathBuf::from("metadata.json"));
    let should_create_source_metadata = !source_metadata_path.exists();

    if should_create_source_metadata && interactive_metadata_setup {
        let a = Ansi::new();
        eprintln!(
            "{}",
            a.bold("No metadata.json found. Let's create one for this model.")
        );
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        project_name = prompt_non_empty_with_default(&mut reader, "Project name", &project_name)?;
        let format_default = format_axis
            .as_deref()
            .unwrap_or_else(|| infer_format_from_model_path(&model_path));
        let format =
            prompt_select_with_default(&mut reader, "Format", FORMAT_OPTIONS, format_default)?;
        let model_type_default = model_type_axis
            .as_deref()
            .unwrap_or_else(|| default_model_type_for_format(&format));
        let model_type = prompt_select_with_default(
            &mut reader,
            "Model type",
            MODEL_TYPE_OPTIONS,
            model_type_default,
        )?;
        let task = prompt_task_for_model_type(&mut reader, &model_type, task_axis.as_deref())?;
        framework = legacy_framework_for(&format, &model_type, &task);
        format_axis = Some(format);
        model_type_axis = Some(model_type);
        task_axis = Some(task);
        version = prompt_non_empty_with_default(&mut reader, "Version", &version)?;
        hardware_requirements.preferred_provider = prompt_provider_with_default(
            &mut reader,
            hardware_requirements.preferred_provider.as_deref(),
        )?;
        eprintln!();
    }

    let package = move || -> Result<PackageKapslResponse, String> {
        let mut output_path = request
            .output_path
            .as_ref()
            .map(|path| PathBuf::from(path.trim()))
            .unwrap_or_else(|| PathBuf::from(format!("{}.aimod", project_name)));

        if output_path.extension().and_then(|ext| ext.to_str()) != Some("aimod") {
            output_path.set_extension("aimod");
        }

        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create parent directory {}: {}",
                        parent.display(),
                        e
                    )
                })?;
            }
        }

        let created_at = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System clock error: {}", e))?
            .as_secs();

        let metadata = match request.metadata.as_ref() {
            Some(metadata) => Some(
                serde_yaml::to_value(metadata)
                    .map_err(|e| format!("Failed to convert metadata payload: {}", e))?,
            ),
            None => None,
        };

        let manifest = Manifest {
            project_name: project_name.clone(),
            framework: framework.clone(),
            version: version.clone(),
            created_at: created_at.to_string(),
            model_file: model_file_name.clone(),
            format: format_axis,
            model_type: model_type_axis,
            task: task_axis,
            metadata,
            hardware_requirements,
            cron_jobs: Vec::new(),
        };
        EngineKind::validate(&manifest)?;

        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| format!("Failed to encode metadata.json: {}", e))?;
        let output_file = File::create(&output_path).map_err(|e| {
            format!(
                "Failed to create output package {}: {}",
                output_path.display(),
                e
            )
        })?;
        // Model weights are binary float data — nearly incompressible. Use level 1
        // (fast) instead of the default level 6 to avoid burning CPU for no gain.
        // The 8 MiB BufWriter reduces syscall overhead from one call per 32 KiB
        // GzEncoder output chunk to one call per 8 MiB.
        let encoder = GzEncoder::new(
            BufWriter::with_capacity(8 << 20, output_file),
            Compression::fast(),
        );
        let mut archive = Builder::new(encoder);

        append_tar_bytes_entry(&mut archive, "metadata.json", &manifest_bytes)?;

        archive
            .append_path_with_name(&model_path, &model_file_name)
            .map_err(|e| format!("Failed to add model file to archive: {}", e))?;

        let encoder = archive
            .into_inner()
            .map_err(|e| format!("Failed to finalize tar archive: {}", e))?;
        let mut buf_writer = encoder
            .finish()
            .map_err(|e| format!("Failed to finalize gzip stream: {}", e))?;
        buf_writer
            .flush()
            .map_err(|e| format!("Failed to flush output package: {}", e))?;

        let created_metadata_path = if should_create_source_metadata {
            create_source_metadata_if_missing(&source_metadata_path, &manifest_bytes)?
        } else {
            None
        };

        let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);

        Ok(PackageKapslResponse {
            status: "ok".to_string(),
            kapsl_path: absolute_output_path.to_string_lossy().to_string(),
            project_name,
            framework,
            version,
            metadata_path: created_metadata_path.map(|path| path.to_string_lossy().to_string()),
        })
    };

    // When prompting was shown above, the caller suppressed its spinner so the
    // prompts stayed legible. Drive the spinner here, around the actual packaging.
    if interactive_metadata_setup {
        run_with_loading("Building package", package)
    } else {
        package()
    }
}

pub(crate) fn find_model_file_in_context(context_dir: &Path) -> Result<PathBuf, String> {
    let mut stack = vec![context_dir.to_path_buf()];
    let mut matches = Vec::new();
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read context directory {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|e| format!("Failed to inspect {}: {}", path.display(), e))?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if matches!(
                ext.as_str(),
                "onnx" | "gguf" | "safetensors" | "pt" | "pth" | "pb"
            ) {
                matches.push(path);
            }
        }
    }

    matches.sort();
    if matches.is_empty() {
        return Err(format!(
            "No model file found in context {}. Pass --model explicitly.",
            context_dir.display()
        ));
    }
    if matches.len() > 1 {
        return Err(format!(
            "Multiple model files found in context {}. Pass --model explicitly.",
            context_dir.display()
        ));
    }
    Ok(matches.remove(0))
}

pub(crate) type ContextManifest = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<kapsl_core::HardwareRequirements>,
    Option<serde_json::Value>,
);

pub(crate) fn parse_context_manifest(context_dir: &Path) -> Result<ContextManifest, String> {
    let metadata_path = context_dir.join("metadata.json");
    if !metadata_path.exists() {
        return Ok((None, None, None, None, None, None));
    }

    let raw = fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Failed to read {}: {}", metadata_path.display(), e))?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "Invalid metadata.json in {}: {}",
            metadata_path.display(),
            e
        )
    })?;
    let Some(obj) = value.as_object() else {
        return Err(format!(
            "metadata.json in {} must be a JSON object",
            metadata_path.display()
        ));
    };

    let project_name = obj
        .get("project_name")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let framework = obj
        .get("framework")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let version = obj
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let model_file = obj
        .get("model_file")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let hardware_requirements = obj
        .get("hardware_requirements")
        .cloned()
        .map(serde_json::from_value::<kapsl_core::HardwareRequirements>)
        .transpose()
        .map_err(|e| format!("Invalid hardware_requirements in metadata.json: {}", e))?;

    let metadata = obj.get("metadata").cloned();

    Ok((
        project_name,
        framework,
        version,
        model_file,
        hardware_requirements,
        metadata,
    ))
}

pub(crate) fn create_source_metadata_if_missing(
    metadata_path: &Path,
    manifest_bytes: &[u8],
) -> Result<Option<PathBuf>, String> {
    let mut file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(metadata_path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(None),
        Err(error) => {
            return Err(format!(
                "Failed to create {}: {}",
                metadata_path.display(),
                error
            ));
        }
    };
    file.write_all(manifest_bytes).map_err(|e| {
        format!(
            "Failed to write generated metadata.json {}: {}",
            metadata_path.display(),
            e
        )
    })?;
    Ok(Some(metadata_path.to_path_buf()))
}

pub(crate) fn prompt_with_default(
    reader: &mut impl BufRead,
    label: &str,
    default: &str,
) -> Result<String, String> {
    let a = Ansi::new();
    eprint!(
        "{} {} {}: ",
        a.teal("?"),
        label,
        a.dim(&format!("({})", default))
    );
    std::io::stderr()
        .flush()
        .map_err(|e| format!("Failed to flush prompt: {}", e))?;

    let mut input = String::new();
    let bytes = reader
        .read_line(&mut input)
        .map_err(|e| format!("Failed to read prompt input: {}", e))?;
    // read_line returns 0 only at end of input. Treat that as an abort rather
    // than as "use the default", so a closed stdin (or Ctrl-D) can never spin a
    // retry loop forever.
    if bytes == 0 {
        return Err("unexpected end of input while reading prompt".to_string());
    }
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

pub(crate) fn prompt_non_empty_with_default(
    reader: &mut impl BufRead,
    label: &str,
    default: &str,
) -> Result<String, String> {
    loop {
        let value = prompt_with_default(reader, label, default)?;
        if !value.trim().is_empty() {
            return Ok(value.trim().to_string());
        }
        eprintln!("  {}", Ansi::new().red("Value cannot be empty."));
    }
}

/// (value, description) pairs shown in the interactive selection menus.
pub(crate) const FORMAT_OPTIONS: &[(&str, &str)] = &[
    ("onnx", "ONNX graph"),
    ("gguf", "GGUF file — tokenizer embedded in the file"),
    ("safetensors", "safetensors weights (custom kernels)"),
];
pub(crate) const MODEL_TYPE_OPTIONS: &[(&str, &str)] = &[
    ("causal-lm", "autoregressive LLM (text generation)"),
    ("embedding", "embedding / encoder model"),
    ("seq-classifier", "sequence classifier"),
    ("seq2seq", "encoder-decoder (seq2seq)"),
    ("opaque", "raw graph — run as-is, tensors in/out"),
];
pub(crate) const TASK_OPTIONS_CAUSAL_LM: &[(&str, &str)] = &[
    ("generate", "autoregressive text generation"),
    ("embed", "embeddings from hidden states"),
    ("forward", "raw forward pass"),
];
pub(crate) const TASK_OPTIONS_SEQ2SEQ: &[(&str, &str)] = &[
    ("generate", "sequence generation"),
    ("forward", "raw forward pass"),
];

/// Model file format inferred from a model path's extension, constrained to the
/// known `format` vocabulary (`onnx`/`gguf`/`safetensors`).
pub(crate) fn infer_format_from_model_path(model_path: &Path) -> &'static str {
    match model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "gguf" => "gguf",
        "safetensors" => "safetensors",
        _ => "onnx",
    }
}

/// Default model type for a format when the user hasn't said otherwise.
pub(crate) fn default_model_type_for_format(format: &str) -> &'static str {
    match format {
        "gguf" => "causal-lm",
        _ => "opaque",
    }
}

/// Default serving task for a model type.
pub(crate) fn default_task_for_model_type(model_type: &str) -> &'static str {
    match model_type {
        "causal-lm" => "generate",
        "embedding" => "embed",
        "seq-classifier" => "classify",
        "seq2seq" => "generate",
        _ => "forward",
    }
}

/// The selectable tasks for a model type. A single-element slice means the task
/// is fixed and is not prompted for.
pub(crate) fn task_options_for_model_type(
    model_type: &str,
) -> &'static [(&'static str, &'static str)] {
    match model_type {
        "causal-lm" => TASK_OPTIONS_CAUSAL_LM,
        "seq2seq" => TASK_OPTIONS_SEQ2SEQ,
        "embedding" => &[("embed", "embeddings")],
        "seq-classifier" => &[("classify", "classification")],
        _ => &[("forward", "raw forward pass")],
    }
}

/// Legacy `framework` value equivalent to a `(format, model_type, task)` triple,
/// kept so packages stay loadable by readers that predate the split.
pub(crate) fn legacy_framework_for(format: &str, _model_type: &str, task: &str) -> String {
    match format {
        "gguf" => "gguf",
        "safetensors" => "safetensors",
        "onnx" if task == "generate" => "llm",
        _ => "onnx",
    }
    .to_string()
}

/// CLI overrides for the orthogonal model axes (`--format` / `--model-type` /
/// `--task`).
#[derive(Default, Clone, Copy)]
pub(crate) struct AxisOverrides<'a> {
    pub(crate) format: Option<&'a str>,
    pub(crate) model_type: Option<&'a str>,
    pub(crate) task: Option<&'a str>,
}

impl AxisOverrides<'_> {
    fn any(&self) -> bool {
        self.format.is_some() || self.model_type.is_some() || self.task.is_some()
    }
}

/// Resolve the full `(format, model_type, task, framework)` from optional axis
/// overrides, filling unspecified axes with format-derived defaults. The legacy
/// `framework` is derived to stay consistent with the chosen axes.
pub(crate) fn resolve_axis_triple(
    default_format: &str,
    axes: AxisOverrides,
) -> (String, String, String, String) {
    let pick = |o: Option<&str>| {
        o.map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let format = pick(axes.format).unwrap_or_else(|| default_format.to_string());
    let task_hint = pick(axes.task);
    // When only a task is given, infer the model type it implies so e.g.
    // `--task embed` alone is a coherent (embedding, embed) pair rather than
    // (opaque, embed).
    let model_type = pick(axes.model_type).unwrap_or_else(|| {
        match task_hint.as_deref() {
            Some("embed") => "embedding",
            Some("classify") => "seq-classifier",
            Some("generate") => "causal-lm",
            _ => default_model_type_for_format(&format),
        }
        .to_string()
    });
    let task = task_hint.unwrap_or_else(|| default_task_for_model_type(&model_type).to_string());
    let framework = legacy_framework_for(&format, &model_type, &task);
    (format, model_type, task, framework)
}

/// Prompt for the serving task, skipping the prompt when the model type allows
/// only one task.
pub(crate) fn prompt_task_for_model_type(
    reader: &mut impl BufRead,
    model_type: &str,
    preferred: Option<&str>,
) -> Result<String, String> {
    let options = task_options_for_model_type(model_type);
    // Use a preferred task (e.g. from --task) as the default when it's valid for
    // this model type, else the model type's natural default.
    let default = preferred
        .map(str::trim)
        .filter(|p| options.iter().any(|(v, _)| v.eq_ignore_ascii_case(p)))
        .unwrap_or_else(|| default_task_for_model_type(model_type));
    if options.len() <= 1 {
        return Ok(default.to_string());
    }
    prompt_select_with_default(reader, "Task", options, default)
}

pub(crate) const PROVIDER_OPTIONS: &[(&str, &str)] = &[
    ("cpu", "CPU execution"),
    ("cuda", "NVIDIA GPU (CUDA)"),
    ("tensorrt", "NVIDIA TensorRT"),
    ("coreml", "Apple CoreML / Metal"),
    ("rocm", "AMD GPU (ROCm)"),
    ("directml", "Windows DirectML"),
];

/// Prompt the user to pick one of `options` (each a `(value, description)` pair).
/// Accepts the option's number or its value (case-insensitive); an empty line
/// keeps `default`. Re-prompts on an invalid choice so the returned value is
/// always one of `options`.
pub(crate) fn prompt_select_with_default(
    reader: &mut impl BufRead,
    label: &str,
    options: &[(&str, &str)],
    default: &str,
) -> Result<String, String> {
    let a = Ansi::new();
    let default_index = options
        .iter()
        .position(|(value, _)| value.eq_ignore_ascii_case(default));
    let name_width = options
        .iter()
        .map(|(value, _)| value.len())
        .max()
        .unwrap_or(0);

    eprintln!("{} {}:", a.teal("?"), label);
    for (index, (value, description)) in options.iter().enumerate() {
        let suffix = if Some(index) == default_index {
            " (default)"
        } else {
            ""
        };
        eprintln!(
            "    {}) {:<width$}  {}",
            index + 1,
            value,
            a.dim(&format!("{}{}", description, suffix)),
            width = name_width
        );
    }

    loop {
        eprint!("  {} {}: ", a.teal("›"), a.dim(&format!("[{}]", default)));
        std::io::stderr()
            .flush()
            .map_err(|e| format!("Failed to flush prompt: {}", e))?;

        let mut input = String::new();
        reader
            .read_line(&mut input)
            .map_err(|e| format!("Failed to read prompt input: {}", e))?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return Ok(default.to_string());
        }
        if let Ok(choice) = trimmed.parse::<usize>() {
            if choice >= 1 && choice <= options.len() {
                return Ok(options[choice - 1].0.to_string());
            }
        }
        if let Some((value, _)) = options
            .iter()
            .find(|(value, _)| value.eq_ignore_ascii_case(trimmed))
        {
            return Ok(value.to_string());
        }
        eprintln!(
            "  {}",
            a.red("Enter a number from the list or one of the option names.")
        );
    }
}

pub(crate) fn prompt_provider_with_default(
    reader: &mut impl BufRead,
    default: Option<&str>,
) -> Result<Option<String>, String> {
    let default = default
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("cpu");
    let value =
        prompt_select_with_default(reader, "Preferred provider", PROVIDER_OPTIONS, default)?;
    Ok(Some(value))
}

pub(crate) fn prompt_model_file_with_default(
    reader: &mut impl BufRead,
    context_dir: &Path,
    default: &str,
) -> Result<PathBuf, String> {
    // Compare against the canonicalized context dir: `canonicalize()` below
    // resolves symlinks (e.g. macOS /var -> /private/var), so the bound must be
    // resolved too or every valid in-context file is wrongly rejected.
    let context_canonical = context_dir
        .canonicalize()
        .unwrap_or_else(|_| context_dir.to_path_buf());
    loop {
        let value = prompt_non_empty_with_default(reader, "Model file", default)?;
        let candidate = context_dir.join(&value);
        if candidate.exists() && candidate.is_file() {
            let canonical = candidate.canonicalize().map_err(|e| {
                format!(
                    "Failed to resolve model file path {}: {}",
                    candidate.display(),
                    e
                )
            })?;
            if canonical.starts_with(&context_canonical) {
                return Ok(canonical);
            }
        }
        eprintln!(
            "  {}",
            Ansi::new().red("Model file must exist inside the build context.")
        );
    }
}

pub(crate) fn normalize_output_path_for_context(
    context_dir: &Path,
    output: Option<&Path>,
    project_name: &str,
) -> PathBuf {
    let mut output_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| context_dir.join(format!("{}.aimod", project_name)));
    if output_path.extension().and_then(|v| v.to_str()) != Some("aimod") {
        output_path.set_extension("aimod");
    }
    output_path
}

pub(crate) fn collect_existing_file_references_from_metadata(
    context_dir: &Path,
    value: &serde_yaml::Value,
    out: &mut HashSet<PathBuf>,
) {
    match value {
        serde_yaml::Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return;
            }
            let candidate = context_dir.join(trimmed);
            if !candidate.exists() || !candidate.is_file() {
                return;
            }
            if let Ok(relative) = candidate.strip_prefix(context_dir) {
                out.insert(relative.to_path_buf());
            }
        }
        serde_yaml::Value::Sequence(seq) => {
            for item in seq {
                collect_existing_file_references_from_metadata(context_dir, item, out);
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (_, item) in map {
                collect_existing_file_references_from_metadata(context_dir, item, out);
            }
        }
        _ => {}
    }
}

pub(crate) fn collect_context_files(
    context_dir: &Path,
    output_path: &Path,
    primary_model_ext: &str,
    keep_primary_model_files: &HashSet<PathBuf>,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let output_canonical = output_path.canonicalize().ok();
    let output_in_context_relative = output_path
        .strip_prefix(context_dir)
        .ok()
        .map(PathBuf::from);

    let primary_model_ext = primary_model_ext.trim().to_ascii_lowercase();
    let mut keep_onnx_stems_by_dir: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    if primary_model_ext == "onnx" {
        for rel in keep_primary_model_files {
            if rel
                .extension()
                .and_then(|v| v.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("onnx"))
                .unwrap_or(false)
            {
                let dir = rel.parent().unwrap_or(Path::new("")).to_path_buf();
                let stem = rel
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .to_string();
                if !stem.is_empty() {
                    keep_onnx_stems_by_dir.entry(dir).or_default().insert(stem);
                }
            }
        }
    }

    // First collect all file paths and (when applicable) the ONNX stems per directory, so we can
    // reliably decide which *.onnx_data* files belong to which model file.
    let mut all_files = Vec::new();
    let mut onnx_stems_by_dir: HashMap<PathBuf, Vec<String>> = HashMap::new();
    let mut primary_model_count_by_dir: HashMap<PathBuf, usize> = HashMap::new();
    let mut stack = vec![context_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read context directory {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|e| format!("Failed to inspect {}: {}", path.display(), e))?;
            if file_type.is_dir() {
                let rel = path
                    .strip_prefix(context_dir)
                    .map_err(|e| format!("Failed to resolve dir path {}: {}", path.display(), e))?
                    .to_path_buf();
                if rel.components().any(|c| {
                    matches!(
                        c.as_os_str().to_str(),
                        Some(".git" | "__pycache__" | ".pytest_cache")
                    )
                }) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            let relative = path
                .strip_prefix(context_dir)
                .map_err(|e| format!("Failed to resolve file path {}: {}", path.display(), e))?
                .to_path_buf();

            let ext = relative
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            if !primary_model_ext.is_empty() && ext == primary_model_ext {
                let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                *primary_model_count_by_dir.entry(dir_rel).or_insert(0) += 1;
            }

            if primary_model_ext == "onnx" && ext == "onnx" {
                let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                let stem = relative
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .to_string();
                if !stem.is_empty() {
                    onnx_stems_by_dir.entry(dir_rel).or_default().push(stem);
                }
            }

            all_files.push((path, relative));
        }
    }

    let mut files = Vec::new();
    for (path, relative) in all_files {
        // Skip common local/system noise.
        if relative
            .file_name()
            .and_then(|v| v.to_str())
            .map(|name| name == ".DS_Store")
            .unwrap_or(false)
        {
            continue;
        }
        if relative.components().any(|c| {
            matches!(
                c.as_os_str().to_str(),
                Some(".git" | "__pycache__" | ".pytest_cache")
            )
        }) {
            continue;
        }

        // metadata.json is generated for each build.
        if relative == Path::new("metadata.json") {
            continue;
        }
        // Avoid recursively embedding existing packages.
        if relative.extension().and_then(|ext| ext.to_str()) == Some("aimod") {
            continue;
        }
        if output_in_context_relative
            .as_ref()
            .map(|p| p == &relative)
            .unwrap_or(false)
        {
            continue;
        }
        if output_canonical
            .as_ref()
            .map(|out| out == &path)
            .unwrap_or(false)
        {
            continue;
        }

        let ext = relative
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if !primary_model_ext.is_empty()
            && ext == primary_model_ext
            && !keep_primary_model_files.is_empty()
            && primary_model_count_by_dir
                .get(relative.parent().unwrap_or(Path::new("")))
                .copied()
                .unwrap_or(0)
                > 1
            && !keep_primary_model_files.contains(&relative)
        {
            continue;
        }

        if primary_model_ext == "onnx" && !keep_primary_model_files.is_empty() {
            if let Some(name) = relative.file_name().and_then(|v| v.to_str()) {
                if name.contains(".onnx_data") {
                    let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                    if primary_model_count_by_dir
                        .get(&dir_rel)
                        .copied()
                        .unwrap_or(0)
                        <= 1
                    {
                        files.push((path, relative));
                        continue;
                    }
                    let stems = onnx_stems_by_dir.get(&dir_rel);
                    let mut associated_stem: Option<&str> = None;
                    if let Some(stems) = stems {
                        for stem in stems {
                            let needle = format!("{}.onnx_data", stem);
                            if name.starts_with(&needle) {
                                associated_stem = Some(stem.as_str());
                                break;
                            }
                        }
                        if associated_stem.is_none() && stems.len() == 1 {
                            associated_stem = stems.first().map(|s| s.as_str());
                        }
                    }

                    if let Some(stem) = associated_stem {
                        if let Some(keep) = keep_onnx_stems_by_dir.get(&dir_rel) {
                            if !keep.contains(stem) {
                                continue;
                            }
                        } else {
                            // We can associate this data shard with a concrete ONNX model in this
                            // directory, but none of the ONNX models in this directory are kept.
                            continue;
                        }
                    }
                }
            }
        }

        files.push((path, relative));
    }

    files.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(files)
}

pub(crate) fn create_kapsl_package_from_context(
    context_path: &Path,
    model_override: Option<&Path>,
    output_override: Option<&Path>,
    project_name_override: Option<&str>,
    framework_override: Option<&str>,
    version_override: Option<&str>,
    metadata_override: Option<&serde_json::Value>,
    axes: AxisOverrides,
    interactive_metadata_setup: bool,
) -> Result<PackageKapslResponse, String> {
    let context_input = PathBuf::from(context_path);
    if !context_input.exists() {
        return Err(format!(
            "Build context does not exist: {}",
            context_input.display()
        ));
    }
    if !context_input.is_dir() {
        return Err(format!(
            "Build context must be a directory: {}",
            context_input.display()
        ));
    }
    let context_dir = context_input.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve build context {}: {}",
            context_input.display(),
            e
        )
    })?;

    let source_metadata_path = context_dir.join("metadata.json");
    let should_create_source_metadata = !source_metadata_path.exists();

    let (
        project_name_from_manifest,
        framework_from_manifest,
        version_from_manifest,
        model_file_from_manifest,
        hardware_requirements_from_manifest,
        metadata_from_manifest,
    ) = parse_context_manifest(&context_dir)?;

    let mut model_path = if let Some(model_path) = model_override {
        let candidate = if model_path.is_absolute() {
            model_path.to_path_buf()
        } else {
            let in_context = context_dir.join(model_path);
            if in_context.exists() {
                in_context
            } else {
                model_path.to_path_buf()
            }
        };
        if !candidate.exists() || !candidate.is_file() {
            return Err(format!(
                "Model file does not exist: {}",
                candidate.display()
            ));
        }
        candidate.canonicalize().map_err(|e| {
            format!(
                "Failed to resolve model file path {}: {}",
                candidate.display(),
                e
            )
        })?
    } else if let Some(model_file) = model_file_from_manifest.as_deref() {
        let candidate = context_dir.join(model_file);
        if !candidate.exists() || !candidate.is_file() {
            return Err(format!(
                "metadata.json model_file does not exist in context: {}",
                candidate.display()
            ));
        }
        candidate
    } else {
        find_model_file_in_context(&context_dir)?
    };

    if !model_path.starts_with(&context_dir) {
        return Err(format!(
            "Model path {} must be inside build context {}",
            model_path.display(),
            context_dir.display()
        ));
    }

    let mut model_file = model_path
        .strip_prefix(&context_dir)
        .map_err(|e| {
            format!(
                "Failed to compute model path relative to context {}: {}",
                model_path.display(),
                e
            )
        })?
        .to_string_lossy()
        .to_string();

    let mut project_name = project_name_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(project_name_from_manifest)
        .or_else(|| {
            context_dir
                .file_name()
                .and_then(|v| v.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "kapsl-model".to_string());

    let mut framework = framework_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(framework_from_manifest)
        .unwrap_or_else(|| infer_framework_from_model_path(&model_path));

    let mut version = version_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(version_from_manifest)
        .unwrap_or_else(|| "1.0.0".to_string());

    let mut hardware_requirements = hardware_requirements_from_manifest.unwrap_or_default();

    // Orthogonal model axes; populated from CLI overrides and/or the interactive
    // flow, else None so the loader infers them from `framework`.
    let mut format_axis: Option<String> = None;
    let mut model_type_axis: Option<String> = None;
    let mut task_axis: Option<String> = None;

    // Non-interactive: --format / --model-type / --task fill the axes.
    if axes.any() {
        let (fmt, mt, tk, fw) =
            resolve_axis_triple(infer_format_from_model_path(&model_path), axes);
        framework = fw;
        format_axis = Some(fmt);
        model_type_axis = Some(mt);
        task_axis = Some(tk);
    }

    if should_create_source_metadata && interactive_metadata_setup {
        let a = Ansi::new();
        eprintln!(
            "{}",
            a.bold("No metadata.json found. Let's create one for this model.")
        );
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        model_path = prompt_model_file_with_default(&mut reader, &context_dir, &model_file)?;
        model_file = model_path
            .strip_prefix(&context_dir)
            .map_err(|e| {
                format!(
                    "Failed to compute model path relative to context {}: {}",
                    model_path.display(),
                    e
                )
            })?
            .to_string_lossy()
            .to_string();

        project_name = prompt_non_empty_with_default(&mut reader, "Project name", &project_name)?;
        let format_default = format_axis
            .as_deref()
            .unwrap_or_else(|| infer_format_from_model_path(&model_path));
        let format =
            prompt_select_with_default(&mut reader, "Format", FORMAT_OPTIONS, format_default)?;
        let model_type_default = model_type_axis
            .as_deref()
            .unwrap_or_else(|| default_model_type_for_format(&format));
        let model_type = prompt_select_with_default(
            &mut reader,
            "Model type",
            MODEL_TYPE_OPTIONS,
            model_type_default,
        )?;
        let task = prompt_task_for_model_type(&mut reader, &model_type, task_axis.as_deref())?;
        framework = legacy_framework_for(&format, &model_type, &task);
        format_axis = Some(format);
        model_type_axis = Some(model_type);
        task_axis = Some(task);
        version = prompt_non_empty_with_default(&mut reader, "Version", &version)?;
        hardware_requirements.preferred_provider = prompt_provider_with_default(
            &mut reader,
            hardware_requirements.preferred_provider.as_deref(),
        )?;
        eprintln!();
    }

    let package = move || -> Result<PackageKapslResponse, String> {
        let output_path =
            normalize_output_path_for_context(&context_dir, output_override, &project_name);
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create parent directory {}: {}",
                        parent.display(),
                        e
                    )
                })?;
            }
        }

        let created_at = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System clock error: {}", e))?
            .as_secs()
            .to_string();

        let metadata_value = metadata_override
            .cloned()
            .or(metadata_from_manifest)
            .map(|value| {
                serde_yaml::to_value(value)
                    .map_err(|e| format!("Failed to convert metadata payload: {}", e))
            })
            .transpose()?;

        let primary_model_ext = model_path
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();

        let mut referenced_files = HashSet::new();
        referenced_files.insert(PathBuf::from(&model_file));
        if let Some(metadata) = metadata_value.as_ref() {
            collect_existing_file_references_from_metadata(
                &context_dir,
                metadata,
                &mut referenced_files,
            );
        }

        let mut keep_primary_model_files: HashSet<PathBuf> = HashSet::new();
        if !primary_model_ext.is_empty() {
            for rel in &referenced_files {
                let ext = rel
                    .extension()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase();
                if !ext.is_empty() && ext == primary_model_ext {
                    keep_primary_model_files.insert(rel.to_path_buf());
                }
            }
        }

        let manifest = Manifest {
            project_name: project_name.clone(),
            framework: framework.clone(),
            version: version.clone(),
            created_at,
            model_file,
            format: format_axis,
            model_type: model_type_axis,
            task: task_axis,
            metadata: metadata_value,
            hardware_requirements,
            cron_jobs: Vec::new(),
        };
        EngineKind::validate(&manifest)?;

        let output_file = File::create(&output_path).map_err(|e| {
            format!(
                "Failed to create output package {}: {}",
                output_path.display(),
                e
            )
        })?;
        let encoder = GzEncoder::new(
            BufWriter::with_capacity(8 << 20, output_file),
            Compression::fast(),
        );
        let mut archive = Builder::new(encoder);

        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| format!("Failed to encode metadata.json: {}", e))?;
        append_tar_bytes_entry(&mut archive, "metadata.json", &manifest_bytes)?;

        for (absolute, relative) in collect_context_files(
            &context_dir,
            &output_path,
            &primary_model_ext,
            &keep_primary_model_files,
        )? {
            archive
                .append_path_with_name(&absolute, &relative)
                .map_err(|e| format!("Failed to add {} to archive: {}", relative.display(), e))?;
        }

        let encoder = archive
            .into_inner()
            .map_err(|e| format!("Failed to finalize tar archive: {}", e))?;
        let mut buf_writer = encoder
            .finish()
            .map_err(|e| format!("Failed to finalize gzip stream: {}", e))?;
        buf_writer
            .flush()
            .map_err(|e| format!("Failed to flush output package: {}", e))?;

        let created_metadata_path = if should_create_source_metadata {
            create_source_metadata_if_missing(&source_metadata_path, &manifest_bytes)?
        } else {
            None
        };

        let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);
        Ok(PackageKapslResponse {
            status: "ok".to_string(),
            kapsl_path: absolute_output_path.to_string_lossy().to_string(),
            project_name,
            framework,
            version,
            metadata_path: created_metadata_path.map(|path| path.to_string_lossy().to_string()),
        })
    };

    // Prompts (above) ran without a spinner; show it here for the packaging work.
    if interactive_metadata_setup {
        run_with_loading("Building package", package)
    } else {
        package()
    }
}

pub(crate) fn push_kapsl_to_placeholder_remote(
    request: &PushKapslRequest,
) -> Result<PushKapslResponse, String> {
    let target = parse_model_target(&request.target)?;
    let input_path = PathBuf::from(request.kapsl_path.trim());
    if !input_path.exists() {
        return Err(format!(
            ".aimod file does not exist: {}",
            input_path.display()
        ));
    }
    if !input_path.is_file() {
        return Err(format!(
            "Provided .aimod path is not a file: {}",
            input_path.display()
        ));
    }

    if input_path.extension().and_then(|ext| ext.to_str()) != Some("aimod") {
        return Err(format!(
            "Push expects a .aimod file, got: {}",
            input_path.display()
        ));
    }

    let absolute_path = input_path.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve kapsl path {}: {}",
            input_path.display(),
            e
        )
    })?;
    let filename = absolute_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid kapsl filename: {}", absolute_path.display()))?
        .to_string();

    let remote_url = resolved_remote_url(request.remote_url.as_deref());
    if is_oci_remote_url(&remote_url) {
        return push_kapsl_to_oci_remote(&remote_url, &absolute_path, &target, &filename);
    }
    let artifact_url = artifact_url_for_remote(&remote_url, &target);
    let mut remote_token = resolved_remote_token(&remote_url, request.remote_token.as_deref());
    let request_has_explicit_token = request
        .remote_token
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty());

    let (mirrored_path, bytes_uploaded) = if is_default_placeholder_remote(&remote_url) {
        let mirrored_path = placeholder_remote_artifact_path(&target);
        let parent_dir = mirrored_path.parent().ok_or_else(|| {
            format!(
                "Invalid placeholder storage path: {}",
                mirrored_path.display()
            )
        })?;
        fs::create_dir_all(parent_dir).map_err(|e| {
            format!(
                "Failed to prepare placeholder remote directory {}: {}",
                parent_dir.display(),
                e
            )
        })?;
        let bytes_uploaded =
            stage_link_or_copy_file(&absolute_path, &mirrored_path, "placeholder-push").map_err(
                |e| {
                    format!(
                        "Failed to mirror .aimod into placeholder remote {}: {}",
                        mirrored_path.display(),
                        e
                    )
                },
            )?;

        (mirrored_path.to_string_lossy().to_string(), bytes_uploaded)
    } else {
        let bytes_uploaded =
            match push_kapsl_to_http_remote(&artifact_url, &absolute_path, remote_token.as_deref())
            {
                Ok(bytes) => bytes,
                Err(http_error) => {
                    if maybe_auto_login_for_remote(
                        &remote_url,
                        request_has_explicit_token,
                        request.interactive_login,
                        &mut remote_token,
                        &http_error,
                    )? {
                        push_kapsl_to_http_remote(
                            &artifact_url,
                            &absolute_path,
                            remote_token.as_deref(),
                        )
                        .map_err(|e| e.message)?
                    } else {
                        return Err(http_error.message);
                    }
                }
            };
        // For real remote backends there is no local mirror path.
        (artifact_url.clone(), bytes_uploaded)
    };

    Ok(PushKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        mirrored_path,
        bytes_uploaded,
        manifest_digest: None,
    })
}

pub(crate) fn pull_kapsl_from_placeholder_remote(
    request: &PullKapslRequest,
) -> Result<PullKapslResponse, String> {
    let target = parse_model_target(&request.target)?;
    let filename = format!("{}.aimod", target.model);

    let destination_dir = request
        .destination_dir
        .as_ref()
        .map(|path| PathBuf::from(path.trim()))
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&destination_dir).map_err(|e| {
        format!(
            "Failed to create destination directory {}: {}",
            destination_dir.display(),
            e
        )
    })?;

    let remote_url = resolved_remote_url(request.remote_url.as_deref());
    if is_oci_remote_url(&remote_url) {
        return pull_kapsl_from_oci_remote(
            &remote_url,
            &target,
            request.reference.as_deref(),
            &destination_dir,
        );
    }
    let output_path = destination_dir.join(&filename);
    let artifact_url = artifact_url_for_remote(&remote_url, &target);
    let mut remote_token = resolved_remote_token(&remote_url, request.remote_token.as_deref());
    let request_has_explicit_token = request
        .remote_token
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty());
    let bytes_downloaded = if is_default_placeholder_remote(&remote_url) {
        let mirrored_path = placeholder_remote_artifact_path(&target);
        if !mirrored_path.exists() {
            return Err(format!(
                "Placeholder remote artifact not found: {} for target {}. Push the package first or set KAPSL_REMOTE_PLACEHOLDER_DIR.",
                mirrored_path.display(),
                target.as_string()
            ));
        }
        stage_link_or_copy_file(&mirrored_path, &output_path, "placeholder-pull").map_err(|e| {
            format!(
                "Failed to pull placeholder remote artifact to {}: {}",
                output_path.display(),
                e
            )
        })?
    } else {
        match pull_kapsl_from_http_remote(&artifact_url, remote_token.as_deref(), &output_path) {
            Ok(bytes_downloaded) => bytes_downloaded,
            Err(http_error) => {
                if maybe_auto_login_for_remote(
                    &remote_url,
                    request_has_explicit_token,
                    request.interactive_login,
                    &mut remote_token,
                    &http_error,
                )? {
                    pull_kapsl_from_http_remote(
                        &artifact_url,
                        remote_token.as_deref(),
                        &output_path,
                    )
                    .map_err(|e| e.message)?
                } else {
                    return Err(http_error.message);
                }
            }
        }
    };

    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);

    Ok(PullKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        bytes_downloaded,
    })
}
