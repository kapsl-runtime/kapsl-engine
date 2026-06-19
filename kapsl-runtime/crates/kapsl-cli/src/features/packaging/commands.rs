use super::*;

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
