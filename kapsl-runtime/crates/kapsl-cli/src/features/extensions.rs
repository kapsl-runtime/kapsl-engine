use super::*;

pub(crate) fn extension_marketplace_url(custom_url: Option<&str>) -> String {
    if let Some(url) = custom_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(url) = optional_env_var(EXTENSION_MARKETPLACE_URL_ENV) {
        return url;
    }

    EXTENSION_MARKETPLACE_URL.to_string()
}

pub(crate) fn is_valid_extension_id(extension_id: &str) -> bool {
    !extension_id.trim().is_empty()
        && extension_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
}

pub(crate) fn fetch_extension_marketplace(
    query: Option<&str>,
    marketplace_url: Option<&str>,
) -> Result<serde_json::Value, String> {
    let marketplace_url = extension_marketplace_url(marketplace_url);
    let agent = native_tls_http_agent();
    let mut request = agent.get(&marketplace_url);

    if let Some(q) = query {
        let trimmed = q.trim();
        if !trimmed.is_empty() {
            request = request.query("q", trimmed);
        }
    }

    let mut response = request.call().map_err(|e| {
        format!(
            "Failed to query extension marketplace {}: {}",
            marketplace_url,
            format_remote_http_error(e)
        )
    })?;

    let raw = response.body_mut().read_to_string().map_err(|e| {
        format!(
            "Failed to read extension marketplace response from {}: {}",
            marketplace_url, e
        )
    })?;

    serde_json::from_str::<serde_json::Value>(&raw).map_err(|e| {
        format!(
            "Failed to parse extension marketplace response as JSON from {}: {}",
            marketplace_url, e
        )
    })
}

pub(crate) fn collect_extension_manifest_dirs(
    dir: &Path,
    matches: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| {
        format!(
            "Failed to inspect extracted extension archive directory {}: {}",
            dir.display(),
            e
        )
    })? {
        let entry = entry.map_err(|e| format!("Failed to read archive directory entry: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            collect_extension_manifest_dirs(&path, matches)?;
            continue;
        }

        if path.file_name().and_then(|n| n.to_str()) == Some("rag-extension.toml") {
            if let Some(parent) = path.parent() {
                matches.push(parent.to_path_buf());
            }
        }
    }

    Ok(())
}

pub(crate) fn find_extension_manifest_root(extract_dir: &Path) -> Result<PathBuf, String> {
    let mut matches = Vec::new();
    collect_extension_manifest_dirs(extract_dir, &mut matches)?;

    if matches.is_empty() {
        return Err(format!(
            "Marketplace archive did not contain rag-extension.toml under {}",
            extract_dir.display()
        ));
    }

    if matches.len() > 1 {
        return Err(format!(
            "Marketplace archive contained multiple extension manifests under {}",
            extract_dir.display()
        ));
    }

    Ok(matches.remove(0))
}

pub(crate) fn unpack_marketplace_archive(
    archive_bytes: &[u8],
    target_dir: &Path,
) -> Result<(), String> {
    let decoder = GzDecoder::new(Cursor::new(archive_bytes));
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|e| format!("Failed to read extension marketplace archive: {}", e))?;

    for entry in entries {
        let mut entry =
            entry.map_err(|e| format!("Failed to read extension archive entry: {}", e))?;
        let unpacked = entry.unpack_in(target_dir).map_err(|e| {
            format!(
                "Failed to unpack extension archive into {}: {}",
                target_dir.display(),
                e
            )
        })?;
        if !unpacked {
            return Err("Extension archive contains invalid paths".to_string());
        }
    }

    Ok(())
}

pub(crate) fn install_extension_from_marketplace(
    registry: &ExtensionRegistry,
    extension_id: &str,
    marketplace_url: Option<&str>,
) -> Result<InstalledExtension, String> {
    let extension_id = extension_id.trim();
    if !is_valid_extension_id(extension_id) {
        return Err(format!("Invalid extension_id `{}`", extension_id));
    }

    let marketplace_url = extension_marketplace_url(marketplace_url);
    let download_url = format!(
        "{}/{}/download",
        marketplace_url.trim_end_matches('/'),
        extension_id
    );

    let agent = native_tls_http_agent();
    let mut response = agent.get(&download_url).call().map_err(|e| {
        format!(
            "Failed to download extension `{}` from marketplace {}: {}",
            extension_id,
            marketplace_url,
            format_remote_http_error(e)
        )
    })?;

    let archive_bytes = response.body_mut().read_to_vec().map_err(|e| {
        format!(
            "Failed to read downloaded extension `{}` archive from {}: {}",
            extension_id, download_url, e
        )
    })?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let temp_dir = std::env::temp_dir().join(format!(
        "kapsl-extension-marketplace-{}-{}",
        std::process::id(),
        timestamp
    ));
    fs::create_dir_all(&temp_dir).map_err(|e| {
        format!(
            "Failed to prepare temporary extension directory {}: {}",
            temp_dir.display(),
            e
        )
    })?;

    let install_result = (|| {
        unpack_marketplace_archive(&archive_bytes, &temp_dir)?;
        let extracted_root = find_extension_manifest_root(&temp_dir)?;
        registry
            .install_from_dir(&extracted_root)
            .map_err(|e| e.to_string())
    })();

    let _ = fs::remove_dir_all(&temp_dir);
    install_result
}

pub(crate) fn execute_extension_command(args: ExtensionCommandArgs) -> Result<(), DynError> {
    match args.command {
        ExtensionSubcommand::Install(args) => execute_extension_install_command(args),
    }
}

fn execute_extension_install_command(args: ExtensionInstallCommandArgs) -> Result<(), DynError> {
    let extension_id = args.extension_id.trim();
    if !is_valid_extension_id(extension_id) {
        return Err(dyn_error_from_message(format!(
            "Invalid extension ID `{}`. Use letters, numbers, dots, underscores, or hyphens.",
            args.extension_id
        )));
    }

    let base_url = match &args.http_url {
        Some(url) => url.trim_end_matches('/').to_string(),
        None => format!("http://{}:{}", args.http_host, args.http_port),
    };
    let install_url = format!("{}/api/extensions/install", base_url);
    let timeout = std::time::Duration::from_millis(args.timeout_ms.max(1));
    let agent_config = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .build();
    let agent: ureq::Agent = agent_config.into();

    let mut payload = serde_json::json!({ "extension_id": extension_id });
    if let Some(marketplace_url) = args.marketplace_url.as_deref() {
        let marketplace_url = marketplace_url.trim();
        if !marketplace_url.is_empty() {
            payload["marketplace_url"] = serde_json::Value::String(marketplace_url.to_string());
        }
    }
    let payload = serde_json::to_string(&payload)
        .map_err(|e| dyn_error_from_message(format!("Failed to serialize request: {}", e)))?;

    let mut request = agent
        .post(&install_url)
        .header("Content-Type", "application/json");
    if let Some(token) = args.auth_token.as_deref() {
        let token = token.trim();
        if !token.is_empty() {
            request = request.header("Authorization", &format!("Bearer {}", token));
        }
    }

    let mut response = request.send(payload).map_err(|error| {
        dyn_error_from_message(format!(
            "Kapsl Engine could not install `{}` via {}: {}",
            extension_id,
            install_url,
            format_remote_http_error(error)
        ))
    })?;
    let body = response
        .body_mut()
        .read_to_string()
        .unwrap_or_else(|_| String::new());
    let installed = serde_json::from_str::<serde_json::Value>(&body).unwrap_or_default();
    let manifest = installed
        .get("extension")
        .and_then(|extension| extension.get("manifest"));
    let display_name = manifest
        .and_then(|manifest| manifest.get("name"))
        .and_then(|name| name.as_str())
        .unwrap_or(extension_id);
    let version = manifest
        .and_then(|manifest| manifest.get("version"))
        .and_then(|version| version.as_str());

    let a = Ansi::new();
    eprintln!();
    eprintln!(
        "  {}  {}{}",
        a.green("✓"),
        a.bold(display_name),
        version
            .map(|version| format!(" {}", a.dim(&format!("v{}", version))))
            .unwrap_or_default()
    );
    eprintln!("     {}", a.dim("Installed in the running Kapsl Engine"));
    eprintln!();
    Ok(())
}

#[cfg(test)]
mod command_tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn extension_install_command_parses_hub_extension_id() {
        let cli = Cli::try_parse_from([
            "kapsl",
            "extension",
            "install",
            "connector.s3",
            "--http-port",
            "9195",
        ])
        .expect("parse extension install command");

        assert!(matches!(
            cli.command,
            Some(KapslCommand::Extension(ExtensionCommandArgs {
                command: ExtensionSubcommand::Install(ExtensionInstallCommandArgs {
                    extension_id,
                    http_port: 9195,
                    ..
                })
            })) if extension_id == "connector.s3"
        ));
    }
}
