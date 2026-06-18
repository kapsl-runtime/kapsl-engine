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
