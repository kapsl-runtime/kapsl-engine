use super::*;

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
