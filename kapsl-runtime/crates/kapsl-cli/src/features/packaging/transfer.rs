use super::*;
use crate::features::http_client::{
    format_remote_http_error, http_agent_for_transfer, native_tls_http_agent,
};

/// Error returned by an artifact transport operation.
#[derive(Debug)]
pub(crate) struct RemoteHttpRequestError {
    pub(crate) status_code: Option<u16>,
    pub(crate) message: String,
}

impl RemoteHttpRequestError {
    fn local(message: impl Into<String>) -> Self {
        Self {
            status_code: None,
            message: message.into(),
        }
    }

    fn from_ureq(context: impl std::fmt::Display, error: ureq::Error) -> Self {
        let status_code = match &error {
            ureq::Error::StatusCode(code) => Some(*code),
            _ => None,
        };
        Self {
            status_code,
            message: format!("{context}: {}", format_remote_http_error(error)),
        }
    }
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

fn remote_inventory_url_for_remote(remote_url: &str) -> String {
    format!(
        "{}/kapsl/repositories/current/models",
        remote_url.trim_end_matches('/')
    )
}

/// Transport boundary used by package orchestration.
///
/// The default implementation speaks HTTP. Keeping orchestration dependent on
/// this narrow interface allows tests and future transports to be injected
/// without passing `ureq` configuration through command handlers.
pub(crate) trait ArtifactTransport {
    fn fetch_inventory(
        &self,
        remote_url: &str,
        authorization_header: Option<&str>,
    ) -> Result<RemoteArtifactInventoryResponse, String>;

    fn upload(
        &self,
        artifact_url: &str,
        source_path: &Path,
        authorization_header: Option<&str>,
    ) -> Result<u64, RemoteHttpRequestError>;

    fn download(
        &self,
        artifact_url: &str,
        authorization_header: Option<&str>,
        output_path: &Path,
    ) -> Result<u64, RemoteHttpRequestError>;
}

/// Reusable HTTP implementation of [`ArtifactTransport`].
pub(crate) struct HttpArtifactTransport {
    api_agent: ureq::Agent,
    transfer_agent: ureq::Agent,
}

impl HttpArtifactTransport {
    pub(crate) fn new(api_agent: ureq::Agent, transfer_agent: ureq::Agent) -> Self {
        Self {
            api_agent,
            transfer_agent,
        }
    }

    /// Requests a presigned upload URL when the remote supports that protocol.
    fn request_presigned_upload_url(
        &self,
        artifact_url: &str,
        authorization_header: Option<&str>,
    ) -> Result<Option<PresignedUploadResponse>, RemoteHttpRequestError> {
        let upload_endpoint = format!("{artifact_url}/upload");
        let mut request = self
            .transfer_agent
            .post(&upload_endpoint)
            .header("Content-Type", "application/json");
        if let Some(header) = authorization_header {
            request = request.header("Authorization", header);
        }
        match request.send_empty() {
            Ok(response) => {
                let body = response.into_body().read_to_string().map_err(|error| {
                    RemoteHttpRequestError::local(format!(
                        "Failed to read presigned upload response: {error}"
                    ))
                })?;
                let parsed: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
                    RemoteHttpRequestError::local(format!(
                        "Invalid presigned upload response JSON: {error}"
                    ))
                })?;
                let upload_url = parsed["upload_url"]
                    .as_str()
                    .ok_or_else(|| {
                        RemoteHttpRequestError::local(
                            "Presigned upload response missing 'upload_url' field",
                        )
                    })?
                    .to_string();
                let method = parsed["method"].as_str().unwrap_or("PUT").to_string();
                let headers = parsed["headers"]
                    .as_object()
                    .map(|object| {
                        object
                            .iter()
                            .filter_map(|(key, value)| {
                                value.as_str().map(|value| (key.clone(), value.to_string()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(Some(PresignedUploadResponse {
                    upload_url,
                    method,
                    headers,
                }))
            }
            Err(ureq::Error::StatusCode(404 | 501)) => Ok(None),
            Err(error) => Err(RemoteHttpRequestError::from_ureq(
                format!("Failed to request presigned upload URL from {upload_endpoint}"),
                error,
            )),
        }
    }

    /// Requests a presigned download URL when the remote supports that protocol.
    fn request_presigned_download_url(
        &self,
        artifact_url: &str,
        authorization_header: Option<&str>,
    ) -> Result<Option<String>, RemoteHttpRequestError> {
        let download_endpoint = format!("{artifact_url}/download");
        let mut request = self
            .transfer_agent
            .post(&download_endpoint)
            .header("Content-Type", "application/json");
        if let Some(header) = authorization_header {
            request = request.header("Authorization", header);
        }
        match request.send_empty() {
            Ok(response) => {
                let body = response.into_body().read_to_string().map_err(|error| {
                    RemoteHttpRequestError::local(format!(
                        "Failed to read presigned download response: {error}"
                    ))
                })?;
                let parsed: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
                    RemoteHttpRequestError::local(format!(
                        "Invalid presigned download response JSON: {error}"
                    ))
                })?;
                let download_url = parsed["download_url"]
                    .as_str()
                    .ok_or_else(|| {
                        RemoteHttpRequestError::local(
                            "Presigned download response missing 'download_url' field",
                        )
                    })?
                    .to_string();
                Ok(Some(download_url))
            }
            Err(ureq::Error::StatusCode(404 | 501)) => Ok(None),
            Err(error) => Err(RemoteHttpRequestError::from_ureq(
                format!("Failed to request presigned download URL from {download_endpoint}"),
                error,
            )),
        }
    }

    fn download_to_file(
        &self,
        url: &str,
        output_path: &Path,
        authorization_header: Option<&str>,
    ) -> Result<u64, RemoteHttpRequestError> {
        let mut request = self.transfer_agent.get(url);
        if let Some(header) = authorization_header {
            request = request.header("Authorization", header);
        }
        let mut response = request.call().map_err(|error| {
            RemoteHttpRequestError::from_ureq(
                format!("Failed to download .aimod from {url}"),
                error,
            )
        })?;

        let staged_path = staged_output_path(output_path, "download");
        let write_result = (|| -> Result<u64, RemoteHttpRequestError> {
            let file = File::create(&staged_path).map_err(|error| {
                RemoteHttpRequestError::local(format!(
                    "Failed to create staging file for pull {} -> {}: {}",
                    url,
                    staged_path.display(),
                    error
                ))
            })?;
            let mut writer = BufWriter::new(file);
            let mut reader = response.body_mut().as_reader();
            let bytes_downloaded = std::io::copy(&mut reader, &mut writer).map_err(|error| {
                RemoteHttpRequestError::local(format!(
                    "Failed to stream .aimod response body from {} to {}: {}",
                    url,
                    staged_path.display(),
                    error
                ))
            })?;
            writer.flush().map_err(|error| {
                RemoteHttpRequestError::local(format!(
                    "Failed to flush pulled .aimod to {}: {}",
                    staged_path.display(),
                    error
                ))
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

        replace_output_file(&staged_path, output_path).map_err(|error| {
            let _ = fs::remove_file(&staged_path);
            RemoteHttpRequestError::local(format!(
                "Failed to finalize pulled .aimod {} -> {}: {}",
                staged_path.display(),
                output_path.display(),
                error
            ))
        })?;

        Ok(bytes_downloaded)
    }
}

impl Default for HttpArtifactTransport {
    fn default() -> Self {
        Self::new(native_tls_http_agent(), http_agent_for_transfer())
    }
}

impl ArtifactTransport for HttpArtifactTransport {
    fn fetch_inventory(
        &self,
        remote_url: &str,
        authorization_header: Option<&str>,
    ) -> Result<RemoteArtifactInventoryResponse, String> {
        let inventory_url = remote_inventory_url_for_remote(remote_url);
        let mut request = self
            .api_agent
            .get(&inventory_url)
            .header("Accept", "application/json");
        if let Some(header) = authorization_header {
            request = request.header("Authorization", header);
        }

        let mut response = request.call().map_err(|error| match error {
            ureq::Error::StatusCode(401 | 403) => format!(
                "Remote artifact inventory requires authentication for {remote_url}. Run `kapsl login --remote-url {remote_url}` first."
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
        serde_json::from_str(&body).map_err(|error| {
            format!(
                "Failed to parse remote artifact inventory response from {}: {}",
                inventory_url, error
            )
        })
    }

    fn upload(
        &self,
        artifact_url: &str,
        source_path: &Path,
        authorization_header: Option<&str>,
    ) -> Result<u64, RemoteHttpRequestError> {
        let file_size = fs::metadata(source_path).map_err(|error| {
            RemoteHttpRequestError::local(format!(
                "Failed to read .aimod metadata for upload {}: {}",
                source_path.display(),
                error
            ))
        })?;

        if let Some(presigned) =
            self.request_presigned_upload_url(artifact_url, authorization_header)?
        {
            eprintln!(
                "  {}",
                Ansi::new().dim(&format!("Uploading {} bytes...", file_size.len()))
            );
            let file = File::open(source_path).map_err(|error| {
                RemoteHttpRequestError::local(format!(
                    "Failed to open .aimod for upload {}: {}",
                    source_path.display(),
                    error
                ))
            })?;
            let mut request = if presigned.method == "PUT" {
                self.transfer_agent.put(&presigned.upload_url)
            } else {
                self.transfer_agent.post(&presigned.upload_url)
            };
            request = request
                .header("Content-Type", "application/octet-stream")
                .header("Content-Length", &file_size.len().to_string());
            for (key, value) in &presigned.headers {
                request = request.header(key, value);
            }
            request.send(file).map_err(|error| {
                RemoteHttpRequestError::from_ureq("Failed to upload .aimod to presigned URL", error)
            })?;
            return Ok(file_size.len());
        }

        let file = File::open(source_path).map_err(|error| {
            RemoteHttpRequestError::local(format!(
                "Failed to open .aimod for upload {}: {}",
                source_path.display(),
                error
            ))
        })?;
        let mut request = self
            .transfer_agent
            .put(artifact_url)
            .header("Content-Type", "application/octet-stream")
            .header("Content-Length", &file_size.len().to_string());
        if let Some(header) = authorization_header {
            request = request.header("Authorization", header);
        }
        request.send(file).map_err(|error| {
            RemoteHttpRequestError::from_ureq(
                format!("Failed to upload .aimod to remote backend {artifact_url}"),
                error,
            )
        })?;
        Ok(file_size.len())
    }

    fn download(
        &self,
        artifact_url: &str,
        authorization_header: Option<&str>,
        output_path: &Path,
    ) -> Result<u64, RemoteHttpRequestError> {
        if let Some(download_url) =
            self.request_presigned_download_url(artifact_url, authorization_header)?
        {
            eprintln!("  {}", Ansi::new().dim("Downloading..."));
            return self.download_to_file(&download_url, output_path, None);
        }
        self.download_to_file(artifact_url, output_path, authorization_header)
    }
}

struct PresignedUploadResponse {
    upload_url: String,
    method: String,
    headers: Vec<(String, String)>,
}

pub(crate) fn fetch_remote_artifact_inventory(
    custom_remote_url: Option<&str>,
) -> Result<RuntimeRemoteArtifactInventoryResponse, String> {
    fetch_remote_artifact_inventory_with_transport(
        custom_remote_url,
        &HttpArtifactTransport::default(),
    )
}

fn fetch_remote_artifact_inventory_with_transport(
    custom_remote_url: Option<&str>,
    transport: &dyn ArtifactTransport,
) -> Result<RuntimeRemoteArtifactInventoryResponse, String> {
    let remote_url = resolved_login_remote_url(custom_remote_url);
    let authorization_header = resolved_remote_token(&remote_url, None);
    let payload = transport.fetch_inventory(&remote_url, authorization_header.as_deref())?;
    Ok(RuntimeRemoteArtifactInventoryResponse {
        status: payload.status,
        remote_url,
        repo: payload.repo,
        available_repos: payload.available_repos,
        models: payload.models,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct InventoryTransport;

    impl ArtifactTransport for InventoryTransport {
        fn fetch_inventory(
            &self,
            _remote_url: &str,
            _authorization_header: Option<&str>,
        ) -> Result<RemoteArtifactInventoryResponse, String> {
            Ok(RemoteArtifactInventoryResponse {
                status: "ok".to_string(),
                repo: "team".to_string(),
                available_repos: vec!["team".to_string()],
                models: Vec::new(),
            })
        }

        fn upload(
            &self,
            _artifact_url: &str,
            _source_path: &Path,
            _authorization_header: Option<&str>,
        ) -> Result<u64, RemoteHttpRequestError> {
            unreachable!()
        }

        fn download(
            &self,
            _artifact_url: &str,
            _authorization_header: Option<&str>,
            _output_path: &Path,
        ) -> Result<u64, RemoteHttpRequestError> {
            unreachable!()
        }
    }

    #[test]
    fn inventory_wrapper_accepts_an_injected_transport() {
        let response = fetch_remote_artifact_inventory_with_transport(
            Some("https://example.invalid/api/v1"),
            &InventoryTransport,
        )
        .expect("fetch injected inventory");

        assert_eq!(response.remote_url, "https://example.invalid/api/v1");
        assert_eq!(response.repo, "team");
    }
}
