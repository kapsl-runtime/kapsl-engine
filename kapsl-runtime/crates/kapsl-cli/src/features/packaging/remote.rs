use super::*;

pub(crate) fn push_kapsl_to_remote(
    request: &PushKapslRequest,
) -> Result<PushKapslResponse, String> {
    push_kapsl_to_remote_with_transport(request, &HttpArtifactTransport::default())
}

fn transfer_with_auth_retry(
    remote_url: &str,
    request_has_explicit_token: bool,
    interactive_login: bool,
    remote_token: &mut Option<String>,
    mut operation: impl FnMut(Option<&str>) -> Result<u64, RemoteHttpRequestError>,
) -> Result<u64, String> {
    match operation(remote_token.as_deref()) {
        Ok(bytes) => Ok(bytes),
        Err(http_error) => {
            if maybe_auto_login_for_remote(
                remote_url,
                request_has_explicit_token,
                interactive_login,
                remote_token,
                &http_error,
            )? {
                operation(remote_token.as_deref()).map_err(|error| error.message)
            } else {
                Err(http_error.message)
            }
        }
    }
}

fn push_kapsl_to_remote_with_transport(
    request: &PushKapslRequest,
    transport: &dyn ArtifactTransport,
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
    let remote_url = resolved_remote_url(request.remote_url.as_deref());
    let artifact_url = artifact_url_for_remote(&remote_url, &target);
    let mut remote_token = resolved_remote_token(&remote_url, request.remote_token.as_deref());
    let request_has_explicit_token = request
        .remote_token
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty());

    let bytes_uploaded = transfer_with_auth_retry(
        &remote_url,
        request_has_explicit_token,
        request.interactive_login,
        &mut remote_token,
        |authorization_header| {
            transport.upload(&artifact_url, &absolute_path, authorization_header)
        },
    )?;

    Ok(PushKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        bytes_uploaded,
    })
}

pub(crate) fn pull_kapsl_from_remote(
    request: &PullKapslRequest,
) -> Result<PullKapslResponse, String> {
    pull_kapsl_from_remote_with_transport(request, &HttpArtifactTransport::default())
}

fn pull_kapsl_from_remote_with_transport(
    request: &PullKapslRequest,
    transport: &dyn ArtifactTransport,
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
    let output_path = destination_dir.join(&filename);
    let artifact_url = artifact_url_for_remote(&remote_url, &target);
    let mut remote_token = resolved_remote_token(&remote_url, request.remote_token.as_deref());
    let request_has_explicit_token = request
        .remote_token
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty());
    let bytes_downloaded = transfer_with_auth_retry(
        &remote_url,
        request_has_explicit_token,
        request.interactive_login,
        &mut remote_token,
        |authorization_header| {
            transport.download(&artifact_url, authorization_header, &output_path)
        },
    )?;

    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);

    Ok(PullKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        bytes_downloaded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingTransport {
        uploads: Mutex<Vec<(String, PathBuf, Option<String>)>>,
    }

    impl ArtifactTransport for RecordingTransport {
        fn fetch_inventory(
            &self,
            _remote_url: &str,
            _authorization_header: Option<&str>,
        ) -> Result<RemoteArtifactInventoryResponse, String> {
            unreachable!()
        }

        fn upload(
            &self,
            artifact_url: &str,
            source_path: &Path,
            authorization_header: Option<&str>,
        ) -> Result<u64, RemoteHttpRequestError> {
            self.uploads.lock().expect("record upload").push((
                artifact_url.to_string(),
                source_path.to_path_buf(),
                authorization_header.map(str::to_string),
            ));
            Ok(fs::metadata(source_path).expect("package metadata").len())
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
    fn push_orchestration_uses_the_injected_transport() {
        let temporary = TempDirGuard::new("kapsl-injected-artifact-transport")
            .expect("create temporary directory");
        let package_path = temporary.path().join("model.aimod");
        fs::write(&package_path, b"package").expect("write package");
        let transport = RecordingTransport::default();
        let request = PushKapslRequest {
            kapsl_path: package_path.to_string_lossy().to_string(),
            target: "team/model:latest".to_string(),
            remote_url: Some("https://example.invalid/api/v1".to_string()),
            remote_token: Some("secret".to_string()),
            interactive_login: false,
        };

        let response = push_kapsl_to_remote_with_transport(&request, &transport)
            .expect("push through injected transport");

        assert_eq!(response.bytes_uploaded, 7);
        let uploads = transport.uploads.lock().expect("read uploads");
        assert_eq!(uploads.len(), 1);
        assert_eq!(
            uploads[0].0,
            "https://example.invalid/api/v1/aimod/team/model:latest"
        );
        assert_eq!(uploads[0].2.as_deref(), Some("Bearer secret"));
    }
}
