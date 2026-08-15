use super::*;

pub(crate) fn push_kapsl_to_remote(
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
    let remote_url = resolved_remote_url(request.remote_url.as_deref());
    let artifact_url = artifact_url_for_remote(&remote_url, &target);
    let mut remote_token = resolved_remote_token(&remote_url, request.remote_token.as_deref());
    let request_has_explicit_token = request
        .remote_token
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty());

    let bytes_uploaded =
        match push_kapsl_to_http_remote(&artifact_url, &absolute_path, remote_token.as_deref()) {
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
    let bytes_downloaded =
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
