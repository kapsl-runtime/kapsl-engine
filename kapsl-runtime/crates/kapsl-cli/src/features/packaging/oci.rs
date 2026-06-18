use super::*;

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
